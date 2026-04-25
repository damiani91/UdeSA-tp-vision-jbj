"""Training multi-task para clasificadores pants/tops."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.csv_dataset import LABEL_MISSING, CSVImageDataset
from src.classification.model import MultiTaskFashionClassifier

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def multi_task_loss(
    outputs: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    head_weights: dict[str, float],
    class_weights: Optional[dict[str, torch.Tensor]] = None,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Suma ponderada de CrossEntropyLoss por head, ignora LABEL_MISSING.

    Args:
        outputs: {head_name: logits (B, C)}.
        labels: {head_name: targets (B,)} con -1 para missing.
        head_weights: Peso por head en la suma final.
        class_weights: Si se provee, pesos por clase para balancear.
        label_smoothing: Factor de label smoothing.

    Returns:
        (loss_total, dict per-head loss).
    """
    total = torch.zeros(1, device=next(iter(outputs.values())).device, requires_grad=True)
    total = total.sum()
    per_head: dict[str, float] = {}
    any_valid = False

    for name, logits in outputs.items():
        target = labels[name].to(logits.device)
        mask = target != LABEL_MISSING
        if mask.sum() == 0:
            per_head[name] = 0.0
            continue
        cw = None
        if class_weights is not None and name in class_weights:
            cw = class_weights[name].to(logits.device)
        loss = F.cross_entropy(
            logits[mask],
            target[mask],
            weight=cw,
            label_smoothing=label_smoothing,
        )
        w = float(head_weights.get(name, 1.0))
        total = total + w * loss
        per_head[name] = float(loss.item())
        any_valid = True

    if not any_valid:
        total = torch.tensor(0.0, requires_grad=True)
    return total, per_head


def evaluate(
    model: MultiTaskFashionClassifier,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """Evalua accuracy y F1-macro por head."""
    from sklearn.metrics import f1_score

    model.eval()
    preds = {name: [] for name in model.heads.keys()}
    targets = {name: [] for name in model.heads.keys()}

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            outputs = model(pixel_values)
            for name, logits in outputs.items():
                target = batch["labels"][name]
                mask = target != LABEL_MISSING
                if mask.sum() == 0:
                    continue
                pred = logits.argmax(dim=-1).cpu().numpy()
                preds[name].extend(pred[mask.numpy()].tolist())
                targets[name].extend(target[mask].numpy().tolist())

    out = {}
    for name in model.heads.keys():
        if not targets[name]:
            out[name] = {"accuracy": 0.0, "f1_macro": 0.0, "n": 0}
            continue
        y_pred = np.array(preds[name])
        y_true = np.array(targets[name])
        out[name] = {
            "accuracy": float((y_pred == y_true).mean()),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "n": int(len(y_true)),
        }
    return out


def train_from_csv(
    config: dict,
    dataset_key: str,
    train_csv: str | Path,
    val_csv: str | Path,
    cache_dir: str | Path,
    history_path: Optional[str | Path] = None,
) -> dict:
    """Entrena un clasificador (pants o tops) desde CSVs ya splitteados.

    Args:
        config: Config completa del pipeline.
        dataset_key: "pants" o "tops". Determina que seccion del config usar.
        train_csv: Path al CSV de train.
        val_csv: Path al CSV de val.
        cache_dir: Directorio con imagenes descargadas.
        history_path: Opcional, path para guardar historial de metricas.

    Returns:
        Dict con `best_metric`, `checkpoint`, `history`.
    """
    if dataset_key not in ("pants", "tops"):
        raise ValueError(f"dataset_key debe ser 'pants' o 'tops', no {dataset_key!r}")

    set_seed(int(config.get("seed", 42)))

    device_str = config.get("device", "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA no disponible, usando CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    model_cfg = config[dataset_key]
    train_cfg = model_cfg["training"]
    head_config = model_cfg["heads"]
    image_size = tuple(config.get("image_size", [224, 224]))

    train_ds = CSVImageDataset(train_csv, cache_dir, head_config, image_size, split="train")
    val_ds = CSVImageDataset(val_csv, cache_dir, head_config, image_size, split="val")
    logger.info("Train: %d ejemplos | Val: %d ejemplos", len(train_ds), len(val_ds))

    batch_size = int(train_cfg.get("batch_size", 32))
    num_workers = int(train_cfg.get("num_workers", 2))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=device.type == "cuda")

    model = MultiTaskFashionClassifier.from_config(model_cfg, pretrained=True).to(device)

    head_weights = {name: float(cfg.get("weight", 1.0)) for name, cfg in head_config.items()}
    class_weights = None
    if train_cfg.get("use_class_weights", False):
        logger.info("Calculando class weights desde train set...")
        class_weights = train_ds.compute_class_weights()
        for name, w in class_weights.items():
            logger.info("  %s: range=[%.2f, %.2f] mean=%.2f", name, float(w.min()), float(w.max()), float(w.mean()))

    total_epochs = int(train_cfg.get("epochs", 15))
    freeze_epochs = int(train_cfg.get("freeze_backbone_epochs", 3))
    lr = float(train_cfg.get("learning_rate", 3e-4))
    wd = float(train_cfg.get("weight_decay", 0.01))
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
    patience = int(train_cfg.get("early_stopping_patience", 4))

    model.freeze_backbone()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    ckpt_path = Path(model_cfg.get("checkpoint", f"models/best_{dataset_key}.pth"))
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_metric = -1.0
    patience_counter = 0
    history: list[dict] = []
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    for epoch in range(total_epochs):
        if epoch == freeze_epochs:
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW([
                {"params": model.backbone.parameters(), "lr": lr / 10.0},
                {"params": model.heads.parameters(), "lr": lr},
            ], weight_decay=wd)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, total_epochs - freeze_epochs),
            )

        model.train()
        epoch_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"[{dataset_key}] Epoca {epoch + 1}/{total_epochs}")
        for batch in pbar:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            optimizer.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(pixel_values)
                    loss, _ = multi_task_loss(outputs, batch["labels"], head_weights,
                                               class_weights, label_smoothing)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(pixel_values)
                loss, _ = multi_task_loss(outputs, batch["labels"], head_weights,
                                           class_weights, label_smoothing)
                loss.backward()
                optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
            pbar.set_postfix(loss=epoch_loss / max(1, n_batches))
        scheduler.step()

        metrics = evaluate(model, val_loader, device)
        mean_f1 = float(np.mean([m["f1_macro"] for m in metrics.values()]))
        mean_acc = float(np.mean([m["accuracy"] for m in metrics.values()]))
        history.append({
            "epoch": epoch + 1,
            "train_loss": epoch_loss / max(1, n_batches),
            "val_acc_mean": mean_acc,
            "val_f1_mean": mean_f1,
            "per_head": metrics,
        })
        logger.info("Epoca %d: train_loss=%.4f val_acc=%.3f val_f1=%.3f",
                    epoch + 1, epoch_loss / max(1, n_batches), mean_acc, mean_f1)

        if mean_f1 > best_metric:
            best_metric = mean_f1
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "head_config": head_config,
                "backbone_name": model_cfg.get("backbone"),
                "epoch": epoch + 1,
                "metric": best_metric,
            }, ckpt_path)
            logger.info("Nuevo mejor modelo: %s (F1=%.3f)", ckpt_path, best_metric)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping en epoca %d.", epoch + 1)
                break

    if history_path is not None:
        Path(history_path).parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    return {
        "best_metric": best_metric,
        "checkpoint": str(ckpt_path),
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset", choices=["pants", "tops"], required=True)
    parser.add_argument("--train-csv", type=str, required=True)
    parser.add_argument("--val-csv", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, required=True)
    parser.add_argument("--history", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_from_csv(
        config, args.dataset, args.train_csv, args.val_csv, args.cache_dir, args.history,
    )


if __name__ == "__main__":
    main()
