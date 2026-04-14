"""Training loop multi-task para el clasificador de prendas."""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import LABEL_MISSING, DeepFashionDataset
from .model import MultiTaskFashionClassifier

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Fija seeds para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def multi_task_loss(
    outputs: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    weights: dict[str, float],
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Suma ponderada de CrossEntropyLoss por head.

    Samples con label = LABEL_MISSING se excluyen del calculo.

    Args:
        outputs: {head_name: logits (B, C)}.
        labels: {head_name: targets (B,)}.
        weights: {head_name: peso escalar}.
        label_smoothing: Factor de label smoothing en [0, 1).

    Returns:
        Tupla (loss_total, dict con loss por head en float).
    """
    total = 0.0
    per_head = {}
    any_valid = False
    for name, logits in outputs.items():
        target = labels[name].to(logits.device)
        mask = target != LABEL_MISSING
        if mask.sum() == 0:
            per_head[name] = 0.0
            continue
        loss = F.cross_entropy(
            logits[mask],
            target[mask],
            label_smoothing=label_smoothing,
        )
        w = float(weights.get(name, 1.0))
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
    """Evalua el modelo computando accuracy y macro-F1 por head.

    Args:
        model: Modelo multi-task.
        dataloader: DataLoader de validacion.
        device: Device.

    Returns:
        Dict {head_name: {"accuracy": f, "f1_macro": f, "n": int}}.
    """
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

    results = {}
    for name in model.heads.keys():
        if not targets[name]:
            results[name] = {"accuracy": 0.0, "f1_macro": 0.0, "n": 0}
            continue
        y_pred = np.array(preds[name])
        y_true = np.array(targets[name])
        acc = float((y_pred == y_true).mean())
        f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        results[name] = {"accuracy": acc, "f1_macro": f1, "n": int(len(y_true))}

    return results


def train(config: dict, annotations_train: list[dict], annotations_val: list[dict],
          image_root: str) -> dict:
    """Entrena el modelo multi-task con estrategia de dos fases.

    Fase 1: backbone congelado por `freeze_backbone_epochs`.
    Fase 2: todo descongelado con lr reducido (lr/10 en backbone).

    Args:
        config: Config completa del pipeline.
        annotations_train: Lista de anotaciones de entrenamiento.
        annotations_val: Lista de anotaciones de validacion.
        image_root: Directorio base de imagenes.

    Returns:
        Dict con metricas finales y path al mejor checkpoint.
    """
    set_seed(int(config.get("seed", 42)))

    device_str = config.get("device", "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA no disponible, usando CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    cls_cfg = config["classification"]
    train_cfg = cls_cfg["training"]

    train_ds = DeepFashionDataset(annotations_train, image_root, config, split="train")
    val_ds = DeepFashionDataset(annotations_val, image_root, config, split="val")

    batch_size = int(train_cfg.get("batch_size", 32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = MultiTaskFashionClassifier(config, pretrained=True).to(device)

    head_weights = {name: float(cfg.get("weight", 1.0)) for name, cfg in cls_cfg["heads"].items()}

    freeze_epochs = int(cls_cfg.get("freeze_backbone_epochs", 3))
    total_epochs = int(train_cfg.get("epochs", 25))
    lr = float(train_cfg.get("learning_rate", 3e-4))
    wd = float(train_cfg.get("weight_decay", 0.01))
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))

    model.freeze_backbone()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=wd,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    models_dir = Path(config.get("paths", {}).get("models", "models"))
    models_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = models_dir / "best_multitask.pth"

    best_metric = -1.0
    patience = int(cls_cfg.get("finetune", {}).get("early_stopping_patience", 5))
    patience_counter = 0

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    for epoch in range(total_epochs):
        # Transicion: descongelar backbone al pasar la fase 1
        if epoch == freeze_epochs:
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.backbone.parameters(), "lr": lr / 10.0},
                    {"params": model.heads.parameters(), "lr": lr},
                ],
                weight_decay=wd,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_epochs - freeze_epochs
            )

        model.train()
        epoch_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoca {epoch + 1}/{total_epochs}")
        for batch in pbar:
            pixel_values = batch["pixel_values"].to(device)
            optimizer.zero_grad()

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(pixel_values)
                    loss, _ = multi_task_loss(
                        outputs, batch["labels"], head_weights, label_smoothing
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(pixel_values)
                loss, _ = multi_task_loss(
                    outputs, batch["labels"], head_weights, label_smoothing
                )
                loss.backward()
                optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1
            pbar.set_postfix(loss=epoch_loss / max(1, n_batches))

        scheduler.step()

        metrics = evaluate(model, val_loader, device)
        mean_f1 = float(np.mean([m["f1_macro"] for m in metrics.values()]))
        logger.info("Epoca %d | val macro-F1 promedio: %.4f", epoch + 1, mean_f1)
        for name, m in metrics.items():
            logger.info("  %s: acc=%.3f f1=%.3f n=%d", name, m["accuracy"], m["f1_macro"], m["n"])

        if mean_f1 > best_metric:
            best_metric = mean_f1
            patience_counter = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "metric": best_metric,
                },
                best_ckpt,
            )
            logger.info("Nuevo mejor modelo guardado: %s", best_ckpt)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping en epoca %d.", epoch + 1)
                break

    return {
        "best_metric": best_metric,
        "checkpoint": str(best_ckpt),
    }


def main() -> None:
    """CLI de entrenamiento."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--train-annotations", type=str, required=True)
    parser.add_argument("--val-annotations", type=str, required=True)
    parser.add_argument("--image-root", type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    with open(args.config) as f:
        config = yaml.safe_load(f)

    import json

    with open(args.train_annotations) as f:
        train_ann = json.load(f)
    with open(args.val_annotations) as f:
        val_ann = json.load(f)

    train(config, train_ann, val_ann, args.image_root)


if __name__ == "__main__":
    main()
