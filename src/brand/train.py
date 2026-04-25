"""Training de BrandClassifier sobre Logo-2K+."""

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

from src.brand.classifier import BrandClassifier
from src.brand.dataset import LogoDataset

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model: BrandClassifier, loader: DataLoader, device: torch.device) -> dict:
    """Accuracy y F1-macro sobre validation."""
    from sklearn.metrics import f1_score

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["pixel_values"].to(device)
            y = batch["label"].numpy()
            logits = model(x)
            pred = logits.argmax(dim=-1).cpu().numpy()
            preds.extend(pred.tolist())
            targets.extend(y.tolist())

    if not targets:
        return {"accuracy": 0.0, "f1_macro": 0.0, "n": 0}
    y_pred = np.array(preds)
    y_true = np.array(targets)
    return {
        "accuracy": float((y_pred == y_true).mean()),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "n": int(len(y_true)),
    }


def train_brand(
    config: dict,
    history_path: Optional[str | Path] = None,
) -> dict:
    """Entrena el clasificador de marca sobre Logo-2K+ /Clothes."""
    set_seed(int(config.get("seed", 42)))

    device_str = config.get("device", "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA no disponible, usando CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    brand_cfg = config["brand"]
    train_cfg = brand_cfg["training"]
    image_size = (train_cfg.get("image_size", 224), train_cfg.get("image_size", 224))

    full_ds = LogoDataset(brand_cfg["classes_source"], image_size=image_size, split="train")
    train_ds, val_ds = full_ds.split_train_val(
        val_ratio=train_cfg.get("val_ratio", 0.15), seed=int(config.get("seed", 42)),
    )
    logger.info("Logo-2K+: %d clases | train=%d | val=%d",
                len(full_ds.classes), len(train_ds), len(val_ds))

    batch_size = int(train_cfg.get("batch_size", 32))
    num_workers = int(train_cfg.get("num_workers", 2))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=device.type == "cuda")

    model = BrandClassifier(num_classes=len(full_ds.classes), pretrained=True).to(device)

    epochs = int(train_cfg.get("epochs", 20))
    freeze_epochs = int(train_cfg.get("freeze_backbone_epochs", 5))
    lr = float(train_cfg.get("learning_rate", 1e-3))
    wd = float(train_cfg.get("weight_decay", 0.01))
    label_smoothing = float(train_cfg.get("label_smoothing", 0.1))
    patience = int(train_cfg.get("early_stopping_patience", 4))

    model.freeze_backbone()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ckpt_path = Path(brand_cfg.get("checkpoint", "models/best_brand.pth"))
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_metric = -1.0
    patience_counter = 0
    history: list[dict] = []
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    for epoch in range(epochs):
        if epoch == freeze_epochs:
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW([
                {"params": [p for n, p in model.backbone.named_parameters()
                            if not n.startswith("classifier")],
                 "lr": lr / 10.0},
                {"params": model.backbone.classifier.parameters(), "lr": lr},
            ], weight_decay=wd)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, epochs - freeze_epochs),
            )

        model.train()
        epoch_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"[brand] Epoca {epoch + 1}/{epochs}")
        for batch in pbar:
            x = batch["pixel_values"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = model(x)
                    loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
                loss.backward()
                optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
            pbar.set_postfix(loss=epoch_loss / max(1, n_batches))
        scheduler.step()

        metrics = evaluate(model, val_loader, device)
        history.append({
            "epoch": epoch + 1,
            "train_loss": epoch_loss / max(1, n_batches),
            **metrics,
        })
        logger.info("Epoca %d: train_loss=%.4f val_acc=%.3f val_f1=%.3f",
                    epoch + 1, epoch_loss / max(1, n_batches),
                    metrics["accuracy"], metrics["f1_macro"])

        if metrics["f1_macro"] > best_metric:
            best_metric = metrics["f1_macro"]
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "classes": full_ds.classes,
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
        "n_classes": len(full_ds.classes),
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--history", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    with open(args.config) as f:
        config = yaml.safe_load(f)
    train_brand(config, args.history)


if __name__ == "__main__":
    main()
