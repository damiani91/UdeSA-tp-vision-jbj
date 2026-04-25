"""Clasificador de marca basado en EfficientNet-B0."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)


class BrandClassifier(nn.Module):
    """EfficientNet-B0 con head para N marcas + threshold de incertidumbre."""

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        from torchvision import models

        if pretrained:
            self.backbone = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
            )
        else:
            self.backbone = models.efficientnet_b0(weights=None)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        for name, p in self.backbone.named_parameters():
            if not name.startswith("classifier"):
                p.requires_grad = False
        logger.info("EfficientNet backbone congelado.")

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True
        logger.info("EfficientNet backbone descongelado.")


class BrandPredictor:
    """Wrapper para inferencia con un BrandClassifier ya entrenado."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
        confidence_threshold: float = 0.6,
        unknown_label: str = "sin_marca",
    ) -> None:
        """Carga un modelo entrenado.

        Args:
            checkpoint_path: Path al .pth con keys `model_state` y `classes`.
            device: Device para inferencia.
            confidence_threshold: Debajo de esto retorna `unknown_label`.
            unknown_label: Label cuando la confianza es baja.
        """
        self.device = torch.device(device)
        self.confidence_threshold = confidence_threshold
        self.unknown_label = unknown_label

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.classes: list[str] = ckpt["classes"]
        self.model = BrandClassifier(num_classes=len(self.classes), pretrained=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device).eval()

        self._build_transform()

    def _build_transform(self) -> None:
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        """Clasifica una imagen y retorna {label, confidence, top_k}."""
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        label = self.classes[idx] if conf >= self.confidence_threshold else self.unknown_label

        top_k = sorted(
            [(self.classes[i], float(probs[i])) for i in np.argsort(probs)[-5:][::-1]],
            key=lambda x: -x[1],
        )
        return {
            "label": label,
            "confidence": round(conf, 4),
            "top_k": [{"brand": b, "confidence": round(c, 4)} for b, c in top_k],
        }
