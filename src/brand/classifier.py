"""Clasificador de marca sobre crops de logos."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)


def _build_efficientnet(n_classes: int, pretrained: bool = True) -> nn.Module:
    """Construye un EfficientNet-B0 con cabeza ajustada a n_classes.

    Args:
        n_classes: Numero de clases de salida.
        pretrained: Si carga pesos ImageNet.

    Returns:
        nn.Module.
    """
    from torchvision import models

    if pretrained:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    else:
        weights = None
    model = models.efficientnet_b0(weights=weights)
    in_feats = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_feats, n_classes)
    return model


class BrandClassifier:
    """EfficientNet-B0 fine-tuneado para clasificar logos de marcas.

    Clases = target_brands + ["otra", "sin_marca"].
    """

    def __init__(
        self, config: dict, weights_path: Optional[str] = None
    ) -> None:
        """Inicializa el clasificador.

        Args:
            config: Config del pipeline (usa brand.classifier).
            weights_path: Path opcional a checkpoint fine-tuneado.
        """
        from torchvision import transforms

        brand_cfg = config.get("brand", {})
        cls_cfg = brand_cfg.get("classifier", {})
        self.target_brands: list[str] = list(cls_cfg.get("target_brands", []))
        self.classes: list[str] = self.target_brands + ["otra", "sin_marca"]
        self.n_classes = len(self.classes)

        device_str = config.get("device", "cpu")
        if device_str == "cuda" and not torch.cuda.is_available():
            device_str = "cpu"
        self.device = torch.device(device_str)

        self.model = _build_efficientnet(self.n_classes, pretrained=True).to(self.device)

        if weights_path and Path(weights_path).exists():
            logger.info("Cargando pesos de BrandClassifier: %s", weights_path)
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state.get("model_state", state))

        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    @torch.no_grad()
    def classify(self, logo_crop: Image.Image) -> dict:
        """Clasifica un crop de logo.

        Args:
            logo_crop: Imagen PIL del crop.

        Returns:
            Dict {"brand": str, "confidence": float}.
        """
        if logo_crop.mode != "RGB":
            logo_crop = logo_crop.convert("RGB")

        x = self.transform(logo_crop).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=-1)[0]
        idx = int(torch.argmax(probs).item())
        return {"brand": self.classes[idx], "confidence": float(probs[idx].item())}

    def classify_from_detections(
        self, crops: list[Image.Image]
    ) -> dict:
        """Agrega clasificaciones de varios crops y devuelve la mejor prediccion.

        Args:
            crops: Lista de crops PIL. Si vacia, retorna marca nula.

        Returns:
            Dict {"brand": str|None, "confidence": float}.
        """
        if not crops:
            return {"brand": None, "confidence": 0.0}

        best = {"brand": "sin_marca", "confidence": 0.0}
        for crop in crops:
            pred = self.classify(crop)
            if pred["confidence"] > best["confidence"]:
                best = pred
        return best
