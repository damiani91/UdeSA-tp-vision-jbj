"""Modelo multi-task de clasificacion de prendas basado en ViT."""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MultiTaskFashionClassifier(nn.Module):
    """ViT backbone compartido + heads independientes por atributo.

    Attributes:
        backbone: Modelo ViT pre-entrenado de HuggingFace.
        heads: ModuleDict con una cabeza por atributo (tipo, estilo, etc.).
        hidden_dim: Dimension del embedding del backbone.
    """

    def __init__(
        self,
        config: dict,
        pretrained: bool = True,
    ) -> None:
        """Inicializa el modelo.

        Args:
            config: Diccionario de configuracion. Espera:
                - classification.backbone: modelo HuggingFace
                - classification.heads: dict {head_name: {"classes": [...]}}
            pretrained: Si carga pesos pre-entrenados del backbone.
        """
        super().__init__()
        from transformers import ViTModel

        cls_cfg = config.get("classification", {})
        self.backbone_name = cls_cfg.get("backbone", "google/vit-base-patch16-224")
        self.head_config: dict[str, dict] = cls_cfg.get("heads", {})

        if pretrained:
            self.backbone = ViTModel.from_pretrained(self.backbone_name)
        else:
            from transformers import ViTConfig

            vit_cfg = ViTConfig.from_pretrained(self.backbone_name)
            self.backbone = ViTModel(vit_cfg)

        self.hidden_dim = self.backbone.config.hidden_size

        self.heads = nn.ModuleDict()
        for head_name, head_cfg in self.head_config.items():
            n_classes = len(head_cfg["classes"])
            self.heads[head_name] = nn.Sequential(
                nn.Linear(self.hidden_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, n_classes),
            )

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            pixel_values: Tensor (B, 3, H, W) normalizado para ViT.

        Returns:
            Dict {head_name: logits (B, n_classes)}.
        """
        outputs = self.backbone(pixel_values=pixel_values)
        # Usamos el CLS token (posicion 0)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        return {name: head(cls_embedding) for name, head in self.heads.items()}

    def freeze_backbone(self) -> None:
        """Congela los pesos del backbone."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        logger.info("Backbone congelado.")

    def unfreeze_backbone(self) -> None:
        """Descongela los pesos del backbone."""
        for p in self.backbone.parameters():
            p.requires_grad = True
        logger.info("Backbone descongelado.")
