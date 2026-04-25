"""Modelo multi-task de clasificacion de prendas basado en ViT."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MultiTaskFashionClassifier(nn.Module):
    """ViT compartido + heads independientes por atributo.

    Disenado para ser entrenado por separado para `pants` o `tops`
    (cada uno con su propio set de heads).
    """

    def __init__(
        self,
        head_config: dict[str, dict],
        backbone_name: str = "google/vit-base-patch16-224",
        pretrained: bool = True,
        head_hidden_dim: int = 256,
        head_dropout: float = 0.3,
    ) -> None:
        """Inicializa el modelo.

        Args:
            head_config: Dict {head_name: {"classes": [...], "weight": float}}.
            backbone_name: Nombre del modelo ViT en HuggingFace.
            pretrained: Si carga pesos pre-entrenados del backbone.
            head_hidden_dim: Dim oculta de cada head MLP.
            head_dropout: Dropout entre las dos capas de cada head.
        """
        super().__init__()
        from transformers import ViTConfig, ViTModel

        self.backbone_name = backbone_name
        self.head_config = head_config

        if pretrained:
            self.backbone = ViTModel.from_pretrained(backbone_name)
        else:
            cfg = ViTConfig.from_pretrained(backbone_name)
            self.backbone = ViTModel(cfg)

        self.hidden_dim = self.backbone.config.hidden_size

        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(self.hidden_dim, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden_dim, len(cfg["classes"])),
            )
            for name, cfg in head_config.items()
        })

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.backbone(pixel_values=pixel_values)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return {name: head(cls_embedding) for name, head in self.heads.items()}

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        logger.info("Backbone congelado.")

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True
        logger.info("Backbone descongelado.")

    @classmethod
    def from_config(cls, model_cfg: dict, pretrained: bool = True):
        """Construye desde una seccion del config (ej. config['pants']).

        Args:
            model_cfg: Dict con keys: backbone, heads.
            pretrained: Si carga pesos pre-entrenados.
        """
        return cls(
            head_config=model_cfg["heads"],
            backbone_name=model_cfg.get("backbone", "google/vit-base-patch16-224"),
            pretrained=pretrained,
        )
