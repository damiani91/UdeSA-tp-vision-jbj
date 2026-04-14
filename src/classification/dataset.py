"""Dataset multi-label para fine-tuning con DeepFashion u otros."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Mapeo manual de las categorias de DeepFashion a nuestro esquema.
# Este dict debe expandirse segun el benchmark que se use.
DEEPFASHION_TIPO_MAP: dict[str, str] = {
    "Tee": "remera",
    "Blouse": "camisa",
    "Shirt": "camisa",
    "Jacket": "campera",
    "Pants": "pantalon",
    "Jeans": "jean",
    "Shorts": "short",
    "Skirt": "falda",
    "Dress": "vestido",
    "Sweater": "sweater",
    "Hoodie": "buzo",
    "Blazer": "blazer",
    "Coat": "tapado",
}

LABEL_MISSING = -1


class DeepFashionDataset(Dataset):
    """Dataset de DeepFashion para clasificacion multi-task.

    Cada item retorna:
        - pixel_values: tensor de la imagen procesada.
        - labels: dict {head_name: class_idx or -1 si no hay anotacion}.
    """

    def __init__(
        self,
        annotations: list[dict],
        image_root: str | Path,
        config: dict,
        split: str = "train",
        transform: Optional[object] = None,
    ) -> None:
        """Inicializa el dataset.

        Args:
            annotations: Lista de dicts con keys: image_path, tipo, estilo,
                fit, cuello, manga, material. Valores pueden ser None o str.
            image_root: Directorio base de imagenes.
            config: Config del pipeline (para leer heads y augmentations).
            split: "train", "val" o "test".
            transform: Albumentations Compose opcional. Si None, se construye
                desde config para train y se usa transform basico para val/test.
        """
        self.annotations = annotations
        self.image_root = Path(image_root)
        self.split = split
        cls_cfg = config.get("classification", {})
        self.head_configs: dict[str, dict] = cls_cfg.get("heads", {})
        self.image_size = config.get("image_size", [224, 224])

        # Construir class_to_idx por head desde la config
        self.class_to_idx: dict[str, dict[str, int]] = {}
        for name, cfg in self.head_configs.items():
            self.class_to_idx[name] = {c: i for i, c in enumerate(cfg["classes"])}

        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._build_default_transform(config)

    def _build_default_transform(self, config: dict):
        """Construye una transform de albumentations desde la config."""
        try:
            import albumentations as A
            from albumentations.pytorch import ToTensorV2
        except ImportError:
            logger.warning("albumentations no instalado, usando transform minimo.")
            return None

        h, w = int(self.image_size[0]), int(self.image_size[1])
        if self.split == "train":
            aug_cfg = (
                config.get("classification", {}).get("training", {}).get("augmentation", {})
            )
            transforms = [A.Resize(h, w)]
            if aug_cfg.get("random_crop"):
                transforms.append(A.RandomResizedCrop(size=(h, w), scale=(0.8, 1.0)))
            if aug_cfg.get("horizontal_flip"):
                transforms.append(A.HorizontalFlip(p=0.5))
            if aug_cfg.get("color_jitter"):
                transforms.append(
                    A.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5
                    )
                )
            rot = aug_cfg.get("random_rotation", 0)
            if rot:
                transforms.append(A.Rotate(limit=int(rot), p=0.5))
            transforms.extend(
                [
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )
            return A.Compose(transforms)

        return A.Compose(
            [
                A.Resize(h, w),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> dict:
        ann = self.annotations[idx]
        img_path = self.image_root / ann["image_path"]
        try:
            image = np.array(Image.open(img_path).convert("RGB"))
        except Exception as exc:
            logger.warning("Error leyendo %s: %s. Usando imagen en blanco.", img_path, exc)
            h, w = int(self.image_size[0]), int(self.image_size[1])
            image = np.zeros((h, w, 3), dtype=np.uint8)

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        labels = {}
        for head_name in self.head_configs.keys():
            value = ann.get(head_name)
            if value is None or value == "":
                labels[head_name] = torch.tensor(LABEL_MISSING, dtype=torch.long)
            else:
                cls_map = self.class_to_idx[head_name]
                idx_val = cls_map.get(str(value), LABEL_MISSING)
                labels[head_name] = torch.tensor(idx_val, dtype=torch.long)

        return {"pixel_values": image, "labels": labels}
