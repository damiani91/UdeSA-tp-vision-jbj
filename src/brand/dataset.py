"""Dataset estilo ImageFolder sobre Logo-2K+ /Clothes."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def discover_brands(root: str | Path) -> list[str]:
    """Lista las clases (subcarpetas) en orden alfabetico."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"No existe: {root}")
    return sorted([p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")])


def collect_samples(root: str | Path, classes: list[str]) -> list[tuple[Path, int]]:
    """Recorre `root/<class>/*` y retorna lista de (path, class_idx)."""
    root = Path(root)
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    samples: list[tuple[Path, int]] = []
    for cls in classes:
        cls_dir = root / cls
        for p in cls_dir.iterdir():
            if p.suffix.lower() in VALID_EXTS:
                samples.append((p, cls_to_idx[cls]))
    return samples


class LogoDataset(Dataset):
    """Dataset de imagenes con labels por carpeta padre."""

    def __init__(
        self,
        root: str | Path,
        classes: Optional[list[str]] = None,
        samples: Optional[list[tuple[Path, int]]] = None,
        image_size: tuple[int, int] = (224, 224),
        split: str = "train",
        transform=None,
    ) -> None:
        """Inicializa el dataset.

        Args:
            root: Directorio raiz con subcarpetas por clase.
            classes: Lista de nombres de clase (en orden de indice).
                Si None, se descubre desde `root`.
            samples: Si se provee, lista (path, class_idx) ya pre-armada.
                Util para pasar splits especificos.
            image_size: (H, W).
            split: "train" o "val/test".
            transform: Albumentations Compose. Si None, default segun split.
        """
        self.root = Path(root)
        self.classes = classes if classes is not None else discover_brands(self.root)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.image_size = tuple(image_size)
        self.split = split

        if samples is None:
            self.samples = collect_samples(self.root, self.classes)
        else:
            self.samples = list(samples)

        self.transform = transform if transform is not None else self._default_transform()

    def _default_transform(self):
        try:
            import albumentations as A
            from albumentations.pytorch import ToTensorV2
        except ImportError:
            logger.warning("albumentations no disponible.")
            return None

        h, w = self.image_size
        norm = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if self.split == "train":
            return A.Compose([
                A.Resize(h, w),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03, p=0.5),
                A.Rotate(limit=15, p=0.4),
                norm,
                ToTensorV2(),
            ])
        return A.Compose([A.Resize(h, w), norm, ToTensorV2()])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        path, label = self.samples[idx]
        try:
            image = np.array(Image.open(path).convert("RGB"))
        except Exception as exc:
            logger.warning("Error leyendo %s: %s", path, exc)
            image = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return {"pixel_values": image, "label": torch.tensor(label, dtype=torch.long)}

    def split_train_val(self, val_ratio: float = 0.15, seed: int = 42) -> tuple["LogoDataset", "LogoDataset"]:
        """Genera dos LogoDataset (train/val) estratificados por clase."""
        from sklearn.model_selection import train_test_split

        paths, labels = zip(*self.samples)
        # Filtrar clases con <2 ejemplos
        from collections import Counter
        cnt = Counter(labels)
        valid_idx = [i for i, l in enumerate(labels) if cnt[l] >= 2]
        paths = [paths[i] for i in valid_idx]
        labels = [labels[i] for i in valid_idx]

        train_idx, val_idx = train_test_split(
            range(len(paths)), test_size=val_ratio, stratify=labels, random_state=seed,
        )
        train_samples = [(paths[i], labels[i]) for i in train_idx]
        val_samples = [(paths[i], labels[i]) for i in val_idx]

        train_ds = LogoDataset(self.root, self.classes, train_samples,
                               self.image_size, split="train")
        val_ds = LogoDataset(self.root, self.classes, val_samples,
                             self.image_size, split="val")
        return train_ds, val_ds
