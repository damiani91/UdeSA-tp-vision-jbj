"""Dataset multi-task que lee un CSV con URLs y atributos."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.downloader import url_to_filename
from src.data.mappings import MAPPING_EN_ES, apply_mapping

logger = logging.getLogger(__name__)

LABEL_MISSING = -1


class CSVImageDataset(Dataset):
    """Dataset multi-task sobre un CSV con URLs y atributos en ingles.

    El dataset:
        1. Lee el CSV (puede ser train/val/test ya splitteado).
        2. Aplica `MAPPING_EN_ES` a cada columna de atributos.
        3. Mapea cada label al indice de su clase segun `head_config`.
        4. Las clases NO presentes en `head_config` se mapean a LABEL_MISSING.
        5. Carga la imagen desde el cache local (descargada previamente).

    Si la imagen no esta en cache, retorna una imagen negra y label
    LABEL_MISSING para todas las heads (la fila se ignora en el loss).
    """

    def __init__(
        self,
        csv_path: str | Path,
        cache_dir: str | Path,
        head_config: dict[str, dict],
        image_size: tuple[int, int] = (224, 224),
        split: str = "train",
        transform: Optional[object] = None,
        url_col: str = "image_url",
    ) -> None:
        """Inicializa el dataset.

        Args:
            csv_path: Path al CSV con columnas `image_url` + atributos.
            cache_dir: Directorio donde estan las imagenes descargadas.
            head_config: Dict por head: {head_name: {"classes": [...], "weight": ...}}.
                Las keys deben ser nombres de columnas del CSV.
            image_size: (H, W) target.
            split: "train", "val" o "test". Determina las augmentations.
            transform: Albumentations Compose opcional. Si None, se construye
                un default segun el split.
            url_col: Nombre de la columna con URLs.
        """
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.cache_dir = Path(cache_dir)
        self.head_config = head_config
        self.image_size = tuple(image_size)
        self.split = split
        self.url_col = url_col

        self.class_to_idx: dict[str, dict[str, int]] = {
            name: {c: i for i, c in enumerate(cfg["classes"])}
            for name, cfg in head_config.items()
        }

        self.transform = transform if transform is not None else self._default_transform()

    def _default_transform(self):
        """Construye una transform de albumentations apropiada al split."""
        try:
            import albumentations as A
            from albumentations.pytorch import ToTensorV2
        except ImportError:
            logger.warning("albumentations no disponible, usando transform minimo.")
            return None

        h, w = self.image_size
        norm = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if self.split == "train":
            return A.Compose([
                A.Resize(h, w),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03, p=0.5),
                A.Rotate(limit=10, p=0.4),
                norm,
                ToTensorV2(),
            ])
        return A.Compose([A.Resize(h, w), norm, ToTensorV2()])

    def __len__(self) -> int:
        return len(self.df)

    def _label_index(self, head_name: str, raw_value) -> int:
        """Aplica EN-ES y mapea a indice. Retorna LABEL_MISSING si no aplica."""
        if raw_value is None or (isinstance(raw_value, float) and pd.isna(raw_value)):
            return LABEL_MISSING
        mapped = apply_mapping(raw_value, MAPPING_EN_ES.get(head_name, {}), default=None)
        if mapped is None:
            mapped = "otro"
        return self.class_to_idx[head_name].get(mapped, LABEL_MISSING)

    def _load_image(self, url: str) -> np.ndarray:
        """Carga la imagen desde el cache. Retorna negro si no existe."""
        path = self.cache_dir / url_to_filename(url)
        try:
            return np.array(Image.open(path).convert("RGB"))
        except Exception as exc:
            logger.debug("Imagen ausente o corrupta %s (%s)", path, exc)
            return np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        url = row[self.url_col]
        image = self._load_image(url)

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        labels = {
            name: torch.tensor(self._label_index(name, row.get(name)), dtype=torch.long)
            for name in self.head_config.keys()
        }
        return {
            "pixel_values": image,
            "labels": labels,
            "id": int(row.get("id", idx)),
        }

    def compute_class_weights(self) -> dict[str, torch.Tensor]:
        """Calcula class weights (inverse frequency normalizado) por head.

        Returns:
            Dict {head_name: tensor de pesos shape (n_classes,)}.
            Las clases sin ejemplos reciben peso 1.0.
        """
        result = {}
        for head_name, cfg in self.head_config.items():
            n_classes = len(cfg["classes"])
            counts = np.zeros(n_classes, dtype=np.float64)
            for i in range(len(self.df)):
                idx = self._label_index(head_name, self.df.iloc[i].get(head_name))
                if 0 <= idx < n_classes:
                    counts[idx] += 1
            total = counts.sum()
            if total == 0:
                result[head_name] = torch.ones(n_classes)
                continue
            freq = counts / total
            with np.errstate(divide="ignore", invalid="ignore"):
                weights = np.where(freq > 0, 1.0 / freq, 1.0)
            mean = weights[counts > 0].mean() if (counts > 0).any() else 1.0
            weights = weights / mean
            result[head_name] = torch.tensor(weights, dtype=torch.float32)
        return result
