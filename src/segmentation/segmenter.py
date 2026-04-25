"""Segmentador de prendas basado en SegFormer."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

# Mapeo de categorias del modelo mattmdjaga/segformer_b2_clothes (ATR-like labels)
# A nuestras categorias de alto nivel.
DEFAULT_LABEL_MAP = {
    "upper_body": [4, 7],           # Upper-clothes, Coat
    "lower_body": [5, 6, 9, 10],    # Skirt, Pants, ...
    "full_body": [8],               # Dress
    "footwear": [9, 10],            # Left/Right shoe
    "accessories": [1, 16, 17, 18], # Hat, bag, scarf
}


class FashionSegmenter:
    """Wrapper sobre SegFormer de HuggingFace para segmentar prendas.

    Attributes:
        model: SegformerForSemanticSegmentation cargado.
        processor: SegformerImageProcessor asociado.
        device: torch.device donde corre inferencia.
        config: Diccionario de configuracion del modulo.
    """

    def __init__(self, config: dict) -> None:
        """Inicializa el segmentador cargando el modelo de HuggingFace.

        Args:
            config: Dict con las claves:
                - segmentation.model_name: checkpoint HuggingFace
                - segmentation.confidence_threshold: umbral minimo
                - device: "cpu" o "cuda"
        """
        from transformers import (
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
        )

        self.config = config
        seg_cfg = config.get("segmentation", {})
        self.model_name = seg_cfg.get(
            "model_name", "mattmdjaga/segformer_b2_clothes"
        )
        self.confidence_threshold = float(
            seg_cfg.get("confidence_threshold", 0.5)
        )
        self.target_categories = seg_cfg.get(
            "target_categories", list(DEFAULT_LABEL_MAP.keys())
        )

        device_str = config.get("device", "cpu")
        if device_str == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA no disponible, usando CPU.")
            device_str = "cpu"
        self.device = torch.device(device_str)

        logger.info("Cargando SegFormer: %s", self.model_name)
        self.processor = SegformerImageProcessor.from_pretrained(self.model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            self.model_name
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image: Image.Image) -> np.ndarray:
        """Corre inferencia y retorna mapa de segmentacion al tamano original.

        Args:
            image: Imagen PIL en modo RGB.

        Returns:
            Array (H, W) con el indice de clase por pixel.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        logits = outputs.logits  # (1, C, h, w)

        # Interpolar al tamano original (W, H en PIL)
        upsampled = F.interpolate(
            logits,
            size=(image.size[1], image.size[0]),
            mode="bilinear",
            align_corners=False,
        )
        pred = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.int32)
        return pred

    def get_garment_mask(
        self,
        segmentation_map: np.ndarray,
        target_category: Optional[str] = None,
    ) -> np.ndarray:
        """Extrae mascara binaria para una categoria de prenda.

        Args:
            segmentation_map: Mapa de clases por pixel.
            target_category: Una de las categorias en DEFAULT_LABEL_MAP. Si es
                None, retorna mascara de la categoria con mayor area
                (excluyendo fondo).

        Returns:
            Mascara binaria (H, W) uint8 con valores 0 o 1.
        """
        if target_category is not None:
            label_ids = DEFAULT_LABEL_MAP.get(target_category)
            if label_ids is None:
                raise ValueError(
                    f"Categoria desconocida: {target_category}. "
                    f"Opciones: {list(DEFAULT_LABEL_MAP.keys())}"
                )
            mask = np.isin(segmentation_map, label_ids).astype(np.uint8)
            return mask

        # Auto: tomar categoria con mayor area (excluyendo fondo=0)
        best_cat = None
        best_area = 0
        for cat, ids in DEFAULT_LABEL_MAP.items():
            if cat not in self.target_categories:
                continue
            area = int(np.isin(segmentation_map, ids).sum())
            if area > best_area:
                best_area = area
                best_cat = cat

        if best_cat is None:
            logger.warning("No se detecto ninguna prenda, retornando mascara vacia.")
            return np.zeros_like(segmentation_map, dtype=np.uint8)

        logger.debug("Categoria detectada: %s (area=%d)", best_cat, best_area)
        return np.isin(
            segmentation_map, DEFAULT_LABEL_MAP[best_cat]
        ).astype(np.uint8)

    def get_dominant_category(self, segmentation_map: np.ndarray) -> Optional[str]:
        """Retorna el nombre de la categoria con mayor area, o None si vacia."""
        best_cat = None
        best_area = 0
        for cat, ids in DEFAULT_LABEL_MAP.items():
            if cat not in self.target_categories:
                continue
            area = int(np.isin(segmentation_map, ids).sum())
            if area > best_area:
                best_area = area
                best_cat = cat
        return best_cat
