"""Detector de logos basado en YOLOv8."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class LogoDetector:
    """Wrapper sobre YOLOv8 de ultralytics para detectar logos.

    Attributes:
        model: Modelo YOLO cargado.
        confidence_threshold: Umbral minimo de confianza.
        iou_threshold: Umbral de IoU para NMS.
    """

    def __init__(self, config: dict, weights_path: Optional[str] = None) -> None:
        """Inicializa el detector.

        Args:
            config: Config del pipeline. Lee de config['brand']['detector'].
            weights_path: Path a pesos fine-tuneados. Si None, usa el modelo
                base de la config.
        """
        from ultralytics import YOLO

        det_cfg = config.get("brand", {}).get("detector", {})
        self.model_name = det_cfg.get("model", "yolov8m")
        self.confidence_threshold = float(det_cfg.get("confidence_threshold", 0.4))
        self.iou_threshold = float(det_cfg.get("iou_threshold", 0.5))

        source = weights_path or f"{self.model_name}.pt"
        logger.info("Cargando YOLO desde: %s", source)
        self.model = YOLO(source)

    def detect(self, image: Image.Image) -> list[dict]:
        """Corre deteccion sobre la imagen.

        Args:
            image: Imagen PIL en modo RGB.

        Returns:
            Lista de detecciones: {"bbox": [x1,y1,x2,y2], "confidence": float}.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        arr = np.array(image)

        results = self.model.predict(
            arr,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )
        if not results:
            return []

        r = results[0]
        detections = []
        if r.boxes is None:
            return []
        for box in r.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(float).tolist()
            conf = float(box.conf[0].cpu().numpy())
            detections.append({"bbox": xyxy, "confidence": conf})

        return detections

    def crop_detections(
        self, image: Image.Image, detections: list[dict]
    ) -> list[Image.Image]:
        """Recorta las regiones detectadas de la imagen original.

        Args:
            image: Imagen PIL.
            detections: Lista de dicts con clave 'bbox'.

        Returns:
            Lista de imagenes PIL recortadas.
        """
        crops = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            crop = image.crop((int(x1), int(y1), int(x2), int(y2)))
            crops.append(crop)
        return crops
