"""Orquestador del pipeline de extraccion de atributos de moda."""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

PIPELINE_VERSION = "0.1.0"


def _set_seed(seed: int) -> None:
    """Fija seeds globales para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_config(config_path: str | Path) -> dict:
    """Carga un YAML de configuracion."""
    with open(config_path) as f:
        return yaml.safe_load(f)


class FashionPipeline:
    """Pipeline end-to-end: imagen -> JSON de atributos.

    Orquesta segmentacion, color, clasificacion multi-task y deteccion de marca.
    Cada modulo se puede habilitar/deshabilitar via config.
    """

    def __init__(self, config_path: str | Path) -> None:
        """Inicializa el pipeline cargando los modulos habilitados.

        Args:
            config_path: Path al YAML de configuracion.
        """
        self.config = _load_config(config_path)
        _set_seed(int(self.config.get("seed", 42)))

        log_level = self.config.get("log_level", "INFO")
        logging.basicConfig(
            level=getattr(logging, log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        pipe_cfg = self.config.get("pipeline", {})
        self.run_segmentation = bool(pipe_cfg.get("run_segmentation", True))
        self.run_color = bool(pipe_cfg.get("run_color", True))
        self.run_classification = bool(pipe_cfg.get("run_classification", True))
        self.run_brand = bool(pipe_cfg.get("run_brand", True))
        self.save_intermediate = bool(pipe_cfg.get("save_intermediate", False))
        self.confidence_threshold = float(pipe_cfg.get("confidence_threshold", 0.5))

        self.outputs_dir = Path(self.config.get("paths", {}).get("outputs", "outputs"))
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        self.segmenter = None
        self.color_extractor = None
        self.classifier = None
        self.logo_detector = None
        self.brand_classifier = None

        self._load_modules()

    def _load_modules(self) -> None:
        """Carga perezosamente los modulos habilitados."""
        if self.run_segmentation:
            try:
                from src.segmentation.segmenter import FashionSegmenter

                self.segmenter = FashionSegmenter(self.config)
            except Exception as exc:
                logger.error("Fallo al cargar segmenter: %s", exc)
                self.segmenter = None

        if self.run_color:
            try:
                from src.color.extractor import ColorExtractor

                self.color_extractor = ColorExtractor(self.config)
            except Exception as exc:
                logger.error("Fallo al cargar color extractor: %s", exc)
                self.color_extractor = None

        if self.run_classification:
            try:
                from src.classification.model import MultiTaskFashionClassifier

                self.classifier = MultiTaskFashionClassifier(
                    self.config, pretrained=True
                )
                ckpt = Path(
                    self.config.get("paths", {}).get("models", "models")
                ) / "best_multitask.pth"
                if ckpt.exists():
                    state = torch.load(ckpt, map_location="cpu")
                    self.classifier.load_state_dict(state.get("model_state", state))
                    logger.info("Clasificador cargado desde %s", ckpt)
                else:
                    logger.warning(
                        "No hay checkpoint en %s, usando pesos pre-entrenados.", ckpt
                    )

                device_str = self.config.get("device", "cpu")
                if device_str == "cuda" and not torch.cuda.is_available():
                    device_str = "cpu"
                self.classifier.to(torch.device(device_str))
                self.classifier.eval()
            except Exception as exc:
                logger.error("Fallo al cargar classifier: %s", exc)
                self.classifier = None

        if self.run_brand:
            try:
                from src.brand.detector import LogoDetector
                from src.brand.classifier import BrandClassifier

                self.logo_detector = LogoDetector(self.config)
                self.brand_classifier = BrandClassifier(self.config)
            except Exception as exc:
                logger.error("Fallo al cargar brand modules: %s", exc)
                self.logo_detector = None
                self.brand_classifier = None

    def _load_image(self, image_path: str | Path) -> Optional[Image.Image]:
        """Carga una imagen desde disco, manejando errores."""
        try:
            img = Image.open(image_path).convert("RGB")
            return img
        except Exception as exc:
            logger.error("Error abriendo imagen %s: %s", image_path, exc)
            return None

    def _run_segmentation(
        self, image: Image.Image
    ) -> tuple[Optional[Image.Image], Optional[np.ndarray]]:
        """Corre segmentacion y retorna (imagen RGBA con mascara aplicada, mascara binaria).

        Si falla, retorna (None, None).
        """
        from src.segmentation.postprocess import (
            apply_mask,
            clean_mask,
            crop_to_content,
            largest_connected_component,
        )

        if self.segmenter is None:
            return None, None

        try:
            seg_map = self.segmenter.predict(image)
            mask = self.segmenter.get_garment_mask(seg_map)
            mask = clean_mask(mask)
            mask = largest_connected_component(mask)
            rgba = apply_mask(image, mask)
            cropped = crop_to_content(rgba)
            return cropped, mask
        except Exception as exc:
            logger.error("Error en segmentacion: %s", exc)
            return None, None

    def _run_classification(self, image: Image.Image) -> dict[str, dict]:
        """Corre el clasificador multi-task y retorna predicciones."""
        if self.classifier is None:
            return {}

        from torchvision import transforms

        device = next(self.classifier.parameters()).device
        tf = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        img_rgb = image.convert("RGB")
        x = tf(img_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = self.classifier(x)

        results = {}
        head_configs = self.config["classification"]["heads"]
        for name, logits in outputs.items():
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            idx = int(np.argmax(probs))
            label = head_configs[name]["classes"][idx]
            conf = float(probs[idx])
            if conf >= self.confidence_threshold:
                results[name] = {"label": label, "confidence": round(conf, 4)}
            else:
                results[name] = {"label": None, "confidence": round(conf, 4)}
        return results

    def _run_brand(self, image: Image.Image) -> dict:
        """Corre deteccion + clasificacion de marca sobre la imagen original."""
        if self.logo_detector is None or self.brand_classifier is None:
            return {"label": None, "confidence": 0.0}

        try:
            detections = self.logo_detector.detect(image)
            if not detections:
                return {"label": None, "confidence": 0.0}
            crops = self.logo_detector.crop_detections(image, detections)
            pred = self.brand_classifier.classify_from_detections(crops)
            return {"label": pred["brand"], "confidence": round(pred["confidence"], 4)}
        except Exception as exc:
            logger.error("Error en brand detection: %s", exc)
            return {"label": None, "confidence": 0.0}

    def process(self, image_path: str | Path) -> dict[str, Any]:
        """Procesa una imagen y retorna el JSON de atributos.

        Args:
            image_path: Path a la imagen.

        Returns:
            Dict con los atributos extraidos.
        """
        t0 = time.time()
        image_path = Path(image_path)
        image = self._load_image(image_path)

        if image is None:
            return {
                "image_path": str(image_path),
                "error": "no_se_pudo_cargar_imagen",
                "pipeline_version": PIPELINE_VERSION,
            }

        # 1. Segmentacion
        segmented = None
        segmented_path = None
        if self.run_segmentation:
            segmented, _mask = self._run_segmentation(image)
            if self.save_intermediate and segmented is not None:
                segmented_path = self.outputs_dir / f"{image_path.stem}_segmented.png"
                segmented.save(segmented_path)

        source_for_attrs = segmented if segmented is not None else image

        result: dict[str, Any] = {
            "image_path": str(image_path),
            "segmented_path": str(segmented_path) if segmented_path else None,
            "pipeline_version": PIPELINE_VERSION,
        }

        # 2. Color
        if self.run_color and self.color_extractor is not None:
            try:
                color_result = self.color_extractor.extract(source_for_attrs)
                result["color"] = {
                    "dominant": {
                        "hex": color_result.get("dominant_color"),
                        "name": color_result.get("dominant_color_name"),
                    },
                    "palette": color_result.get("palette", []),
                    "pattern": {
                        "label": color_result.get("pattern"),
                        "confidence": color_result.get("pattern_confidence", 0.0),
                    },
                }
            except Exception as exc:
                logger.error("Error en color: %s", exc)
                result["color"] = None

        # 3. Clasificacion multi-task
        if self.run_classification:
            classif = self._run_classification(source_for_attrs.convert("RGB"))
            result.update(classif)

        # 4. Marca (sobre imagen original, el logo puede estar en contexto)
        if self.run_brand:
            result["marca"] = self._run_brand(image)

        result["processing_time_ms"] = int((time.time() - t0) * 1000)
        return result

    def process_batch(
        self, image_paths: list[str | Path]
    ) -> list[dict[str, Any]]:
        """Procesa una lista de imagenes con barra de progreso.

        Args:
            image_paths: Lista de paths a imagenes.

        Returns:
            Lista de resultados JSON.
        """
        results = []
        for p in tqdm(image_paths, desc="Procesando"):
            try:
                results.append(self.process(p))
            except Exception as exc:
                logger.error("Fallo procesando %s: %s", p, exc)
                results.append({"image_path": str(p), "error": str(exc)})
        return results


def main() -> None:
    """CLI: procesa una imagen o un directorio."""
    parser = argparse.ArgumentParser(description="Fashion Feature Extraction Pipeline")
    parser.add_argument("--config", type=str, default="config/pipeline_config.yaml")
    parser.add_argument("--image", type=str, help="Path a una imagen unica")
    parser.add_argument("--dir", type=str, help="Directorio con imagenes a procesar")
    parser.add_argument("--output", type=str, default="outputs/results.json")
    args = parser.parse_args()

    pipeline = FashionPipeline(args.config)

    if args.image:
        result = pipeline.process(args.image)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    elif args.dir:
        d = Path(args.dir)
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        paths = [p for p in d.iterdir() if p.suffix.lower() in exts]
        results = pipeline.process_batch(paths)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Procesadas {len(results)} imagenes. Resultados en {out}.")
    else:
        parser.error("Debe especificar --image o --dir")


if __name__ == "__main__":
    main()
