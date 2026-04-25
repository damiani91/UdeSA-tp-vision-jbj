"""Orquestador del pipeline de extraccion de atributos de moda.

El pipeline procesa una imagen y produce un JSON con:
    - Segmentacion (mascara de la prenda + categoria dominante)
    - Color (paleta dominante + patron)
    - Atributos del clasificador (pants o tops segun routing)
    - Marca (Logo-2K+)
"""

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

PIPELINE_VERSION = "0.2.0"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_config(config_path: str | Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class FashionPipeline:
    """Pipeline end-to-end: imagen -> JSON de atributos.

    El routing pants vs tops se hace en base a la categoria dominante
    de la segmentacion (`upper_body` vs `lower_body`).
    """

    def __init__(self, config_path: str | Path) -> None:
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

        device_str = self.config.get("device", "cpu")
        if device_str == "cuda" and not torch.cuda.is_available():
            device_str = "cpu"
        self.device = torch.device(device_str)

        self.segmenter = None
        self.color_extractor = None
        self.pants_classifier = None
        self.tops_classifier = None
        self.brand_predictor = None

        routing = self.config.get("segmentation", {}).get("routing", {})
        self.upper_label = routing.get("upper_body_label", "tops")
        self.lower_label = routing.get("lower_body_label", "pants")
        self.full_body_fallback = routing.get("full_body_fallback", "tops")

        self._load_modules()

    def _load_classifier(self, key: str):
        """Carga un clasificador (pants o tops) si existe checkpoint."""
        from src.classification.model import MultiTaskFashionClassifier

        model_cfg = self.config.get(key, {})
        if not model_cfg:
            return None
        ckpt_path = Path(model_cfg.get("checkpoint", f"models/best_{key}.pth"))
        try:
            model = MultiTaskFashionClassifier.from_config(model_cfg, pretrained=False)
            if ckpt_path.exists():
                state = torch.load(ckpt_path, map_location="cpu")
                model.load_state_dict(state["model_state"])
                logger.info("Clasificador %s cargado desde %s", key, ckpt_path)
            else:
                logger.warning("Sin checkpoint para %s en %s, modelo aleatorio.",
                               key, ckpt_path)
            model.to(self.device).eval()
            return model
        except Exception as exc:
            logger.error("Fallo cargando clasificador %s: %s", key, exc)
            return None

    def _load_modules(self) -> None:
        if self.run_segmentation:
            try:
                from src.segmentation.segmenter import FashionSegmenter
                self.segmenter = FashionSegmenter(self.config)
            except Exception as exc:
                logger.error("Fallo al cargar segmenter: %s", exc)

        if self.run_color:
            try:
                from src.color.extractor import ColorExtractor
                self.color_extractor = ColorExtractor(self.config)
            except Exception as exc:
                logger.error("Fallo al cargar color extractor: %s", exc)

        if self.run_classification:
            self.pants_classifier = self._load_classifier("pants")
            self.tops_classifier = self._load_classifier("tops")

        if self.run_brand:
            try:
                from src.brand.classifier import BrandPredictor
                ckpt = Path(self.config["brand"].get("checkpoint", "models/best_brand.pth"))
                if ckpt.exists():
                    self.brand_predictor = BrandPredictor(
                        ckpt,
                        device=str(self.device),
                        confidence_threshold=self.config["brand"].get(
                            "confidence_threshold", 0.6),
                        unknown_label=self.config["brand"].get(
                            "unknown_label", "sin_marca"),
                    )
                    logger.info("BrandPredictor cargado desde %s", ckpt)
                else:
                    logger.warning("Sin checkpoint de brand en %s, deshabilitado.", ckpt)
            except Exception as exc:
                logger.error("Fallo al cargar brand predictor: %s", exc)

    def _route_classifier(self, dominant_category: Optional[str]) -> tuple[Optional[str], Optional[Any]]:
        """Decide que clasificador usar segun la categoria de segmentacion."""
        if dominant_category == "upper_body":
            return self.upper_label, self.tops_classifier
        if dominant_category == "lower_body":
            return self.lower_label, self.pants_classifier
        if dominant_category == "full_body":
            label = self.full_body_fallback
            return label, self.tops_classifier if label == "tops" else self.pants_classifier
        return None, None

    def _classify(self, image: Image.Image, model_key: str, model) -> dict[str, dict]:
        """Inferencia del clasificador multi-task."""
        if model is None:
            return {}
        from torchvision import transforms

        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        x = tf(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = model(x)

        head_config = self.config[model_key]["heads"]
        results = {}
        for head_name, logits in outputs.items():
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            idx = int(np.argmax(probs))
            label = head_config[head_name]["classes"][idx]
            conf = float(probs[idx])
            results[head_name] = {
                "label": label if conf >= self.confidence_threshold else None,
                "confidence": round(conf, 4),
            }
        return results

    def process(self, image_path: str | Path) -> dict[str, Any]:
        """Procesa una imagen y retorna el JSON con todos los atributos."""
        t0 = time.time()
        image_path = Path(image_path)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            return {
                "image_path": str(image_path),
                "error": f"no_se_pudo_cargar_imagen: {exc}",
                "pipeline_version": PIPELINE_VERSION,
            }

        result: dict[str, Any] = {
            "image_path": str(image_path),
            "pipeline_version": PIPELINE_VERSION,
        }

        # 1. Segmentacion + routing
        segmented_img = image
        dominant_category = None
        if self.run_segmentation and self.segmenter is not None:
            try:
                from src.segmentation.postprocess import (
                    apply_mask, clean_mask, crop_to_content, largest_connected_component,
                )
                seg_map = self.segmenter.predict(image)
                dominant_category = self.segmenter.get_dominant_category(seg_map)
                mask = self.segmenter.get_garment_mask(seg_map, dominant_category)
                mask = clean_mask(mask)
                mask = largest_connected_component(mask)
                rgba = apply_mask(image, mask)
                segmented_img = crop_to_content(rgba)
                if self.save_intermediate:
                    seg_path = self.outputs_dir / f"{image_path.stem}_segmented.png"
                    segmented_img.save(seg_path)
                    result["segmented_path"] = str(seg_path)
                result["segmentation"] = {
                    "dominant_category": dominant_category,
                    "mask_area_px": int(mask.sum()),
                }
            except Exception as exc:
                logger.error("Error en segmentacion: %s", exc)
                result["segmentation"] = {"error": str(exc)}

        # 2. Color
        if self.run_color and self.color_extractor is not None:
            try:
                color_result = self.color_extractor.extract(segmented_img)
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
                result["color"] = {"error": str(exc)}

        # 3. Clasificacion (routing por categoria)
        if self.run_classification:
            model_key, model = self._route_classifier(dominant_category)
            if model is not None and model_key is not None:
                rgb_for_clf = segmented_img.convert("RGB") if segmented_img.mode != "RGB" else segmented_img
                attrs = self._classify(rgb_for_clf, model_key, model)
                result["garment_type"] = model_key
                result["attributes"] = attrs
            else:
                result["garment_type"] = None
                result["attributes"] = {}

        # 4. Marca
        if self.run_brand and self.brand_predictor is not None:
            try:
                rgb_for_brand = segmented_img.convert("RGB") if segmented_img.mode != "RGB" else segmented_img
                result["brand"] = self.brand_predictor.predict(rgb_for_brand)
            except Exception as exc:
                logger.error("Error en brand: %s", exc)
                result["brand"] = {"error": str(exc)}

        result["processing_time_ms"] = int((time.time() - t0) * 1000)
        return result

    def process_batch(self, image_paths: list[str | Path]) -> list[dict[str, Any]]:
        results = []
        for p in tqdm(image_paths, desc="Procesando"):
            try:
                results.append(self.process(p))
            except Exception as exc:
                logger.error("Fallo procesando %s: %s", p, exc)
                results.append({"image_path": str(p), "error": str(exc)})
        return results


def main() -> None:
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
