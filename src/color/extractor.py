"""Extraccion de paleta de color y clasificacion de patron."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PIL import Image

from .color_names import find_nearest_color_name, rgb_to_hex

logger = logging.getLogger(__name__)


class ColorExtractor:
    """Extrae color dominante, paleta y patron de una prenda.

    Usa K-Means en espacio LAB (perceptualmente uniforme) y mapea centroides
    a nombres en espanol via Delta E CIE2000.
    """

    def __init__(self, config: dict) -> None:
        """Inicializa el extractor desde la config.

        Args:
            config: Dict de configuracion con clave 'color'.
        """
        cfg = config.get("color", {})
        self.n_clusters = int(cfg.get("n_clusters", 5))
        self.color_space = str(cfg.get("color_space", "LAB")).upper()
        self.min_cluster_ratio = float(cfg.get("min_cluster_ratio", 0.05))
        self.dominant_threshold = float(cfg.get("dominant_threshold", 0.3))
        self.color_naming = bool(cfg.get("color_naming", True))
        pattern_cfg = cfg.get("pattern_classifier", {})
        self.pattern_enabled = bool(pattern_cfg.get("enabled", True))
        self.pattern_classes = pattern_cfg.get(
            "classes", ["liso", "rayas", "cuadros", "estampado"]
        )
        self.seed = int(config.get("seed", 42))

    def extract(self, image: Image.Image) -> dict:
        """Pipeline completo de extraccion de color.

        Args:
            image: Imagen PIL (idealmente RGBA con transparencia como mascara).

        Returns:
            Dict con dominant_color, palette y pattern.
        """
        pixels_lab, pixels_rgb = self._prepare_pixels(image)
        if len(pixels_lab) == 0:
            logger.warning("No hay pixeles validos, retornando resultado vacio.")
            return {
                "dominant_color": None,
                "dominant_color_name": None,
                "palette": [],
                "pattern": None,
                "pattern_confidence": 0.0,
            }

        centroids_lab, centroids_rgb, percentages = self._cluster_colors(
            pixels_lab, pixels_rgb
        )
        palette = self._rank_clusters(centroids_lab, centroids_rgb, percentages)

        if not palette:
            return {
                "dominant_color": None,
                "dominant_color_name": None,
                "palette": [],
                "pattern": None,
                "pattern_confidence": 0.0,
            }

        dominant = palette[0]
        pattern_label, pattern_conf = (None, 0.0)
        if self.pattern_enabled:
            pattern_label, pattern_conf = self._classify_pattern(image)

        return {
            "dominant_color": dominant["hex"],
            "dominant_color_name": dominant.get("name"),
            "palette": palette,
            "pattern": pattern_label,
            "pattern_confidence": pattern_conf,
        }

    def _prepare_pixels(
        self, image: Image.Image
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extrae pixeles validos y los convierte a LAB.

        Args:
            image: Imagen PIL. Si tiene canal alpha, se usan solo los pixeles
                con alpha > 0.

        Returns:
            Tupla (pixels_lab (N,3), pixels_rgb (N,3) en 0-255).
        """
        from skimage.color import rgb2lab

        if image.mode == "RGBA":
            arr = np.array(image)
            rgb = arr[:, :, :3]
            alpha = arr[:, :, 3]
            mask = alpha > 0
            pixels_rgb = rgb[mask]
        else:
            rgb = np.array(image.convert("RGB"))
            pixels_rgb = rgb.reshape(-1, 3)

        if len(pixels_rgb) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)

        pixels_float = pixels_rgb.astype(np.float64) / 255.0
        pixels_lab = rgb2lab(pixels_float.reshape(1, -1, 3)).reshape(-1, 3)
        return pixels_lab, pixels_rgb

    def _cluster_colors(
        self, pixels_lab: np.ndarray, pixels_rgb: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Corre K-Means sobre los pixeles en LAB.

        Args:
            pixels_lab: Array (N, 3) de pixeles en LAB.
            pixels_rgb: Array (N, 3) de pixeles en RGB correspondientes.

        Returns:
            Tupla (centroids_lab (K,3), centroids_rgb (K,3), percentages (K,)).
        """
        from sklearn.cluster import KMeans

        # Limitar tamano para velocidad si hay muchos pixeles
        max_samples = 20000
        if len(pixels_lab) > max_samples:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(len(pixels_lab), max_samples, replace=False)
            pixels_lab_s = pixels_lab[idx]
            pixels_rgb_s = pixels_rgb[idx]
        else:
            pixels_lab_s = pixels_lab
            pixels_rgb_s = pixels_rgb

        k = min(self.n_clusters, len(pixels_lab_s))
        kmeans = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=self.seed,
        )
        labels = kmeans.fit_predict(pixels_lab_s)
        centroids_lab = kmeans.cluster_centers_

        # Centroides RGB: promedio de pixeles RGB en cada cluster
        centroids_rgb = np.zeros((k, 3), dtype=np.float64)
        percentages = np.zeros(k, dtype=np.float64)
        total = len(labels)
        for i in range(k):
            mask = labels == i
            cnt = int(mask.sum())
            percentages[i] = cnt / total if total > 0 else 0.0
            if cnt > 0:
                centroids_rgb[i] = pixels_rgb_s[mask].mean(axis=0)

        return centroids_lab, centroids_rgb, percentages

    def _rank_clusters(
        self,
        centroids_lab: np.ndarray,
        centroids_rgb: np.ndarray,
        percentages: np.ndarray,
    ) -> list[dict]:
        """Ordena clusters por porcentaje y filtra pequenos.

        Args:
            centroids_lab: Centroides en LAB.
            centroids_rgb: Centroides en RGB.
            percentages: Porcentaje de cada cluster.

        Returns:
            Lista de dicts ordenados por porcentaje descendente.
        """
        order = np.argsort(-percentages)
        results = []
        for i in order:
            pct = float(percentages[i])
            if pct < self.min_cluster_ratio:
                continue
            rgb_tup = tuple(int(round(c)) for c in centroids_rgb[i])
            entry = {
                "hex": rgb_to_hex(rgb_tup),
                "rgb": list(rgb_tup),
                "percentage": round(pct, 4),
            }
            if self.color_naming:
                name, delta = find_nearest_color_name(centroids_lab[i])
                entry["name"] = name
                entry["delta_e"] = round(float(delta), 3)
            results.append(entry)

        return results

    def _classify_pattern(
        self, image: Image.Image, grid: int = 8
    ) -> tuple[str, float]:
        """Clasifica el patron cromatico mediante heuristica de varianza.

        Calcula varianza de color dentro y entre patches de una grilla. Patches
        uniformes sugieren "liso", varianza alta sugiere "estampado".

        Args:
            image: Imagen PIL (usa zona con alpha>0 si es RGBA).
            grid: Tamano de la grilla (grid x grid patches).

        Returns:
            Tupla (nombre_patron, confidence).
        """
        # Recortar al bounding box del contenido si es RGBA
        if image.mode == "RGBA":
            arr = np.array(image)
            alpha = arr[:, :, 3]
            ys, xs = np.where(alpha > 0)
            if len(xs) == 0:
                return "liso", 0.5
            y1, y2 = int(ys.min()), int(ys.max())
            x1, x2 = int(xs.min()), int(xs.max())
            rgb_crop = arr[y1:y2 + 1, x1:x2 + 1, :3]
            mask_crop = alpha[y1:y2 + 1, x1:x2 + 1] > 0
        else:
            rgb_crop = np.array(image.convert("RGB"))
            mask_crop = np.ones(rgb_crop.shape[:2], dtype=bool)

        h, w = rgb_crop.shape[:2]
        if h < grid * 2 or w < grid * 2:
            return "liso", 0.5

        patch_means = []
        patch_stds = []
        ph, pw = h // grid, w // grid
        for i in range(grid):
            for j in range(grid):
                y0 = i * ph
                y1 = (i + 1) * ph
                x0 = j * pw
                x1 = (j + 1) * pw
                patch = rgb_crop[y0:y1, x0:x1]
                pmask = mask_crop[y0:y1, x0:x1]
                valid = patch[pmask]
                if len(valid) == 0:
                    continue
                patch_means.append(valid.mean(axis=0))
                patch_stds.append(valid.std(axis=0).mean())

        if not patch_means:
            return "liso", 0.5

        patch_means = np.array(patch_means)
        patch_stds = np.array(patch_stds)

        inter_patch_var = patch_means.std(axis=0).mean()
        intra_patch_var = patch_stds.mean()

        # Reglas heuristicas simples
        if inter_patch_var < 8 and intra_patch_var < 15:
            return "liso", 0.85
        if inter_patch_var >= 8 and intra_patch_var < 20:
            # Varianza entre patches pero baja dentro -> patron regular
            return "rayas", 0.6
        if inter_patch_var > 20 and intra_patch_var > 25:
            return "estampado", 0.7
        return "estampado", 0.5
