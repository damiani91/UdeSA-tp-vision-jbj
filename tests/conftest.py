"""Fixtures compartidos para los tests."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image


@pytest.fixture
def dummy_rgb_image() -> Image.Image:
    """Imagen RGB 64x64 con dos bloques de color."""
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[:, :32, 0] = 255
    arr[:, 32:, 2] = 255
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def dummy_red_image() -> Image.Image:
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    arr[:, :, 0] = 230
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def dummy_rgba_image() -> Image.Image:
    arr = np.zeros((64, 64, 4), dtype=np.uint8)
    arr[16:48, 16:48, 0] = 200
    arr[16:48, 16:48, 1] = 50
    arr[16:48, 16:48, 2] = 50
    arr[16:48, 16:48, 3] = 255
    return Image.fromarray(arr, mode="RGBA")


@pytest.fixture
def dummy_mask() -> np.ndarray:
    m = np.zeros((64, 64), dtype=np.uint8)
    m[16:48, 16:48] = 1
    return m


@pytest.fixture
def head_config_pants() -> dict:
    """Head config minima para tests de pants."""
    return {
        "color_family": {
            "classes": ["neutrals_dark", "blues", "reds", "otro"],
            "weight": 1.0,
        },
        "pattern": {"classes": ["liso", "estampado", "otro"], "weight": 1.0},
        "fit_silhouette": {"classes": ["slim", "regular", "otro"], "weight": 0.8},
        "waist_rise": {"classes": ["mid_rise", "high_rise", "otro"], "weight": 0.5},
    }


@pytest.fixture
def dummy_image_dir(tmp_path) -> Path:
    """Directorio con 6 imagenes dummy nombradas con md5(url)."""
    import hashlib
    d = tmp_path / "cache"
    d.mkdir()
    urls = []
    for i in range(6):
        url = f"https://example.com/img_{i}.jpg"
        urls.append(url)
        arr = np.full((32, 32, 3), (i * 40) % 255, dtype=np.uint8)
        Image.fromarray(arr).save(d / (hashlib.md5(url.encode()).hexdigest() + ".jpg"))
    return d, urls


@pytest.fixture
def dummy_pants_csv(tmp_path, dummy_image_dir) -> Path:
    """CSV con 6 filas de pants y URLs apuntando al cache local."""
    _, urls = dummy_image_dir
    rows = [
        {"id": 1, "image_url": urls[0], "color_family": "Neutrals-Dark",
         "pattern": "Non-Pattern/Solid", "fit_silhouette": "Slim-Fit",
         "fabric_content": "Polyester", "dressing_syle": "Minimalist",
         "waist_rise": "Mid-Rise"},
        {"id": 2, "image_url": urls[1], "color_family": "Blues",
         "pattern": "Stripes", "fit_silhouette": "Regular-Fit",
         "fabric_content": "Cotton", "dressing_syle": "Casual",
         "waist_rise": "High-Rise"},
        {"id": 3, "image_url": urls[2], "color_family": "Reds",
         "pattern": "Non-Pattern/Solid", "fit_silhouette": "Slim-Fit",
         "fabric_content": "Cotton", "dressing_syle": "Classic",
         "waist_rise": "Mid-Rise"},
        {"id": 4, "image_url": urls[3], "color_family": "Neutrals-Dark",
         "pattern": "Non-Pattern/Solid", "fit_silhouette": "Regular-Fit",
         "fabric_content": "Polyester", "dressing_syle": "Minimalist",
         "waist_rise": "Mid-Rise"},
        {"id": 5, "image_url": urls[4], "color_family": "Blues",
         "pattern": "Printed", "fit_silhouette": "Slim-Fit",
         "fabric_content": "Cotton", "dressing_syle": "Casual",
         "waist_rise": "High-Rise"},
        {"id": 6, "image_url": urls[5], "color_family": "Reds",
         "pattern": "Non-Pattern/Solid", "fit_silhouette": "Regular-Fit",
         "fabric_content": "Polyester", "dressing_syle": "Classic",
         "waist_rise": "Low-Rise"},
    ]
    p = tmp_path / "pants.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


@pytest.fixture
def tmp_image_path(tmp_path, dummy_rgb_image):
    p = tmp_path / "dummy.png"
    dummy_rgb_image.save(p)
    return p
