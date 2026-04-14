"""Fixtures compartidos para los tests."""

from __future__ import annotations

import numpy as np
import pytest
import yaml
from PIL import Image


@pytest.fixture
def minimal_config() -> dict:
    """Config minima para tests (sin modelos pesados)."""
    return {
        "device": "cpu",
        "seed": 42,
        "image_size": [224, 224],
        "paths": {
            "raw_data": "data/raw",
            "processed_data": "data/processed",
            "splits": "data/splits",
            "outputs": "outputs",
            "models": "models",
        },
        "segmentation": {
            "model_name": "mattmdjaga/segformer_b2_clothes",
            "confidence_threshold": 0.5,
            "target_categories": ["upper_body", "lower_body"],
        },
        "color": {
            "n_clusters": 3,
            "color_space": "LAB",
            "min_cluster_ratio": 0.05,
            "dominant_threshold": 0.3,
            "color_naming": True,
            "pattern_classifier": {"enabled": True, "classes": ["liso", "estampado"]},
        },
        "classification": {
            "backbone": "google/vit-base-patch16-224",
            "freeze_backbone_epochs": 1,
            "heads": {
                "tipo": {"classes": ["remera", "camisa", "pantalon"], "weight": 1.0},
                "estilo": {"classes": ["casual", "formal"], "weight": 0.8},
            },
            "training": {
                "epochs": 1,
                "batch_size": 2,
                "learning_rate": 3.0e-4,
                "weight_decay": 0.01,
                "label_smoothing": 0.0,
                "augmentation": {"horizontal_flip": True},
            },
        },
        "brand": {
            "detector": {
                "model": "yolov8n",
                "confidence_threshold": 0.4,
                "iou_threshold": 0.5,
            },
            "classifier": {
                "target_brands": ["nike", "adidas"],
                "model": "efficientnet_b0",
            },
        },
        "pipeline": {
            "run_segmentation": False,
            "run_color": True,
            "run_classification": False,
            "run_brand": False,
            "save_intermediate": False,
            "confidence_threshold": 0.0,
        },
    }


@pytest.fixture
def dummy_rgb_image() -> Image.Image:
    """Imagen RGB 64x64 con tres bloques de color."""
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[:, :32, 0] = 255        # Mitad izq rojo puro
    arr[:, 32:, 2] = 255        # Mitad der azul puro
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def dummy_red_image() -> Image.Image:
    """Imagen RGB 32x32 toda roja."""
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    arr[:, :, 0] = 230
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def dummy_rgba_image() -> Image.Image:
    """Imagen RGBA 64x64 con contenido en centro y bordes transparentes."""
    arr = np.zeros((64, 64, 4), dtype=np.uint8)
    arr[16:48, 16:48, 0] = 200
    arr[16:48, 16:48, 1] = 50
    arr[16:48, 16:48, 2] = 50
    arr[16:48, 16:48, 3] = 255
    return Image.fromarray(arr, mode="RGBA")


@pytest.fixture
def dummy_mask() -> np.ndarray:
    """Mascara binaria 64x64 con un cuadrado central."""
    m = np.zeros((64, 64), dtype=np.uint8)
    m[16:48, 16:48] = 1
    return m


@pytest.fixture
def tmp_image_path(tmp_path, dummy_rgb_image):
    """Guarda dummy_rgb_image en tmp_path y retorna el path."""
    p = tmp_path / "dummy.png"
    dummy_rgb_image.save(p)
    return p


@pytest.fixture
def tmp_config_path(tmp_path, minimal_config):
    """Guarda minimal_config como YAML y retorna el path."""
    p = tmp_path / "config.yaml"
    with open(p, "w") as f:
        yaml.safe_dump(minimal_config, f)
    return p
