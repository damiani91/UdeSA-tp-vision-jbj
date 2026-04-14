"""Tests del modulo de segmentacion (solo postprocess, sin cargar modelo)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.segmentation.postprocess import (
    apply_mask,
    clean_mask,
    crop_to_content,
    largest_connected_component,
)


def test_apply_mask_produces_rgba(dummy_rgb_image, dummy_mask):
    result = apply_mask(dummy_rgb_image, dummy_mask)
    assert result.mode == "RGBA"
    assert result.size == dummy_rgb_image.size


def test_apply_mask_alpha_matches_mask(dummy_rgb_image, dummy_mask):
    result = apply_mask(dummy_rgb_image, dummy_mask)
    arr = np.array(result)
    # Donde mask=0 -> alpha=0
    assert (arr[:, :, 3][dummy_mask == 0] == 0).all()
    # Donde mask=1 -> alpha=255
    assert (arr[:, :, 3][dummy_mask == 1] == 255).all()


def test_crop_to_content_smaller_than_input(dummy_rgba_image):
    cropped = crop_to_content(dummy_rgba_image, padding=0)
    assert cropped.size[0] <= dummy_rgba_image.size[0]
    assert cropped.size[1] <= dummy_rgba_image.size[1]


def test_clean_mask_removes_small_noise():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:50, 10:50] = 1
    # Ruido pequeno aislado
    mask[0, 0] = 1
    mask[2, 2] = 1
    cleaned = clean_mask(mask, kernel_size=3)
    # El cuadrado grande debe persistir
    assert cleaned[25, 25] == 1
    # Los pixeles aislados de ruido deben desaparecer tras opening
    assert cleaned[0, 0] == 0


def test_largest_connected_component_keeps_biggest():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[5:15, 5:15] = 1     # pequeno
    mask[20:60, 20:60] = 1   # grande
    result = largest_connected_component(mask)
    # El componente grande debe quedar
    assert result[30, 30] == 1
    # El componente chico debe haberse eliminado
    assert result[10, 10] == 0
