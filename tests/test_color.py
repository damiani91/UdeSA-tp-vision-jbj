"""Tests del modulo de extraccion de color."""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.color.color_names import (
    COLOR_REFERENCES_LAB,
    find_nearest_color_name,
    rgb_to_hex,
)
from src.color.extractor import ColorExtractor


def test_rgb_to_hex():
    assert rgb_to_hex((255, 0, 0)) == "#FF0000"
    assert rgb_to_hex((0, 0, 0)) == "#000000"
    assert rgb_to_hex((255, 255, 255)) == "#FFFFFF"


def test_find_nearest_color_name_red():
    from skimage.color import rgb2lab

    lab = rgb2lab(np.array([[[1.0, 0.0, 0.0]]]))[0, 0]
    name, _ = find_nearest_color_name(lab)
    assert "rojo" in name.lower()


def test_find_nearest_color_name_black():
    from skimage.color import rgb2lab

    lab = rgb2lab(np.array([[[0.0, 0.0, 0.0]]]))[0, 0]
    name, _ = find_nearest_color_name(lab)
    assert name == "negro"


def test_color_names_dict_has_no_duplicates():
    names = list(COLOR_REFERENCES_LAB.keys())
    assert len(names) == len(set(names))


def test_extractor_red_image(minimal_config, dummy_red_image):
    ext = ColorExtractor(minimal_config)
    result = ext.extract(dummy_red_image)
    assert result["palette"], "Debe haber al menos un color"
    dominant = result["palette"][0]
    # Debe haber al menos un R alto
    assert dominant["rgb"][0] > 150
    # Nombre debe contener 'rojo' o similar
    assert "rojo" in (dominant.get("name") or "").lower() or "carmesí" in (
        dominant.get("name") or ""
    ).lower() or "coral" in (dominant.get("name") or "").lower()


def test_extractor_percentages_sum_close_to_one(minimal_config, dummy_rgb_image):
    ext = ColorExtractor(minimal_config)
    result = ext.extract(dummy_rgb_image)
    total = sum(item["percentage"] for item in result["palette"])
    assert 0.9 <= total <= 1.0


def test_pattern_solid_is_liso(minimal_config, dummy_red_image):
    ext = ColorExtractor(minimal_config)
    result = ext.extract(dummy_red_image)
    assert result["pattern"] == "liso"
