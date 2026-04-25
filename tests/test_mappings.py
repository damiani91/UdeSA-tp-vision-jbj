"""Tests para src/data/mappings.py."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.mappings import (
    DEFAULT_OTHER,
    LONG_TAIL_THRESHOLD,
    MAPPING_EN_ES,
    apply_mapping,
    group_long_tail,
    map_series,
)


def test_apply_mapping_known_value():
    assert apply_mapping("Slim-Fit", MAPPING_EN_ES["fit_silhouette"]) == "slim"
    assert apply_mapping("V-Neck", MAPPING_EN_ES["neck_style"]) == "en_v"


def test_apply_mapping_unknown_returns_default():
    assert apply_mapping("UnknownThing", MAPPING_EN_ES["pattern"]) == DEFAULT_OTHER


def test_apply_mapping_none_returns_none():
    assert apply_mapping(None, MAPPING_EN_ES["color_family"]) is None
    assert apply_mapping(float("nan"), MAPPING_EN_ES["color_family"]) is None


def test_map_series_full_column():
    s = pd.Series(["Non-Pattern/Solid", "Stripes", "Floral", "Bogus"])
    out = map_series(s, "pattern")
    assert out.tolist() == ["liso", "rayas", "floral", DEFAULT_OTHER]


def test_map_series_unknown_attribute_raises():
    with pytest.raises(KeyError):
        map_series(pd.Series(["x"]), "no_existe")


def test_long_tail_grouping():
    s = pd.Series(["a"] * 95 + ["b"] * 4 + ["c"] * 1)
    out = group_long_tail(s, threshold=0.05)
    assert "a" in out.tolist()
    # b y c estan debajo del 5% -> deben ir a "otro"
    assert (out == "otro").sum() == 5
    assert "b" not in out.tolist()
    assert "c" not in out.tolist()


def test_long_tail_threshold_constant():
    assert 0 < LONG_TAIL_THRESHOLD < 0.1


def test_mapping_covers_main_attributes():
    """Los atributos clave de los CSV deben estar en MAPPING_EN_ES."""
    expected = {"color_family", "pattern", "fit_silhouette",
                "fabric_content", "dressing_syle", "waist_rise", "neck_style"}
    assert expected.issubset(MAPPING_EN_ES.keys())
