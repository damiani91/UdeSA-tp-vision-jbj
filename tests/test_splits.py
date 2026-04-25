"""Tests para src/data/splits.py."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.splits import generate_splits


def test_splits_create_three_files(dummy_pants_csv, tmp_path):
    """Con 6 filas y 3 clases (2 ejemplos cada una) y min_samples=2 debe armar splits."""
    paths = generate_splits(
        csv_path=dummy_pants_csv,
        output_dir=tmp_path / "splits",
        stratify_col="color_family",
        train_ratio=0.5, val_ratio=0.25, test_ratio=0.25,
        min_samples_per_class=2,
        seed=42,
    )
    assert set(paths.keys()) == {"train", "val", "test"}
    for p in paths.values():
        assert p.exists()


def test_splits_sum_equals_input(dummy_pants_csv, tmp_path):
    paths = generate_splits(
        csv_path=dummy_pants_csv,
        output_dir=tmp_path / "splits",
        stratify_col="color_family",
        train_ratio=0.5, val_ratio=0.25, test_ratio=0.25,
        min_samples_per_class=2,
        seed=42,
    )
    n_total = sum(len(pd.read_csv(p)) for p in paths.values())
    assert n_total == 6


def test_splits_invalid_ratios_raises(dummy_pants_csv, tmp_path):
    with pytest.raises(ValueError):
        generate_splits(
            csv_path=dummy_pants_csv, output_dir=tmp_path,
            train_ratio=0.5, val_ratio=0.5, test_ratio=0.5,
        )


def test_splits_filter_urls(dummy_pants_csv, dummy_image_dir, tmp_path):
    _, urls = dummy_image_dir
    keep = set(urls[:3])
    paths = generate_splits(
        csv_path=dummy_pants_csv,
        output_dir=tmp_path / "splits",
        stratify_col="color_family",
        train_ratio=0.34, val_ratio=0.33, test_ratio=0.33,
        min_samples_per_class=1,
        filter_urls=keep,
    )
    n_total = sum(len(pd.read_csv(p)) for p in paths.values())
    assert n_total == 3
