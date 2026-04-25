"""Tests para src/data/csv_dataset.py."""

from __future__ import annotations

import torch

from src.data.csv_dataset import LABEL_MISSING, CSVImageDataset


def test_dataset_loads_rows(dummy_pants_csv, dummy_image_dir, head_config_pants):
    cache_dir, _ = dummy_image_dir
    ds = CSVImageDataset(
        dummy_pants_csv, cache_dir, head_config_pants,
        image_size=(32, 32), split="val",
    )
    assert len(ds) == 6


def test_getitem_returns_tensors(dummy_pants_csv, dummy_image_dir, head_config_pants):
    cache_dir, _ = dummy_image_dir
    ds = CSVImageDataset(
        dummy_pants_csv, cache_dir, head_config_pants,
        image_size=(32, 32), split="val",
    )
    item = ds[0]
    assert isinstance(item["pixel_values"], torch.Tensor)
    assert item["pixel_values"].shape == (3, 32, 32)
    assert set(item["labels"].keys()) == set(head_config_pants.keys())
    for v in item["labels"].values():
        assert isinstance(v, torch.Tensor)


def test_label_mapping_en_es(dummy_pants_csv, dummy_image_dir, head_config_pants):
    cache_dir, _ = dummy_image_dir
    ds = CSVImageDataset(
        dummy_pants_csv, cache_dir, head_config_pants,
        image_size=(32, 32), split="val",
    )
    # Fila 0: Neutrals-Dark, Non-Pattern/Solid, Slim-Fit, Mid-Rise
    item = ds[0]
    assert item["labels"]["color_family"].item() == 0  # neutrals_dark
    assert item["labels"]["pattern"].item() == 0  # liso
    assert item["labels"]["fit_silhouette"].item() == 0  # slim
    assert item["labels"]["waist_rise"].item() == 0  # mid_rise


def test_unknown_label_maps_to_otro_or_missing(dummy_pants_csv, dummy_image_dir):
    cache_dir, _ = dummy_image_dir
    head_config = {"color_family": {"classes": ["neutrals_dark", "otro"], "weight": 1.0}}
    ds = CSVImageDataset(
        dummy_pants_csv, cache_dir, head_config, image_size=(32, 32), split="val",
    )
    # Fila 1 (Blues) no esta en classes -> debe ir a "otro"
    item = ds[1]
    assert item["labels"]["color_family"].item() == 1  # otro


def test_compute_class_weights_shape(dummy_pants_csv, dummy_image_dir, head_config_pants):
    cache_dir, _ = dummy_image_dir
    ds = CSVImageDataset(
        dummy_pants_csv, cache_dir, head_config_pants,
        image_size=(32, 32), split="val",
    )
    weights = ds.compute_class_weights()
    assert set(weights.keys()) == set(head_config_pants.keys())
    for name, cfg in head_config_pants.items():
        assert weights[name].shape == (len(cfg["classes"]),)


def test_missing_image_returns_zero_tensor(tmp_path, head_config_pants):
    """Si la imagen no esta en cache, debe retornar tensor (no crashear)."""
    import pandas as pd
    csv = tmp_path / "x.csv"
    pd.DataFrame([
        {"id": 1, "image_url": "https://nope.com/x.jpg",
         "color_family": "Neutrals-Dark", "pattern": "Solid",
         "fit_silhouette": "Slim-Fit", "waist_rise": "Mid-Rise"}
    ]).to_csv(csv, index=False)
    ds = CSVImageDataset(csv, tmp_path, head_config_pants,
                         image_size=(32, 32), split="val")
    item = ds[0]
    assert item["pixel_values"].shape == (3, 32, 32)
