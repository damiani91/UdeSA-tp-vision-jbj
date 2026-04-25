"""Tests para src/brand/ (sin GPU, sin descargas)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from src.brand.dataset import LogoDataset, collect_samples, discover_brands


@pytest.fixture
def dummy_logo_dir(tmp_path):
    """Crea estructura ImageFolder dummy con 3 marcas y 3 imagenes c/u."""
    for brand in ["nike", "adidas", "puma"]:
        d = tmp_path / brand
        d.mkdir()
        for i in range(3):
            arr = np.full((32, 32, 3), (i * 60 + 30) % 255, dtype=np.uint8)
            Image.fromarray(arr).save(d / f"{i}.jpg")
    return tmp_path


def test_discover_brands_returns_sorted(dummy_logo_dir):
    brands = discover_brands(dummy_logo_dir)
    assert brands == ["adidas", "nike", "puma"]


def test_collect_samples_count(dummy_logo_dir):
    brands = discover_brands(dummy_logo_dir)
    samples = collect_samples(dummy_logo_dir, brands)
    assert len(samples) == 9
    labels = [s[1] for s in samples]
    assert set(labels) == {0, 1, 2}


def test_logo_dataset_getitem(dummy_logo_dir):
    ds = LogoDataset(dummy_logo_dir, image_size=(32, 32), split="val")
    assert len(ds) == 9
    item = ds[0]
    assert isinstance(item["pixel_values"], torch.Tensor)
    assert item["pixel_values"].shape == (3, 32, 32)
    assert item["label"].dtype == torch.long


def test_logo_dataset_split_train_val(dummy_logo_dir):
    ds = LogoDataset(dummy_logo_dir, image_size=(32, 32), split="train")
    train_ds, val_ds = ds.split_train_val(val_ratio=0.34, seed=42)
    assert len(train_ds) + len(val_ds) == 9
    assert len(val_ds) >= 1


def test_brand_classifier_forward():
    from src.brand.classifier import BrandClassifier

    model = BrandClassifier(num_classes=5, pretrained=False).eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 5)


def test_brand_classifier_freeze_unfreeze():
    from src.brand.classifier import BrandClassifier

    model = BrandClassifier(num_classes=3, pretrained=False)
    model.freeze_backbone()
    n_train = sum(p.requires_grad for p in model.backbone.parameters())
    # Solo classifier layers deberian ser trainable
    assert n_train > 0  # algunas (classifier)
    model.unfreeze_backbone()
    assert all(p.requires_grad for p in model.backbone.parameters())
