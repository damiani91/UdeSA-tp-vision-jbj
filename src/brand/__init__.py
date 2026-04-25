"""Modulo de clasificacion de marca sobre Logo-2K+."""

from .classifier import BrandClassifier, BrandPredictor
from .dataset import LogoDataset, collect_samples, discover_brands
from .train import evaluate, train_brand

__all__ = [
    "BrandClassifier",
    "BrandPredictor",
    "LogoDataset",
    "collect_samples",
    "discover_brands",
    "evaluate",
    "train_brand",
]
