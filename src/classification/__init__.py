"""Modulo de clasificacion multi-task de prendas."""

from .model import MultiTaskFashionClassifier
from .dataset import DeepFashionDataset, LABEL_MISSING

__all__ = [
    "MultiTaskFashionClassifier",
    "DeepFashionDataset",
    "LABEL_MISSING",
]
