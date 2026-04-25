"""Modulo de clasificacion multi-task de prendas."""

from .model import MultiTaskFashionClassifier
from .train import evaluate, multi_task_loss, train_from_csv

__all__ = [
    "MultiTaskFashionClassifier",
    "evaluate",
    "multi_task_loss",
    "train_from_csv",
]
