"""Modulo de reconocimiento de marca."""

from .detector import LogoDetector
from .classifier import BrandClassifier

__all__ = ["LogoDetector", "BrandClassifier"]
