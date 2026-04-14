"""Tests del modulo de clasificacion multi-task.

Se testea con pretrained=False para evitar descargar pesos reales en CI.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers", reason="transformers no instalado")

from src.classification.model import MultiTaskFashionClassifier


@pytest.fixture
def model(minimal_config):
    return MultiTaskFashionClassifier(minimal_config, pretrained=False)


def test_forward_returns_dict_with_head_keys(model, minimal_config):
    x = torch.randn(1, 3, 224, 224)
    outputs = model(x)
    assert set(outputs.keys()) == set(minimal_config["classification"]["heads"].keys())


def test_forward_output_shapes(model, minimal_config):
    x = torch.randn(2, 3, 224, 224)
    outputs = model(x)
    for name, cfg in minimal_config["classification"]["heads"].items():
        assert outputs[name].shape == (2, len(cfg["classes"]))


def test_freeze_backbone_sets_requires_grad_false(model):
    model.freeze_backbone()
    for p in model.backbone.parameters():
        assert p.requires_grad is False


def test_unfreeze_backbone_restores_requires_grad(model):
    model.freeze_backbone()
    model.unfreeze_backbone()
    for p in model.backbone.parameters():
        assert p.requires_grad is True
