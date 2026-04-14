"""Tests del pipeline orquestador."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from src.pipeline import FashionPipeline


def test_pipeline_loads_config(tmp_config_path):
    pipe = FashionPipeline(tmp_config_path)
    assert pipe.config is not None
    assert pipe.run_color is True


def test_pipeline_process_returns_dict_with_required_keys(tmp_config_path, tmp_image_path):
    pipe = FashionPipeline(tmp_config_path)
    result = pipe.process(tmp_image_path)
    assert isinstance(result, dict)
    assert result["image_path"] == str(tmp_image_path)
    assert "pipeline_version" in result
    assert "processing_time_ms" in result


def test_pipeline_process_nonexistent_image_returns_error(tmp_config_path, tmp_path):
    pipe = FashionPipeline(tmp_config_path)
    fake = tmp_path / "nope.jpg"
    result = pipe.process(fake)
    assert "error" in result


def test_pipeline_process_batch_returns_list(tmp_config_path, tmp_image_path):
    pipe = FashionPipeline(tmp_config_path)
    results = pipe.process_batch([tmp_image_path, tmp_image_path])
    assert isinstance(results, list)
    assert len(results) == 2


def test_pipeline_color_only_produces_palette(tmp_config_path, tmp_image_path):
    pipe = FashionPipeline(tmp_config_path)
    result = pipe.process(tmp_image_path)
    assert "color" in result
    assert "palette" in result["color"]
    assert len(result["color"]["palette"]) > 0
