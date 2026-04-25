"""Tests para src/data/downloader.py (sin red)."""

from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image

from src.data.downloader import ImageDownloader, url_to_filename


def test_url_to_filename_deterministic():
    url = "https://example.com/test.jpg"
    assert url_to_filename(url) == hashlib.md5(url.encode()).hexdigest() + ".jpg"
    assert url_to_filename(url) == url_to_filename(url)


def test_url_to_filename_different_urls():
    a = url_to_filename("https://a.com/img.jpg")
    b = url_to_filename("https://b.com/img.jpg")
    assert a != b


def test_is_cached_returns_false_for_missing(tmp_path):
    d = ImageDownloader(tmp_path)
    assert d.is_cached("https://nope.com/foo.jpg") is False


def test_is_cached_returns_true_for_valid(tmp_path):
    d = ImageDownloader(tmp_path)
    url = "https://example.com/img.jpg"
    arr = np.full((10, 10, 3), 128, dtype=np.uint8)
    Image.fromarray(arr).save(d.cache_path(url))
    assert d.is_cached(url) is True


def test_is_cached_deletes_corrupt_file(tmp_path):
    d = ImageDownloader(tmp_path)
    url = "https://example.com/bad.jpg"
    p = d.cache_path(url)
    p.write_bytes(b"not an image")
    assert d.is_cached(url) is False
    assert not p.exists()


def test_cache_path_under_cache_dir(tmp_path):
    d = ImageDownloader(tmp_path / "cache")
    p = d.cache_path("https://example.com/x.jpg")
    assert (tmp_path / "cache").exists()
    assert p.parent == tmp_path / "cache"
