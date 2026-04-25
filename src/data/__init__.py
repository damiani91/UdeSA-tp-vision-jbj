"""Data layer: mapeos EN-ES, descarga de imagenes, datasets y splits."""

from src.data.csv_dataset import LABEL_MISSING, CSVImageDataset
from src.data.downloader import ImageDownloader, download_csv_images
from src.data.mappings import (
    LONG_TAIL_THRESHOLD,
    MAPPING_EN_ES,
    apply_mapping,
    group_long_tail,
    map_series,
)
from src.data.splits import generate_splits

__all__ = [
    "CSVImageDataset",
    "LABEL_MISSING",
    "ImageDownloader",
    "download_csv_images",
    "MAPPING_EN_ES",
    "LONG_TAIL_THRESHOLD",
    "apply_mapping",
    "group_long_tail",
    "map_series",
    "generate_splits",
]
