"""Generacion de splits estratificados train/val/test desde un CSV."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def _join(base: str | Path, name: str) -> str | Path:
    """Une base + name respetando el esquema gs:// (Path los destruye)."""
    if str(base).startswith("gs://"):
        return f"{str(base).rstrip('/')}/{name}"
    return Path(base) / name


def generate_splits(
    csv_path: str | Path,
    output_dir: str | Path,
    stratify_col: str = "color_family",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    min_samples_per_class: int = 3,
    seed: int = 42,
    name_prefix: Optional[str] = None,
    filter_urls: Optional[set[str]] = None,
) -> dict[str, Path]:
    """Genera splits estratificados y los guarda como CSVs.

    Args:
        csv_path: Path al CSV original con todas las columnas.
        output_dir: Directorio donde escribir los splits.
        stratify_col: Columna a usar para estratificar.
        train_ratio: Fraccion train.
        val_ratio: Fraccion val.
        test_ratio: Fraccion test.
        min_samples_per_class: Minimo de ejemplos por clase. Las clases con
            menos se descartan (no se pueden estratificar).
        seed: Semilla.
        name_prefix: Prefijo para los archivos. Si None, usa el stem del CSV.
        filter_urls: Si se provee, conserva solo filas cuya `image_url` este en
            este set (util para descartar imagenes no descargadas).

    Returns:
        Dict {"train": Path, "val": Path, "test": Path}.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Los ratios deben sumar 1.0")

    df = pd.read_csv(csv_path)
    name_prefix = name_prefix or Path(csv_path).stem
    if not str(output_dir).startswith("gs://"):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    n_orig = len(df)

    if filter_urls is not None:
        df = df[df["image_url"].isin(filter_urls)].reset_index(drop=True)
        logger.info("Filtro por URLs disponibles: %d -> %d filas", n_orig, len(df))

    df = df.dropna(subset=[stratify_col]).reset_index(drop=True)

    counts = df[stratify_col].value_counts()
    valid_classes = counts[counts >= min_samples_per_class].index
    n_filtered = (~df[stratify_col].isin(valid_classes)).sum()
    if n_filtered > 0:
        logger.warning(
            "Descartando %d filas con clases <%d ejemplos en '%s'",
            n_filtered, min_samples_per_class, stratify_col,
        )
    df = df[df[stratify_col].isin(valid_classes)].reset_index(drop=True)

    y = df[stratify_col]
    df_trainval, df_test = train_test_split(
        df, test_size=test_ratio, stratify=y, random_state=seed,
    )
    val_relative = val_ratio / (train_ratio + val_ratio)
    df_train, df_val = train_test_split(
        df_trainval,
        test_size=val_relative,
        stratify=df_trainval[stratify_col],
        random_state=seed,
    )

    paths = {
        "train": _join(output_dir, f"{name_prefix}_train.csv"),
        "val": _join(output_dir, f"{name_prefix}_val.csv"),
        "test": _join(output_dir, f"{name_prefix}_test.csv"),
    }
    df_train.to_csv(paths["train"], index=False)
    df_val.to_csv(paths["val"], index=False)
    df_test.to_csv(paths["test"], index=False)

    logger.info(
        "Splits %s: train=%d val=%d test=%d (descartadas %d)",
        name_prefix, len(df_train), len(df_val), len(df_test), n_orig - len(df),
    )
    return paths
