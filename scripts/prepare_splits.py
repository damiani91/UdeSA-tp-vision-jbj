"""Genera splits estratificados train/val/test desde un CSV.

Opcionalmente filtra por las imagenes efectivamente descargadas (CSV log).

Ejemplo:
    python scripts/prepare_splits.py \
        --csv data/preprocessed/pants_1.csv \
        --output data/splits \
        --stratify color_family \
        --download-log data/splits/pants_download_log.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.splits import generate_splits  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera splits estratificados")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--output", type=str, default="data/splits")
    parser.add_argument("--stratify", type=str, default="color_family")
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--download-log", type=str, default=None,
                        help="Si se provee, filtra solo URLs con success=True")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    filter_urls = None
    if args.download_log is not None:
        log = pd.read_csv(args.download_log)
        filter_urls = set(log[log["success"]]["url"].tolist())
        print(f"Filtro: {len(filter_urls)} URLs descargadas exitosamente")

    paths = generate_splits(
        csv_path=args.csv,
        output_dir=args.output,
        stratify_col=args.stratify,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        min_samples_per_class=args.min_samples,
        seed=args.seed,
        name_prefix=args.prefix,
        filter_urls=filter_urls,
    )
    print("\n✓ Splits generados:")
    for split, path in paths.items():
        print(f"  {split}: {path} ({len(pd.read_csv(path)):,} filas)")


if __name__ == "__main__":
    main()
