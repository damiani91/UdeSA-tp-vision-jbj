"""Descarga imagenes de un CSV con URLs en paralelo.

Ejemplo:
    python scripts/download_images.py \
        --csv data/preprocessed/pants_1.csv \
        --output data/images/pants \
        --sample 15000 \
        --workers 8
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Permitir ejecutar desde la raiz del repo sin instalar
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.downloader import download_csv_images  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga imagenes desde un CSV con URLs")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--output", type=str, required=True, help="Directorio cache")
    parser.add_argument("--url-col", type=str, default="image_url")
    parser.add_argument("--sample", type=int, default=None,
                        help="Si se provee, descarga solo N filas (sample estratificado)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log", type=str, default=None,
                        help="Path para guardar el log de descarga")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    log = download_csv_images(
        csv_path=args.csv,
        cache_dir=args.output,
        url_col=args.url_col,
        sample=args.sample,
        workers=args.workers,
        timeout=args.timeout,
        seed=args.seed,
        log_path=args.log,
    )
    n_ok = int(log["success"].sum())
    print(f"\n✓ Descargadas {n_ok}/{len(log)} imagenes en {args.output}")
    if n_ok < len(log):
        print(f"  Fallaron {len(log) - n_ok}. Revisa --log para detalles.")


if __name__ == "__main__":
    main()
