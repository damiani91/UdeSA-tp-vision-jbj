"""Entry point para entrenar el clasificador de pants."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classification.train import train_from_csv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/pipeline_config.yaml")
    parser.add_argument("--train-csv", type=str, default="data/splits/pants_1_train.csv")
    parser.add_argument("--val-csv", type=str, default="data/splits/pants_1_val.csv")
    parser.add_argument("--cache-dir", type=str, default="data/images/pants")
    parser.add_argument("--history", type=str, default="outputs/history_pants.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    with open(args.config) as f:
        config = yaml.safe_load(f)

    result = train_from_csv(
        config=config,
        dataset_key="pants",
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        cache_dir=args.cache_dir,
        history_path=args.history,
    )
    print(f"\n✓ Best F1-macro: {result['best_metric']:.3f}")
    print(f"✓ Checkpoint: {result['checkpoint']}")


if __name__ == "__main__":
    main()
