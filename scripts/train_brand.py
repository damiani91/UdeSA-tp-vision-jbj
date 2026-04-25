"""Entry point para entrenar el clasificador de marca sobre Logo-2K+."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.brand.train import train_brand  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/pipeline_config.yaml")
    parser.add_argument("--history", type=str, default="outputs/history_brand.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    with open(args.config) as f:
        config = yaml.safe_load(f)

    result = train_brand(config, history_path=args.history)
    print(f"\n✓ Best F1-macro: {result['best_metric']:.3f}")
    print(f"✓ Clases: {result['n_classes']}")
    print(f"✓ Checkpoint: {result['checkpoint']}")


if __name__ == "__main__":
    main()
