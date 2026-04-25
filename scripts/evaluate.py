"""Evaluacion del pipeline contra ground truth desde CSV.

Compara las predicciones del pipeline (output JSON) con los labels
del CSV original, tras aplicar el mapeo EN-ES.

Ejemplo:
    python scripts/evaluate.py \
        --predictions outputs/results.json \
        --csv data/preprocessed/pants_1.csv \
        --dataset pants \
        --output outputs/evaluation/report_pants.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.mappings import MAPPING_EN_ES, apply_mapping  # noqa: E402

logger = logging.getLogger(__name__)


def evaluate_attribute(
    predictions: list[dict],
    gt_df: pd.DataFrame,
    attribute: str,
    id_col: str = "id",
) -> dict:
    """Compara predicciones vs ground truth para un atributo.

    Las predicciones se asumen con shape:
        {"id": int, "attributes": {attr: {"label": str, "confidence": float}}}
    El ground truth se mapea EN -> ES con MAPPING_EN_ES.
    """
    from sklearn.metrics import classification_report, f1_score

    gt_lookup = gt_df.set_index(id_col)[attribute].to_dict() if attribute in gt_df.columns else {}
    mapping = MAPPING_EN_ES.get(attribute, {})

    y_pred, y_true = [], []
    for p in predictions:
        pid = p.get("id") or p.get("image_id")
        if pid is None or pid not in gt_lookup:
            continue
        pred = (p.get("attributes", {}) or {}).get(attribute, {}).get("label")
        gt_raw = gt_lookup[pid]
        gt = apply_mapping(gt_raw, mapping)
        if pred is None or gt is None:
            continue
        y_pred.append(pred)
        y_true.append(gt)

    if not y_true:
        return {"accuracy": 0.0, "f1_macro": 0.0, "n": 0}

    acc = float(np.mean(np.array(y_pred) == np.array(y_true)))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return {
        "accuracy": acc,
        "f1_macro": f1,
        "n": len(y_true),
        "report": classification_report(y_true, y_pred, zero_division=0, output_dict=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path al JSON de predicciones (lista o dict)")
    parser.add_argument("--csv", type=str, required=True,
                        help="CSV con ground truth (pants_1.csv o tops_1.csv)")
    parser.add_argument("--dataset", choices=["pants", "tops"], required=True)
    parser.add_argument("--output", type=str, default="outputs/evaluation/report.json")
    parser.add_argument("--id-col", type=str, default="id")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.predictions) as f:
        preds = json.load(f)
    if not isinstance(preds, list):
        preds = [preds]

    gt_df = pd.read_csv(args.csv)

    pants_attrs = ["color_family", "pattern", "fit_silhouette", "fabric_content",
                   "dressing_syle", "waist_rise"]
    tops_attrs = ["color_family", "pattern", "fit_silhouette", "fabric_content",
                  "dressing_syle", "neck_style"]
    attrs = pants_attrs if args.dataset == "pants" else tops_attrs

    report: dict[str, Any] = {
        "dataset": args.dataset,
        "n_predictions": len(preds),
        "attributes": {},
    }
    for attr in attrs:
        m = evaluate_attribute(preds, gt_df, attr, args.id_col)
        report["attributes"][attr] = m
        logger.info("%s: acc=%.3f f1=%.3f n=%d", attr, m["accuracy"], m["f1_macro"], m["n"])

    if any(m["n"] > 0 for m in report["attributes"].values()):
        report["overall"] = {
            "mean_accuracy": float(np.mean([m["accuracy"] for m in report["attributes"].values() if m["n"] > 0])),
            "mean_f1_macro": float(np.mean([m["f1_macro"] for m in report["attributes"].values() if m["n"] > 0])),
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Reporte: %s", out)


if __name__ == "__main__":
    main()
