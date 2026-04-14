"""Script de evaluacion modular del pipeline.

Computa metricas por modulo a partir de predicciones y ground truth en JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Calcula IoU entre dos mascaras binarias."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter / union) if union > 0 else 0.0


def evaluate_classification(
    predictions: list[dict], ground_truth: list[dict], head: str
) -> dict:
    """Calcula accuracy y macro-F1 para una head especifica."""
    from sklearn.metrics import classification_report, f1_score

    y_pred, y_true = [], []
    for p, g in zip(predictions, ground_truth):
        pred_label = (p.get(head) or {}).get("label")
        gt_label = g.get(head)
        if gt_label is None or pred_label is None:
            continue
        y_pred.append(pred_label)
        y_true.append(gt_label)

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
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--ground-truth", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/evaluation/report.json")
    parser.add_argument(
        "--heads",
        nargs="+",
        default=["tipo", "estilo", "fit", "cuello", "manga", "material"],
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.predictions) as f:
        preds = json.load(f)
    with open(args.ground_truth) as f:
        gts = json.load(f)

    if not isinstance(preds, list):
        preds = [preds]
    if not isinstance(gts, list):
        gts = [gts]

    assert len(preds) == len(gts), "predictions y ground-truth deben tener misma longitud"

    report: dict[str, Any] = {"n_samples": len(preds), "heads": {}}
    for head in args.heads:
        report["heads"][head] = evaluate_classification(preds, gts, head)
        logger.info(
            "%s: acc=%.3f f1=%.3f n=%d",
            head,
            report["heads"][head]["accuracy"],
            report["heads"][head]["f1_macro"],
            report["heads"][head]["n"],
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Reporte guardado en %s", out)


if __name__ == "__main__":
    main()
