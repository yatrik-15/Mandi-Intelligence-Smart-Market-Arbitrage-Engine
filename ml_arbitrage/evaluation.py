"""
evaluation.py — Unified Metrics Block
======================================
Covers:
  - RMSE for TFT price forecast quality (Stages 1A + 1B)
  - Precision / Recall / F1-Score for CatBoost classification (Stage 3)
  - Per-class breakdown and confusion matrix printing
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_squared_error,
    precision_recall_fscore_support,
)

logger = logging.getLogger(__name__)

_DIVIDER = "─" * 68


class ModelEvaluator:
    """Centralised evaluation utilities for all three pipeline stages."""

    # ─────────────────────────────────────────────────────────────────────────
    # Regression (TFT Price Forecasts)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def rmse(y_true: List[float], y_pred: List[float]) -> float:
        """
        Root Mean Squared Error — primary metric for price forecast quality.
        Penalises large errors more than MAE, appropriate for commodity pricing
        where large mispredictions lead to catastrophic arbitrage decisions.
        """
        arr_true = np.array(y_true, dtype=float)
        arr_pred = np.array(y_pred, dtype=float)
        mask     = ~(np.isnan(arr_true) | np.isnan(arr_pred))
        if mask.sum() == 0:
            logger.warning("RMSE: no valid (non-NaN) pairs found; returning NaN.")
            return float("nan")
        return float(np.sqrt(mean_squared_error(arr_true[mask], arr_pred[mask])))

    @staticmethod
    def mae(y_true: List[float], y_pred: List[float]) -> float:
        """Mean Absolute Error — interpretable in price units (₹/quintal)."""
        arr_true = np.array(y_true, dtype=float)
        arr_pred = np.array(y_pred, dtype=float)
        return float(np.mean(np.abs(arr_true - arr_pred)))

    @staticmethod
    def mape(y_true: List[float], y_pred: List[float]) -> float:
        """
        Mean Absolute Percentage Error — scale-invariant, useful for
        comparing across different crops with very different price levels.
        NOTE: Undefined when y_true contains zeros.
        """
        arr_true = np.array(y_true, dtype=float)
        arr_pred = np.array(y_pred, dtype=float)
        nonzero  = arr_true != 0
        return float(np.mean(np.abs((arr_true[nonzero] - arr_pred[nonzero]) / arr_true[nonzero])) * 100)

    # ─────────────────────────────────────────────────────────────────────────
    # Classification (CatBoost Decision Classifier)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def classification_report(
        y_true: List[str],
        y_pred: List[str],
        labels: Optional[List[str]] = None,
    ) -> Dict:
        """
        Compute per-class and macro-averaged Precision, Recall, F1-Score.

        Returns
        -------
        dict with keys: per_class, macro_avg, weighted_avg, accuracy
        """
        if labels is None:
            labels = sorted(set(y_true) | set(y_pred))

        report_str = classification_report(y_true, y_pred, labels=labels, zero_division=0)
        logger.info("\n%s\nCATBOOST CLASSIFICATION REPORT\n%s\n%s", _DIVIDER, _DIVIDER, report_str)

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )

        per_class = {
            lbl: {
                "precision": float(precision[i]),
                "recall":    float(recall[i]),
                "f1_score":  float(f1[i]),
                "support":   int(support[i]),
            }
            for i, lbl in enumerate(labels)
        }

        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        wp, wr, wf1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        accuracy = float(np.mean(np.array(y_true) == np.array(y_pred)))

        return {
            "per_class":    per_class,
            "macro_avg":    {"precision": float(macro_p), "recall": float(macro_r), "f1_score": float(macro_f1)},
            "weighted_avg": {"precision": float(wp),      "recall": float(wr),      "f1_score": float(wf1)},
            "accuracy":     accuracy,
        }

    @staticmethod
    def confusion_matrix_df(y_true: List[str], y_pred: List[str]) -> pd.DataFrame:
        """Returns a labelled confusion matrix as a DataFrame."""
        labels = sorted(set(y_true) | set(y_pred))
        cm     = confusion_matrix(y_true, y_pred, labels=labels)
        return pd.DataFrame(cm, index=[f"actual_{l}" for l in labels],
                                columns=[f"pred_{l}" for l in labels])

    # ─────────────────────────────────────────────────────────────────────────
    # Unified Report Printer
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def print_report(results: Dict) -> None:
        """Pretty-print the full evaluation results dict to stdout."""
        print(f"\n{'═' * 68}")
        print("  MANDI ARBITRAGE ENGINE — EVALUATION REPORT")
        print(f"{'═' * 68}")

        if "tft_rmse_t3" in results:
            print(f"\n  [TFT Price Forecast]")
            print(f"  {'RMSE (t+3)':<28}  ₹ {results['tft_rmse_t3']:.4f} / quintal")
            print(f"  {'RMSE (t+7)':<28}  ₹ {results['tft_rmse_t7']:.4f} / quintal")

        if "classification" in results:
            clf = results["classification"]
            print(f"\n  [CatBoost Decision Classifier]")
            print(f"  {'Accuracy':<28}  {clf['accuracy']:.4f}")
            print(f"\n  {'Class':<28}  {'Precision':>10}  {'Recall':>8}  {'F1-Score':>10}  {'Support':>8}")
            print(f"  {_DIVIDER}")
            for cls, m in clf["per_class"].items():
                print(f"  {cls:<28}  {m['precision']:>10.4f}  {m['recall']:>8.4f}  {m['f1_score']:>10.4f}  {m['support']:>8d}")
            print(f"  {_DIVIDER}")
            ma = clf["macro_avg"]
            wa = clf["weighted_avg"]
            print(f"  {'macro avg':<28}  {ma['precision']:>10.4f}  {ma['recall']:>8.4f}  {ma['f1_score']:>10.4f}")
            print(f"  {'weighted avg':<28}  {wa['precision']:>10.4f}  {wa['recall']:>8.4f}  {wa['f1_score']:>10.4f}")

        print(f"\n{'═' * 68}\n")
