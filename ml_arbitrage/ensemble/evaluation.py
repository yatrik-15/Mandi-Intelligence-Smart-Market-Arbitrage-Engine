"""
evaluation.py — Accuracy metrics for TFT and CatBoost.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, precision_recall_fscore_support

class ModelEvaluator:
    """
    Calculates RMSE and Classification metrics.
    """

    @staticmethod
    def rmse(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mask = ~np.isnan(y_true)
        if not np.any(mask): return 0.0
        return np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))

    @staticmethod
    def classification_metrics(y_true, y_pred):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
