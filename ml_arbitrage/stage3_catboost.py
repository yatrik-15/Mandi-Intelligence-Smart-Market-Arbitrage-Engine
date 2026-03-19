"""
stage3_catboost.py — Stage 3: Optimal Action Classification via CatBoost
=========================================================================
WHY CATBOOST OVER PLAIN XGBOOST HERE:
  - CatBoost has native, order-statistics-based encoding for high-cardinality
    categoricals like `mandi_id` (hundreds of mandis) and `crop_name`.
    XGBoost requires manual OHE or target-encoding — introducing target
    leakage risk and a preprocessing pipeline to maintain.
  - CatBoost's symmetric trees and oblivious decision process make it
    significantly more robust to feature noise and outliers in the
    net_profit features that are derived from TFT forecasts (which carry
    their own uncertainty).
  - Ordered boosting (CatBoost's default) prevents overfitting on small
    agricultural datasets where labelled examples are scarce.
  - Feature importance in CatBoost is computed per-class for multiclass,
    enabling interpretable "why did the engine choose STORE?" explanations
    for farmers and field agents.

Classes
-------
    0: SELL_LOCAL_NOW       — Immediate sale at current mandi
    1: TRAVEL_TO_FAR_MANDI  — Transport to higher-price destination
    2: STORE_FOR_T_DAYS     — Hold in cold storage / warehouse
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from .config import CatBoostConfig

logger = logging.getLogger(__name__)

_MODEL_FNAME = "catboost_classifier.cbm"


class CatBoostDecisionClassifier:
    """
    Wraps CatBoost for three-class action prediction.

    Input
    -----
    Feature DataFrame produced by MandiArbitrageEngine._build_feature_row(),
    containing both raw Mandi attributes and the economic signals computed
    by Prophet + TFT + Arbitrage Math in Stages 1-2.

    Output
    ------
    Per-sample action label + probability dictionary.
    """

    CLASS_NAMES = ["SELL_LOCAL_NOW", "TRAVEL_TO_FAR_MANDI", "STORE_FOR_T_DAYS"]

    def __init__(self, config: CatBoostConfig, model_dir: Path) -> None:
        self.config    = config
        self.model_dir = model_dir
        self._model: Optional[CatBoostClassifier] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────────────────

    def train(
        self,
        X:          pd.DataFrame,
        y:          pd.Series,
        X_val:      Optional[pd.DataFrame] = None,
        y_val:      Optional[pd.Series]    = None,
    ) -> None:
        """
        Fit CatBoost with optional eval set for early stopping.

        Parameters
        ----------
        X, y          : Training features + labels
        X_val, y_val  : Optional validation features + labels.
                        If None, 20 % of training data is used.
        """
        cfg = self.config

        if X_val is None:
            split   = int(len(X) * 0.80)
            X_val   = X.iloc[split:].copy()
            y_val   = y.iloc[split:].copy()
            X       = X.iloc[:split].copy()
            y       = y.iloc[:split].copy()

        X_train_feat = X[cfg.feature_cols]
        X_val_feat   = X_val[cfg.feature_cols]

        # Locate positional indices of categorical features
        cat_indices = [X_train_feat.columns.get_loc(c)
                       for c in cfg.cat_features
                       if c in X_train_feat.columns]

        train_pool = Pool(X_train_feat, label=y,     cat_features=cat_indices)
        eval_pool  = Pool(X_val_feat,   label=y_val, cat_features=cat_indices)

        model = CatBoostClassifier(
            iterations          = cfg.iterations,
            learning_rate       = cfg.learning_rate,
            depth               = cfg.depth,
            l2_leaf_reg         = cfg.l2_leaf_reg,
            random_seed         = cfg.random_seed,
            eval_metric         = cfg.eval_metric,
            loss_function       = cfg.loss_function,
            verbose             = cfg.verbose,
            early_stopping_rounds = cfg.early_stopping_rounds,
            class_names         = cfg.class_names,
        )

        logger.info("Training CatBoost (max %d iterations) …", cfg.iterations)
        model.fit(train_pool, eval_set=eval_pool)

        self._model = model
        save_path   = self.model_dir / _MODEL_FNAME
        model.save_model(str(save_path))
        logger.info("CatBoost model saved to %s", save_path)

        # Log top-5 feature importances
        importances = model.get_feature_importance(train_pool)
        fi_df = (
            pd.DataFrame({"feature": cfg.feature_cols, "importance": importances})
            .sort_values("importance", ascending=False)
            .head(5)
        )
        logger.info("Top-5 CatBoost feature importances:\n%s", fi_df.to_string(index=False))

    # ─────────────────────────────────────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────────────────────────────────────

    def predict(
        self, X: pd.DataFrame
    ) -> Tuple[List[str], List[Dict[str, float]]]:
        """
        Predict optimal action labels + per-class probabilities.

        Returns
        -------
        (labels, probas)
            labels : List[str]  — predicted class names
            probas : List[Dict[str, float]]  — {class_name: probability}
        """
        if self._model is None:
            self._load_model()

        cfg          = self.config
        X_feat       = X[cfg.feature_cols]
        cat_indices  = [X_feat.columns.get_loc(c)
                        for c in cfg.cat_features if c in X_feat.columns]
        pool         = Pool(X_feat, cat_features=cat_indices)

        raw_labels   = self._model.predict(pool).flatten()
        raw_probas   = self._model.predict_proba(pool)

        class_order  = self._model.classes_
        labels  = [str(l) for l in raw_labels]
        probas  = [
            {str(cls): float(p) for cls, p in zip(class_order, row)}
            for row in raw_probas
        ]
        return labels, probas

    def predict_proba_df(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convenience: return probabilities as a DataFrame."""
        _, probas = self.predict(X)
        return pd.DataFrame(probas)

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        path = self.model_dir / _MODEL_FNAME
        if not path.exists():
            raise FileNotFoundError(
                f"CatBoost model not found at {path}. "
                "Call CatBoostDecisionClassifier.train() first."
            )
        logger.info("Loading CatBoost model from %s", path)
        model = CatBoostClassifier()
        model.load_model(str(path))
        self._model = model
