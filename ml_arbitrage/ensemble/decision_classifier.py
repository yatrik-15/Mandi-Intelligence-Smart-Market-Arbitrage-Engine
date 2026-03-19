"""
Stage 3: Optimal Action Classification via CatBoost
====================================================
Improved version with Pool management and structured probas.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from catboost import CatBoostClassifier, Pool
from .config import CatBoostConfig

logger = logging.getLogger(__name__)

class CatBoostDecisionClassifier:
    """
    Wraps CatBoost for three-class action prediction.
    """

    def __init__(self, config: CatBoostConfig, model_dir: Path) -> None:
        self.config = config
        self.model_dir = model_dir
        self._model: Optional[CatBoostClassifier] = None

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        X_feat = X[self.config.feature_cols]
        cat_indices = [X_feat.columns.get_loc(c) for c in self.config.cat_features if c in X_feat.columns]
        
        train_pool = Pool(X_feat, label=y, cat_features=cat_indices)
        
        self._model = CatBoostClassifier(
            iterations=self.config.iterations,
            learning_rate=self.config.learning_rate,
            depth=self.config.depth,
            loss_function=self.config.loss_function,
            random_seed=self.config.random_seed,
            verbose=self.config.verbose
        )
        
        self._model.fit(train_pool)
        
    def predict(self, X: pd.DataFrame) -> Tuple[List[str], List[Dict[str, float]]]:
        if self._model is None:
            # Fallback for skeleton if not trained
            actions = ["SELL_LOCAL_NOW"] * len(X)
            probas = [{"SELL_LOCAL_NOW": 0.9, "TRAVEL_TO_FAR_MANDI": 0.05, "STORE_FOR_T_DAYS": 0.05}] * len(X)
            return actions, probas

        X_feat = X[self.config.feature_cols]
        cat_indices = [X_feat.columns.get_loc(c) for c in self.config.cat_features if c in X_feat.columns]
        pool = Pool(X_feat, cat_features=cat_indices)

        raw_labels = self._model.predict(pool).flatten()
        raw_probas = self._model.predict_proba(pool)
        
        class_order = self._model.classes_
        labels = [str(l) for l in raw_labels]
        probas = [
            {str(cls): float(p) for cls, p in zip(class_order, row)}
            for row in raw_probas
        ]
        return labels, probas
