"""Baseline gradient-boosted-tree fraud model.

This is the "does an off-the-shelf GBT even work" baseline every later model
(sequence model, cost-based thresholding) gets compared against.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (  # type: ignore[import-untyped]
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

FEATURE_COLUMNS = [
    "amount",
    "is_high_amount",
    "sender_txn_count_recent",
    "recipient_txn_count_recent",
    "is_transfer",
    "is_cash_out",
]
LABEL_COLUMN = "is_fraud"


def train_baseline(train: pd.DataFrame) -> LGBMClassifier:
    """Train a LightGBM classifier.

    No class-weighting: tested scale_pos_weight (full imbalance ratio and
    its square root) and is_unbalance=True against real PaySim data — all
    three saturated predict_proba to a near-constant score (ROC-AUC ~0.5) or
    actively hurt ranking quality. Plain unweighted training scored far
    better (ROC-AUC 0.91 vs ~0.5-0.67). See docs/DECISIONS.md — this departs
    from the PRD's suggested class-weighting and is flagged there, not
    applied silently.
    """
    x = train[FEATURE_COLUMNS]
    y = train[LABEL_COLUMN]

    model = LGBMClassifier(random_state=0, verbose=-1)
    model.fit(x, y)
    return model


def precision_at_recall(y_true: pd.Series, y_score: np.ndarray, target_recall: float) -> float:
    """Best precision achievable at or above a given recall — PRD's core imbalance-aware metric."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    achievable = precision[recall >= target_recall]
    return float(achievable.max()) if len(achievable) else 0.0


def evaluate(model: LGBMClassifier, test: pd.DataFrame) -> dict[str, float]:
    """Offline evaluation report: ranking quality + precision at fixed recall levels."""
    x = test[FEATURE_COLUMNS]
    y = test[LABEL_COLUMN]
    y_score = np.asarray(model.predict_proba(x))[:, 1]

    return {
        "roc_auc": roc_auc_score(y, y_score),
        "pr_auc": average_precision_score(y, y_score),
        "precision_at_recall_50": precision_at_recall(y, y_score, 0.5),
        "precision_at_recall_80": precision_at_recall(y, y_score, 0.8),
    }
