"""Train a model end-to-end and save its artifacts to the registry.

One function, used both for the very first baseline (scripts/train_baseline.py)
and for automated retraining (scripts/check_drift.py) — so both paths save
the same three things the same way: the model, its evaluation metrics, and a
drift reference fitted on its own training data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from src.models import registry
from src.models.baseline import FEATURE_COLUMNS, evaluate, train_baseline
from src.monitoring.drift import fit_reference, fit_references, save_references
from src.pipelines.build_training_table import build_feature_table, time_based_split


@dataclass
class TrainingRun:
    model: LGBMClassifier
    metrics: dict[str, float]
    train: pd.DataFrame
    test: pd.DataFrame


def run_training(transactions: pd.DataFrame) -> TrainingRun:
    features = build_feature_table(transactions)
    train, test = time_based_split(features)
    model = train_baseline(train)
    metrics = evaluate(model, test)
    return TrainingRun(model=model, metrics=metrics, train=train, test=test)


def save_artifacts(run: TrainingRun, version: str) -> None:
    """Saves the model, its metrics, and a drift reference fitted on `run.train`."""
    registry.REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(run.model, registry.model_path(version))
    registry.metrics_path(version).write_text(json.dumps(run.metrics))

    references = fit_references(run.train)
    train_scores = np.asarray(run.model.predict_proba(run.train[FEATURE_COLUMNS]))[:, 1]
    references["fraud_probability"] = fit_reference(pd.Series(train_scores))
    save_references(references, registry.drift_reference_path(version))
