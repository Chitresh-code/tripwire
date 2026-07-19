"""Runs every offline feature over a transactions table and splits it for training.

This is the offline half of train/serve parity: the same feature functions
`src/serving/` will call live are run here in batch, over the whole table.
"""

from __future__ import annotations

import pandas as pd

from src.config import PipelineSettings
from src.features import amount_features, velocity_features

_settings = PipelineSettings()  # type: ignore[call-arg]  # fields load from configs/pipeline.yaml


def build_feature_table(transactions: pd.DataFrame) -> pd.DataFrame:
    """Attach every offline feature column to a canonical transactions table."""
    features = amount_features.compute_features(transactions)
    features = velocity_features.compute_features(features)
    return features


def time_based_split(
    features: pd.DataFrame, test_fraction: float = _settings.test_fraction
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time, not randomly: earlier transactions train, later ones test.

    A random split would leak information — the model could train on
    transactions that, in the real world, hadn't happened yet at scoring
    time. Splitting by time is the only leakage-safe way to evaluate.
    """
    ordered = features.sort_values("timestamp").reset_index(drop=True)
    split_index = int(len(ordered) * (1 - test_fraction))
    split_time = ordered.loc[split_index, "timestamp"]

    train = ordered[ordered["timestamp"] < split_time]
    test = ordered[ordered["timestamp"] >= split_time]
    return train, test
