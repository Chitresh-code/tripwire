"""Amount-based transaction features.

`is_high_amount` is the one function both the live API and the training
pipeline call — that shared function is what keeps them in sync.
"""

from __future__ import annotations

import pandas as pd

from src.config import FeatureSettings

_settings = FeatureSettings()  # type: ignore[call-arg]  # fields load from configs/features.yaml


def is_high_amount(amount: float, threshold: float = _settings.high_amount_threshold) -> bool:
    return amount >= threshold


def score_transaction(transaction: dict[str, float]) -> dict[str, bool]:
    """Online path: compute features for a single live transaction."""
    return {"is_high_amount": is_high_amount(transaction["amount"])}


def compute_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Offline path: compute the same features for a full training table."""
    out = transactions.copy()
    out["is_high_amount"] = out["amount"].apply(is_high_amount)
    return out
