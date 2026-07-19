"""Turns the raw PaySim CSV into Tripwire's canonical transaction shape.

Canonical columns: transaction_id, account_id, recipient_id, timestamp,
amount, transaction_type, is_fraud. Everything downstream (src/features/,
src/models/) reads this shape, not PaySim's raw column names — so swapping
datasets later only means changing the loader, not every feature.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import IngestionSettings

_settings = IngestionSettings()  # type: ignore[call-arg]  # fields load from configs/ingestion.yaml


def load_paysim(path: Path | str) -> pd.DataFrame:
    """Read a PaySim CSV and return it in Tripwire's canonical transaction schema."""
    raw = pd.read_csv(path, usecols=["step", "type", "nameOrig", "nameDest", "amount", "isFraud"])

    return pd.DataFrame(
        {
            "transaction_id": raw.index,
            "account_id": raw["nameOrig"],
            "recipient_id": raw["nameDest"],
            "timestamp": _settings.paysim_epoch + pd.to_timedelta(raw["step"], unit="h"),
            "amount": raw["amount"],
            "transaction_type": raw["type"],
            "is_fraud": raw["isFraud"].astype(bool),
        }
    )
