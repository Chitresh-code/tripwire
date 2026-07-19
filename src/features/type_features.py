"""Transaction-type feature: is this a type PaySim's fraud is restricted to?

Verified against the real data (not assumed): PaySim's fraud generator only
ever labels TRANSFER (0.77% fraud) and CASH_OUT (0.18% fraud) transactions as
fraud — CASH_IN, DEBIT, and PAYMENT are always 0% fraud. This is the single
strongest signal in the dataset.
"""

from __future__ import annotations

import pandas as pd


def is_transfer(transaction_type: str) -> bool:
    return transaction_type == "TRANSFER"


def is_cash_out(transaction_type: str) -> bool:
    return transaction_type == "CASH_OUT"


def score_transaction(transaction_type: str) -> dict[str, bool]:
    """Online path: compute type features for a single live transaction."""
    return {
        "is_transfer": is_transfer(transaction_type),
        "is_cash_out": is_cash_out(transaction_type),
    }


def compute_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Offline path: compute the same features for a full training table."""
    out = transactions.copy()
    out["is_transfer"] = out["transaction_type"].apply(is_transfer)
    out["is_cash_out"] = out["transaction_type"].apply(is_cash_out)
    return out
