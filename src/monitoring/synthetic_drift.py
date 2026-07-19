"""Injects an obvious, controlled distribution shift for demonstrating drift detection.

Public fraud datasets don't reliably drift on their own (docs/PRD.md's own
risk note), so this hand-engineers one: a burst of unusually large-amount
transactions, simulating a new fraud pattern showing up mid-stream.
"""

from __future__ import annotations

import pandas as pd


def inject_amount_spike(transactions: pd.DataFrame, multiplier: float = 8.0) -> pd.DataFrame:
    """Scales every amount up, simulating a shift toward much larger transactions."""
    shifted = transactions.copy()
    shifted["amount"] = shifted["amount"] * multiplier
    return shifted
