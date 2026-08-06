"""Simulates delayed label arrival (PRD FR12, §5): a transaction's true
fraud label isn't actually knowable the instant it happens — a chargeback
takes days to get filed. Training "as of" some moment must only use labels
that would genuinely have arrived by then. Using today's known outcome for
a transaction that happened yesterday — when in the real world nobody would
know that outcome yet — is leakage from the future into training data, the
exact failure mode FR12 exists to prevent (and `docs/ARCHITECTURE.md` §4
flags the point-in-time label join as leakage-critical).

Simplification, stated plainly (this was PRD §10's own open question — "a
simpler offline-simulated version... document the tradeoff"): every
transaction gets the same delay distribution, fraud and legitimate alike. A
real system would model "no chargeback within N days -> treat as negative"
separately from "chargeback arrived -> positive" (an asymmetric process —
you only ever get an explicit signal for the positive case). That's a real
second axis of complexity, not needed here to demonstrate and test the
leakage-safe join itself, which is what FR12 actually asks for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import LabelDelaySettings

_settings = LabelDelaySettings()  # type: ignore[call-arg]  # fields load from configs/labels.yaml


def simulate_label_delay(transactions: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Attaches `label_available_at`: when this transaction's true label would
    actually become known, not when the transaction itself happened."""
    rng = np.random.default_rng(seed)
    delay_days = rng.uniform(_settings.min_days, _settings.max_days, size=len(transactions))
    out = transactions.copy()
    out["label_available_at"] = out["timestamp"] + pd.to_timedelta(delay_days, unit="D")
    return out


def labels_available_as_of(transactions: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Only the rows whose delayed label has actually arrived by `as_of` — the
    leakage-safe view for a retraining job running "now"."""
    return transactions[transactions["label_available_at"] <= as_of]
