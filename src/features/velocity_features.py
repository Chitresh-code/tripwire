"""Velocity features: how many transactions has this entity been part of recently?

Two versions of the same idea:
- sender velocity: is this account suddenly transacting a lot? (account takeover)
- recipient velocity: is this account suddenly receiving a lot? (money-mule / cash-out burst)

Both reuse the exact same shared math (`count_in_window`) and the same
generic history tracker (`TransactionHistory`) — only the window length and
which ID column feeds them differs.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timedelta

import pandas as pd

from src.config import FeatureSettings

_settings = FeatureSettings()  # type: ignore[call-arg]  # fields load from configs/features.yaml
_SENDER_WINDOW = timedelta(minutes=_settings.sender_velocity_window_minutes)
_RECIPIENT_WINDOW = timedelta(minutes=_settings.recipient_velocity_window_minutes)


def count_in_window(
    timestamps: Sequence[datetime], current_time: datetime, window: timedelta
) -> int:
    """How many of `timestamps` fall in the window right before `current_time`."""
    return sum(1 for t in timestamps if current_time - window <= t < current_time)


class TransactionHistory:
    """Online path: remembers, per ID, the recent timestamps it's shown up at.

    Generic on purpose — the same class tracks "sender" and "recipient"
    history, they're just two separate instances with two separate windows.

    ponytail: process-local dict, fine for one scoring process. Swap for
    Redis (a shared online feature store) once scoring runs on more than
    one instance, so every instance sees the same history.
    """

    def __init__(self, window: timedelta) -> None:
        self._window = window
        self._by_id: dict[str, list[datetime]] = defaultdict(list)

    def count_recent(self, entity_id: str, timestamp: datetime) -> int:
        count = count_in_window(self._by_id[entity_id], timestamp, self._window)
        self._by_id[entity_id].append(timestamp)
        return count


def score_transaction(
    sender_history: TransactionHistory,
    recipient_history: TransactionHistory,
    account_id: str,
    recipient_id: str,
    timestamp: datetime,
) -> dict[str, int]:
    """Online path: compute both velocity features for a single live transaction."""
    return {
        "sender_txn_count_recent": sender_history.count_recent(account_id, timestamp),
        "recipient_txn_count_recent": recipient_history.count_recent(recipient_id, timestamp),
    }


def compute_features(
    transactions: pd.DataFrame,
    sender_window: timedelta = _SENDER_WINDOW,
    recipient_window: timedelta = _RECIPIENT_WINDOW,
) -> pd.DataFrame:
    """Offline path: replay accounts and recipients in timestamp order, same counting logic."""
    out = transactions.sort_values("timestamp").reset_index(drop=True)

    sender_history = TransactionHistory(sender_window)
    recipient_history = TransactionHistory(recipient_window)
    sender_counts: list[int] = []
    recipient_counts: list[int] = []
    for account_id, recipient_id, timestamp in zip(
        out["account_id"], out["recipient_id"], out["timestamp"]
    ):
        sender_counts.append(sender_history.count_recent(account_id, timestamp))
        recipient_counts.append(recipient_history.count_recent(recipient_id, timestamp))

    out["sender_txn_count_recent"] = sender_counts
    out["recipient_txn_count_recent"] = recipient_counts
    return out
