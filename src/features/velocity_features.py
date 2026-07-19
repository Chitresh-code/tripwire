"""Velocity feature: how many transactions has this account made recently?

Unlike `amount_features`, this one needs memory — you can't answer "how many
transactions in the last hour" by looking at a single transaction alone.
`count_in_window` is the one shared piece of math; the online and offline
paths just feed it history in different shapes.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timedelta

import pandas as pd

from src.config import FeatureSettings

_settings = FeatureSettings()  # type: ignore[call-arg]  # fields load from configs/features.yaml
_WINDOW = timedelta(minutes=_settings.velocity_window_minutes)


def count_in_window(timestamps: Sequence[datetime], current_time: datetime, window: timedelta) -> int:
    """How many of `timestamps` fall in the window right before `current_time`."""
    return sum(1 for t in timestamps if current_time - window <= t < current_time)


class TransactionHistory:
    """Online path: an in-memory log of each account's past transaction times.

    ponytail: process-local dict, fine for one scoring process. Swap for
    Redis (a shared online feature store) once scoring runs on more than
    one instance, so every instance sees the same history.
    """

    def __init__(self) -> None:
        self._by_account: dict[str, list[datetime]] = defaultdict(list)

    def score_transaction(self, account_id: str, timestamp: datetime) -> dict[str, int]:
        count = count_in_window(self._by_account[account_id], timestamp, _WINDOW)
        self._by_account[account_id].append(timestamp)
        return {"txn_count_last_hour": count}


def compute_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Offline path: replay the same accounts in timestamp order, calling the same count."""
    out = transactions.sort_values("timestamp").reset_index(drop=True)

    history: dict[str, list[datetime]] = defaultdict(list)
    counts: list[int] = []
    for account_id, timestamp in zip(out["account_id"], out["timestamp"]):
        counts.append(count_in_window(history[account_id], timestamp, _WINDOW))
        history[account_id].append(timestamp)

    out["txn_count_last_hour"] = counts
    return out
