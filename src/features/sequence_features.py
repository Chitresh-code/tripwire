"""Sequence features: an account's last N transactions, for the sequence model (FR5).

`velocity_features.TransactionHistory` only remembers *how many* prior
transactions an account had — enough for a count, not enough for a sequence
model to learn an order-dependent pattern. This tracks the transactions
themselves (a small per-step feature vector each, reusing `amount_features`/
`type_features` so a "large transfer" means the same thing here as
everywhere else in the codebase), fixed-length and zero-padded so every
sequence is the same shape going into the model.

Only prior transactions are ever included — never the current one — so
there's no leakage from the thing being predicted into its own input.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque

import pandas as pd

from src.config import SequenceSettings
from src.features.amount_features import is_high_amount
from src.features.type_features import is_cash_out, is_transfer

_settings = SequenceSettings()  # type: ignore[call-arg]  # fields load from configs/sequence.yaml
SEQUENCE_LENGTH = _settings.sequence_length
STEP_FEATURES = ["log_amount", "is_high_amount", "is_transfer", "is_cash_out"]
SEQUENCE_COLUMNS = [f"seq{i}_{name}" for i in range(SEQUENCE_LENGTH) for name in STEP_FEATURES]


def _step_features(amount: float, transaction_type: str) -> tuple[float, float, float, float]:
    return (
        math.log1p(amount),
        float(is_high_amount(amount)),
        float(is_transfer(transaction_type)),
        float(is_cash_out(transaction_type)),
    )


class TransactionSequenceStore:
    """Online path: remembers, per account, its last `sequence_length` transactions."""

    def __init__(self, sequence_length: int = SEQUENCE_LENGTH) -> None:
        self._length = sequence_length
        self._by_account: dict[str, deque[tuple[float, float, float, float]]] = defaultdict(
            lambda: deque(maxlen=sequence_length)
        )

    def get_and_update(self, account_id: str, amount: float, transaction_type: str) -> list[float]:
        """Returns the padded sequence of *prior* transactions, then records this one."""
        history = self._by_account[account_id]
        padding = [0.0] * (len(STEP_FEATURES) * (self._length - len(history)))
        flattened = [value for step in history for value in step]
        history.append(_step_features(amount, transaction_type))
        return padding + flattened


def score_transaction(
    store: TransactionSequenceStore, account_id: str, amount: float, transaction_type: str
) -> dict[str, float]:
    """Online path: compute the sequence feature columns for a single live transaction."""
    values = store.get_and_update(account_id, amount, transaction_type)
    return dict(zip(SEQUENCE_COLUMNS, values))


def compute_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Offline path: replay accounts in timestamp order, same store logic."""
    out = transactions.sort_values("timestamp").reset_index(drop=True)

    store = TransactionSequenceStore()
    rows: list[list[float]] = []
    for account_id, amount, transaction_type in zip(
        out["account_id"], out["amount"], out["transaction_type"]
    ):
        rows.append(store.get_and_update(account_id, amount, transaction_type))

    sequence_df = pd.DataFrame(rows, columns=SEQUENCE_COLUMNS, index=out.index)
    return pd.concat([out, sequence_df], axis=1)
