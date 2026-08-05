from datetime import datetime, timedelta

import pandas as pd

from src.features.sequence_features import (
    SEQUENCE_COLUMNS,
    TransactionSequenceStore,
    compute_features,
    score_transaction,
)


def test_sequence_is_padded_then_fills_with_prior_transactions_only():
    store = TransactionSequenceStore(sequence_length=2)

    first = score_transaction(store, "A", 100.0, "PAYMENT")
    second = score_transaction(store, "A", 200.0, "TRANSFER")
    third = score_transaction(store, "A", 300.0, "CASH_OUT")

    # first call: no prior transactions yet, fully padded (zeros).
    assert all(v == 0.0 for v in first.values())
    # second call: sees only the first transaction, still one slot padded.
    assert second[f"{list(SEQUENCE_COLUMNS)[0]}"] == 0.0
    # third call: window is full, both prior amounts show up, current (300.0) never does.
    assert 300.0 not in third.values()


def test_online_and_offline_agree():
    base = datetime(2026, 1, 1, 12, 0, 0)
    transactions = [
        {"account_id": "A", "amount": 100.0, "transaction_type": "PAYMENT", "timestamp": base},
        {
            "account_id": "A",
            "amount": 200.0,
            "transaction_type": "TRANSFER",
            "timestamp": base + timedelta(minutes=10),
        },
        {
            "account_id": "B",
            "amount": 50.0,
            "transaction_type": "CASH_OUT",
            "timestamp": base + timedelta(minutes=20),
        },
        {
            "account_id": "A",
            "amount": 300.0,
            "transaction_type": "CASH_OUT",
            "timestamp": base + timedelta(minutes=30),
        },
    ]
    sorted_txns = sorted(transactions, key=lambda t: t["timestamp"])

    store = TransactionSequenceStore()
    online_results = [
        score_transaction(store, t["account_id"], t["amount"], t["transaction_type"])
        for t in sorted_txns
    ]

    offline_df = compute_features(pd.DataFrame(transactions))
    offline_results = offline_df[SEQUENCE_COLUMNS].to_dict(orient="records")

    assert online_results == offline_results
