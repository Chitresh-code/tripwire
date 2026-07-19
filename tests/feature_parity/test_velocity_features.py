from datetime import datetime, timedelta

import pandas as pd

from src.features.velocity_features import TransactionHistory, compute_features, count_in_window


def test_count_in_window_boundary():
    now = datetime(2026, 1, 1, 12, 0, 0)
    window = timedelta(minutes=60)

    exactly_one_hour_ago = now - timedelta(minutes=60)
    just_over_one_hour_ago = now - timedelta(minutes=61)

    assert count_in_window([exactly_one_hour_ago], now, window) == 1
    assert count_in_window([just_over_one_hour_ago], now, window) == 0


def test_online_and_offline_agree():
    base = datetime(2026, 1, 1, 12, 0, 0)
    # account A sends twice within an hour; recipient X receives from both A and B within an hour.
    transactions = [
        {"account_id": "A", "recipient_id": "X", "timestamp": base},
        {"account_id": "A", "recipient_id": "Y", "timestamp": base + timedelta(minutes=10)},
        {"account_id": "B", "recipient_id": "X", "timestamp": base + timedelta(minutes=20)},
        {"account_id": "A", "recipient_id": "X", "timestamp": base + timedelta(minutes=90)},
    ]
    sorted_txns = sorted(transactions, key=lambda t: t["timestamp"])

    sender_window = timedelta(minutes=60)
    recipient_window = timedelta(minutes=60)
    sender_history = TransactionHistory(sender_window)
    recipient_history = TransactionHistory(recipient_window)
    online_results = [
        (
            sender_history.count_recent(t["account_id"], t["timestamp"]),
            recipient_history.count_recent(t["recipient_id"], t["timestamp"]),
        )
        for t in sorted_txns
    ]

    offline_df = compute_features(
        pd.DataFrame(transactions), sender_window=sender_window, recipient_window=recipient_window
    )
    offline_results = list(
        zip(offline_df["sender_txn_count_recent"], offline_df["recipient_txn_count_recent"])
    )

    assert online_results == offline_results
    # order: A->X@0min, A->Y@10min, B->X@20min, A->X@90min
    assert online_results == [(0, 0), (1, 0), (0, 1), (0, 0)]
