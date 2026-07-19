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
    transactions = [
        {"account_id": "A", "timestamp": base},
        {"account_id": "A", "timestamp": base + timedelta(minutes=10)},
        {"account_id": "A", "timestamp": base + timedelta(minutes=90)},  # both prior A txns have aged out
        {"account_id": "B", "timestamp": base + timedelta(minutes=5)},  # different account, own count
    ]

    history = TransactionHistory()
    sorted_txns = sorted(transactions, key=lambda t: t["timestamp"])
    online_results = [
        history.score_transaction(t["account_id"], t["timestamp"])["txn_count_last_hour"] for t in sorted_txns
    ]

    offline_df = compute_features(pd.DataFrame(transactions))
    offline_results = offline_df["txn_count_last_hour"].tolist()

    assert online_results == offline_results
    assert online_results == [0, 0, 1, 0]  # order: A@0min, B@5min, A@10min, A@90min
