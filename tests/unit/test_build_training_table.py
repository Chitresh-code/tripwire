from datetime import datetime, timedelta

import pandas as pd

from src.pipelines.build_training_table import build_feature_table, time_based_split


def _fake_transactions() -> pd.DataFrame:
    base = datetime(2026, 1, 1)
    return pd.DataFrame(
        {
            "account_id": ["A", "A", "B", "A", "B"],
            "timestamp": [base + timedelta(hours=h) for h in [0, 1, 2, 3, 4]],
            "amount": [10.0, 600.0, 50.0, 700.0, 20.0],
        }
    )


def test_build_feature_table_adds_both_feature_columns():
    features = build_feature_table(_fake_transactions())

    assert "is_high_amount" in features.columns
    assert "txn_count_last_hour" in features.columns
    assert len(features) == 5


def test_time_based_split_has_no_leakage():
    features = build_feature_table(_fake_transactions())
    train, test = time_based_split(features, test_fraction=0.4)

    assert len(train) + len(test) == len(features)
    assert train["timestamp"].max() < test["timestamp"].min()
