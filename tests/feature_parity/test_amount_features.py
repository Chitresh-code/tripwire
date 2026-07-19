import pandas as pd

from src.features.amount_features import compute_features, is_high_amount, score_transaction


def test_is_high_amount_boundary():
    assert is_high_amount(499.99) is False
    assert is_high_amount(500.0) is True


def test_online_and_offline_agree():
    transactions = [{"amount": 10.0}, {"amount": 500.0}, {"amount": 999.99}]

    online_results = [score_transaction(t)["is_high_amount"] for t in transactions]

    offline_df = compute_features(pd.DataFrame(transactions))
    offline_results = offline_df["is_high_amount"].tolist()

    assert online_results == offline_results
