import pandas as pd

from src.features.type_features import compute_features, is_cash_out, is_transfer, score_transaction


def test_is_transfer_and_is_cash_out():
    assert is_transfer("TRANSFER") is True
    assert is_transfer("PAYMENT") is False
    assert is_cash_out("CASH_OUT") is True
    assert is_cash_out("PAYMENT") is False


def test_online_and_offline_agree():
    transactions = [
        {"transaction_type": "PAYMENT"},
        {"transaction_type": "TRANSFER"},
        {"transaction_type": "CASH_OUT"},
        {"transaction_type": "CASH_IN"},
    ]

    online_results = [
        (
            score_transaction(t["transaction_type"])["is_transfer"],
            score_transaction(t["transaction_type"])["is_cash_out"],
        )
        for t in transactions
    ]

    offline_df = compute_features(pd.DataFrame(transactions))
    offline_results = list(zip(offline_df["is_transfer"], offline_df["is_cash_out"]))

    assert online_results == offline_results
    assert online_results == [(False, False), (True, False), (False, True), (False, False)]
