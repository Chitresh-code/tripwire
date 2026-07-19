import numpy as np
import pandas as pd

from src.models.baseline import evaluate, precision_at_recall, train_baseline


def _learnable_dataset(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Fraud is deliberately made to depend on the features, so a working model should score well."""
    rng = np.random.default_rng(seed)
    is_high_amount = rng.integers(0, 2, n)
    sender_txn_count_recent = rng.integers(0, 3, n)
    recipient_txn_count_recent = rng.integers(0, 3, n)
    is_transfer = rng.integers(0, 2, n)
    is_cash_out = np.where(is_transfer == 1, 0, rng.integers(0, 2, n))
    amount = rng.uniform(0, 1_000_000, n)

    # fraud is likely when a high amount transaction is also a transfer/cash-out
    fraud_signal = (is_high_amount == 1) & ((is_transfer == 1) | (is_cash_out == 1))
    is_fraud = np.where(fraud_signal, rng.random(n) < 0.9, rng.random(n) < 0.02)

    return pd.DataFrame(
        {
            "amount": amount,
            "is_high_amount": is_high_amount,
            "sender_txn_count_recent": sender_txn_count_recent,
            "recipient_txn_count_recent": recipient_txn_count_recent,
            "is_transfer": is_transfer,
            "is_cash_out": is_cash_out,
            "is_fraud": is_fraud,
        }
    )


def test_baseline_learns_a_known_pattern():
    train = _learnable_dataset(seed=0)
    test = _learnable_dataset(seed=1)

    model = train_baseline(train)
    metrics = evaluate(model, test)

    assert metrics["roc_auc"] > 0.8
    assert metrics["pr_auc"] > 0.3


def test_precision_at_recall_is_between_zero_and_one():
    y_true = pd.Series([0, 0, 1, 1, 0, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9])

    result = precision_at_recall(y_true, y_score, target_recall=0.5)

    assert 0.0 <= result <= 1.0
