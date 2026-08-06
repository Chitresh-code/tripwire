from datetime import datetime, timedelta

import pandas as pd

from src.pipelines.label_delay import labels_available_as_of, simulate_label_delay


def test_every_available_row_has_a_label_that_arrived_by_as_of():
    base = datetime(2026, 1, 1)
    transactions = pd.DataFrame(
        {
            "timestamp": [base + timedelta(days=i) for i in range(30)],
            "is_fraud": [i % 5 == 0 for i in range(30)],
        }
    )
    delayed = simulate_label_delay(transactions, seed=0)
    as_of = base + timedelta(days=15)

    available = labels_available_as_of(delayed, as_of)

    assert len(available) > 0
    assert len(available) < len(transactions)  # some labels genuinely haven't arrived yet
    assert (available["label_available_at"] <= as_of).all()


def test_label_available_at_is_never_before_the_transaction_happened():
    transactions = pd.DataFrame(
        {"timestamp": [datetime(2026, 1, 1), datetime(2026, 1, 2)], "is_fraud": [0, 1]}
    )

    delayed = simulate_label_delay(transactions, seed=0)

    assert (delayed["label_available_at"] > delayed["timestamp"]).all()
