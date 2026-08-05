import pandas as pd

from src.models.rules_baseline import RulesOnlyModel


def test_flags_only_large_transfers_and_cash_outs() -> None:
    x = pd.DataFrame(
        {
            "is_high_amount": [True, True, False, True],
            "is_transfer": [True, False, False, False],
            "is_cash_out": [False, True, False, False],
        }
    )

    proba = RulesOnlyModel().predict_proba(x)

    assert list(proba[:, 1]) == [1.0, 1.0, 0.0, 0.0]
    assert (proba.sum(axis=1) == 1.0).all()
