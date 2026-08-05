import numpy as np
import pandas as pd

from src.features.sequence_features import SEQUENCE_COLUMNS
from src.models.baseline import LABEL_COLUMN
from src.models.sequence_model import evaluate, predict_proba, train_sequence_model


def _learnable_sequence_dataset(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Fraud iff the (fake) most recent prior amount, seq0's log_amount, is large."""
    rng = np.random.default_rng(seed)
    data = {col: rng.normal(size=n) for col in SEQUENCE_COLUMNS}
    data[LABEL_COLUMN] = (data["seq0_log_amount"] > 0).astype(int)
    return pd.DataFrame(data)


def test_trained_model_beats_chance_on_a_learnable_signal() -> None:
    train = _learnable_sequence_dataset(seed=0)
    test = _learnable_sequence_dataset(seed=1)

    model = train_sequence_model(train, epochs=30, batch_size=32)
    metrics = evaluate(model, test)

    assert metrics["roc_auc"] > 0.7


def test_predict_proba_returns_two_columns_summing_to_one() -> None:
    model = train_sequence_model(_learnable_sequence_dataset(n=50), epochs=1, batch_size=32)
    proba = predict_proba(model, _learnable_sequence_dataset(n=10, seed=2))

    assert proba.shape == (10, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
