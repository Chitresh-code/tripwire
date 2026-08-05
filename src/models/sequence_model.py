"""Sequence-aware model (FR5): a GRU over an account's recent transactions,
evaluated directly against the GBT baseline (src/models/baseline.py) on the
identical test split — same requirement `src/models/rules_baseline.py`'s
comparison satisfies for the rules-only floor.

Small on purpose (hidden_size=16, 3 epochs): this is a comparison model, not
a latency-critical production path yet, and PaySim's sequences are short
(most accounts see very few repeat transactions — see configs/features.yaml's
velocity comment), so there's little signal a bigger network would find.
"""

from __future__ import annotations

import os

# ponytail: torch and lightgbm each bundle their own OpenMP runtime; loading
# both in one process aborts/segfaults on this platform under real work
# (reproduced: passes on tiny data, crashes on the full ~6M-row training
# run). Must be set before `import torch` below — this is the standard,
# widely-shipped workaround for duplicate-OpenMP crashes, not a hack unique
# to this codebase. torch.set_num_threads(1) further down reduces thread
# contention on top of this; neither alone was enough in testing.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.features.sequence_features import SEQUENCE_COLUMNS, SEQUENCE_LENGTH, STEP_FEATURES
from src.models.baseline import LABEL_COLUMN, evaluate_scores

torch.set_num_threads(1)

_HIDDEN_SIZE = 16
_EPOCHS = 3
_BATCH_SIZE = 8192
_LEARNING_RATE = 1e-3


class SequenceGRU(nn.Module):
    def __init__(
        self, input_size: int = len(STEP_FEATURES), hidden_size: int = _HIDDEN_SIZE
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        result: torch.Tensor = self.head(hidden[-1]).squeeze(-1)  # raw logits
        return result


def _to_tensor(df: pd.DataFrame) -> torch.Tensor:
    values = df[SEQUENCE_COLUMNS].to_numpy(dtype="float32", copy=True)
    return torch.from_numpy(values).reshape(len(df), SEQUENCE_LENGTH, len(STEP_FEATURES))


def train_sequence_model(
    train: pd.DataFrame, epochs: int = _EPOCHS, batch_size: int = _BATCH_SIZE
) -> SequenceGRU:
    torch.manual_seed(
        0
    )  # reproducible weight init/shuffling, matching train_baseline's random_state=0
    x = _to_tensor(train)
    y = torch.from_numpy(train[LABEL_COLUMN].to_numpy(dtype="float32"))

    # Unweighted BCE on <1% positive rate collapses to "always predict not
    # fraud" (near-zero loss, ROC-AUC ~0.5) — confirmed, not hypothetical,
    # see docs/DECISIONS.md's M8 entry. pos_weight counteracts it for a
    # gradient-descent-trained model; note this is the opposite of what
    # worked for the GBT baseline (src/models/baseline.py's train_baseline
    # docstring) — LightGBM's leaf-wise boosting and BCE-over-minibatches
    # respond to imbalance in genuinely different ways, so the same fix
    # doesn't transfer between the two model families.
    positive_rate = float(y.mean())
    pos_weight = torch.tensor((1 - positive_rate) / positive_rate)

    model = SequenceGRU()
    optimizer = torch.optim.Adam(model.parameters(), lr=_LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        print(f"epoch {epoch + 1}/{epochs}: loss={total_loss / len(x):.4f}")

    model.eval()
    return model


def load_sequence_model(path: str) -> SequenceGRU:
    model = SequenceGRU()
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


def predict_proba(model: SequenceGRU, df: pd.DataFrame) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        fraud_score = torch.sigmoid(model(_to_tensor(df))).numpy()
    return np.column_stack([1 - fraud_score, fraud_score])


def evaluate(model: SequenceGRU, test: pd.DataFrame) -> dict[str, float]:
    y_score = predict_proba(model, test)[:, 1]
    return evaluate_scores(test[LABEL_COLUMN], y_score)
