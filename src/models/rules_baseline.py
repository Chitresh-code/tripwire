"""Rules-only heuristic — the comparison floor the PRD's Goals table promises
("beats a rules-only baseline by a defined margin") but that M1's offline
baseline never actually built (it only compared GBT against itself).

Flags a transaction as fraud when it's a large TRANSFER or CASH_OUT — the
same combination PaySim's own fraud generation logic uses, and the same
`is_high_amount`/`is_transfer`/`is_cash_out` features the GBT model trains
on, so this isn't a strawman with weaker inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class RulesOnlyModel:
    """Duck-types `.predict_proba` so `src/models/baseline.py`'s `evaluate()` works
    unchanged on it — the GBT and the heuristic get scored identically."""

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        flagged = x["is_high_amount"].astype(bool) & (
            x["is_transfer"].astype(bool) | x["is_cash_out"].astype(bool)
        )
        fraud_score = flagged.astype(float).to_numpy()
        return np.column_stack([1 - fraud_score, fraud_score])
