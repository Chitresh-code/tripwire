"""Per-decision explainability (PRD FR7): which features drove this score.

Uses LightGBM's built-in `pred_contrib` (exact SHAP values for tree models)
instead of the separate `shap` package — no new dependency needed for
something the model library already computes natively.
"""

from __future__ import annotations

import pandas as pd
from lightgbm import LGBMClassifier

_TOP_N = 3


def top_contributing_features(
    model: LGBMClassifier, x: pd.DataFrame, feature_columns: list[str], n: int = _TOP_N
) -> list[tuple[str, float]]:
    """The `n` features with the largest |SHAP contribution| for this one-row prediction."""
    contributions = model.predict(x, pred_contrib=True)[0][: len(feature_columns)]
    ranked = sorted(
        zip(feature_columns, contributions), key=lambda pair: abs(pair[1]), reverse=True
    )
    return [(name, float(value)) for name, value in ranked[:n]]
