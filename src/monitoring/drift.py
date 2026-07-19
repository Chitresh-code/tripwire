"""Population Stability Index (PSI): how much has a distribution shifted?

Fit a `DriftReference` once on a known-good distribution (the training set),
save it alongside the model, then compare any later window of data against
it — offline (a later time slice) or online (a rolling window of live
scores). PSI bands (< 0.1 stable, 0.1-0.25 moderate, >= 0.25 alert) are a
standard credit-risk-modeling convention, not invented here — see
configs/drift.yaml.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DriftSettings

_settings = DriftSettings()  # type: ignore[call-arg]  # fields load from configs/drift.yaml

_EPSILON = 1e-6  # avoids log(0)/divide-by-zero when a bin is empty


@dataclass
class DriftReference:
    """A fitted reference distribution: bin edges + the reference's share in each bin."""

    breakpoints: list[float]
    reference_pct: list[float]


def fit_reference(values: pd.Series, bins: int = _settings.bins) -> DriftReference:
    """Bin `values` into `bins` quantile buckets and record the reference's share of each."""
    breakpoints = sorted(set(np.quantile(values, np.linspace(0, 1, bins + 1))))
    counts, _ = np.histogram(values, bins=breakpoints)
    pct = np.clip(counts / len(values), _EPSILON, None)
    return DriftReference(breakpoints=breakpoints, reference_pct=list(pct))


def population_stability_index(reference: DriftReference, current: pd.Series) -> float:
    """PSI of `current` against the fitted `reference`. 0 = identical, larger = more shifted."""
    counts, _ = np.histogram(current, bins=reference.breakpoints)
    current_pct = np.clip(counts / len(current), _EPSILON, None)
    reference_pct = np.array(reference.reference_pct)
    return float(np.sum((current_pct - reference_pct) * np.log(current_pct / reference_pct)))


def classify_psi(psi: float) -> str:
    if psi >= _settings.alert_threshold:
        return "alert"
    if psi >= _settings.moderate_threshold:
        return "moderate"
    return "stable"


def fit_references(
    data: pd.DataFrame, columns: list[str] = _settings.monitored_columns
) -> dict[str, DriftReference]:
    return {column: fit_reference(data[column]) for column in columns if column in data.columns}


def check_drift(
    references: dict[str, DriftReference], current: pd.DataFrame
) -> dict[str, tuple[float, str]]:
    """PSI + classification for every monitored column present in `current`."""
    results = {}
    for column, reference in references.items():
        if column not in current.columns:
            continue
        psi = population_stability_index(reference, current[column])
        results[column] = (psi, classify_psi(psi))
    return results


def save_references(references: dict[str, DriftReference], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {column: asdict(reference) for column, reference in references.items()}
    path.write_text(json.dumps(payload))


def load_references(path: Path) -> dict[str, DriftReference]:
    payload = json.loads(path.read_text())
    return {column: DriftReference(**values) for column, values in payload.items()}
