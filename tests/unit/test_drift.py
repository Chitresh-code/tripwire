import numpy as np
import pandas as pd

from src.monitoring.drift import (
    check_drift,
    classify_psi,
    fit_reference,
    load_references,
    population_stability_index,
    save_references,
)


def test_identical_distributions_have_near_zero_psi() -> None:
    rng = np.random.default_rng(0)
    values = pd.Series(rng.uniform(0, 1000, 5000))

    reference = fit_reference(values, bins=10)
    psi = population_stability_index(reference, values)

    assert psi < 0.01
    assert classify_psi(psi) == "stable"


def test_shifted_distribution_has_high_psi() -> None:
    rng = np.random.default_rng(0)
    reference_values = pd.Series(rng.uniform(0, 1000, 5000))
    shifted_values = pd.Series(rng.uniform(5000, 10000, 5000))

    reference = fit_reference(reference_values, bins=10)
    psi = population_stability_index(reference, shifted_values)

    assert psi >= 0.25
    assert classify_psi(psi) == "alert"


def test_save_and_load_reference_round_trips(tmp_path) -> None:
    rng = np.random.default_rng(0)
    values = pd.Series(rng.uniform(0, 1000, 500))
    reference = fit_reference(values, bins=5)

    path = tmp_path / "reference.json"
    save_references({"amount": reference}, path)
    loaded = load_references(path)

    assert loaded["amount"].breakpoints == reference.breakpoints
    assert loaded["amount"].reference_pct == reference.reference_pct


def test_check_drift_skips_columns_not_present() -> None:
    rng = np.random.default_rng(0)
    reference = fit_reference(pd.Series(rng.uniform(0, 1000, 500)))
    current = pd.DataFrame({"other_column": [1, 2, 3]})

    result = check_drift({"amount": reference}, current)

    assert result == {}
