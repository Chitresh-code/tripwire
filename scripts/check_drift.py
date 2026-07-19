"""Checks production's drift reference against a current data window.

If any monitored column alerts (PSI >= configs/drift.yaml's alert
threshold), retrains a candidate model, evaluates it against production's
saved metrics, and — only if it clears the evaluation gate — registers it
as a shadow candidate. Never auto-promotes to production; see
scripts/promote_shadow.py for that step.

Run: uv run python scripts/check_drift.py [--inject-drift]
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib  # type: ignore[import-untyped]
import numpy as np
import structlog

from src.config import DriftSettings
from src.ingestion.paysim_loader import load_paysim
from src.models import registry
from src.models.baseline import FEATURE_COLUMNS
from src.monitoring.drift import check_drift, load_references
from src.monitoring.synthetic_drift import inject_amount_spike
from src.pipelines.build_training_table import build_feature_table, time_based_split
from src.pipelines.train_and_register import run_training, save_artifacts

log = structlog.get_logger()
_settings = DriftSettings()  # type: ignore[call-arg]  # fields load from configs/drift.yaml

RAW_PATH = Path("data/raw/paysim.csv")


def next_version(current: str) -> str:
    prefix, _, n = current.rpartition("_v")
    return f"{prefix}_v{int(n) + 1}"


def main() -> None:
    inject = "--inject-drift" in sys.argv

    production_version = registry.get_production()
    if production_version is None:
        raise RuntimeError("no production model registered — run scripts/train_baseline.py first")

    references = load_references(registry.drift_reference_path(production_version))
    production_model = joblib.load(registry.model_path(production_version))

    transactions = load_paysim(RAW_PATH)
    features = build_feature_table(transactions)
    _, test = time_based_split(features)
    current = inject_amount_spike(test) if inject else test.copy()
    current["fraud_probability"] = np.asarray(
        production_model.predict_proba(current[FEATURE_COLUMNS])
    )[:, 1]

    results = check_drift(references, current)
    alerted_columns = []
    for column, (psi, status) in results.items():
        log.info("drift_check", column=column, psi=round(psi, 4), status=status)
        print(f"  {column}: PSI={psi:.4f} ({status})")
        if status == "alert":
            alerted_columns.append(column)

    if not alerted_columns:
        print("no drift alert — nothing to do")
        return

    log.warning("drift_alert_fired", columns=alerted_columns)
    print(f"\ndrift alert on {alerted_columns} — retraining a candidate...")

    run = run_training(transactions)
    candidate_version = next_version(production_version)
    save_artifacts(run, candidate_version)

    production_metrics = registry.load_metrics(production_version)
    regression = production_metrics["roc_auc"] - run.metrics["roc_auc"]

    if regression <= _settings.max_roc_auc_regression:
        registry.set_shadow(candidate_version)
        log.info(
            "candidate_promoted_to_shadow",
            version=candidate_version,
            candidate_roc_auc=run.metrics["roc_auc"],
            production_roc_auc=production_metrics["roc_auc"],
        )
        print(
            f"candidate '{candidate_version}' cleared the evaluation gate "
            f"(ROC-AUC {run.metrics['roc_auc']:.4f} vs production {production_metrics['roc_auc']:.4f}) "
            f"— registered as shadow"
        )
    else:
        log.warning(
            "candidate_rejected",
            version=candidate_version,
            candidate_roc_auc=run.metrics["roc_auc"],
            production_roc_auc=production_metrics["roc_auc"],
            regression=round(regression, 4),
        )
        print(
            f"candidate '{candidate_version}' rejected: ROC-AUC regressed {regression:.4f} "
            f"(gate: {_settings.max_roc_auc_regression})"
        )


if __name__ == "__main__":
    main()
