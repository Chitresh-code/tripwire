"""Prometheus metrics for the scoring API — the "system health" half of
docs/ARCHITECTURE.md §2.6's dashboard list (latency, throughput, prediction
distribution, drift PSI). The precision/recall proxy panel PRD FR13 also asks
for is skipped: it needs the delayed-feedback labeling loop (FR12), which was
left as an open question in docs/PRD.md §10 and never built — see
docs/DECISIONS.md.
"""

from __future__ import annotations

import json

from prometheus_client import Counter, Gauge, Histogram

from src.models import registry

SCORING_LATENCY = Histogram("scoring_latency_seconds", "Time to score one transaction")
SCORING_DECISIONS = Counter("scoring_decisions_total", "Scoring decisions made", ["decision"])
FRAUD_PROBABILITY = Histogram(
    "fraud_probability",
    "Distribution of predicted fraud probabilities",
    buckets=[i / 10 for i in range(11)],
)
DRIFT_PSI = Gauge("drift_psi", "Latest PSI per monitored column", ["column"])

DRIFT_STATUS_PATH = registry.REGISTRY_DIR / "latest_drift_status.json"


def record_drift_status(results: dict[str, tuple[float, str]]) -> None:
    """Called by scripts/check_drift.py after each run — persisted so the long-lived
    serving process (a separate Prometheus scrape target) can expose the latest values."""
    DRIFT_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    DRIFT_STATUS_PATH.write_text(json.dumps({col: psi for col, (psi, _) in results.items()}))


def refresh_drift_gauges() -> None:
    """Load the latest persisted drift check (if any) into the Gauges before a scrape."""
    if not DRIFT_STATUS_PATH.exists():
        return
    for column, psi in json.loads(DRIFT_STATUS_PATH.read_text()).items():
        DRIFT_PSI.labels(column=column).set(psi)
