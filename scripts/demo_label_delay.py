"""Demonstrates the delayed-feedback training loop (FR12) on real data: naive
"train on everything we have" vs. leakage-safe "train only on labels that
have actually arrived by now," side by side, with a real GBT trained on each.

Deliberately not wired into scripts/train_baseline.py or scripts/check_drift.py
(the shared src.pipelines.train_and_register.run_training path those use) —
see docs/DECISIONS.md's M9 entry for why this stays additive rather than
becoming every retrain's default behavior.

Run: uv run python scripts/demo_label_delay.py
"""

from __future__ import annotations

from pathlib import Path

from src.ingestion.paysim_loader import load_paysim
from src.models.baseline import evaluate, train_baseline
from src.pipelines.build_training_table import build_feature_table, time_based_split
from src.pipelines.label_delay import labels_available_as_of, simulate_label_delay

RAW_PATH = Path("data/raw/paysim.csv")


def main() -> None:
    transactions = load_paysim(RAW_PATH)
    features = build_feature_table(transactions)

    naive_train, naive_test = time_based_split(features)
    print(f"naive (leaky) split:      train={len(naive_train):,}  test={len(naive_test):,}")

    delayed = simulate_label_delay(features)
    as_of = delayed["timestamp"].max()
    available = labels_available_as_of(delayed, as_of)
    dropped = len(features) - len(available)
    print(
        f"as of {as_of}: {dropped:,} of {len(features):,} rows "
        f"({dropped / len(features):.1%}) have a label that hasn't arrived yet — excluded"
    )

    safe_train, safe_test = time_based_split(available)
    print(f"delay-aware (safe) split: train={len(safe_train):,}  test={len(safe_test):,}")

    print("\ntraining GBT on the naive split...")
    naive_model = train_baseline(naive_train)
    naive_metrics = evaluate(naive_model, naive_test)

    print("training GBT on the delay-aware split...")
    safe_model = train_baseline(safe_train)
    safe_metrics = evaluate(safe_model, safe_test)

    print(f"\n{'metric':<25}{'naive (leaky)':>15}{'delay-aware':>15}")
    for name in naive_metrics:
        print(f"{name:<25}{naive_metrics[name]:>15.4f}{safe_metrics[name]:>15.4f}")


if __name__ == "__main__":
    main()
