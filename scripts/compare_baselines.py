"""Rules-only heuristic vs. the trained GBT baseline, on the identical test split.

The PRD's Goals table promises the GBT "beats a rules-only baseline by a
defined margin" but M1 never actually built or measured against one — this
is that missing comparison.

Run: uv run python scripts/compare_baselines.py
"""

from __future__ import annotations

from pathlib import Path

from src.ingestion.paysim_loader import load_paysim
from src.models.baseline import evaluate, train_baseline
from src.models.rules_baseline import RulesOnlyModel
from src.pipelines.build_training_table import build_feature_table, time_based_split

RAW_PATH = Path("data/raw/paysim.csv")


def main() -> None:
    transactions = load_paysim(RAW_PATH)
    features = build_feature_table(transactions)
    train, test = time_based_split(features)

    gbt = train_baseline(train)
    gbt_metrics = evaluate(gbt, test)
    rules_metrics = evaluate(RulesOnlyModel(), test)

    print(f"test set: {len(test):,} rows (fraud rate {test['is_fraud'].mean():.4%})\n")
    print(f"{'metric':<25}{'rules-only':>15}{'GBT':>15}")
    for name in gbt_metrics:
        print(f"{name:<25}{rules_metrics[name]:>15.4f}{gbt_metrics[name]:>15.4f}")


if __name__ == "__main__":
    main()
