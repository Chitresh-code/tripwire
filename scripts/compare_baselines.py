"""Rules-only heuristic vs. GBT baseline vs. sequence model, on the identical
test split — the full three-way comparison the PRD's Evaluation Plan (§8)
asks for ("rules-only heuristic vs. GBT vs. sequence model, reported side by
side"), never assembled until now.

Runs the sequence model's training as a separate subprocess
(scripts/train_sequence_model.py), never importing torch in this process:
LightGBM (here) and PyTorch (there) each bundle their own OpenMP runtime,
and running both under real load in one process segfaults — see
docs/DECISIONS.md's M8 entry and that script's docstring.

Run: uv run python scripts/compare_baselines.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.ingestion.paysim_loader import load_paysim
from src.models import registry
from src.models.baseline import evaluate, train_baseline
from src.models.rules_baseline import RulesOnlyModel
from src.pipelines.build_training_table import build_feature_table, time_based_split

RAW_PATH = Path("data/raw/paysim.csv")


def main() -> None:
    transactions = load_paysim(RAW_PATH)
    features = build_feature_table(transactions)
    train, test = time_based_split(features)

    rules_metrics = evaluate(RulesOnlyModel(), test)

    gbt = train_baseline(train)
    gbt_metrics = evaluate(gbt, test)

    print("training sequence model (GRU) in a separate process...")
    subprocess.run(
        [sys.executable, "scripts/train_sequence_model.py"],
        check=True,
    )
    gru_metrics = json.loads(registry.SEQUENCE_METRICS_PATH.read_text())

    print(f"\ntest set: {len(test):,} rows (fraud rate {test['is_fraud'].mean():.4%})\n")
    print(f"{'metric':<25}{'rules-only':>15}{'GBT':>15}{'sequence (GRU)':>18}")
    for name in gbt_metrics:
        print(
            f"{name:<25}{rules_metrics[name]:>15.4f}{gbt_metrics[name]:>15.4f}"
            f"{gru_metrics[name]:>18.4f}"
        )


if __name__ == "__main__":
    main()
