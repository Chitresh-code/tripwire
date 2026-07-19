# Coding Standards & Guidelines

These standards apply to all code in this repository. The goal is consistency, readability, and making the ML-specific failure modes (data leakage, train/serve skew, silent metric regressions) hard to introduce by accident.

---

## 1. Language & Style

- **Language:** Python 3.11+
- **Formatter:** `black` (line length 100), enforced via pre-commit hook.
- **Linter:** `ruff` (replaces flake8 + isort + more), config in `pyproject.toml`.
- **Type checking:** `mypy` in strict mode for all files under `src/`. Type hints are required on all public function signatures.
- **Docstrings:** Google-style docstrings on every public function/class. Minimum: one-line summary, `Args`, `Returns`, and `Raises` if applicable.

```python
def compute_velocity_feature(
    transactions: pd.DataFrame,
    window_minutes: int,
) -> pd.Series:
    """Compute transaction velocity (count) over a trailing time window.

    Args:
        transactions: Transaction events with a 'timestamp' and 'card_id' column.
        window_minutes: Size of the trailing window in minutes.

    Returns:
        A Series indexed identically to `transactions`, with the count of
        transactions for the same card_id in the preceding window.

    Raises:
        ValueError: If `transactions` is missing required columns.
    """
```

## 2. Project Structure

```
tripwire/
├── src/
│   ├── ingestion/          # Kafka producers/consumers, event schemas
│   ├── features/           # Feature definitions — SHARED between online & offline
│   ├── models/             # Training code, model architectures
│   ├── serving/            # FastAPI app, inference logic, decision engine
│   ├── monitoring/         # Drift detection, metrics emitters
│   └── pipelines/          # Orchestration DAGs (training, retraining)
├── tests/
│   ├── unit/
│   ├── integration/
│   └── feature_parity/     # Online/offline feature equivalence tests — critical
├── configs/                # YAML configs (thresholds, model params, infra)
├── docs/
├── notebooks/              # Exploration ONLY — no production logic lives here
└── scripts/                # One-off / operational scripts
```

**Rule:** Anything under `src/features/` must be importable and identically callable from both the streaming (online) path and the batch (offline/training) path. If a feature can't be written this way, that's a signal the feature definition needs to change, not that it's OK to duplicate the logic.

## 3. Naming Conventions

- Modules/files: `snake_case.py`
- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Feature names (in code and in the feature store): `domain_entity_metric_window`, e.g. `card_txn_count_10m`, `merchant_avg_amount_7d`. Consistent naming makes drift dashboards and feature parity tests far easier to reason about.

## 4. Testing Standards

- **Framework:** `pytest`
- **Minimum coverage:** 80% on `src/features/`, `src/models/`, and `src/serving/` (the core correctness-critical paths). Coverage on orchestration/glue code is not held to the same bar.
- **Required test categories:**
  1. **Unit tests** — pure function correctness (e.g., a windowed feature computes the right count given synthetic input).
  2. **Feature parity tests** — assert that a feature computed via the online path and the offline path produce identical values on the same input. This is the single most important test category in this repo.
  3. **Leakage tests** — assert that no feature uses information that would not have been available at the actual scoring timestamp (critical for the delayed-label training join).
  4. **API contract tests** — the scoring endpoint's request/response schema is validated against a fixed contract; breaking changes must be explicit and versioned.
- Every bug fix that stems from a production incident must ship with a regression test that would have caught it.

## 5. Configuration Management

- All thresholds, model hyperparameters, and infra settings live in versioned YAML under `configs/` — never hardcoded in application code.
- Use `pydantic` (or `pydantic-settings`) for typed, validated config loading. A config that fails validation should fail fast at startup, not silently misbehave at request time.
- Secrets (API keys, credentials) are never committed; loaded via environment variables, documented in `.env.example`.

## 6. Logging & Observability

- Use structured logging (`structlog` or equivalent JSON-formatted logs) — not bare `print()` or unstructured string logs.
- Every scoring decision logs (at minimum): transaction ID, model version, feature values used, score, decision, threshold applied. This is what makes the system auditable (see Architecture doc §1.2).
- Log levels: `DEBUG` for local development detail, `INFO` for normal operational events, `WARNING` for degraded-but-recovered situations (e.g., feature fallback used), `ERROR` for failures requiring attention.

## 7. Git Workflow

- **Branching:** `main` is always deployable. Feature work happens on `feature/<short-description>` branches, merged via PR.
- **Commit messages:** Conventional Commits format — `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`. Example: `feat(features): add card velocity window feature with parity test`.
- **PR requirements:** at least one passing CI run (lint, type-check, tests) before merge; PR description must state what changed and why, and link to the relevant PRD section if applicable.

## 8. Code Review Checklist

Before approving a PR, check:
- [ ] Are new features implemented once and shared between online/offline paths?
- [ ] Is there a feature-parity test for any new feature?
- [ ] Could this change introduce label leakage (using future information)?
- [ ] Are thresholds/config values externalized, not hardcoded?
- [ ] Does this change affect the latency budget? If so, is it benchmarked?
- [ ] Are logs sufficient to debug a production issue from this code path after the fact?

## 9. Dependency Management

- Managed via `pyproject.toml` + `uv` (or `poetry`) — no unpinned `pip install` in application code.
- New dependencies require a one-line justification in the PR description (what problem it solves, why an existing dependency doesn't cover it).