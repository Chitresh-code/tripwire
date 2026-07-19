# CLAUDE.md

Guidance for Claude (or any AI coding assistant) working in this repository. Read this before making changes. If something here conflicts with a specific instruction from the user in a session, the user's explicit instruction wins — but flag the conflict rather than silently resolving it.

---

## Project Summary

Tripwire: a real-time fraud detection platform — transaction events → feature pipeline → real-time scoring API → cost-based decision, with automated drift detection and retraining. Full context in [`docs/PRD.md`](docs/PRD.md) and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — read both before non-trivial changes.

## Before Making Any Change

1. Check whether the change touches `src/features/` — if so, it **must** work identically for both the online (streaming) and offline (training) code paths. This is the single most load-bearing constraint in this codebase. See `docs/ARCHITECTURE.md` §2.2.
2. Check whether the change touches label handling or training data joins — verify no leakage is introduced (a feature or label using information not actually available at the original scoring timestamp). See `docs/CODING_STANDARDS.md` §4 ("Leakage tests").
3. Check whether the change affects the scoring path's latency — if so, benchmark before/after and note the numbers in the PR description.

## Commands

```bash
# Install deps
uv sync

# Run all tests
pytest

# Run only feature parity tests (critical path — run these after any feature change)
pytest tests/feature_parity

# Lint + format
ruff check src/ tests/
black src/ tests/

# Type check
mypy src/ --strict

# Run scoring API locally
uvicorn src.serving.app:app --reload --port 8000

# Run the full local stack (Kafka/Redpanda, Redis, Prometheus, Grafana)
docker compose up -d
```

Always run `pytest tests/feature_parity`, `ruff check`, and `mypy` before considering a change complete — these three catch the majority of the ML-specific bug classes this codebase is prone to (skew, style drift, type errors in feature pipelines).

## Directory Map (What Lives Where)

| Path | Purpose | Notes for AI edits |
|---|---|---|
| `src/ingestion/` | Kafka producers/consumers, event schemas | Schema changes need a corresponding versioned migration note in `docs/` |
| `src/features/` | Shared feature definitions | **Highest-risk directory.** Any new feature needs a parity test in `tests/feature_parity/` |
| `src/models/` | Training code, architectures | Keep baseline (GBT) and sequence model code paths clearly separated for comparison |
| `src/serving/` | FastAPI app, inference, decision engine | Latency-sensitive — avoid introducing synchronous blocking calls |
| `src/monitoring/` | Drift detection, metrics | Changes to drift thresholds should be justified with a comment referencing expected false-alarm rate |
| `src/pipelines/` | Orchestration DAGs | Keep idempotent — re-running a DAG stage should not duplicate side effects |
| `tests/feature_parity/` | Online/offline equivalence tests | Never delete or weaken an assertion here without explicit user confirmation |
| `notebooks/` | Exploration only | Never treat notebook code as production-ready; if logic proves out, port it into `src/` properly with tests |

## Conventions to Follow

- Full conventions live in `docs/CODING_STANDARDS.md` — this is a summary of what matters most for AI-driven edits specifically.
- Type hints on all public functions; Google-style docstrings.
- No hardcoded thresholds/config values in code — they belong in `configs/*.yaml`, loaded via `pydantic-settings`.
- Structured logging only (no bare `print`); every scoring decision must log transaction ID, model version, features used, score, decision, threshold.
- Conventional Commits for any commit message you generate (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`).

## Things to Never Do Without Explicit Confirmation

- Do not weaken, skip, or delete a feature-parity or leakage test to make a change "pass."
- Do not change the cost-based decision threshold logic or its underlying cost assumptions silently — these are business-meaningful values (see `docs/PRD.md` §6.2).
- Do not introduce a new external dependency without a one-line justification (per `docs/CODING_STANDARDS.md` §9).
- Do not commit secrets or real data — this project uses public/synthetic datasets only (IEEE-CIS, PaySim).
- Do not modify `docs/PRD.md` scope sections without flagging it — the PRD is the source of truth for what's in/out of scope.

## When You're Unsure

If a change could affect train/serve parity, latency budget, label leakage, or the cost-based decision logic, and it's not obvious from existing tests/docs how to proceed correctly — stop and ask, rather than guessing. These four areas are where subtle mistakes look fine in a quick test but silently break the system's actual purpose.

## Useful Context for Generating Code

- Target Python version: 3.11+
- Preferred libraries already in use: `pandas`, `lightgbm`/`xgboost`, `fastapi`, `pydantic`, `structlog`, `pytest`. Prefer these over introducing alternatives unless there's a clear gap.
- The "reference" cost function for decisioning lives in `src/serving/decision_engine.py` — read it before touching any threshold-related logic.