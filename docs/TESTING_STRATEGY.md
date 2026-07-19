# Testing Strategy

This document explains **why** each test category exists — what specific failure mode it's designed to catch — not just how to run them (see `docs/CODING_STANDARDS.md` §4 for the how).

---

## 1. Unit Tests (`tests/unit/`)

**Catches:** basic logical errors in isolated functions (off-by-one window errors, incorrect aggregation math, malformed schema handling).

**Scope:** every function in `src/features/`, `src/models/`, `src/serving/decision_engine.py`. Run on every commit via CI.

## 2. Feature Parity Tests (`tests/feature_parity/`)

**Catches:** train/serve skew — the single most common and most damaging failure mode in real ML systems. A model can look great offline and perform badly (or dangerously) in production if the features it sees at serving time are computed even slightly differently than the features it was trained on.

**How it works:** for a fixed set of synthetic transaction sequences, compute each feature via (a) the online/streaming code path and (b) the offline/batch code path, and assert bit-for-bit (or tolerance-bound, for floats) equivalence.

**Rule:** no new feature merges to `main` without an accompanying parity test. This is enforced by code review, not just CI, because it's easy to write a parity test that technically passes but doesn't actually exercise the skew-prone logic (e.g., testing only the "no data in window" edge case).

## 3. Leakage Tests

**Catches:** a feature or training label using information that would not actually have been available at the moment the model made its real-time decision. This is the classic subtle bug in fraud/credit modeling — e.g., a "days until chargeback" feature accidentally computed using the chargeback date itself, or a label joined using post-hoc knowledge.

**How it works:** synthetic test transactions are constructed with a known "as of" timestamp; tests assert that no feature value used in scoring or training reflects data with a timestamp later than the "as of" time.

## 4. API Contract Tests

**Catches:** accidental breaking changes to the scoring API's request/response schema (see `docs/API_SPEC.md`). Consumers of this API (in a real org, this would be the payment authorization system) depend on a stable contract.

**How it works:** schema validation tests against the documented contract; any change that breaks these tests must be accompanied by an explicit version bump and update to `API_SPEC.md`.

## 5. Integration Tests (`tests/integration/`)

**Catches:** issues that only appear when components interact — e.g., the scoring API correctly reading from a real (test) instance of the online feature store, or the training pipeline correctly consuming from the offline store after a real ingestion run.

**Scope:** run against a local Docker Compose stack (Redpanda, Redis, test Postgres/warehouse), not mocks, for at least the critical paths — because mocked integration tests tend to pass even when the real integration is broken.

## 6. Drift Injection Tests

**Catches:** whether the drift detection system actually works, as opposed to just existing. Passive monitoring code with no test of its actual detection capability is a common way drift detection silently stops working (e.g., after a refactor changes what "reference distribution" means).

**How it works:** inject a synthetic, known distribution shift (e.g., shift the mean transaction amount by 3 standard deviations for a subset of synthetic traffic) and assert that the drift metric crosses the alert threshold within an expected time window.

## 7. Business-Cost Regression Tests

**Catches:** silent regressions in the *decision quality* of the system, as distinct from raw model accuracy. A model can have identical AUC to a previous version but a worse expected-cost outcome if the threshold or cost assumptions have drifted.

**How it works:** a fixed backtest set with known outcomes and defined cost assumptions is scored on every model candidate; the expected-cost metric is compared against the current production model as a release gate.

---

## Test Coverage Targets

| Directory | Target Coverage | Rationale |
|---|---|---|
| `src/features/` | 90%+ | Highest-risk code for skew/leakage |
| `src/serving/` | 85%+ | Latency- and decision-critical |
| `src/models/` | 80%+ | Correctness matters, but some paths are exploratory by nature |
| `src/pipelines/` | 60%+ | Orchestration glue — correctness matters more at the integration level than line coverage |

## CI Gate Summary

A PR cannot merge unless:
1. All unit, feature-parity, and leakage tests pass.
2. `mypy --strict` and `ruff check` pass with zero errors.
3. If the PR touches `src/models/` or `src/serving/decision_engine.py`, the business-cost regression test has been run and the result included in the PR description.