# Decisions Log

A running record of real choices made while building Tripwire, and why — updated whenever we pick between options. This documents the actual approach taken, not just an explanation of it (see `docs/learning/` for that, kept local).

---

## 2026-07-19 — Dataset: IEEE-CIS Fraud Detection

**Decision:** Use the IEEE-CIS Fraud Detection Kaggle dataset (`train_transaction.csv` + `train_identity.csv`) as the primary dataset, over PaySim.

**Options considered:**
- **PaySim** — synthetic mobile-money transactions, one simple file, hourly `step` instead of real timestamps. Easier to start with.
- **IEEE-CIS** — real anonymized card transactions. 590,540 rows / 394 columns in `train_transaction.csv`; 144,233 rows / 41 columns in `train_identity.csv` (only some transactions have identity data), joined on `TransactionID`.

**Why:** Chosen for realism — messier and closer to what a production fraud system actually deals with, matching the goal of building a real system rather than a toy one.

**Consequences:**
- Raw files (`data/raw/*.csv`, ~650MB + 25MB) are never committed — `data/` is git-ignored, per `AGENTs.md`'s "no data committed" rule.
- The raw data has no `account_id` or real `timestamp` column, and our features (`amount_features.py`, `velocity_features.py`) expect a canonical schema (`account_id`, `timestamp`, `amount`). A loader/adapter is needed to bridge raw Kaggle columns to that schema — this keeps `src/features/` dataset-agnostic. See the next entry once that mapping is decided.
