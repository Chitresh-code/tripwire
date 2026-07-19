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

**Superseded 2026-07-19 — see next entry.** After actually looking at the raw files, IEEE-CIS's lack of a native account column and real timestamp turned out to be more friction than expected, and PaySim fits the project's later streaming-simulation goal better. Reversed below.

---

## 2026-07-19 — Dataset: switched to PaySim

**Decision:** Use PaySim (`ealaxi/paysim1` on Kaggle) instead of IEEE-CIS.

**Why:**
- PaySim already has an account identifier (`nameOrig` / `nameDest`) — IEEE-CIS would have needed a synthetic one guessed from card/address fields.
- PaySim's `step` field (hour 1–744 across a simulated month) is purpose-built to replay as an event stream — a direct match for the PRD's streaming-simulation goal (M3) and for a clean, leakage-free time-based train/test split.
- Simpler schema overall (11 columns vs. 394+41), which matters while still building up the codebase incrementally.

**Consequences:**
- IEEE-CIS's `train_transaction.csv` / `train_identity.csv` were deleted from `data/raw/` (never committed — regenerable by re-downloading if ever needed).
- `docs/PRD.md` §5 updated to point at PaySim.
- The loader/adapter still needs writing, but mapping is simpler: `nameOrig` → `account_id`, `amount` → `amount` (already named that), `step` → `timestamp` (hour number converted to a real datetime by picking an arbitrary start date and adding `step` hours), `isFraud` → label.
