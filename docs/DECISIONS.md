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

**Note — PaySim is a static file, not a service:** there is no "PaySim API." It's a one-time simulator output (a fixed CSV), used only for offline training and for *our own* streaming-replay script (M3) to feed rows in as if they were live. Production would connect to a real transaction event stream instead — PaySim never appears in a production path.

---

## 2026-07-19 — Recalibrated features against real PaySim statistics

**Problem found:** running the two existing features on the real 6.36M-row file showed both were miscalibrated, invented before we had real data to check against:
- `is_high_amount` (threshold `$500`) fired on 98.9% of transactions — PaySim amounts are on a much larger scale (median ~$74,871, 90th percentile ~$365,423) than the toy value assumed.
- `txn_count_last_hour` (sender-side, 60-minute window) was almost always 0 — only 9,298 of 6,353,307 accounts (0.15%) ever send more than once in the entire 31-day dataset.

**Decision:**
1. Recalibrated `high_amount_threshold` to `350000.0` (~90th percentile of real amounts).
2. Split the single velocity feature into two: `sender_txn_count_recent` (window widened to 7 days — median gap between a repeat sender's transactions is ~139h) and a new `recipient_txn_count_recent` (window 24h — median gap between a repeat recipient's incoming payments is ~15h; 16.9% of recipients get repeat payments, vs. 0.15% of senders).
3. Canonical schema gained a `recipient_id` column (from PaySim's `nameDest`), alongside the existing `account_id` (from `nameOrig`).

**Why:** Recipient-side repeat activity is both far more common and faster in this dataset, and matches PaySim's actual fraud pattern (money funneled into an account and cashed out quickly — the "mule account" pattern) better than sender-side velocity alone. Verified on real data: `sender_txn_count_recent` still stays near-zero (max 2) even after widening the window, confirming this is a real property of the dataset, not a bug — while `recipient_txn_count_recent` now shows real variance (mean 1.06, max 91).

**Consequences:**
- `src/features/velocity_features.py`'s `TransactionHistory` was generalized (`count_recent(entity_id, timestamp)`) so the same class tracks sender and recipient history as two separate instances, instead of one sender-only implementation.
- All thresholds/windows stay config-driven (`configs/features.yaml`), with the reasoning behind each value written as comments in that file.
- Noticed but not yet acted on: fraud rate is ~4x higher in the test split (0.33%) than the train split (0.08%) — the time-based split means train and test come from different weeks, so this could be real drift within the dataset. Worth watching once a model is trained; this is exactly the kind of thing the drift-monitoring milestone (M4) exists to catch.

---

## 2026-07-19 — First baseline model: added the missing signal, dropped class-weighting

**Problem found:** the first LightGBM baseline (trained on `amount`, `is_high_amount`, `sender_txn_count_recent`, `recipient_txn_count_recent`) scored ROC-AUC exactly `0.5000` — no better than random. Root-caused, not guessed at:
1. `model.predict_proba` was returning a **constant `1.0`** for every test row — `scale_pos_weight` set to the real class-imbalance ratio (~1281:1) was saturating the model's output. Confirmed by testing `scale_pos_weight` at full ratio, √ratio, and LightGBM's `is_unbalance=True` — all three either saturated (`roc_auc≈0.5`) or actively hurt ranking (`sqrt ratio` gave `roc_auc=0.44`, worse than random) on the real data.
2. Separately, and more importantly: `groupby('type')['isFraud'].mean()` on the raw PaySim data showed fraud is **only ever** `TRANSFER` (0.77%) or `CASH_OUT` (0.18%) — `CASH_IN`, `DEBIT`, `PAYMENT` are exactly 0% fraud, by construction of the PaySim simulator. This is the single strongest signal in the dataset, and none of our features captured it.

**Decision:**
1. Added a new feature pair, `is_transfer` / `is_cash_out` (`src/features/type_features.py`), following the same online/offline-parity pattern as the other features.
2. Trained the baseline **without** `scale_pos_weight` or any class-weighting. Tested with the new features added — unweighted training scored ROC-AUC `0.91` / PR-AUC `0.22`, vastly outperforming every weighted variant (`0.5`–`0.67` ROC-AUC).

**Why this deviates from the PRD:** `docs/PRD.md` FR4 calls for a baseline "trained with class-imbalance handling (class weighting or focal loss)." Every class-weighting approach tried made the model measurably worse on real data. Flagging this rather than silently either (a) ignoring the FR6.2 requirement or (b) shipping a worse model to satisfy it on paper. Open follow-up: a more carefully tuned weighted approach (e.g. lower learning rate + moderate weight, or focal loss) might still beat unweighted — not attempted yet, since a first baseline is meant to be a fast, honest first cut, not a fully tuned model.

**Final offline evaluation report** (`scripts/train_baseline.py`, real PaySim data, time-based split):

| Metric | Value |
| --- | --- |
| ROC-AUC | 0.9123 |
| PR-AUC | 0.2240 |
| Precision @ 50% recall | 0.0556 |
| Precision @ 80% recall | 0.0203 |

PR-AUC of 0.224 vs. a 0.33% base rate in the test set is roughly a 68x lift over random guessing.

---

## 2026-07-19 — Serving API (M2 first cut): in-process online store, no decision logic yet

**Decision:**
1. `src/serving/app.py`'s velocity features are tracked with a process-local `TransactionHistory` (plain in-memory dict), not Redis or another shared store.
2. `POST /v1/score` returns `fraud_probability` only — no `decision` field, despite `docs/API_SPEC.md` specifying one.

**Why:**
1. There is exactly one server process right now. A shared store only matters once more than one instance needs to agree on an account's recent activity — adding one now would be unused infrastructure. Documented in code as a `ponytail:` comment naming the upgrade path (Redis) and the trigger for taking it (more than one serving instance).
2. `docs/PRD.md` FR6 requires the allow/review/block decision to come from an explicit cost function (expected $ loss from missed fraud vs. expected $ friction from a wrongly blocked customer). `AGENTs.md` explicitly forbids setting the cost-based decision threshold or its underlying $ assumptions silently. Those dollar figures are a business input, not something to invent — deferred until provided.

**Consequences:**
- `models/registry/` now holds the trained model artifact (`baseline_v1.joblib`), gitignored like all trained artifacts, regenerated by `scripts/train_baseline.py`.
- `src/serving/decision_engine.py` (referenced in `AGENTs.md` and `docs/TESTING_STRATEGY.md`) does not exist yet — next real step once cost inputs are available.

**Latency benchmark** (300 sequential local `/v1/score` requests, real trained model, single process): p50 0.82ms, p95 1.04ms, p99 1.35ms, max 3.20ms — well under the PRD's 100ms budget. Expected to grow once the online store moves off-process and under real concurrent load; revisit then.
