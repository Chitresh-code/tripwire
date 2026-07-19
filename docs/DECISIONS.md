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

---

## 2026-07-19 — Cost-based decision threshold: placeholder $ figure, not a business input

**Context — flagging a conflict, per `AGENTs.md`'s own instruction to do so:** `AGENTs.md` explicitly says not to set the cost-based decision threshold or its underlying $ assumptions silently — these are supposed to be real business-supplied figures. Asked; the answer was "I don't have a rough $, add it yourself, then move to M3." That's an explicit instruction, which per `AGENTs.md` wins over the file's own default — but it's flagged here rather than resolved quietly, exactly as `AGENTs.md` asks.

**Decision:** Implemented `src/serving/decision_engine.py` using the standard Bayes-optimal cost-sensitive threshold (Elkan, *"The Foundations of Cost-Sensitive Learning"*, 2001): `p* = C_FP / (C_FP + C_FN)`, with `C_FN` = the transaction's own `amount` (the real $ loss if it's fraud and goes uncaught) and `C_FP` = a fixed cost of wrongly blocking a legitimate transaction. `C_FP` is set to **$10** in `configs/decision.yaml`, chosen as a commonly-cited rough figure for a manual-review/customer-support contact — **not** a number anyone at this business has validated.

**A "review" band was added below the block threshold** (at half of it, `review_band_fraction: 0.5` in config) so borderline cases aren't immediately auto-blocked. This split is an arbitrary, conservative heuristic — not derived from the cost function like the block threshold itself.

**Observed consequence, worth knowing before trusting this in anything real:** PaySim amounts run into the hundreds of thousands to millions, while $10 is tiny by comparison. The formula makes the threshold collapse toward ~0 for any large amount (e.g. `$900,000` → threshold ≈ 0.000011), so almost any transaction above a few thousand dollars gets blocked the moment the model assigns it *any* non-trivial fraud probability — verified live: a $900K TRANSFER with fraud probability 0.011 (barely above the 0.33% base rate) still triggered `block`. This is the formula working correctly, not a bug — but it strongly suggests $10 is too low relative to this dataset's amount scale, and a real false-positive-friction figure from the business would very likely raise it substantially. Revisit `false_positive_cost` the moment a real number exists — nothing else in the decision engine needs to change.

---

## 2026-07-19 — M3 streaming: Redpanda only (not the full stack yet), kafka-python client

**Decision:**
1. `docker-compose.yml` starts only Redpanda for now — not Redis, Prometheus, or Grafana, even though `AGENTs.md`'s documented `docker compose up -d` command describes starting all four.
2. Client library: `kafka-python` (pure Python, no native/C dependency, installs cleanly, Redpanda is Kafka-API compatible).
3. The stream consumer (`src/ingestion/stream_consumer.py`) doesn't score transactions itself — it calls the already-running `/v1/score` HTTP API for each event, reusing the exact same code path as any other caller.

**Why:**
1. Redis isn't used anywhere yet (the online store is still an in-process dict, see the earlier M2 entry) and Prometheus/Grafana belong to the M5 dashboard milestone — starting them now would be unused infrastructure with nothing to talk to.
2. No native compiled dependency (unlike `confluent-kafka`, which needs `librdkafka`) — least friction for local dev, matching `CODING_STANDARDS.md` §9's one-line-justification rule for new dependencies.
3. Keeps `src/ingestion/` and `src/serving/` cleanly separated per the architecture diagram — the consumer is a caller of the scoring API, not a second scoring implementation that could drift from it.

**Consequences:**
- `docker-compose.yml` will grow (Redis, Prometheus, Grafana) as M4/M5 actually need them — not added preemptively.
- The demo consumer (`scripts/consume_stream.py`) doesn't set a Kafka consumer group, so re-running it replays the whole topic from the start rather than resuming — fine for local demoing, would need a group ID for a real "don't double-process" consumer.

**Verified end-to-end on real data:** started Redpanda, replayed real PaySim transactions onto the `transactions` topic, ran the consumer against the live `/v1/score` API — structured logs showed real transaction IDs, features, probabilities, decisions, and thresholds for every event, including a live `block` decision on a real high-probability transaction. `tests/integration/test_stream_roundtrip.py` (produce → consume against a real broker) passes when Redpanda is running, and self-skips otherwise so a normal `pytest` run isn't affected.

---

## 2026-07-19 — M4: drift detection resolves the earlier open item — it was real

**Finding, not invented:** the very first drift check (`scripts/check_drift.py`, no synthetic injection) run against `baseline_v1`'s reference distribution vs. the real held-out test split flagged `recipient_txn_count_recent` with **PSI = 0.4766** (well past the 0.25 alert band) and `fraud_probability` with PSI = 0.2105 (moderate). This is the concrete explanation for the open item logged back on 2026-07-19 in the "Recalibrated features" entry — "fraud rate is ~4x higher in the test split (0.33%) than train split (0.08%)." It wasn't a fluke: recipient-side transaction velocity genuinely shifts between the two time periods in this dataset, and the model's own prediction distribution shifts with it. `amount` and `sender_txn_count_recent` stayed stable (PSI ≈ 0), consistent with earlier findings that sender velocity barely varies in PaySim at all.

**PSI thresholds used** (`configs/drift.yaml`): < 0.1 stable, 0.1–0.25 moderate, ≥ 0.25 alert — a standard, published convention from credit-risk modeling (not invented for this project, unlike the decision-engine's cost figure above).

**Decision — drift detection implementation (`src/monitoring/drift.py`):** fit a `DriftReference` (quantile bin edges + reference bin percentages) once per model version, at training time, on that model's own training split; save it alongside the model artifact. Comparing later data against it only needs the saved reference, not the original training data — cheap enough to run repeatedly, and works the same whether "current" is an offline test split or a rolling window of live scores.

---

## 2026-07-19 — M4: synthetic drift injection, automated retrain, and a model registry

**Synthetic drift injection** (`src/monitoring/synthetic_drift.py`): scales every transaction amount 8x, simulating a burst of unusually large transactions — a stand-in for "a new fraud pattern showed up." Chosen because `docs/PRD.md`'s own risk note anticipated this: public datasets may not drift on their own, so a hand-engineered scenario is needed to prove the detection loop actually works. Verified: `amount` PSI jumped to 1.39, `fraud_probability` PSI to 1.77 — both an order of magnitude past the alert threshold, an unambiguous signal.

**Model registry** (`src/models/registry.py`), the "Model Registry" component named in `docs/ARCHITECTURE.md`'s diagram but not built until now: a `models/registry/pointers.json` file tracking two pointers, `production` and `shadow`, over a directory of versioned artifacts (`{version}.joblib`, `{version}_metrics.json`, `{version}_drift_reference.json`).

**Automated retrain loop** (`scripts/check_drift.py`): on any alert, retrains a candidate, evaluates it, and compares its ROC-AUC against the current production model's saved metrics. If it doesn't regress by more than `max_roc_auc_regression` (0.01, `configs/drift.yaml` — an engineering tolerance, not a $ business figure, but still a judgment call worth adjusting if it proves too strict or loose), the candidate is registered as **shadow** — never auto-promoted to production. `scripts/promote_shadow.py` is a separate, explicitly human-triggered step. This matches `docs/ARCHITECTURE.md` §2.5's loop ("if it passes evaluation gates, deployed via shadow mode first") and finally completes FR8 / the shadow-mode piece deferred from M3, since there's now a second model to actually compare against.

**Known limitation, stated plainly:** both retrains in this session (real-drift-triggered and synthetic-injection-triggered) produced a candidate with **effectively identical ROC-AUC** to production (0.91225... in both cases). That's expected, not a bug: `scripts/check_drift.py` retrains on the *same* static PaySim file, using the *same* time-based train/test split every time — there's no genuinely new, post-drift labeled data to learn from, because this project has one fixed historical file, not a live label feed. A real production retraining job would train on data collected *after* the drift was detected, which is exactly what would let a new candidate actually learn the new pattern instead of just reproducing the old one. Worth knowing before reading too much into any drift-triggered retrain's improvement (or lack of it) in this project.

**Verified end-to-end on real data:** trained `baseline_v1` → real drift check flagged `recipient_txn_count_recent` → retrained `baseline_v2`, cleared the evaluation gate, registered as shadow → confirmed live via the serving API that shadow-mode scores every request through both models silently (`shadow_scored` log line) without changing the response → promoted `baseline_v2` to production via `scripts/promote_shadow.py` → ran the synthetic amount-spike injection, got an unambiguous alert, retrained `baseline_v3`, cleared the gate, registered as the new shadow.
