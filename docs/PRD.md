# PRD: Tripwire — Real-Time Transaction Fraud & Anomaly Detection Platform

**Status:** Draft v1.0
**Owner:** Chitresh Gyanani  
**Category:** ML Systems / MLOps
**Related skills demonstrated:** streaming systems, feature engineering under train/serve skew, cost-sensitive modeling, drift detection & automated retraining, production monitoring

---

## 1. Problem Statement

Payment fraud costs the financial industry tens of billions of dollars annually. The core difficulty isn't fitting a classifier — it's operating one under constraints that don't show up in a notebook:

- Decisions must be made in **under ~100ms** at the point of transaction.
- Fraud labels arrive **days to weeks late** (a chargeback is filed long after the transaction), so naive supervised retraining loops don't work.
- Fraud patterns **drift constantly** as adversaries adapt, so yesterday's model degrades silently.
- Classes are **severely imbalanced** (often <1% positive), so naive accuracy is meaningless and the real objective is a business cost function, not F1.
- Every false positive has a cost (blocked legitimate customer, support burden) and every false negative has a cost (fraud loss) — these costs are asymmetric and change over time.

This project builds a system that makes real-time fraud decisions, monitors its own health, detects when it's going stale, and retrains itself — the full lifecycle a real risk team owns.

## 2. Goals


| Goal                     | Success Metric                                                                                      |
| ------------------------ | --------------------------------------------------------------------------------------------------- |
| Real-time scoring        | p99 inference latency < 100ms end-to-end                                                            |
| Detect fraud effectively | Precision @ fixed recall (e.g., recall=80%) beats a rules-only baseline by a defined margin         |
| Handle delayed labels    | Working delayed-feedback training loop, documented and tested                                       |
| Detect model/data drift  | Automated drift alert fires on injected synthetic drift within N hours                              |
| Business alignment       | Cost-based threshold selection, not F1/accuracy-based                                               |
| Operational trust        | Full observability: latency, throughput, prediction distribution, drift metrics on a live dashboard |


## 3. Non-Goals (Explicitly Out of Scope)

- Not building a real payment processor or handling real PII/PCI data (use public/synthetic datasets only).
- Not optimizing for massive horizontal scale (this is a single-node/small-cluster proof of architecture, not a claim of hyperscale).
- Not building a case-management UI for fraud analysts (a stub/API is enough; this is not the focus).

## 4. Users / Stakeholders (Simulated Personas)

- **Risk Operations team:** consumes model decisions, needs explainability for blocked transactions.
- **ML Engineer (you):** owns training, deployment, drift response.
- **Finance leadership:** cares about $ loss avoided vs. false-positive customer friction cost.

## 5. Data

- **Primary dataset:** PaySim synthetic mobile money dataset (Kaggle) — decided 2026-07-19, switched from IEEE-CIS for its native account IDs and built-in hourly steps that suit streaming simulation. See `docs/DECISIONS.md`.
- **Streaming simulation:** replay dataset as a synthetic event stream (configurable events/sec) through Kafka/Redpanda to simulate production traffic.
- **Label delay simulation:** artificially delay label availability (e.g., 3–14 days) to force a realistic delayed-feedback training design.

## 6. Functional Requirements

### 6.1 Ingestion & Feature Pipeline

- FR1: System ingests transaction events from a streaming source in real time.
- FR2: Feature computation must produce **identical values** whether computed online (serving) or offline (training) — i.e., no train/serve skew. This is validated with an automated parity test.
- FR3: Support both point-in-time features (transaction amount, merchant category) and aggregated/windowed features (e.g., "txn count for this card in last 10 min").

### 6.2 Modeling

- FR4: Baseline model: gradient-boosted trees (XGBoost/LightGBM) trained with class-imbalance handling (class weighting or focal loss).
- FR5: Advanced model: sequence-aware model (e.g., transformer or GRU over a customer's recent transaction sequence) — built to directly compare against the baseline in the eval report.
- FR6: Decision threshold is selected via an explicit **cost function** (expected $ loss from false negative vs. expected $ friction cost from false positive), not a default 0.5 cutoff or pure F1 optimization.

### 6.3 Serving

- FR7: Expose a scoring endpoint (FastAPI) returning a fraud probability + decision + top contributing features (SHAP or similar) within the latency budget.
- FR8: Support shadow-mode deployment: new model scores traffic silently alongside the production model for comparison before cutover.
- FR9: Support canary rollout (e.g., 5% → 25% → 100% traffic) with automatic rollback if error/latency budget is breached.

### 6.4 Drift Detection & Retraining

- FR10: Monitor feature distributions and prediction distributions in production; compute a drift metric (e.g., Population Stability Index or KL divergence) on a rolling window.
- FR11: When drift exceeds a defined threshold, trigger an alert and an automated retraining job.
- FR12: Implement a delayed-feedback training loop: labels arrive asynchronously and are joined back to the original feature snapshot (not recomputed later, to avoid label leakage from future information).

### 6.5 Monitoring & Observability

- FR13: Dashboard showing: request latency (p50/p95/p99), throughput, prediction volume/distribution over time, drift metric over time, and live precision/recall proxy metrics (using delayed labels as they arrive).

## 7. Non-Functional Requirements

- **Latency:** p99 < 100ms per scoring request.
- **Reproducibility:** training pipeline must be re-runnable end-to-end from raw data to deployed model artifact via a single command/DAG.
- **Auditability:** every production decision must be traceable back to the model version and feature values used.

## 8. Evaluation Plan


| Metric                                   | Purpose                                                                |
| ---------------------------------------- | ---------------------------------------------------------------------- |
| Precision @ fixed recall                 | Core model quality metric, imbalance-aware                             |
| Expected $ cost (business cost function) | Ties model choice to business impact, not abstract metrics             |
| Latency (p50/p95/p99)                    | Production viability                                                   |
| Drift detection lead time                | How fast the system notices a synthetic drift injection                |
| Baseline comparison                      | Rules-only heuristic vs. GBT vs. sequence model, reported side by side |


## 9. Milestones

1. **M1 — Offline baseline:** data pipeline + GBT model + offline evaluation report.
2. **M2 — Serving path:** FastAPI scoring service, latency benchmarked, feature parity test passing.
3. **M3 — Streaming + shadow deploy:** Kafka ingestion, shadow-mode comparison against baseline.
4. **M4 — Drift + retraining loop:** synthetic drift injection, alerting, automated retrain triggered and validated.
5. **M5 — Dashboard + write-up:** monitoring dashboard live, architecture doc, results write-up with cost-based threshold justification.

## 10. Risks & Open Questions

- **Risk:** Public fraud datasets may not exhibit realistic drift — may need to hand-engineer synthetic drift scenarios (e.g., inject a new fraud pattern mid-stream) to properly demonstrate the detection loop.
- **Risk:** Sequence model may not meaningfully beat GBT baseline on this dataset — acceptable outcome if honestly reported with analysis of why.
- **Open question:** Whether to implement true delayed-feedback simulation with a message queue delay, or a simpler offline-simulated version — decide based on time budget, document tradeoff either way.

## 11. Deliverables

- Architecture diagram (ingestion → feature store → serving → monitoring → retraining loop).
- Latency/throughput benchmark table.
- A documented "incident": drift injected, detected, retrain triggered, resolved — with before/after metrics.
- README explaining design tradeoffs (why GBT baseline first, why cost-based thresholding, why shadow deploy before canary).

