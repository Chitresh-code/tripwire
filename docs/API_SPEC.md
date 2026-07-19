# API Specification — Scoring Service

**Base URL (local):** `http://localhost:8000`
**Version:** v1

This document defines the request/response contract for the real-time scoring API. Breaking changes to this contract require a version bump (`/v2/...`) and a deprecation note here.

---

## `POST /v1/score`

Scores a single transaction for fraud risk and returns a decision.

### Request

```json
{
  "transaction_id": "txn_9f8a7b6c",
  "card_id": "card_1234",
  "merchant_id": "merchant_5678",
  "amount": 249.99,
  "currency": "USD",
  "timestamp": "2026-07-19T14:32:01Z",
  "merchant_category": "electronics",
  "channel": "card_not_present"
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `transaction_id` | string | yes | Unique per transaction; used for audit logging |
| `card_id` | string | yes | Used to look up online velocity/aggregate features |
| `merchant_id` | string | yes | Used to look up merchant-level features |
| `amount` | float | yes | Transaction amount, in `currency` units |
| `currency` | string | yes | ISO 4217 currency code |
| `timestamp` | string (ISO 8601) | yes | Event time — used for point-in-time feature correctness |
| `merchant_category` | string | yes | Merchant category code / description |
| `channel` | string | yes | e.g. `card_present`, `card_not_present`, `online` |

### Response

```json
{
  "transaction_id": "txn_9f8a7b6c",
  "fraud_probability": 0.0342,
  "decision": "allow",
  "model_version": "gbt_v2.3.1",
  "threshold_used": 0.087,
  "top_contributing_features": [
    {"feature": "card_txn_count_10m", "contribution": 0.021},
    {"feature": "merchant_avg_amount_7d", "contribution": -0.008}
  ],
  "latency_ms": 42,
  "scored_at": "2026-07-19T14:32:01.087Z"
}
```

| Field | Type | Notes |
|---|---|---|
| `fraud_probability` | float [0,1] | Raw model output |
| `decision` | enum: `allow`, `review`, `block` | Result of applying the cost-based threshold to `fraud_probability` |
| `model_version` | string | Exact model artifact version used — required for audit trail |
| `threshold_used` | float | The decision threshold in effect at scoring time (can change between deploys) |
| `top_contributing_features` | array | SHAP-style local explanation, top-N features by contribution |
| `latency_ms` | int | Server-side processing time for this request |
| `scored_at` | string (ISO 8601) | Server timestamp of scoring, for audit correlation |

### Error Responses

| Status | Condition | Body |
|---|---|---|
| 400 | Malformed request / missing required field | `{"error": "invalid_request", "detail": "..."}` |
| 424 | Online feature store unavailable, no safe fallback | `{"error": "feature_unavailable", "detail": "..."}` |
| 500 | Unhandled server error | `{"error": "internal_error", "detail": "..."}` |

**Design note:** a `424` is intentionally distinct from a `500` — it signals a specific, expected failure mode (feature store degradation) that the caller/ops team should treat differently (e.g., fall back to a conservative default decision) rather than treating as a generic outage.

---

## `GET /v1/health`

Liveness/readiness check.

```json
{"status": "ok", "model_version": "gbt_v2.3.1", "feature_store": "connected"}
```

---

## `GET /v1/metrics`

Prometheus-format metrics endpoint (latency histograms, request counts, prediction distribution buckets). Not intended for direct human consumption — scraped by Prometheus per the monitoring setup in `docs/ARCHITECTURE.md` §2.6.

---

## Versioning Policy

- Additive, backward-compatible changes (new optional response fields) do not require a version bump.
- Any change to required request fields, response field types, or decision semantics requires a `/v2/` endpoint alongside `/v1/`, with `/v1/` supported for a documented deprecation window.