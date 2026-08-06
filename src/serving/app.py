"""Real-time scoring API — the online side of train/serve parity.

Reuses the exact same feature functions the offline training pipeline uses
(src/features/*), so there is only one definition of each feature, not two
that can drift apart.

The allow/review/block decision (src/serving/decision_engine.py) is built
from a cost function per PRD FR6, but its false-positive-cost input is a
placeholder, not a real business figure — see docs/DECISIONS.md.

Shadow mode (PRD FR8): if the registry has a "shadow" model registered
(scripts/check_drift.py sets one when a retrained candidate clears its
evaluation gate), every request is also scored by it and logged — silently,
never affecting the response. A human promotes shadow -> production with
scripts/promote_shadow.py after reviewing it; nothing here does that
automatically.

Sequence model (PRD FR5): if scripts/compare_baselines.py has saved one
(models/registry/sequence_model.pt), every request is also scored by it and
logged the same shadow-style way — it's a comparison model against the GBT
baseline, never a candidate for production.

Deployment: if configs/deployment.yaml's model_bucket is set, the registry
is fetched from object storage once at startup (src/models/artifact_store.py)
instead of requiring a model baked into the image. Local dev is unaffected —
that config is empty by default, so this is a no-op unless someone deploys.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import joblib  # type: ignore[import-untyped]
import pandas as pd
import structlog
from fastapi import FastAPI, Response
from lightgbm import LGBMClassifier
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from src.features import amount_features, sequence_features, type_features, velocity_features
from src.models import artifact_store, registry
from src.models import sequence_model as sequence_model_module
from src.models.baseline import FEATURE_COLUMNS
from src.monitoring import metrics
from src.serving import decision_engine
from src.serving.explain import top_contributing_features

log = structlog.get_logger()


class TransactionRequest(BaseModel):
    transaction_id: str
    account_id: str
    recipient_id: str
    amount: float
    timestamp: datetime
    transaction_type: str


class FeatureContribution(BaseModel):
    feature: str
    contribution: float  # signed SHAP value: positive pushes toward fraud, negative away


class ScoreResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    decision: str
    model_version: str
    threshold_used: float
    top_contributing_features: list[FeatureContribution]
    latency_ms: float
    scored_at: datetime


def create_app(model: LGBMClassifier | None = None, version: str | None = None) -> FastAPI:
    """Build the app. Pass `model`/`version` directly in tests to skip the registry."""
    app = FastAPI(title="Tripwire Scoring API")
    app.state.model = model
    app.state.model_version = version
    app.state.shadow_model = None
    app.state.shadow_version = None
    app.state.sequence_model = None
    app.state.sequence_store = sequence_features.TransactionSequenceStore()
    # ponytail: process-local history, one instance per process — see
    # velocity_features.TransactionHistory's docstring for the Redis
    # upgrade path once scoring runs on more than one instance.
    app.state.sender_history = velocity_features.TransactionHistory(
        velocity_features._SENDER_WINDOW
    )
    app.state.recipient_history = velocity_features.TransactionHistory(
        velocity_features._RECIPIENT_WINDOW
    )

    @app.on_event("startup")
    def sync_model_registry() -> None:
        if model is None and artifact_store.is_enabled():
            artifact_store.sync_registry()

    def get_production_model() -> tuple[LGBMClassifier, str]:
        if app.state.model is None:
            prod_version = registry.get_production()
            if prod_version is None:
                raise RuntimeError("no production model registered — run scripts/train_baseline.py")
            app.state.model = joblib.load(registry.model_path(prod_version))
            app.state.model_version = prod_version
        return app.state.model, app.state.model_version

    def get_shadow_model() -> tuple[LGBMClassifier, str] | None:
        shadow_version = registry.get_shadow()
        if shadow_version is None:
            return None
        if app.state.shadow_version != shadow_version:
            app.state.shadow_model = joblib.load(registry.model_path(shadow_version))
            app.state.shadow_version = shadow_version
        return app.state.shadow_model, app.state.shadow_version

    def get_sequence_model() -> sequence_model_module.SequenceGRU | None:
        if not registry.SEQUENCE_MODEL_PATH.exists():
            return None
        if app.state.sequence_model is None:
            app.state.sequence_model = sequence_model_module.load_sequence_model(
                str(registry.SEQUENCE_MODEL_PATH)
            )
        model: sequence_model_module.SequenceGRU = app.state.sequence_model
        return model

    @app.get("/v1/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "production_version": registry.get_production() or "unregistered",
            "shadow_version": registry.get_shadow() or "none",
        }

    @app.get("/metrics")
    def prometheus_metrics() -> Response:
        metrics.refresh_drift_gauges()
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/score", response_model=ScoreResponse)
    def score(txn: TransactionRequest) -> ScoreResponse:
        start = time.perf_counter()

        features: dict[str, float | bool | int] = {"amount": txn.amount}
        features.update(amount_features.score_transaction({"amount": txn.amount}))
        features.update(type_features.score_transaction(txn.transaction_type))
        features.update(
            velocity_features.score_transaction(
                app.state.sender_history,
                app.state.recipient_history,
                txn.account_id,
                txn.recipient_id,
                txn.timestamp,
            )
        )

        x = pd.DataFrame([{col: features[col] for col in FEATURE_COLUMNS}])

        model, model_version = get_production_model()
        fraud_probability = float(model.predict_proba(x)[0][1])
        threshold_used = decision_engine.block_threshold(txn.amount)
        decision = decision_engine.decide(fraud_probability, txn.amount)
        top_features = top_contributing_features(model, x, FEATURE_COLUMNS)

        metrics.SCORING_DECISIONS.labels(decision=decision).inc()
        metrics.FRAUD_PROBABILITY.observe(fraud_probability)

        log.info(
            "scored_transaction",
            transaction_id=txn.transaction_id,
            model_version=model_version,
            features=features,
            fraud_probability=fraud_probability,
            decision=decision,
            threshold_used=threshold_used,
            top_features=top_features,
        )

        shadow = get_shadow_model()
        if shadow is not None:
            shadow_model, shadow_version = shadow
            shadow_probability = float(shadow_model.predict_proba(x)[0][1])
            log.info(
                "shadow_scored",
                transaction_id=txn.transaction_id,
                shadow_version=shadow_version,
                shadow_fraud_probability=shadow_probability,
                production_fraud_probability=fraud_probability,
            )

        sequence_row = pd.DataFrame(
            [
                sequence_features.score_transaction(
                    app.state.sequence_store, txn.account_id, txn.amount, txn.transaction_type
                )
            ]
        )
        gru = get_sequence_model()
        if gru is not None:
            sequence_probability = float(
                sequence_model_module.predict_proba(gru, sequence_row)[0][1]
            )
            log.info(
                "sequence_scored",
                transaction_id=txn.transaction_id,
                sequence_fraud_probability=sequence_probability,
                production_fraud_probability=fraud_probability,
            )

        latency_ms = (time.perf_counter() - start) * 1000
        metrics.SCORING_LATENCY.observe(latency_ms / 1000)

        return ScoreResponse(
            transaction_id=txn.transaction_id,
            fraud_probability=fraud_probability,
            decision=decision,
            model_version=model_version,
            threshold_used=threshold_used,
            top_contributing_features=[
                FeatureContribution(feature=name, contribution=value)
                for name, value in top_features
            ],
            latency_ms=latency_ms,
            scored_at=datetime.now(timezone.utc),
        )

    return app


app = create_app()
