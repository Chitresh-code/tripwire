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
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import joblib  # type: ignore[import-untyped]
import pandas as pd
import structlog
from fastapi import FastAPI
from lightgbm import LGBMClassifier
from pydantic import BaseModel

from src.features import amount_features, type_features, velocity_features
from src.models import registry
from src.models.baseline import FEATURE_COLUMNS
from src.serving import decision_engine

log = structlog.get_logger()


class TransactionRequest(BaseModel):
    transaction_id: str
    account_id: str
    recipient_id: str
    amount: float
    timestamp: datetime
    transaction_type: str


class ScoreResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    decision: str
    model_version: str
    threshold_used: float
    latency_ms: float
    scored_at: datetime


def create_app(model: LGBMClassifier | None = None, version: str | None = None) -> FastAPI:
    """Build the app. Pass `model`/`version` directly in tests to skip the registry."""
    app = FastAPI(title="Tripwire Scoring API")
    app.state.model = model
    app.state.model_version = version
    app.state.shadow_model = None
    app.state.shadow_version = None
    # ponytail: process-local history, one instance per process — see
    # velocity_features.TransactionHistory's docstring for the Redis
    # upgrade path once scoring runs on more than one instance.
    app.state.sender_history = velocity_features.TransactionHistory(
        velocity_features._SENDER_WINDOW
    )
    app.state.recipient_history = velocity_features.TransactionHistory(
        velocity_features._RECIPIENT_WINDOW
    )

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

    @app.get("/v1/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "production_version": registry.get_production() or "unregistered",
            "shadow_version": registry.get_shadow() or "none",
        }

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

        log.info(
            "scored_transaction",
            transaction_id=txn.transaction_id,
            model_version=model_version,
            features=features,
            fraud_probability=fraud_probability,
            decision=decision,
            threshold_used=threshold_used,
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

        return ScoreResponse(
            transaction_id=txn.transaction_id,
            fraud_probability=fraud_probability,
            decision=decision,
            model_version=model_version,
            threshold_used=threshold_used,
            latency_ms=(time.perf_counter() - start) * 1000,
            scored_at=datetime.now(timezone.utc),
        )

    return app


app = create_app()
