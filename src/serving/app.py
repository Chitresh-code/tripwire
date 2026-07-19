"""Real-time scoring API — the online side of train/serve parity.

Reuses the exact same feature functions the offline training pipeline uses
(src/features/*), so there is only one definition of each feature, not two
that can drift apart.

The allow/review/block decision (src/serving/decision_engine.py) is built
from a cost function per PRD FR6, but its false-positive-cost input is a
placeholder, not a real business figure — see docs/DECISIONS.md.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import joblib  # type: ignore[import-untyped]
import pandas as pd
from fastapi import FastAPI
from lightgbm import LGBMClassifier
from pydantic import BaseModel

from src.features import amount_features, type_features, velocity_features
from src.models.baseline import FEATURE_COLUMNS
from src.serving import decision_engine

MODEL_PATH = Path("models/registry/baseline_v1.joblib")
MODEL_VERSION = "baseline_v1"


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


def create_app(model: LGBMClassifier | None = None) -> FastAPI:
    """Build the app. Pass `model` directly in tests to skip loading from disk."""
    app = FastAPI(title="Tripwire Scoring API")
    app.state.model = model
    # ponytail: process-local history, one instance per process — see
    # velocity_features.TransactionHistory's docstring for the Redis
    # upgrade path once scoring runs on more than one instance.
    app.state.sender_history = velocity_features.TransactionHistory(velocity_features._SENDER_WINDOW)
    app.state.recipient_history = velocity_features.TransactionHistory(velocity_features._RECIPIENT_WINDOW)

    def get_model() -> LGBMClassifier:
        if app.state.model is None:
            app.state.model = joblib.load(MODEL_PATH)
        return app.state.model  # type: ignore[no-any-return]

    @app.get("/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model_version": MODEL_VERSION}

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
        fraud_probability = float(get_model().predict_proba(x)[0][1])
        threshold_used = decision_engine.block_threshold(txn.amount)
        decision = decision_engine.decide(fraud_probability, txn.amount)

        return ScoreResponse(
            transaction_id=txn.transaction_id,
            fraud_probability=fraud_probability,
            decision=decision,
            model_version=MODEL_VERSION,
            threshold_used=threshold_used,
            latency_ms=(time.perf_counter() - start) * 1000,
            scored_at=datetime.now(timezone.utc),
        )

    return app


app = create_app()
