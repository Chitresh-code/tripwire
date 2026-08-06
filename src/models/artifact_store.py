"""Fetches the model registry from S3-compatible object storage at startup,
instead of baking a model into the deploy image — an image with a stale
model means redeploying just to pick up a retrain; fetching at startup means
`scripts/publish_model.py` + a restart does it.

Works against AWS S3, Cloudflare R2, MinIO, or anything else that speaks the
S3 API (`configs/deployment.yaml`'s `endpoint_url`) — no vendor lock-in.

Opt-in: `configs/deployment.yaml`'s `model_bucket` is empty by default, and
`is_enabled()` gates every call here — local dev keeps working exactly as
before, straight off `models/registry/` on disk, until a bucket is set.

Credentials are never read from `configs/*.yaml` (never committed) — boto3's
standard `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` env vars or an IAM role.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import boto3  # type: ignore[import-untyped]

from src.config import DeploymentSettings
from src.models import registry

_settings = DeploymentSettings()  # type: ignore[call-arg]  # fields load from configs/deployment.yaml


def is_enabled() -> bool:
    return bool(_settings.model_bucket)


def _client() -> Any:
    kwargs = {"endpoint_url": _settings.endpoint_url} if _settings.endpoint_url else {}
    return boto3.client("s3", **kwargs)


def _download(client: Any, key: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(_settings.model_bucket, key, str(path))


def sync_registry() -> None:
    """Downloads pointers.json and whichever production/shadow model files it
    references. Called once at app startup — never on the request path."""
    client = _client()
    _download(client, registry.POINTERS_PATH.name, registry.POINTERS_PATH)

    pointers = json.loads(registry.POINTERS_PATH.read_text())
    for version in (pointers.get("production"), pointers.get("shadow")):
        if version:
            _download(client, registry.model_path(version).name, registry.model_path(version))


def publish_registry() -> None:
    """Uploads the local pointers.json and its referenced model files — the
    deploy-time counterpart to sync_registry(). Run after training/promoting
    locally, before restarting (or redeploying) an object-storage-backed API."""
    client = _client()
    client.upload_file(
        str(registry.POINTERS_PATH), _settings.model_bucket, registry.POINTERS_PATH.name
    )

    for version in (registry.get_production(), registry.get_shadow()):
        if version:
            path = registry.model_path(version)
            client.upload_file(str(path), _settings.model_bucket, path.name)
