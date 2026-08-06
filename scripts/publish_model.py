"""Uploads the local model registry to object storage — the deploy-time
counterpart to src/models/artifact_store.py's startup fetch. Run this after
training/promoting locally, before restarting (or redeploying) the API.

Run: uv run python scripts/publish_model.py
"""

from __future__ import annotations

from src.models import artifact_store


def main() -> None:
    if not artifact_store.is_enabled():
        raise RuntimeError(
            "configs/deployment.yaml's model_bucket is empty — nothing to publish to. "
            "Set it if you're preparing a deployment; local dev doesn't need this script."
        )
    artifact_store.publish_registry()
    print("published registry to object storage")


if __name__ == "__main__":
    main()
