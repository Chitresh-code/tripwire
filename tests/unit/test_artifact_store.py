import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.models import artifact_store, registry


def test_disabled_when_bucket_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(artifact_store._settings, "model_bucket", "")
    assert artifact_store.is_enabled() is False


def test_enabled_when_bucket_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(artifact_store._settings, "model_bucket", "my-bucket")
    assert artifact_store.is_enabled() is True


def test_sync_registry_downloads_pointers_then_referenced_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(registry, "REGISTRY_DIR", tmp_path)
    monkeypatch.setattr(registry, "POINTERS_PATH", tmp_path / "pointers.json")
    monkeypatch.setattr(artifact_store._settings, "model_bucket", "my-bucket")

    fake_client = MagicMock()

    def fake_download(bucket: str, key: str, dest: str) -> None:
        assert bucket == "my-bucket"
        if key == "pointers.json":
            Path(dest).write_text(json.dumps({"production": "v1", "shadow": "v2"}))
        else:
            Path(dest).touch()

    fake_client.download_file.side_effect = fake_download
    monkeypatch.setattr(artifact_store, "_client", lambda: fake_client)

    artifact_store.sync_registry()

    assert (tmp_path / "pointers.json").exists()
    assert (tmp_path / "v1.joblib").exists()
    assert (tmp_path / "v2.joblib").exists()
