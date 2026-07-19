"""Tracks model artifacts on disk: which version is live vs. a shadow candidate.

The "Model Registry" component from docs/ARCHITECTURE.md's diagram — a
directory of versioned artifacts plus two pointers (`production`, `shadow`),
not a database. Good enough for a single-node deployment (see
docs/ARCHITECTURE.md §6's honest-scope note).
"""

from __future__ import annotations

import json
from pathlib import Path

REGISTRY_DIR = Path("models/registry")
POINTERS_PATH = REGISTRY_DIR / "pointers.json"


def model_path(version: str) -> Path:
    return REGISTRY_DIR / f"{version}.joblib"


def metrics_path(version: str) -> Path:
    return REGISTRY_DIR / f"{version}_metrics.json"


def drift_reference_path(version: str) -> Path:
    return REGISTRY_DIR / f"{version}_drift_reference.json"


def _read_pointers() -> dict[str, str]:
    if not POINTERS_PATH.exists():
        return {}
    result: dict[str, str] = json.loads(POINTERS_PATH.read_text())
    return result


def _write_pointers(pointers: dict[str, str]) -> None:
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    POINTERS_PATH.write_text(json.dumps(pointers))


def set_production(version: str) -> None:
    pointers = _read_pointers()
    pointers["production"] = version
    _write_pointers(pointers)


def get_production() -> str | None:
    return _read_pointers().get("production")


def set_shadow(version: str) -> None:
    pointers = _read_pointers()
    pointers["shadow"] = version
    _write_pointers(pointers)


def clear_shadow() -> None:
    pointers = _read_pointers()
    pointers.pop("shadow", None)
    _write_pointers(pointers)


def get_shadow() -> str | None:
    return _read_pointers().get("shadow")


def load_metrics(version: str) -> dict[str, float]:
    result: dict[str, float] = json.loads(metrics_path(version).read_text())
    return result
