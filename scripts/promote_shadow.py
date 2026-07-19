"""Promotes the current shadow model to production. A human decision, not automatic —
shadow mode exists so a candidate can be observed on real traffic before it's trusted.

Run: uv run python scripts/promote_shadow.py
"""

from __future__ import annotations

from src.models import registry


def main() -> None:
    shadow_version = registry.get_shadow()
    if shadow_version is None:
        print("no shadow model registered — nothing to promote")
        return

    previous_production = registry.get_production()
    registry.set_production(shadow_version)
    registry.clear_shadow()
    print(f"promoted '{shadow_version}' to production (was '{previous_production}')")


if __name__ == "__main__":
    main()
