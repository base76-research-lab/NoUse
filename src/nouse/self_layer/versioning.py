from __future__ import annotations

from pathlib import Path
from typing import Any

from nouse.self_layer.living_core import load_living_core


def living_core_version(path: Path | None = None) -> int:
    state = load_living_core(path)
    try:
        return max(1, int(state.get("version", 1) or 1))
    except (TypeError, ValueError):
        return 1


def has_identity_profile(path: Path | None = None) -> bool:
    state = load_living_core(path)
    identity = state.get("identity")
    return isinstance(identity, dict) and bool(str(identity.get("mission") or "").strip())


def version_snapshot(path: Path | None = None) -> dict[str, Any]:
    state = load_living_core(path)
    return {
        "version": living_core_version(path),
        "updated_at": str(state.get("updated_at") or ""),
        "mode": str((state.get("homeostasis") or {}).get("mode", "steady")),
        "active_drive": str((state.get("drives") or {}).get("active", "maintenance")),
    }
