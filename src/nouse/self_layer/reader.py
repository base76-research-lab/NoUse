from __future__ import annotations

from pathlib import Path
from typing import Any

from nouse.self_layer.living_core import identity_prompt_fragment, load_living_core


def read_self_state(path: Path | None = None) -> dict[str, Any]:
    return load_living_core(path)


def read_identity_prompt(path: Path | None = None) -> str:
    return identity_prompt_fragment(load_living_core(path))
