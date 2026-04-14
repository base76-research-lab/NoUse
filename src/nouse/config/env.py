from __future__ import annotations

import os
import re
from pathlib import Path

_LOADED = False


def load_env_files(force: bool = False) -> None:
    """
    Laddar .env-filer till processmiljön (utan att skriva över existerande env vars).
    """
    global _LOADED
    if _LOADED and not force:
        return

    candidates = _env_candidates()

    for path in candidates:
        _load_single_file(path)

    _LOADED = True


def _load_single_file(path: Path) -> None:
    try:
        if not path.exists() or not path.is_file():
            return
    except OSError:
        # Extern mount kan ge I/O error vid stat(); ignorera och fortsätt.
        return

    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return
    except Exception:
        return

    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value and ((value[0] == value[-1]) and value[0] in {"'", '"'}):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _extra_env_candidates() -> list[Path]:
    """
    Valfria extra .env-sökvägar via env:
    - NOUSE_ENV_FILES="/path/a.env,/path/b.env"
    """
    raw = str(os.getenv("NOUSE_ENV_FILES", "")).strip()
    if not raw:
        return []
    out: list[Path] = []
    seen: set[str] = set()
    for part in re.split(r"[\n,;]+", raw):
        item = part.strip()
        if not item:
            continue
        p = Path(item).expanduser()
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _auto_profile_candidates() -> list[Path]:
    """
    Försök hålla CLI/daemon/journal på samma profil även när användaren inte
    satt NOUSE_ENV_FILES i aktuell shell.

    Prioritera .env.personal i cwd och repo root före vanliga .env-filer.
    """
    repo_root = Path(__file__).resolve().parents[3]
    return [
        Path.cwd() / ".env.personal",
        repo_root / ".env.personal",
        Path.cwd() / ".env",
        repo_root / ".env",
        Path.home() / ".env",
    ]


def _env_candidates() -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for path in [*_extra_env_candidates(), *_auto_profile_candidates()]:
        key = str(path.expanduser())
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out
