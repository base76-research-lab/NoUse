"""
b76.metacognition.snapshot — Forsknings-snapshots
===================================================
Tar kompletta frysta kopior ("snapshots") av hela AI-systemets tillstånd:
 - Grafdatabasen (SQLite WAL + graph state)
 - Limbiska värden (Dopamin, λ, Arousal)
 - Domän-betti (TDA H0/H1 värden)

Denna modul används för att kunna "spola tillbaka" hjärnan för forskning,
eller för att jämföra systemets kreativitet över tid (före/efter en bisociation).
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

from nouse.daemon.lock import BrainLock
from nouse.config.paths import path_from_env
from nouse.field.surface import FieldSurface
from nouse.limbic.signals import load_state

log = logging.getLogger("nouse.snapshot")

SNAPSHOT_DIR = path_from_env("NOUSE_SNAPSHOT_DIR", "snapshots")
LIVE_FIELD_SQLITE = path_from_env("NOUSE_FIELD_DB", "field.sqlite")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def list_snapshots(limit: int = 50) -> list[dict]:
    """Lista snapshots (senaste först)."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for p in SNAPSHOT_DIR.iterdir():
        if not p.is_dir():
            continue
        meta_path = p / "meta.json"
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        sqlite_path = p / "field.sqlite"
        rows.append(
            {
                "name": p.name,
                "path": str(p),
                "has_sqlite": sqlite_path.exists(),
                "size_bytes": int(sqlite_path.stat().st_size) if sqlite_path.exists() else 0,
                "timestamp": str(meta.get("timestamp") or ""),
                "tag": str(meta.get("tag") or ""),
                "mtime": datetime.utcfromtimestamp(p.stat().st_mtime).isoformat() + "Z",
            }
        )
    rows.sort(key=lambda r: str(r.get("mtime") or ""), reverse=True)
    safe_limit = max(1, min(int(limit), 500))
    return rows[:safe_limit]


def _resolve_snapshot_dir(snapshot_ref: str) -> Path:
    raw = str(snapshot_ref or "").strip()
    if not raw:
        raise ValueError("snapshot_ref saknas")
    cand = Path(raw).expanduser()
    if cand.exists() and cand.is_dir():
        return cand
    by_name = SNAPSHOT_DIR / raw
    if by_name.exists() and by_name.is_dir():
        return by_name
    raise FileNotFoundError(f"Hittar inte snapshot: {snapshot_ref}")


def create_snapshot(field: FieldSurface, tag: str = "auto") -> str:
    """
    Skapar ett fryst snapshot av hjärnans nuvarande konfiguration.
    Tar cirka 1 sekund. MÅSTE använda BrainLock runtom.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_name = f"snapshot_{timestamp}_{tag}"
    target_dir = SNAPSHOT_DIR / snapshot_name

    target_dir.mkdir(parents=True, exist_ok=True)

    log.info("Metacognition: Påbörjar snapshot '%s' för forskningsanalys.", snapshot_name)

    with BrainLock(timeout=10.0):
        # 1. SQLite Backup — kopiera databasfilen
        sqlite_path = LIVE_FIELD_SQLITE
        if sqlite_path.exists():
            db_target = target_dir / "field.sqlite"
            shutil.copy2(sqlite_path, db_target)

        # 2. Limbic State
        limbic = load_state()

        # 3. Extrahera TDA Betti Numbers per domän
        domains = field.domains()
        topo_profiles = {}
        for d in domains:
            try:
                topo_profiles[d] = field.domain_tda_profile(d)
            except Exception:
                pass

        # 4. Generell graf-statistik
        stats = field.stats()

        # Spara all metadata
        meta = {
            "timestamp": datetime.now().isoformat(),
            "tag": tag,
            "architecture_phase": "5_Autonomy",
            "stats": stats,
            "limbic": {
                "dopamine": limbic.dopamine,
                "noradrenaline": limbic.noradrenaline,
                "acetylcholine": limbic.acetylcholine,
                "lam": limbic.lam,
                "arousal": limbic.arousal,
                "pruning_aggression": limbic.pruning_aggression,
                "cycle": limbic.cycle,
            },
            "topological_profiles": topo_profiles,
        }

        (target_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("Snapshot '%s' färdigställt: sparat till %s", snapshot_name, target_dir)
    return str(target_dir)


def restore_snapshot(snapshot_ref: str, *, create_backup: bool = True) -> dict:
    """
    Återställ live-databasen från snapshot.
    snapshot_ref kan vara snapshot-namn eller absolut katalogsökväg.
    """
    snap_dir = _resolve_snapshot_dir(snapshot_ref)
    src_sqlite = snap_dir / "field.sqlite"
    if not src_sqlite.exists():
        raise FileNotFoundError(f"Snapshot saknar field.sqlite: {src_sqlite}")

    LIVE_FIELD_SQLITE.parent.mkdir(parents=True, exist_ok=True)

    backup_path = ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with BrainLock(timeout=20.0):
        if LIVE_FIELD_SQLITE.exists() and create_backup:
            backup_dir = SNAPSHOT_DIR / f"pre_restore_{ts}"
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_sqlite = backup_dir / "field.sqlite"
            shutil.copy2(LIVE_FIELD_SQLITE, backup_sqlite)
            (backup_dir / "meta.json").write_text(
                json.dumps(
                    {
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "tag": "auto_pre_restore_backup",
                        "source_snapshot": str(snap_dir),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            backup_path = str(backup_dir)

        # Undvik stale sqlite-sidofiler.
        for side in (
            LIVE_FIELD_SQLITE.with_name(LIVE_FIELD_SQLITE.name + "-wal"),
            LIVE_FIELD_SQLITE.with_name(LIVE_FIELD_SQLITE.name + "-shm"),
        ):
            try:
                if side.exists():
                    side.unlink()
            except Exception:
                pass

        shutil.copy2(src_sqlite, LIVE_FIELD_SQLITE)

    return {
        "status": "ok",
        "restored_from": str(snap_dir),
        "live_path": str(LIVE_FIELD_SQLITE),
        "backup_path": backup_path,
        "live_sha256": _sha256(LIVE_FIELD_SQLITE),
    }
