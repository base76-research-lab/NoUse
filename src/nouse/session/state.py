from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from nouse.config.paths import path_from_env

SESSION_STATE_PATH = path_from_env("NOUSE_SESSION_STATE_PATH", "session_state.json")
_LOCK = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _blank_state() -> dict[str, Any]:
    return {
        "sessions": {},
        "runs": [],
        "updated_at": _now_iso(),
    }


def _normalize_state(raw: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return _blank_state()
    sessions = raw.get("sessions")
    runs = raw.get("runs")
    out = _blank_state()
    out["sessions"] = sessions if isinstance(sessions, dict) else {}
    out["runs"] = runs if isinstance(runs, list) else []
    out["updated_at"] = str(raw.get("updated_at") or out["updated_at"])
    return out


def load_state(path: Path = SESSION_STATE_PATH) -> dict[str, Any]:
    if not path.exists():
        return _blank_state()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _blank_state()
    return _normalize_state(raw)


def save_state(state: dict[str, Any], path: Path = SESSION_STATE_PATH) -> None:
    out = _normalize_state(state)
    out["updated_at"] = _now_iso()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def _sanitize_session_id(session_id: str | None) -> str:
    clean = "".join(ch for ch in str(session_id or "").strip() if ch.isalnum() or ch in {"-", "_"})
    return clean[:64] if clean else ""


def _default_session(session_id: str, *, lane: str, source: str, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    now = _now_iso()
    return {
        "id": session_id,
        "lane": str(lane or "main").strip() or "main",
        "source": str(source or "unknown").strip() or "unknown",
        "status": "idle",
        "created_at": now,
        "updated_at": now,
        "last_seen_at": now,
        "active_run_id": None,
        "energy": 0.5,
        "meta": dict(meta or {}),
        "counters": {
            "started": 0,
            "succeeded": 0,
            "failed": 0,
            "cancelled": 0,
        },
    }


def ensure_session(
    session_id: str,
    *,
    lane: str = "main",
    source: str = "cli",
    meta: dict[str, Any] | None = None,
    path: Path = SESSION_STATE_PATH,
) -> dict[str, Any]:
    sid = _sanitize_session_id(session_id) or "main"
    with _LOCK:
        state = load_state(path)
        sessions = state.setdefault("sessions", {})
        row = sessions.get(sid)
        if not isinstance(row, dict):
            row = _default_session(sid, lane=lane, source=source, meta=meta)
            sessions[sid] = row
        else:
            row.setdefault("id", sid)
            row["lane"] = str(lane or row.get("lane") or "main").strip() or "main"
            row["source"] = str(source or row.get("source") or "unknown").strip() or "unknown"
            row.setdefault("meta", {})
            if isinstance(meta, dict):
                row["meta"].update(meta)
            row.setdefault("counters", {})
            for key in ("started", "succeeded", "failed", "cancelled"):
                row["counters"][key] = int(row["counters"].get(key, 0) or 0)
            row.setdefault("energy", 0.5)
            row.setdefault("status", "idle")
            row.setdefault("created_at", _now_iso())
        row["last_seen_at"] = _now_iso()
        row["updated_at"] = _now_iso()
        sessions[sid] = row
        save_state(state, path)
        return dict(row)


def create_session(
    *,
    session_id: str | None = None,
    lane: str = "main",
    source: str = "cli",
    meta: dict[str, Any] | None = None,
    path: Path = SESSION_STATE_PATH,
) -> dict[str, Any]:
    sid = _sanitize_session_id(session_id) if session_id else ""
    if not sid:
        sid = f"s_{uuid4().hex[:10]}"
    return ensure_session(
        sid,
        lane=lane,
        source=source,
        meta=meta,
        path=path,
    )


def get_session(session_id: str, path: Path = SESSION_STATE_PATH) -> dict[str, Any] | None:
    sid = _sanitize_session_id(session_id)
    if not sid:
        return None
    with _LOCK:
        state = load_state(path)
        row = (state.get("sessions") or {}).get(sid)
        if not isinstance(row, dict):
            return None
        return dict(row)


def list_sessions(
    *,
    status: str | None = None,
    limit: int = 50,
    path: Path = SESSION_STATE_PATH,
) -> list[dict[str, Any]]:
    safe_limit = max(1, min(int(limit), 5000))
    wanted = str(status or "").strip().lower()
    with _LOCK:
        state = load_state(path)
        rows = []
        for row in (state.get("sessions") or {}).values():
            if not isinstance(row, dict):
                continue
            cur_status = str(row.get("status") or "idle").strip().lower()
            if wanted and wanted != "all" and cur_status != wanted:
                continue
            rows.append(dict(row))
    rows.sort(key=lambda r: str(r.get("updated_at") or ""), reverse=True)
    return rows[:safe_limit]


def start_run(
    session_id: str,
    *,
    workload: str,
    model: str = "",
    provider: str = "",
    request_chars: int = 0,
    meta: dict[str, Any] | None = None,
    path: Path = SESSION_STATE_PATH,
) -> dict[str, Any]:
    session = ensure_session(session_id, path=path)
    run_id = f"run_{uuid4().hex[:12]}"
    run = {
        "run_id": run_id,
        "session_id": session["id"],
        "workload": str(workload or "unknown").strip() or "unknown",
        "model": str(model or "").strip(),
        "provider": str(provider or "").strip(),
        "status": "running",
        "started_at": _now_iso(),
        "ended_at": None,
        "error": "",
        "request_chars": max(0, int(request_chars or 0)),
        "response_chars": 0,
        "meta": dict(meta or {}),
        "metrics": {},
    }
    with _LOCK:
        state = load_state(path)
        sessions = state.setdefault("sessions", {})
        row = sessions.get(session["id"])
        if not isinstance(row, dict):
            row = _default_session(session["id"], lane="main", source="unknown")
        counters = row.setdefault("counters", {})
        counters["started"] = int(counters.get("started", 0) or 0) + 1
        row["status"] = "running"
        row["active_run_id"] = run_id
        row["last_seen_at"] = _now_iso()
        row["updated_at"] = _now_iso()
        sessions[session["id"]] = row
        runs = state.setdefault("runs", [])
        runs.append(run)
        if len(runs) > 5000:
            state["runs"] = runs[-5000:]
        save_state(state, path)
    return dict(run)


def finish_run(
    run_id: str,
    *,
    status: str = "succeeded",
    error: str = "",
    response_chars: int = 0,
    metrics: dict[str, Any] | None = None,
    path: Path = SESSION_STATE_PATH,
) -> dict[str, Any] | None:
    wanted_run = str(run_id or "").strip()
    if not wanted_run:
        return None
    normalized_status = str(status or "succeeded").strip().lower()
    if normalized_status not in {"succeeded", "failed", "cancelled"}:
        normalized_status = "failed"
    with _LOCK:
        state = load_state(path)
        runs = state.setdefault("runs", [])
        target: dict[str, Any] | None = None
        for row in reversed(runs):
            if not isinstance(row, dict):
                continue
            if str(row.get("run_id") or "") != wanted_run:
                continue
            target = row
            break
        if target is None:
            return None
        target["status"] = normalized_status
        target["ended_at"] = _now_iso()
        target["error"] = str(error or "")[:1000]
        target["response_chars"] = max(0, int(response_chars or 0))
        target["metrics"] = dict(metrics or {})

        session_id = str(target.get("session_id") or "")
        sessions = state.setdefault("sessions", {})
        session = sessions.get(session_id)
        if isinstance(session, dict):
            counters = session.setdefault("counters", {})
            key = "failed" if normalized_status == "failed" else normalized_status
            counters[key] = int(counters.get(key, 0) or 0) + 1
            if str(session.get("active_run_id") or "") == wanted_run:
                session["active_run_id"] = None
            session["status"] = "idle"
            session["last_seen_at"] = _now_iso()
            session["updated_at"] = _now_iso()
            sessions[session_id] = session
        save_state(state, path)
        return dict(target)


def list_runs(
    *,
    session_id: str | None = None,
    status: str | None = None,
    limit: int = 100,
    path: Path = SESSION_STATE_PATH,
) -> list[dict[str, Any]]:
    sid = _sanitize_session_id(session_id or "")
    wanted_status = str(status or "").strip().lower()
    safe_limit = max(1, min(int(limit), 5000))
    with _LOCK:
        state = load_state(path)
        rows = []
        for row in reversed(state.get("runs") or []):
            if not isinstance(row, dict):
                continue
            if sid and str(row.get("session_id") or "") != sid:
                continue
            cur_status = str(row.get("status") or "").strip().lower()
            if wanted_status and wanted_status != "all" and cur_status != wanted_status:
                continue
            rows.append(dict(row))
            if len(rows) >= safe_limit:
                break
    return rows


def session_stats(path: Path = SESSION_STATE_PATH) -> dict[str, Any]:
    with _LOCK:
        state = load_state(path)
        sessions = [
            row for row in (state.get("sessions") or {}).values()
            if isinstance(row, dict)
        ]
        runs = [
            row for row in (state.get("runs") or [])
            if isinstance(row, dict)
        ]
    running_sessions = sum(1 for row in sessions if str(row.get("status") or "") == "running")
    active_run_ids = {
        str(row.get("active_run_id") or "")
        for row in sessions
        if str(row.get("active_run_id") or "")
    }
    return {
        "sessions_total": len(sessions),
        "sessions_running": running_sessions,
        "runs_total": len(runs),
        "active_runs": len(active_run_ids),
        "updated_at": str(state.get("updated_at") or ""),
    }


def set_session_energy(
    session_id: str,
    energy: float,
    *,
    path: Path = SESSION_STATE_PATH,
) -> dict[str, Any]:
    safe_energy = max(0.0, min(1.0, float(energy)))
    session = ensure_session(session_id, path=path)
    with _LOCK:
        state = load_state(path)
        sessions = state.setdefault("sessions", {})
        row = sessions.get(session["id"])
        if not isinstance(row, dict):
            row = _default_session(session["id"], lane="main", source="unknown")
        row["energy"] = safe_energy
        row["updated_at"] = _now_iso()
        row["last_seen_at"] = _now_iso()
        sessions[session["id"]] = row
        save_state(state, path)
        return dict(row)


def clear_stale_running(
    *,
    max_age_sec: float = 3600.0,
    reason: str = "doctor_stale_cleanup",
    path: Path = SESSION_STATE_PATH,
) -> list[str]:
    safe_age = max(0.1, float(max_age_sec))
    now = datetime.now(timezone.utc).timestamp()
    fixed: list[str] = []
    with _LOCK:
        state = load_state(path)
        sessions = state.setdefault("sessions", {})
        for sid, row in sessions.items():
            if not isinstance(row, dict):
                continue
            if str(row.get("status") or "") != "running":
                continue
            updated_raw = str(row.get("updated_at") or "")
            ts = None
            if updated_raw:
                value = updated_raw[:-1] + "+00:00" if updated_raw.endswith("Z") else updated_raw
                try:
                    dt = datetime.fromisoformat(value)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    ts = dt.astimezone(timezone.utc).timestamp()
                except ValueError:
                    ts = None
            if ts is None:
                continue
            if now - ts <= safe_age:
                continue
            row["status"] = "idle"
            row["active_run_id"] = None
            row["updated_at"] = _now_iso()
            row["last_seen_at"] = _now_iso()
            meta = row.setdefault("meta", {})
            meta["doctor_fix"] = reason
            sessions[sid] = row
            fixed.append(str(sid))
        if fixed:
            save_state(state, path)
    return fixed
