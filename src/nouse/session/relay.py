"""
nouse.session.relay — Cross-model session handoff layer.

A relay session stores the semantic state of ongoing work: goal, decisions,
files touched, open questions, and which NoUse nodes were used.

Any model — Claude, Copilot, Codex, Gemma, llama — can open a relay, update
it as work progresses, and hand off to any other model via relay_continue().
The receiving model gets a compact context block without reloading full history.

Storage: $NOUSE_RELAY_DIR or $NOUSE_HOME/relay/<session_id>.json
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from nouse.config.paths import path_from_env

_LOCK = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relay_dir() -> Path:
    return path_from_env("NOUSE_RELAY_DIR", "relay")


def _session_state_path() -> Path:
    return path_from_env("NOUSE_SESSION_STATE_PATH", "session_state.json")


def _sanitize_id(value: str | None, *, default: str = "") -> str:
    clean = "".join(ch for ch in str(value or "").strip() if ch.isalnum() or ch in {"-", "_"})
    return clean[:64] if clean else default


def _relay_path(session_id: str) -> Path:
    return _relay_dir() / f"{session_id}.json"


def _runtime_snapshot(linked_session_id: str) -> dict[str, Any]:
    safe_sid = _sanitize_id(linked_session_id)
    if not safe_sid:
        return {}
    try:
        from nouse.session.state import get_session, list_runs

        state_path = _session_state_path()
        session = get_session(safe_sid, path=state_path)
        if not session:
            return {}
        runs = list_runs(session_id=safe_sid, limit=1, path=state_path)
        latest = runs[0] if runs else None
        return {
            "session_id": safe_sid,
            "lane": str(session.get("lane") or ""),
            "source": str(session.get("source") or ""),
            "status": str(session.get("status") or ""),
            "energy": float(session.get("energy", 0.5) or 0.5),
            "active_run_id": str(session.get("active_run_id") or ""),
            "latest_run": latest or {},
        }
    except Exception:
        return {}


# ── Core data types ───────────────────────────────────────────────────────────

def _blank_relay(
    session_id: str,
    goal: str,
    model: str,
    *,
    linked_session_id: str = "",
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "goal": goal,
        "status": "active",            # active | relay_ready | closed
        "started_by": model,
        "last_active_model": model,
        "linked_session_id": linked_session_id,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "decisions": [],               # [{what, why, confidence}]
        "open_questions": [],          # [str]
        "files_touched": [],           # [str]
        "nouse_nodes_used": [],        # [str]
        "summary": "",                 # one-line current status
        "tokens_saved": 0,
        "handoffs": 0,
    }


# ── CRUD ──────────────────────────────────────────────────────────────────────

def relay_open(
    goal: str,
    *,
    model: str = "unknown",
    session_id: str | None = None,
    linked_session_id: str | None = None,
) -> dict[str, Any]:
    """Start a new relay session. Returns the session dict."""
    sid = _sanitize_id(session_id) or f"relay_{uuid4().hex[:10]}"
    linked_sid = _sanitize_id(linked_session_id)
    _relay_dir().mkdir(parents=True, exist_ok=True)
    relay = _blank_relay(sid, goal=goal, model=model, linked_session_id=linked_sid)
    with _LOCK:
        _relay_path(sid).write_text(
            json.dumps(relay, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return relay


def relay_update(
    session_id: str,
    *,
    decision: str | None = None,
    decision_why: str = "",
    decision_confidence: float = 0.8,
    open_question: str | None = None,
    file_touched: str | None = None,
    node_used: str | None = None,
    summary: str | None = None,
    status: str | None = None,
    model: str | None = None,
    linked_session_id: str | None = None,
) -> dict[str, Any] | None:
    """Update an active relay session with new work context."""
    path = _relay_path(_sanitize_id(session_id))
    with _LOCK:
        if not path.exists():
            return None
        relay = json.loads(path.read_text(encoding="utf-8"))

        if decision:
            relay["decisions"].append({
                "what": decision,
                "why": decision_why,
                "confidence": round(float(decision_confidence), 2),
                "at": _now_iso(),
            })
        if open_question and open_question not in relay["open_questions"]:
            relay["open_questions"].append(open_question)
        if file_touched and file_touched not in relay["files_touched"]:
            relay["files_touched"].append(file_touched)
        if node_used and node_used not in relay["nouse_nodes_used"]:
            relay["nouse_nodes_used"].append(node_used)
        if summary is not None:
            relay["summary"] = summary
        if status in {"active", "relay_ready", "closed"}:
            relay["status"] = status
        if model:
            relay["last_active_model"] = model
        if linked_session_id is not None:
            relay["linked_session_id"] = _sanitize_id(linked_session_id)

        relay["updated_at"] = _now_iso()
        path.write_text(json.dumps(relay, ensure_ascii=False, indent=2), encoding="utf-8")
        return dict(relay)


def relay_continue(session_id: str, *, model: str = "unknown") -> str:
    """
    Return a context block for the receiving model.

    The block contains goal, decisions, open questions, files touched,
    and which NoUse nodes were used — compact enough to fit in any system prompt.
    """
    safe_sid = _sanitize_id(session_id)
    path = _relay_path(safe_sid)
    if not path.exists():
        return f"[relay] session '{safe_sid or session_id}' not found"

    with _LOCK:
        relay = json.loads(path.read_text(encoding="utf-8"))
        relay["last_active_model"] = model
        relay["handoffs"] = relay.get("handoffs", 0) + 1
        relay["updated_at"] = _now_iso()
        path.write_text(json.dumps(relay, ensure_ascii=False, indent=2), encoding="utf-8")

    runtime = _runtime_snapshot(str(relay.get("linked_session_id") or ""))

    lines = [
        f"[Nous session handoff — {relay['session_id']}]",
        "Resume the existing job. Do not restart broad discovery unless the relay state is insufficient.",
        f"Goal: {relay['goal']}",
        f"Started by: {relay['started_by']}  |  Last active: {relay.get('last_active_model', '?')}",
        f"Status: {relay['status']}",
    ]

    if relay.get("summary"):
        lines.append(f"Current state: {relay['summary']}")
    if runtime:
        lines.append(
            "Runtime session: "
            f"{runtime['session_id']}  |  lane={runtime['lane'] or 'unknown'}"
            f"  |  status={runtime['status'] or 'unknown'}"
            f"  |  energy={runtime['energy']:.2f}"
        )
        latest = runtime.get("latest_run") or {}
        if latest:
            provider_model = f"{latest.get('provider', '')}/{latest.get('model', '')}".strip("/")
            lines.append(
                "Latest run: "
                f"{latest.get('workload', 'unknown')}  |  "
                f"{provider_model or 'unknown'}"
                f"  |  status={latest.get('status', 'unknown')}"
            )

    decisions = relay.get("decisions", [])
    if decisions:
        lines.append("\nDecisions made:")
        for d in decisions[-5:]:  # last 5 to keep context compact
            conf = d.get("confidence", 0)
            why = f" — {d['why']}" if d.get("why") else ""
            lines.append(f"  • {d['what']}{why}  [confidence={conf}]")

    open_q = relay.get("open_questions", [])
    if open_q:
        lines.append("\nOpen questions:")
        for q in open_q[-3:]:
            lines.append(f"  ? {q}")

    files = relay.get("files_touched", [])
    if files:
        lines.append(f"\nFiles touched: {', '.join(files[-8:])}")

    nodes = relay.get("nouse_nodes_used", [])
    if nodes:
        lines.append(f"Nous nodes: {', '.join(nodes[-6:])}")

    lines.append(f"\nHandoffs: {relay.get('handoffs', 0)}  |  Updated: {relay['updated_at'][:19]}")
    return "\n".join(lines)


def relay_get(session_id: str) -> dict[str, Any] | None:
    path = _relay_path(_sanitize_id(session_id))
    if not path.exists():
        return None
    with _LOCK:
        return json.loads(path.read_text(encoding="utf-8"))


def relay_list(*, status: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    _relay_dir().mkdir(parents=True, exist_ok=True)
    sessions = []
    for p in sorted(_relay_dir().glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            relay = json.loads(p.read_text(encoding="utf-8"))
            if status and relay.get("status") != status:
                continue
            sessions.append(relay)
            if len(sessions) >= limit:
                break
        except Exception:
            continue
    return sessions


def relay_close(session_id: str) -> dict[str, Any] | None:
    return relay_update(session_id, status="closed")
