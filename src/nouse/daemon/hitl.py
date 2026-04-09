"""
b76.daemon.hitl — Human-in-the-loop interrupts för kritiska actions
===================================================================
Ger en enkel pause/approve/reject-mekanism som daemon-loop kan använda
för högrisk- eller mission-kritiska tasks.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


HITL_INTERRUPTS_PATH = Path.home() / ".local" / "share" / "nouse" / "hitl_interrupts.json"
_SENSITIVE_QUERY_TOKENS = (
    "delete",
    "radera",
    "remove",
    "unsafe",
    "credential",
    "secret",
    "rm -rf",
    "sudo",
    "drop table",
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path = HITL_INTERRUPTS_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(raw, list):
        return []
    return [r for r in raw if isinstance(r, dict)]


def _save(rows: list[dict[str, Any]], path: Path = HITL_INTERRUPTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def _next_id(rows: list[dict[str, Any]]) -> int:
    max_id = 0
    for row in rows:
        rid = row.get("id")
        if isinstance(rid, int) and rid > max_id:
            max_id = rid
    return max_id + 1


def _to_task_snapshot(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": int(task.get("id", 0) or 0),
        "domain": str(task.get("domain", "okänd")),
        "gap_type": str(task.get("gap_type", "unknown")),
        "priority": float(task.get("priority", 0.0) or 0.0),
        "query": str(task.get("query", "")),
        "concepts": [str(c) for c in (task.get("concepts") or [])][:6],
    }


def _contains_sensitive_query(query: str) -> bool:
    text = str(query or "").strip().lower()
    if not text:
        return False
    return any(tok in text for tok in _SENSITIVE_QUERY_TOKENS)


def critical_task_reason(
    task: dict[str, Any],
    *,
    priority_threshold: float = 0.98,
) -> str | None:
    if bool(task.get("hitl_approved")):
        return None

    gap_type = str(task.get("gap_type", "")).strip().lower()
    if gap_type.startswith("mission_"):
        return f"mission-kritisk task ({gap_type})"

    try:
        priority = float(task.get("priority", 0.0) or 0.0)
    except Exception:
        priority = 0.0
    if priority >= max(0.0, min(1.0, priority_threshold)):
        return f"hög prioritet ({priority:.2f})"

    query = str(task.get("query", "")).lower()
    if _contains_sensitive_query(query):
        return "innehåller känslig operation i query"
    return None


def low_risk_auto_approve_reason(
    task: dict[str, Any],
    *,
    reason: str = "",
    max_priority: float = 0.92,
    allow_gap_types: set[str] | None = None,
) -> str | None:
    """
    Returnerar en note-sträng om tasken kan auto-godkännas trots HITL-trigger.

    Policy:
    - endast mission-kritiska reason/gaptyper
    - endast under prioritetströskeln
    - aldrig när query innehåller känsliga operationer
    """
    if bool(task.get("hitl_approved")):
        return None

    reason_text = str(reason or "").strip().lower()
    if not reason_text.startswith("mission-kritisk task"):
        return None

    gap_type = str(task.get("gap_type", "")).strip().lower()
    if not gap_type.startswith("mission_"):
        return None

    if allow_gap_types:
        normalized = {str(x).strip().lower() for x in allow_gap_types if str(x).strip()}
        if normalized and gap_type not in normalized:
            return None

    try:
        priority = float(task.get("priority", 0.0) or 0.0)
    except Exception:
        priority = 0.0
    safe_max = max(0.0, min(1.0, float(max_priority)))
    if priority > safe_max:
        return None

    if _contains_sensitive_query(str(task.get("query", ""))):
        return None

    return f"auto-approved low-risk mission task ({gap_type}, priority={priority:.2f})"


def pending_interrupt_for_task(
    task_id: int,
    *,
    path: Path = HITL_INTERRUPTS_PATH,
) -> dict[str, Any] | None:
    rows = _load(path)
    for row in rows:
        if str(row.get("status", "")).lower() != "pending":
            continue
        if int(row.get("task_id", -1) or -1) == int(task_id):
            return row
    return None


def create_interrupt(
    *,
    task: dict[str, Any],
    reason: str,
    category: str = "critical_action",
    payload: dict[str, Any] | None = None,
    path: Path = HITL_INTERRUPTS_PATH,
) -> dict[str, Any]:
    rows = _load(path)
    now = _utcnow()
    row = {
        "id": _next_id(rows),
        "status": "pending",
        "category": str(category or "critical_action"),
        "reason": str(reason or "").strip()[:400],
        "created_at": now,
        "updated_at": now,
        "resolved_at": None,
        "task_id": int(task.get("id", 0) or 0),
        "task": _to_task_snapshot(task),
        "payload": payload or {},
        "reviewer": "",
        "note": "",
    }
    rows.append(row)
    _save(rows, path)
    return row


def _resolve_interrupt(
    interrupt_id: int,
    *,
    status: str,
    reviewer: str = "",
    note: str = "",
    path: Path = HITL_INTERRUPTS_PATH,
) -> dict[str, Any] | None:
    rows = _load(path)
    out: dict[str, Any] | None = None
    for row in rows:
        if int(row.get("id", -1) or -1) != int(interrupt_id):
            continue
        if str(row.get("status", "")).lower() != "pending":
            out = dict(row)
            break
        now = _utcnow()
        row["status"] = status
        row["reviewer"] = str(reviewer or "").strip()
        row["note"] = str(note or "").strip()[:800]
        row["updated_at"] = now
        row["resolved_at"] = now
        out = dict(row)
        break
    _save(rows, path)
    return out


def approve_interrupt(
    interrupt_id: int,
    *,
    reviewer: str = "",
    note: str = "",
    path: Path = HITL_INTERRUPTS_PATH,
) -> dict[str, Any] | None:
    return _resolve_interrupt(
        interrupt_id,
        status="approved",
        reviewer=reviewer,
        note=note,
        path=path,
    )


def reject_interrupt(
    interrupt_id: int,
    *,
    reviewer: str = "",
    note: str = "",
    path: Path = HITL_INTERRUPTS_PATH,
) -> dict[str, Any] | None:
    return _resolve_interrupt(
        interrupt_id,
        status="rejected",
        reviewer=reviewer,
        note=note,
        path=path,
    )


def list_interrupts(
    *,
    status: str | None = None,
    limit: int = 20,
    path: Path = HITL_INTERRUPTS_PATH,
) -> list[dict[str, Any]]:
    rows = _load(path)
    if status:
        want = str(status).lower().strip()
        rows = [r for r in rows if str(r.get("status", "")).lower() == want]
    rows.sort(key=lambda r: int(r.get("id", 0) or 0), reverse=True)
    return rows[: max(1, int(limit or 1))]


def interrupt_stats(path: Path = HITL_INTERRUPTS_PATH) -> dict[str, Any]:
    rows = _load(path)
    counts = {"pending": 0, "approved": 0, "rejected": 0}
    for row in rows:
        st = str(row.get("status", "")).lower()
        if st in counts:
            counts[st] += 1
    return {"total": len(rows), **counts, "path": str(path)}
