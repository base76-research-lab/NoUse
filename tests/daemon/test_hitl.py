from __future__ import annotations

from pathlib import Path

from nouse.daemon.hitl import (
    approve_interrupt,
    create_interrupt,
    critical_task_reason,
    list_interrupts,
    low_risk_auto_approve_reason,
    pending_interrupt_for_task,
    reject_interrupt,
)


def test_create_and_approve_interrupt(tmp_path: Path):
    path = tmp_path / "hitl.json"
    row = create_interrupt(
        task={
            "id": 7,
            "domain": "artificiell intelligens",
            "gap_type": "mission_focus_domain",
            "priority": 0.99,
            "query": "test",
            "concepts": ["a", "b"],
        },
        reason="mission-kritisk task",
        path=path,
    )
    assert int(row["id"]) == 1
    assert pending_interrupt_for_task(7, path=path) is not None

    approved = approve_interrupt(1, reviewer="bjorn", note="ok", path=path)
    assert approved is not None
    assert approved["status"] == "approved"
    assert pending_interrupt_for_task(7, path=path) is None


def test_reject_interrupt(tmp_path: Path):
    path = tmp_path / "hitl.json"
    create_interrupt(
        task={"id": 9, "domain": "x", "gap_type": "mission_bootstrap_domain"},
        reason="kritisk",
        path=path,
    )
    rejected = reject_interrupt(1, reviewer="bjorn", note="nej", path=path)
    assert rejected is not None
    assert rejected["status"] == "rejected"
    rows = list_interrupts(status="rejected", limit=5, path=path)
    assert len(rows) == 1


def test_critical_task_reason_prefers_mission_and_priority():
    assert (
        critical_task_reason(
            {"gap_type": "mission_cross_domain", "priority": 0.3},
            priority_threshold=0.98,
        )
        == "mission-kritisk task (mission_cross_domain)"
    )
    assert (
        critical_task_reason(
            {"gap_type": "fragmented_domain", "priority": 0.99},
            priority_threshold=0.98,
        )
        == "hög prioritet (0.99)"
    )
    assert (
        critical_task_reason(
            {"gap_type": "fragmented_domain", "priority": 0.3, "query": "safe"},
            priority_threshold=0.98,
        )
        is None
    )


def test_low_risk_auto_approve_reason_accepts_safe_mission_task():
    note = low_risk_auto_approve_reason(
        {
            "gap_type": "mission_focus_domain",
            "priority": 0.74,
            "query": "Sammanställ öppna frågor i AI-domänen",
        },
        reason="mission-kritisk task (mission_focus_domain)",
        max_priority=0.92,
        allow_gap_types={"mission_focus_domain", "mission_bootstrap_domain"},
    )
    assert note is not None
    assert "auto-approved low-risk mission task" in note


def test_low_risk_auto_approve_reason_rejects_high_risk_or_sensitive():
    assert (
        low_risk_auto_approve_reason(
            {
                "gap_type": "mission_focus_domain",
                "priority": 0.99,
                "query": "safe query",
            },
            reason="mission-kritisk task (mission_focus_domain)",
            max_priority=0.92,
            allow_gap_types={"mission_focus_domain"},
        )
        is None
    )
    assert (
        low_risk_auto_approve_reason(
            {
                "gap_type": "mission_focus_domain",
                "priority": 0.40,
                "query": "delete all credentials now",
            },
            reason="mission-kritisk task (mission_focus_domain)",
            max_priority=0.92,
            allow_gap_types={"mission_focus_domain"},
        )
        is None
    )
