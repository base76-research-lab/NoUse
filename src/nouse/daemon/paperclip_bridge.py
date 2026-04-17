"""
nouse.daemon.paperclip_bridge — Nous goals → Paperclip issues
==============================================================
Postar Nous's aktiva mål (goal_registry) som issues i Paperclip.
Kör varje N:e cykel. Nous hjärnan styr Paperclip-företaget.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import httpx

log = logging.getLogger("nouse.paperclip_bridge")

PAPERCLIP_URL = os.getenv("PAPERCLIP_API_URL", "http://127.0.0.1:3100")
PAPERCLIP_COMPANY = os.getenv("PAPERCLIP_COMPANY_ID", "")
PAPERCLIP_AGENT = os.getenv("PAPERCLIP_TARGET_AGENT_ID", "")  # Agent att tilldela issues
_POSTED: set[str] = set()  # Deduplicering i minnet


def _post_issue(title: str, body: str, agent_id: str = "") -> bool:
    if not PAPERCLIP_COMPANY:
        return False
    payload: dict[str, Any] = {"title": title, "body": body, "status": "todo"}
    if agent_id:
        payload["assigneeAgentId"] = agent_id
    try:
        with httpx.Client(timeout=10.0) as c:
            r = c.post(
                f"{PAPERCLIP_URL}/api/companies/{PAPERCLIP_COMPANY}/issues",
                json=payload,
            )
            r.raise_for_status()
            identifier = r.json().get("identifier", "?")
            log.info("Paperclip: skapade issue %s — %s", identifier, title[:60])
            return True
    except Exception as e:
        log.debug("Paperclip bridge (non-fatal): %s", e)
        return False


def sync_goals_to_paperclip(field: Any, cycle: int, max_goals: int = 3) -> int:
    """
    Hämta aktiva Nous-mål och posta som Paperclip-issues.
    Returnerar antal skapade issues.
    """
    if not PAPERCLIP_COMPANY:
        return 0

    try:
        from nouse.daemon.goal_registry import active_goals
        goals = active_goals()
    except Exception as e:
        log.debug("Paperclip bridge: kunde inte läsa goals: %s", e)
        return 0

    created = 0
    for goal in goals[:max_goals]:
        title = goal.title if hasattr(goal, "title") else str(goal.get("title", ""))
        kind = goal.kind if hasattr(goal, "kind") else str(goal.get("kind", ""))
        priority = goal.priority if hasattr(goal, "priority") else goal.get("priority", 0.5)
        domain = goal.target_domain if hasattr(goal, "target_domain") else goal.get("target_domain", "")

        key = f"{kind}:{title[:40]}"
        if key in _POSTED:
            continue

        body = (
            f"**Nous epistemic goal** (cycle {cycle})\n\n"
            f"- Kind: `{kind}`\n"
            f"- Priority: `{priority:.2f}`\n"
            f"- Domain: `{domain}`\n\n"
            f"This issue was auto-created by the Nous goal_registry. "
            f"The epistemic substrate has identified this as a knowledge gap to fill."
        )

        if _post_issue(title, body, agent_id=PAPERCLIP_AGENT):
            _POSTED.add(key)
            created += 1

    return created
