#!/usr/bin/env python3
"""
scripts/paperclip_ollama_agent.py — Paperclip process-adapter via Ollama.

Paperclip sätter:
  PAPERCLIP_AGENT_ID   — agent-ID
  PAPERCLIP_COMPANY_ID — company-ID
  PAPERCLIP_API_URL    — Paperclip API endpoint (http://127.0.0.1:3100)
"""
from __future__ import annotations
import json, os, sys
import httpx

OLLAMA_BASE = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.getenv("PAPERCLIP_OLLAMA_MODEL", "kimi-k2.5:cloud")
API_URL = os.getenv("PAPERCLIP_API_URL", "http://127.0.0.1:3100")
AGENT_ID = os.getenv("PAPERCLIP_AGENT_ID", "")
COMPANY_ID = os.getenv("PAPERCLIP_COMPANY_ID", "")

SYSTEM = """You are an AI research engineer on the Nous project —
a persistent epistemic substrate (plastic brain) for LLMs at /home/bjorn/projects/nouse.
Complete the assigned task thoroughly and post findings back via Paperclip API."""


def api_get(path: str) -> dict | list:
    with httpx.Client(timeout=15.0) as c:
        r = c.get(f"{API_URL}/api{path}")
        r.raise_for_status()
        return r.json()


def api_post(path: str, body: dict) -> dict:
    with httpx.Client(timeout=15.0) as c:
        r = c.post(f"{API_URL}/api{path}", json=body)
        r.raise_for_status()
        return r.json()


def api_patch(path: str, body: dict) -> dict:
    with httpx.Client(timeout=15.0) as c:
        r = c.patch(f"{API_URL}/api{path}", json=body)
        r.raise_for_status()
        return r.json()


def call_ollama(prompt: str) -> str:
    with httpx.Client(timeout=300.0) as c:
        r = c.post(f"{OLLAMA_BASE}/api/chat", json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        })
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "").strip()


def get_my_issues() -> list[dict]:
    try:
        all_issues = api_get(f"/companies/{COMPANY_ID}/issues?assigneeAgentId={AGENT_ID}&limit=10")
        items = all_issues if isinstance(all_issues, list) else all_issues.get("items", all_issues.get("issues", []))
        active = [i for i in items if i.get("status") in ("todo", "in_progress", "blocked")]
        return active[:3]
    except Exception as e:
        print(f"[warn] Kunde inte hämta issues: {e}", file=sys.stderr)
        return []


def get_issue_context(issue_id: str) -> str:
    try:
        ctx = api_get(f"/issues/{issue_id}/heartbeat-context")
        return json.dumps(ctx, ensure_ascii=False, indent=2)[:4000]
    except Exception:
        return ""


def post_comment(issue_id: str, body: str) -> None:
    try:
        api_post(f"/issues/{issue_id}/comments", {"body": body})
    except Exception as e:
        print(f"[warn] Kommentar misslyckades: {e}", file=sys.stderr)


def mark_done(issue_id: str, comment: str) -> None:
    try:
        api_patch(f"/issues/{issue_id}", {"status": "done", "comment": comment})
    except Exception as e:
        print(f"[warn] mark_done misslyckades: {e}", file=sys.stderr)


def main() -> None:
    print(f"[agent] ID={AGENT_ID[:8]}... | Model={MODEL}", flush=True)

    issues = get_my_issues()
    if not issues:
        print("[agent] Inga tilldelade issues.", flush=True)
        return

    issue = issues[0]
    issue_id = issue.get("id", "")
    title = issue.get("title", "")
    desc = issue.get("body") or issue.get("description") or ""
    identifier = issue.get("identifier", "")

    print(f"[agent] {identifier}: {title[:70]}", flush=True)

    ctx = get_issue_context(issue_id)
    prompt = (
        f"# Issue: {identifier} — {title}\n\n{desc}\n\n"
        f"## Context\n{ctx}\n\n"
        "Complete this task. You have access to /home/bjorn/projects/nouse. "
        "Provide detailed, actionable findings."
    )

    result = call_ollama(prompt)
    print(result[:300], flush=True)

    post_comment(issue_id, f"## Ollama Agent ({MODEL})\n\n{result}")
    mark_done(issue_id, f"Done by {MODEL}")
    print(f"[agent] {identifier} klar.", flush=True)


if __name__ == "__main__":
    main()
