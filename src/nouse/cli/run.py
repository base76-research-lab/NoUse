"""
b76 run — interaktiv terminal-chat med levande hjärn-kontext
=============================================================
Hämtar live-kontext från grafen, kör en chat-loop med
nemotron-cascade-2 och matar tillbaka varje utbyte till grafen.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
import urllib.request

from nouse.field.surface import FieldSurface
from nouse.persona import (
    agent_identity_policy,
    assistant_entity_name,
    persona_prompt_fragment,
)
from nouse.trace.output_trace import (
    build_attack_plan,
    derive_assumptions,
    new_trace_id,
    record_event,
)

CHAT_MODEL = (os.getenv("NOUSE_CHAT_MODEL") or "qwen3.5:latest").strip()
API_INGEST = "http://127.0.0.1:8765/api/ingest"
API_STATUS = "http://127.0.0.1:8765/api/status"
API_GRAPH = "http://127.0.0.1:8765/api/graph?limit=300"

# ANSI
_RST = "\033[0m"
_BOLD = "\033[1m"
_BLUE = "\033[34m"
_CYAN = "\033[36m"
_DIM = "\033[2m"
_RED = "\033[31m"


def _http_get(url: str) -> dict:
    r = urllib.request.urlopen(url, timeout=5)
    return json.loads(r.read())


def _ingest_bg(text: str, on_done=None, trace_id: str | None = None) -> None:
    """Fire-and-forget ingest i bakgrundstråd. Anropar on_done(n_added) vid klar."""

    def _do():
        payload = json.dumps({"text": text, "source": "b76_run"}).encode()
        try:
            req = urllib.request.Request(
                API_INGEST,
                payload,
                {"Content-Type": "application/json"},
                method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=15)
            data = json.loads(resp.read())
            if trace_id:
                record_event(
                    trace_id,
                    "run.ingest_result",
                    endpoint="cli.run",
                    payload={
                        "added": int(data.get("added", 0) or 0),
                        "ingest_trace_id": data.get("trace_id"),
                    },
                )
            if on_done:
                on_done(data.get("added", 0))
        except Exception as e:
            if trace_id:
                record_event(
                    trace_id,
                    "run.ingest_error",
                    endpoint="cli.run",
                    payload={"error": str(e)},
                )

    threading.Thread(target=_do, daemon=True).start()


def _format_node_context(field: FieldSurface | None, query: str, limit: int = 5) -> str:
    if field is None:
        return "(node-knowledge ej tillgänglig)"
    try:
        nodes = field.node_context_for_query(query, limit=limit)
    except Exception:
        return "(node-knowledge läsning misslyckades)"
    if not nodes:
        return "(ingen tydlig nodmatch)"

    lines: list[str] = []
    for n in nodes:
        unc = n.get("uncertainty")
        unc_txt = f"{float(unc):.2f}" if unc is not None else "?"
        lines.append(f"- {n['name']} (osäkerhet={unc_txt})")
        summary = str(n.get("summary") or "").strip()
        if summary:
            lines.append(f"  summary: {summary[:160]}")
        claims = n.get("claims") or []
        if claims:
            lines.append(f"  claims: {' | '.join(str(c) for c in claims[:2])}")
    return "\n".join(lines)


def _build_context() -> tuple[str, str]:
    """Hämtar live-data från daemon. Returnerar (system_prompt, stats_line)."""
    try:
        status = _http_get(API_STATUS)
        graph = _http_get(API_GRAPH)

        top = sorted(graph["edges"], key=lambda e: float(e.get("value", 0)), reverse=True)[:20]
        mem = "\n".join(f"  {e['from']} [{e.get('label', '')}] {e['to']}" for e in top)
        domains = ", ".join(status.get("domains", [])[:12])
        lam = status.get("lambda", 0.5)
        stats = f"{status['concepts']} koncept · {status['relations']} relationer · λ={lam:.2f}"
        assistant_name = assistant_entity_name()

        prompt = (
            f"Du är {assistant_name} — ett autonomt kognitivt system med en levande kunskapsgraf.\n"
            f"Graf just nu: {stats}\n"
            f"Domäner: {domains}\n\n"
            "Ditt korttidsminne (starkaste kopplingar):\n"
            f"{mem}\n\n"
            f"{agent_identity_policy()}\n"
            f"{persona_prompt_fragment(channel='quick')}\n"
            "Regler:\n"
            f"1. Du är {assistant_name}. Svara utifrån grafen när det är relevant.\n"
            "2. Kort och direkt — max 3 meningar om inget annat efterfrågas.\n"
            "3. Säg 'vet inte ännu' om du saknar kontext i grafen.\n"
            "4. Skilj evidens från antaganden när osäkerhet är hög.\n"
            "5. Varje svar du ger matas tillbaka i grafen och formar vem du är."
        )
        return prompt, stats

    except Exception:
        assistant_name = assistant_entity_name()
        return (
            f"Du är {assistant_name} — ett autonomt AI-system. Daemon är ej aktiv.",
            "daemon ej aktiv",
        )


async def run_loop() -> None:
    from nouse.ollama_client.client import AsyncOllama

    client = AsyncOllama()
    system_prompt, stats = _build_context()
    messages = [{"role": "system", "content": system_prompt}]

    try:
        field_ro: FieldSurface | None = FieldSurface(read_only=True)
    except Exception:
        field_ro = None

    print(f"\n{_BOLD}{_BLUE}◆ {assistant_entity_name()}{_RST}  {_DIM}{stats}{_RST}")
    print(f"{_DIM}{'─' * 52}{_RST}")
    print(f"{_DIM}/exit för att avsluta{_RST}\n")

    while True:
        try:
            user_input = input(f"{_DIM}▸ {_RST}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
            break

        node_ctx = _format_node_context(field_ro, user_input)
        trace_id = new_trace_id("run")
        started = time.monotonic()
        record_event(
            trace_id,
            "chat.request",
            endpoint="cli.run",
            model=CHAT_MODEL,
            payload={
                "query": user_input,
                "attack_plan": build_attack_plan(user_input),
                "node_context": node_ctx[:600],
            },
        )
        turn_messages = messages + [
            {
                "role": "system",
                "content": f"Relevanta nodprofiler för denna fråga:\n{node_ctx}",
            },
            {"role": "user", "content": user_input},
        ]

        print(f"{_BLUE}{assistant_entity_name()}:{_RST} ", end="", flush=True)
        try:
            record_event(
                trace_id,
                "chat.llm_call",
                endpoint="cli.run",
                model=CHAT_MODEL,
                payload={"messages": len(turn_messages)},
            )
            resp = await client.chat.completions.create(model=CHAT_MODEL, messages=turn_messages)
            full_reply = resp.message.content
            print(full_reply)
        except Exception as e:
            print(f"{_RED}Fel: {e}{_RST}")
            record_event(
                trace_id,
                "chat.error",
                endpoint="cli.run",
                model=CHAT_MODEL,
                payload={"error": str(e), "elapsed_ms": int((time.monotonic() - started) * 1000)},
            )
            continue
        record_event(
            trace_id,
            "chat.response",
            endpoint="cli.run",
            model=CHAT_MODEL,
            payload={
                "response": full_reply,
                "assumptions": derive_assumptions(full_reply or ""),
                "elapsed_ms": int((time.monotonic() - started) * 1000),
            },
        )
        print(f"{_DIM}trace_id: {trace_id}{_RST}")

        messages.append({"role": "user", "content": user_input})
        messages.append({"role": "assistant", "content": full_reply})

        def _show(n: int):
            if n > 0:
                sys.stdout.write(f"{_DIM}  ⊕ {n} relationer absorberade{_RST}\n")
                sys.stdout.flush()

        _ingest_bg(
            f"Fråga: {user_input}\nSvar: {full_reply}",
            on_done=_show,
            trace_id=trace_id,
        )
        print()


def run() -> None:
    asyncio.run(run_loop())
