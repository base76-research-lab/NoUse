"""
nouse.search.escalator — Knowledge Escalation Pipeline
=======================================================

Triggas när brain.query() ger låg konfidens eller tomt svar.

Nivå 1: Graf (ev ≥ threshold)       → return direkt, ingen escalation
Nivå 2: DDG web-sök + scrape        → injicera i kontext + lär in i graf
Nivå 3: NightRun-queue              → ny fakta → DeepDive async konsolidering

Trigger-kriterier ("flat LLM knowledge"):
  - result.confidence < threshold (default 0.5)
  - result.has_knowledge == False
  - Egennamn i frågan matchar ej graf-noder (token-miss)

Backends (fallback-ordning):
  1. Brave Search API (om BRAVE_API_KEY satt)
  2. DuckDuckGo HTML scrape (ingen nyckel)
  3. Direkt URL-fetch via web_text (om URL skickas)
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import urllib.parse
from dataclasses import dataclass, field

import httpx

_log = logging.getLogger("nouse.escalator")

# ── Config ────────────────────────────────────────────────────────────────────

ESCALATION_THRESHOLD   = float(os.getenv("NOUSE_ESCALATION_THRESHOLD", "0.5"))
ESCALATION_MAX_RESULTS = int(os.getenv("NOUSE_ESCALATION_MAX_RESULTS", "3"))
ESCALATION_TIMEOUT     = float(os.getenv("NOUSE_ESCALATION_TIMEOUT", "20.0"))
BRAVE_API_KEY          = os.getenv("BRAVE_API_KEY", "")


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class EscalationResult:
    query: str
    context_block: str          # formaterad för LLM-prompt
    sources: list[str]          # URL:er som användes
    escalated: bool             # False = graf räckte, True = web användes
    confidence_before: float    # graf-konfidens innan escalation
    snippets: list[str] = field(default_factory=list)
    learned: bool = False       # True om ny fakta skrevs till grafen


# ── Public API ────────────────────────────────────────────────────────────────

async def escalate_query(
    query: str,
    brain,                          # NouseBrain
    *,
    threshold: float = ESCALATION_THRESHOLD,
    learn: bool = True,             # skriv ny fakta till grafen
    max_results: int = ESCALATION_MAX_RESULTS,
) -> EscalationResult:
    """
    Huvud-entry-point. Används från brain.escalate() och run_repl.py.

    brain = nouse.attach()
    result = await escalate_query("vad är KuzuDB?", brain)
    print(result.context_block)   # injicera i LLM-prompt
    """
    graph_result = brain.query(query)
    conf = graph_result.confidence

    # Nivå 1: grafen räcker
    if conf >= threshold and graph_result.has_knowledge:
        return EscalationResult(
            query=query,
            context_block=graph_result.context_block(),
            sources=[],
            escalated=False,
            confidence_before=conf,
        )

    _log.info("Escalating '%s' (conf=%.2f < %.2f)", query[:60], conf, threshold)

    # Nivå 2: LLM bootstrap — seed graph from model weights if domain is unknown
    llm_learned = False
    if learn and not brain._read_only and not graph_result.has_knowledge:
        n = brain.domain_bootstrap(query)
        llm_learned = n > 0
        if llm_learned:
            graph_result = brain.query(query)
            if graph_result.confidence >= threshold and graph_result.has_knowledge:
                return EscalationResult(
                    query=query,
                    context_block=graph_result.context_block(),
                    sources=[],
                    escalated=True,
                    confidence_before=conf,
                    learned=True,
                )

    # Nivå 3: web-sök
    snippets, sources = await _web_search(query, max_results=max_results)
    web_block = _format_web_block(snippets, sources)

    graph_block = graph_result.context_block()
    combined = "\n\n".join(filter(None, [graph_block, web_block]))

    # Nivå 4: lär in ny fakta asynkront (non-blocking)
    learned = llm_learned
    if learn and snippets and not brain._read_only:
        try:
            brain.learn(query, "\n".join(snippets), source="escalator_web")
            learned = True
        except Exception as e:
            _log.warning("learn() failed during escalation: %s", e)

    return EscalationResult(
        query=query,
        context_block=combined,
        sources=sources,
        escalated=True,
        confidence_before=conf,
        snippets=snippets,
        learned=learned,
    )


async def _bootstrap_from_llm(query: str, brain) -> bool:  # kept for backward compat
    """
    Deprecated: use brain.domain_bootstrap() directly.
    Nivå 2: Seed graph from LLM weights when domain is unknown.

    Uses the model router attached to brain (if available) to ask the LLM
    about the query topic, then stores the response as hypothesis-tier
    knowledge via brain.learn().

    Returns True if knowledge was successfully seeded.
    """
    model_router = getattr(brain, "_model_router", None)
    if model_router is None:
        return False

    prompt = (
        f"Explain the concept or topic: '{query}'. "
        f"Describe key relations, subdomains, and connections to other concepts. "
        f"State facts as concrete relations: 'X is Y', 'X causes Z', 'X relates to Y'."
    )
    system = (
        "You are a knowledge distiller. Extract structured, factual knowledge. "
        "Be specific and concrete. Max 300 words."
    )

    try:
        response = await model_router.complete(prompt, system=system, max_tokens=400)
        if not response:
            return False
        brain.learn(prompt, response, source="escalator_llm_bootstrap")
        _log.info("LLM bootstrap: seeded knowledge for '%s'", query[:60])
        return True
    except Exception as e:
        _log.debug("LLM bootstrap failed for '%s': %s", query[:60], e)
        return False


# ── Web search backends ───────────────────────────────────────────────────────

async def _web_search(
    query: str,
    max_results: int = 3,
) -> tuple[list[str], list[str]]:
    """Returns (snippets, urls). Tries Brave → DDG → empty."""
    if BRAVE_API_KEY:
        try:
            return await _brave_search(query, max_results)
        except Exception as e:
            _log.debug("Brave search failed: %s", e)

    try:
        return await _ddg_search(query, max_results)
    except Exception as e:
        _log.debug("DDG search failed: %s", e)

    return [], []


async def _brave_search(
    query: str, max_results: int
) -> tuple[list[str], list[str]]:
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": BRAVE_API_KEY,
    }
    params = {"q": query, "count": max_results, "text_decorations": False}

    async with httpx.AsyncClient(timeout=ESCALATION_TIMEOUT) as client:
        r = await client.get(url, headers=headers, params=params)
        r.raise_for_status()
        data = r.json()

    snippets, urls = [], []
    for item in data.get("web", {}).get("results", [])[:max_results]:
        title = item.get("title", "")
        desc  = item.get("description", "")
        href  = item.get("url", "")
        if desc:
            snippets.append(f"{title}: {desc}")
            urls.append(href)

    return snippets, urls


async def _ddg_search(
    query: str, max_results: int
) -> tuple[list[str], list[str]]:
    """DuckDuckGo HTML scrape — no API key needed."""
    from bs4 import BeautifulSoup

    encoded = urllib.parse.quote_plus(query)
    url = f"https://html.duckduckgo.com/html/?q={encoded}"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; Nouse/0.2)"}

    async with httpx.AsyncClient(
        timeout=ESCALATION_TIMEOUT,
        follow_redirects=True,
        headers=headers,
    ) as client:
        r = await client.get(url)
        r.raise_for_status()

    soup = BeautifulSoup(r.text, "lxml")
    snippets, urls = [], []

    for result in soup.select(".result")[:max_results * 2]:
        title_el   = result.select_one(".result__title")
        snippet_el = result.select_one(".result__snippet")
        url_el     = result.select_one(".result__url")

        title   = title_el.get_text(strip=True)   if title_el   else ""
        snippet = snippet_el.get_text(strip=True) if snippet_el else ""
        href    = url_el.get_text(strip=True)      if url_el     else ""

        if snippet:
            snippets.append(f"{title}: {snippet}" if title else snippet)
            urls.append(href)

        if len(snippets) >= max_results:
            break

    return snippets, urls


# ── Formatting ────────────────────────────────────────────────────────────────

def _format_web_block(snippets: list[str], sources: list[str]) -> str:
    if not snippets:
        return ""
    parts = ["[web search]"]
    for i, (s, u) in enumerate(zip(snippets, sources), 1):
        parts.append(f"  {i}. {s[:300]}")
        if u:
            parts.append(f"     källa: {u}")
    return "\n".join(parts)


# ── Sync wrapper ──────────────────────────────────────────────────────────────

def escalate_query_sync(
    query: str,
    brain,
    *,
    threshold: float = ESCALATION_THRESHOLD,
    learn: bool = True,
) -> EscalationResult:
    """Synchronous wrapper for use outside async context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                return ex.submit(
                    asyncio.run,
                    escalate_query(query, brain, threshold=threshold, learn=learn)
                ).result(timeout=60)
        return loop.run_until_complete(
            escalate_query(query, brain, threshold=threshold, learn=learn)
        )
    except Exception as e:
        _log.warning("escalate_query_sync failed: %s", e)
        result = brain.query(query)
        return EscalationResult(
            query=query,
            context_block=result.context_block(),
            sources=[],
            escalated=False,
            confidence_before=result.confidence,
        )
