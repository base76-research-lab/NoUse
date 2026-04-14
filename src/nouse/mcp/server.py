"""
nouse-mcp — MCP server exposing Nous as a cognitive tool for any LLM agent.

Exposes:
  nouse_query        — query the knowledge graph, returns structured context
  nouse_recall       — get axioms for a specific concept
  nouse_learn        — extract and store relations from a prompt/response pair
  nouse_add          — directly add a typed relation to the graph
  nouse_status       — graph statistics
"""
from __future__ import annotations

import json
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="nouse",
    instructions=(
        "Persistent domain memory for LLMs. "
        "Query, learn from, and add knowledge to the Nous knowledge graph."
    ),
)

_brain = None


def _get_brain():
    global _brain
    if _brain is None:
        import nouse
        _brain = nouse.attach()
    return _brain


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def nouse_query(question: str, top_k: int = 6) -> str:
    """
    Query the Nous knowledge graph with a question or concept.

    Returns a structured context block with relevant concepts, typed relations,
    and confidence scores — ready to inject into any LLM system prompt.

    Use this before answering any domain-specific question to retrieve
    relevant memory context.

    Args:
        question: The question or concept to look up.
        top_k: Number of top nodes to retrieve (default 6).
    """
    brain = _get_brain()
    result = brain.query(question, top_k=top_k)
    if not result.has_knowledge:
        return f"[Nous] No knowledge found for: {question!r}"
    block = result.context_block()
    meta = (
        f"\n[confidence={result.confidence:.2f}, "
        f"axioms={len(result.axioms)}, "
        f"domains={', '.join(result.domains) or 'unknown'}]"
    )
    return block + meta


@mcp.tool()
def nouse_recall(concept: str, top_k: int = 10) -> str:
    """
    Retrieve all known axioms (typed relations) for a specific concept.

    Returns a list of validated and uncertain relations the graph holds
    about this concept, with evidence scores.

    Args:
        concept: The concept name to look up.
        top_k: Max number of axioms to return (default 10).
    """
    brain = _get_brain()
    axioms = brain.recall_axioms(concept, top_k=top_k)
    if not axioms:
        return f"[Nous] No axioms found for concept: {concept!r}"
    lines = [f"[Nous axioms for '{concept}']"]
    for a in axioms:
        lines.append(f"  {a.as_text()}")
    return "\n".join(lines)


@mcp.tool()
def nouse_learn(prompt: str, response: str, source: str = "openclaw") -> str:
    """
    Extract and store knowledge from a prompt/response conversation pair.

    Nous will parse the exchange, extract typed relations between concepts,
    and add them to the knowledge graph with Hebbian weighting.

    Call this after every significant exchange to grow the graph continuously.

    Args:
        prompt: The user prompt or question.
        response: The LLM's response.
        source: Tag indicating the source (default: 'openclaw').
    """
    brain = _get_brain()
    try:
        brain.learn(prompt, response, source=source)
        return f"[Nous] Learning complete. Source tagged as '{source}'."
    except Exception as e:
        return f"[Nous] Learn failed: {e}"


@mcp.tool()
def nouse_add(
    src: str,
    rel_type: str,
    tgt: str,
    why: str = "",
    evidence_score: float = 0.7,
) -> str:
    """
    Directly add a typed relation to the Nous knowledge graph.

    Use this to record a specific insight, decision, or fact that you want
    to persist in long-term memory.

    Args:
        src: Source concept (e.g. 'transformer').
        rel_type: Relation type (e.g. 'USES', 'EXTENDS', 'CONTRADICTS', 'DEPENDS_ON').
        tgt: Target concept (e.g. 'attention mechanism').
        why: Optional provenance note (e.g. 'user confirmed in session').
        evidence_score: Confidence 0.0–1.0 (default 0.7).

    Example:
        nouse_add("SQLite WAL", "USED_BY", "Nous", why="current graph persistence layer")
    """
    brain = _get_brain()
    try:
        brain.add(src, rel_type, tgt, why=why, evidence_score=evidence_score)
        return (
            f"[Nous] Added: {src} —[{rel_type}]→ {tgt}  "
            f"[ev={evidence_score:.2f}]"
            + (f"  why='{why}'" if why else "")
        )
    except Exception as e:
        return f"[Nous] Add failed: {e}"


@mcp.tool()
def nouse_status() -> str:
    """
    Return statistics about the current state of the Nous knowledge graph.

    Shows node count, relation count, and top domains.
    """
    brain = _get_brain()
    try:
        stats = brain.stats()
        return json.dumps(stats, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"[Nous] Status failed: {e}"


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
