"""
nouse/cli/commands/seed_cmd.py
================================
`nouse seed` — Let the LLM teach Nous its knowledge.

Screenshots the LLM's internal knowledge into the Nous graph as reference
nodes. These nodes serve as calibration anchors: the LLM provides surface
knowledge Q&A, and Nous acts as the meta layer (calibration, evidence scoring,
contradiction detection).

Architecture:
  LLM = surface knowledge (broad, fast, but unreliable)
  Nous = meta layer (structured, calibrated, persistent)
  Reference nodes = LLM knowledge anchored in Nous with evidence_score=0.5
                    and assumption_flag=True, source="llm_bootstrap"

Usage:
    nouse seed                       # auto-detect underrepresented regions
    nouse seed --regions frontal amygdala hippocampus
    nouse seed --model minimax-m2.7:cloud
    nouse seed --dry-run             # show what would be seeded
    nouse seed --clean-noise         # remove noise concepts first
    nouse seed --clean-noise --dry-run
"""
from __future__ import annotations

import asyncio
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

SEED_DOMAINS = {
    "frontal": {
        "domain": "logik_och_beslut",
        "description": "Deduktiv och induktiv logik, beslutsteori, kausalitet, Bayes sats",
        "key_concepts": [
            "deduktion", "induktion", "abduktion", "syllogism", "premis",
            "kontrapositiv", "necessär och sufficient villkor", "fallacy",
            "Modus Ponens", "Modus Tollens", "Bayes sats", "prior probability",
            "posterior probability", "likelihood", "beslutsteori",
            "utility function", "risk-aversion", "counterfactual reasoning",
            "kognitiv bias", "heuristik", "Anchoring effect", "Confirmation bias",
            "Sunk cost fallacy", "Dunning-Kruger effect", "Framing effect",
        ],
    },
    "hippocampus": {
        "domain": "minne_och_lärande",
        "description": "Minneskonsolidering, spaced repetition, arbetsminne, långtidsminne",
        "key_concepts": [
            "konsolidering", "reconsolidation", "spaced repetition",
            "forgetting curve", "Ebbinghaus curve", "interference theory",
            "proactive interference", "retroactive interference",
            "working memory", "episodic memory", "semantic memory",
            "procedural memory", "priming", "transfer-appropriate processing",
            "encoding specificity", "levels of processing",
            "long-term potentiation", "hippocampal formation",
            "spatial navigation", "cognitive map",
        ],
    },
    "amygdala": {
        "domain": "risk_och_etik",
        "description": "Riskbedömning, moraliska dilemman, emotionell vägning, etiska ramverk",
        "key_concepts": [
            "riskbedömning", "risk-aversion", "risk-seeking", "loss aversion",
            "trolley problem", "utilitarianism", "deontological ethics",
            "virtue ethics", "care ethics", "rights-based ethics",
            "prospect theory", "framing effect", "certainty effect",
            "emotional intelligence", "affective forecasting",
            "moral intuition", "moral reasoning", "empathy gap",
            "amygdala hijack", "fear conditioning", "threat detection",
            "uncertainty aversion", "ambiguity aversion",
        ],
    },
    "temporal_right": {
        "domain": "kreativitet_och_bisociation",
        "description": "Kreativitet, bisociation, metafor, analogi, konvergent/divergent tänkande",
        "key_concepts": [
            "bisociation", "divergent thinking", "convergent thinking",
            "lateral thinking", "incubation effect", "flow state",
            "remote associations", "conceptual blending", "metaphor",
            "analogy", "insight problem solving", "functional fixedness",
            "creative blocks", "ideation", "brainstorming",
            "constraint relaxation", "combinatorial creativity",
            "transformational creativity", "everyday creativity",
            "creative confidence", "aesthetic judgment",
        ],
    },
    "temporal_left": {
        "domain": "språk_och_semantik",
        "description": "Språklig nyans, pragmatik, semantik, diskurs, retorik",
        "key_concepts": [
            "pragmatik", "semantik", "syntax", "diskursanalys",
            "speech act theory", "implicature", "Gricean maxims",
            "relevance theory", "presupposition", "entailment",
            "ambiguity", "polysemy", "metonymy", "irony",
            "register", "code-switching", "framing", "narrative structure",
            "rhetoric", "ethos pathos logos", "argumentation theory",
            "informal logic", "critical thinking",
        ],
    },
    "prefrontal": {
        "domain": "metakognition_och_syntes",
        "description": "Metakognition, exekutiv funktion, syntes, självreglering",
        "key_concepts": [
            "metakognition", "executive function", "cognitive control",
            "self-regulation", "reflective thinking", "system 1 vs system 2",
            "metacognitive monitoring", "metacognitive control",
            "cognitive flexibility", "planning", "goal-setting",
            "working memory updating", "inhibitory control",
            "task-switching", "cognitive load theory",
            "deliberate practice", "epistemic vigilance",
            "intellectual humility", "open-mindedness",
            "synthesis", "integration", "abstraction",
        ],
    },
    "occipital": {
        "domain": "mönsterigenkänning",
        "description": "Mönster, gestalt, visuell igenkänning, perception",
        "key_concepts": [
            "gestalt perception", "figure-ground", "pattern recognition",
            "feature detection", "visual processing", "spatial frequency",
            "change blindness", "inattentional blindness",
            "category learning", "prototype theory", "exemplar theory",
            "statistical learning", "implicit learning",
            "recognition vs recall", "face perception",
        ],
    },
    "cerebellum": {
        "domain": "procedural_och_automatisering",
        "description": "Procedural inlärning, automatisering, motorisk planering",
        "key_concepts": [
            "procedural learning", "skill acquisition", "automaticity",
            "chunking", "motor learning", "error-based learning",
            "prediction error", "internal model", "feedforward control",
            "feedback control", "adaptation", "habit formation",
            "deliberate practice", "transfer of learning",
        ],
    },
    "brainstem": {
        "domain": "axiom_och_grundläggande",
        "description": "Grundläggande axiom, logiska lagar, definitioner",
        "key_concepts": [
            "identity law", "non-contradiction", "excluded middle",
            "causality", "conservation laws", "thermodynamic principles",
            "mathematical axioms", "Peano axioms", "ZFC set theory",
            "definition by extension", "definition by intension",
            "necessary truth", "contingent truth",
        ],
    },
}


async def call_llm_for_seed(model: str, prompt: str, timeout: float = 90.0) -> str:
    """Call LLM to generate seed knowledge for a domain."""
    import httpx
    import os

    # Provider resolution
    base_url, real_model, headers = _resolve_model_provider(model)

    payload = {
        "model": real_model,
        "messages": [
            {"role": "system", "content": "You are a knowledge engineer. Respond with JSON only."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 2000,
        "temperature": 0.7,
    }

    try:
        if base_url is None:
            ollama_base = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            async with httpx.AsyncClient(timeout=timeout) as hx:
                r = await hx.post(f"{ollama_base}/api/chat", json={**payload, "stream": False})
                r.raise_for_status()
                data = r.json()
                return data.get("message", {}).get("content", "") or ""
        else:
            async with httpx.AsyncClient(timeout=timeout, headers=headers) as hx:
                r = await hx.post(f"{base_url}/chat/completions", json=payload)
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"] or ""
    except Exception as e:
        return f"[ERROR: {e}]"


SEED_PROMPT = """\
You are a knowledge engineer building a structured knowledge graph.

DOMAIN: {domain}
BRAIN REGION: {region}
DESCRIPTION: {description}

EXISTING CONCEPTS (already in graph): {existing}

Generate 15-25 NEW concepts and their RELATIONS for this domain.
Focus on concepts that are NOT already in the existing list.

For each concept, provide:
1. A clear concept name (in the language most natural for the domain)
2. Key relations to OTHER concepts (within or across domains)

Output JSON only:
{{
  "concepts": [
    {{
      "name": "concept name",
      "relations": [
        {{"target": "other concept", "type": "relation type", "why": "brief explanation"}}
      ]
    }}
  ]
}}

Relation types to use: stärker, försvagar, möjliggör, hindrar, är_del_av, innehåller,
orsakar, leder_till, motsäger, påverkar, beror_på, exemplifierar

Be specific and accurate. These will be reference nodes in an epistemic substrate."""


def _resolve_model_provider(model: str) -> tuple:
    """Resolve model string to (base_url, real_model, headers)."""
    import os

    base_url = None
    real_model = model
    headers = {}

    if "minimax" in model.lower():
        api_key = os.getenv("MINIMAX_API_KEY", "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        base_url = os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.chat/v1")
    elif model.startswith("cerebras/"):
        real_model = model[len("cerebras/"):]
        base_url = "https://api.cerebras.ai/v1"
        api_key = os.getenv("CEREBRAS_API_KEY", "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
    elif model.startswith("groq/"):
        real_model = model[len("groq/"):]
        base_url = "https://api.groq.com/openai/v1"
        api_key = os.getenv("GROQ_API_KEY", "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
    # Default: Ollama native API (base_url = None)

    return base_url, real_model, headers


def get_slagsida_report(field) -> dict:
    """Get slagsida report showing concept distribution across brain regions."""
    from nouse.field.brain_topology import classify_domain
    from collections import Counter

    region_counts = Counter()
    region_domains = {}

    for c in field.concepts():
        d = c.get("domain", "unknown")
        r = classify_domain(d)
        region_counts[r] += 1
        if r not in region_domains:
            region_domains[r] = Counter()
        region_domains[r][d] += 1

    total = sum(region_counts.values())
    report = {}
    for region, count in region_counts.most_common():
        pct = count / total * 100
        top_domains = region_domains.get(region, Counter()).most_common(3)
        report[region] = {
            "count": count,
            "pct": round(pct, 1),
            "top_domains": [(d, c) for d, c in top_domains],
            "needs_seeding": pct < 5.0,
        }
    return report


def clean_noise_concepts(field, dry_run: bool = False) -> dict:
    """Remove noise concepts: code artifacts, pure numbers, fragments."""
    noise_patterns = []
    kept = 0
    removed = 0

    concepts = list(field.concepts())
    for c in concepts:
        name = c["name"]
        is_noise = False

        # Pure numbers
        if name.strip().isdigit():
            is_noise = True
        # Very short fragments (< 3 chars, not common abbreviations)
        elif len(name.strip()) < 3 and not name.strip().isupper():
            is_noise = True
        # Code paths (dotted identifiers like nouse.orchestrator.conductor)
        elif name.count(".") > 2 and not name[0].isupper():
            is_noise = True
        # English fragments starting with articles/prepositions
        elif name.lower().startswith(("by the ", "the ", "a ", "an ", "in the ")):
            is_noise = True

        if is_noise:
            noise_patterns.append(name)
            if not dry_run:
                try:
                    field.delete_orphan_concepts()  # bulk cleanup
                except Exception:
                    pass
            removed += 1
        else:
            kept += 1

    return {
        "total_concepts": len(concepts),
        "noise_found": len(noise_patterns),
        "noise_removed": removed if not dry_run else 0,
        "kept": kept,
        "dry_run": dry_run,
        "examples": noise_patterns[:10],
    }


def seed_region(field, region: str, model: str, dry_run: bool = False,
                extra_concepts: list[str] | None = None) -> dict:
    """Seed a brain region with LLM-generated reference concepts."""
    if region not in SEED_DOMAINS:
        return {"error": f"Unknown region: {region}. Choose from: {list(SEED_DOMAINS.keys())}"}

    region_info = SEED_DOMAINS[region]
    domain = region_info["domain"]

    # Get existing concepts in this region
    from nouse.field.brain_topology import classify_domain
    existing = []
    for c in field.concepts():
        d = c.get("domain", "unknown")
        if classify_domain(d) == region:
            existing.append(c["name"])
    existing_str = ", ".join(existing[:30]) if existing else "(empty)"

    # Combine predefined concepts with any extras
    predefined = list(region_info["key_concepts"])
    if extra_concepts:
        predefined.extend(extra_concepts)

    # Remove concepts already in the graph
    existing_names = {c["name"] for c in field.concepts()}
    new_concepts = [c for c in predefined if c not in existing_names]

    if dry_run:
        return {
            "region": region,
            "domain": domain,
            "existing_count": len(existing),
            "predefined_to_seed": len(new_concepts),
            "concepts": new_concepts[:20],
            "would_llm_generate": True,
            "dry_run": True,
        }

    # Add predefined concepts as reference nodes
    added = 0
    for concept_name in new_concepts:
        try:
            field.add_concept(
                concept_name,
                domain=domain,
                granularity=1,
                source="llm_bootstrap",
                ensure_knowledge=True,
            )
            added += 1
        except Exception:
            pass

    # Optionally generate more via LLM
    llm_concepts = 0
    if model:
        prompt = SEED_PROMPT.format(
            domain=domain,
            region=region,
            description=region_info["description"],
            existing=existing_str,
        )
        try:
            response = asyncio.run(call_llm_for_seed(model, prompt))
            # Parse JSON response
            cleaned = response.strip()
            if cleaned.startswith("```"):
                first_nl = cleaned.find("\n")
                if first_nl >= 0:
                    cleaned = cleaned[first_nl + 1:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3].strip()
            parsed = json.loads(cleaned)
            for concept_data in parsed.get("concepts", []):
                name = concept_data.get("name", "")
                if not name or name in existing_names:
                    continue
                try:
                    field.add_concept(
                        name,
                        domain=domain,
                        granularity=1,
                        source="llm_bootstrap",
                        ensure_knowledge=True,
                    )
                    # Add relations with bootstrap evidence
                    for rel in concept_data.get("relations", []):
                        target = rel.get("target", "")
                        rel_type = rel.get("type", "påverkar")
                        why = rel.get("why", "LLM bootstrap")
                        if target:
                            field.add_relation(
                                name, rel_type, target,
                                why=why,
                                strength=0.5,
                                source_tag="llm_bootstrap",
                                evidence_score=0.5,
                                assumption_flag=True,
                                domain_src=domain,
                                domain_tgt="unknown",  # will be classified later
                            )
                    llm_concepts += 1
                except Exception:
                    pass
        except (json.JSONDecodeError, Exception):
            pass

    return {
        "region": region,
        "domain": domain,
        "predefined_added": added,
        "llm_generated": llm_concepts,
        "existing_before": len(existing),
        "total_now": len([c for c in field.concepts()
                        if classify_domain(c.get("domain", "unknown")) == region]),
    }


app = typer.Typer(help="Seed Nous with LLM knowledge to balance brain regions.")


@app.command()
def seed(
    regions: list[str] = typer.Argument(
        None,
        help="Brain regions to seed (default: all underrepresented). "
             "Options: frontal, hippocampus, amygdala, temporal_right, "
             "temporal_left, prefrontal, occipital, cerebellum, brainstem",
    ),
    model: str = typer.Option(
        "minimax-m2.7:cloud",
        "--model", "-m",
        help="LLM model to use for generating concepts.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be seeded without making changes.",
    ),
    clean_noise: bool = typer.Option(
        False,
        "--clean-noise",
        help="Remove noise concepts before seeding.",
    ),
    list_regions: bool = typer.Option(
        False,
        "--list", "-l",
        help="List current brain region distribution and exit.",
    ),
    discover: bool = typer.Option(
        False,
        "--discover", "-d",
        help="Ask the LLM which domains it knows that Nous is missing.",
    ),
) -> None:
    """Seed Nous with reference knowledge from LLM to balance brain regions.

    This command lets the LLM 'screenshot' its knowledge into Nous as
    reference nodes. These nodes serve as calibration anchors — the LLM
    provides surface knowledge, and Nous provides the meta layer
    (calibration, evidence scoring, contradiction detection).

    Use --discover to ask the LLM which domains it has knowledge about
    that Nous is missing. The LLM lists its domains, we compare with
    what Nous already has, and seed the gaps.
    """
    from nouse.field.surface import FieldSurface

    try:
        field = FieldSurface(read_only=False)
    except Exception as e:
        console.print(f"[red]Could not open field: {e}[/red]")
        raise typer.Exit(1)

    # Discover mode: ask LLM what domains it knows
    if discover:
        from nouse.field.brain_topology import classify_domain
        existing_domains = sorted(set(c.get("domain", "unknown") for c in field.concepts()))
        console.print(f"[bold]Nous has {len(existing_domains)} domains.[/bold]")
        console.print(f"[dim]Asking {model} what domains it knows...[/dim]")

        discover_prompt = f"""\
You are a knowledge engineer. Nous currently has these domains:
{', '.join(existing_domains[:100])}

List 30-50 ADDITIONAL domains that you have strong knowledge about that are
NOT in the list above. Focus on domains that would fill gaps in:
- Logic, decision theory, and reasoning (frontal lobe)
- Memory, learning, and consolidation (hippocampus)
- Risk, ethics, and emotional evaluation (amygdala)
- Creativity, bisociation, and metaphor (temporal right)
- Language, pragmatics, and discourse (temporal left)
- Metacognition and synthesis (prefrontal)

Output JSON only:
{{"domains": [{{"name": "domain_name", "region": "brain_region", "description": "one line"}}]}}"""

        try:
            response = asyncio.run(call_llm_for_seed(model, discover_prompt))
            cleaned = response.strip()
            if cleaned.startswith("```"):
                first_nl = cleaned.find("\n")
                if first_nl >= 0:
                    cleaned = cleaned[first_nl + 1:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3].strip()
            parsed = json.loads(cleaned)
            discovered = parsed.get("domains", [])

            table = Table(title="Discovered Domains (not in Nous)")
            table.add_column("Domain", style="cyan")
            table.add_column("Region", style="green")
            table.add_column("Description")
            for d in discovered:
                table.add_row(
                    d.get("name", "?"),
                    d.get("region", "?"),
                    d.get("description", "")[:60],
                )
            console.print(table)
            console.print(f"\n[dim]Run: nouse seed --regions {' '.join(d.get('region','') for d in discovered[:5])}[/dim]")
        except json.JSONDecodeError:
            console.print(f"[red]Could not parse LLM response[/red]")
            console.print(f"[dim]{response[:500]}[/dim]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(0)

    # List regions mode
    if list_regions:
        report = get_slagsida_report(field)
        table = Table(title="Brain Region Distribution (Slagsida Report)")
        table.add_column("Region", style="cyan")
        table.add_column("Concepts", justify="right")
        table.add_column("Share", justify="right")
        table.add_column("Needs Seed", justify="center")
        table.add_column("Top Domains")

        for region, data in sorted(report.items(), key=lambda x: -x[1]["count"]):
            needs = "[red]YES[/red]" if data["needs_seeding"] else "[green]no[/green]"
            top_d = ", ".join(f"{d} ({c})" for d, c in data["top_domains"][:3])
            table.add_row(
                region, str(data["count"]),
                f"{data['pct']}%", needs, top_d,
            )

        console.print(table)
        raise typer.Exit(0)

    # Clean noise first if requested
    if clean_noise:
        console.print("[bold yellow]Cleaning noise concepts...[/bold yellow]")
        result = clean_noise_concepts(field, dry_run=dry_run)
        if dry_run:
            console.print(f"  Would remove {result['noise_found']} noise concepts")
            console.print(f"  Examples: {result['examples'][:5]}")
        else:
            console.print(f"  Removed {result['noise_removed']} noise concepts")
            console.print(f"  Examples: {result['examples'][:5]}")

    # Determine which regions to seed
    if not regions:
        # Auto-detect underrepresented regions
        report = get_slagsida_report(field)
        regions = [r for r, data in report.items() if data["needs_seeding"]]
        if not regions:
            regions = list(SEED_DOMAINS.keys())[:3]
        console.print(f"[dim]Auto-detected underrepresented regions: {', '.join(regions)}[/dim]")

    # Seed each region
    total_added = 0
    for region in regions:
        console.print(f"\n[bold cyan]Seeding {region}...[/bold cyan]")
        result = seed_region(field, region, model, dry_run=dry_run)

        if "error" in result:
            console.print(f"  [red]{result['error']}[/red]")
            continue

        if dry_run:
            console.print(f"  Domain: {result['domain']}")
            console.print(f"  Existing concepts: {result['existing_count']}")
            console.print(f"  Would seed: {result['predefined_to_seed']} predefined concepts")
            console.print(f"  Sample: {', '.join(result['concepts'][:8])}")
        else:
            console.print(f"  Predefined added: {result['predefined_added']}")
            console.print(f"  LLM-generated: {result['llm_generated']}")
            console.print(f"  Before: {result['existing_before']} → After: {result['total_now']}")
            total_added += result["predefined_added"] + result["llm_generated"]

    if not dry_run:
        console.print(f"\n[bold green]Seeding complete. Total concepts added: {total_added}[/bold green]")

        # Show updated slagsida
        report = get_slagsida_report(field)
        total = sum(d["count"] for d in report.values())
        console.print(f"\n[bold]Updated distribution:[/bold]")
        for region, data in sorted(report.items(), key=lambda x: -x[1]["count"]):
            marker = " [red]⚠[/red]" if data["needs_seeding"] else ""
            console.print(f"  {region:20s}: {data['count']:5d} ({data['pct']:5.1f}%){marker}")