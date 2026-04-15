"""
nouse.daemon.percolation — Topology-aware domain density monitoring
====================================================================

Diagnoses and addresses the percolation problem in autonomous knowledge graphs:
when total concept count is high but cross-domain density is below the critical
threshold, bisociation cannot emerge regardless of volume.

This module provides:

1. `domain_density_report()` — classify all domains as absent/thin/marginal/dense
2. `percolation_threshold()` — compute the critical cross-domain connection count
3. `cross_domain_edge_stats()` — measure actual vs needed cross-domain connections
4. `identify_bridge_domains()` — find strategic domains that maximise connectivity
5. `generate_ingestion_tasks()` — create targeted research tasks for thin/bridge domains

The percolation threshold is derived from random graph theory (Erdős–Rényi):
a giant connected component emerges when edge probability exceeds 1/n, which
translates to each domain needing ~ln(n) cross-domain connections on average.

For scale-free knowledge graphs (which Nous is), this is a conservative lower
bound — the actual threshold may be lower due to hub effects, but ln(n) provides
a safe target.

Design influenced by the observation that 32K concepts across 4357 domains
produced zero bisociation candidates, because domains enabling structural
overlap (kreativitetsteori=2, nätverksteori=2, kognitiv vetenskap=4) were
precisely the ones with negligible coverage.
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any

from nouse.field.surface import FieldSurface

log = logging.getLogger("nouse.percolation")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Minimum concepts per domain to be considered "thin" (below percolation)
MIN_DOMAIN_CONCEPTS = int(os.getenv("NOUSE_PERC_MIN_CONCEPTS", "8"))

#: Minimum concepts per domain to be considered "marginal" (approaching threshold)
MARGINAL_DOMAIN_CONCEPTS = int(os.getenv("NOUSE_PERC_MARGINAL_CONCEPTS", "15"))

#: Multiplier for the percolation threshold (ln(n) * factor)
PERC_THRESHOLD_FACTOR = float(os.getenv("NOUSE_PERC_THRESHOLD_FACTOR", "1.0"))

#: Maximum number of ingestion tasks to generate per cycle
MAX_INGESTION_TASKS = int(os.getenv("NOUSE_PERC_MAX_TASKS", "5"))

#: Maximum number of bridge domains to identify
MAX_BRIDGE_DOMAINS = int(os.getenv("NOUSE_PERC_MAX_BRIDGES", "10"))

#: Minimum concepts a domain needs before we consider it as a potential bridge
BRIDGE_DOMAIN_MIN_CONCEPTS = int(os.getenv("NOUSE_PERC_BRIDGE_MIN", "3"))

#: Priority boost for bridge domains vs thin domains
BRIDGE_PRIORITY_BOOST = float(os.getenv("NOUSE_PERC_BRIDGE_BOOST", "0.15"))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DomainProfile:
    """Density profile for a single domain."""
    name: str
    concept_count: int
    cross_domain_edges_in: int   # edges from other domains INTO this domain
    cross_domain_edges_out: int   # edges from this domain INTO other domains
    internal_edges: int           # edges within this domain
    connected_domains: set[str]  # unique domains this domain connects to
    density_class: str            # "absent" | "thin" | "marginal" | "dense"

    @property
    def cross_domain_edges_total(self) -> int:
        return self.cross_domain_edges_in + self.cross_domain_edges_out

    @property
    def connectivity_ratio(self) -> float:
        """Ratio of actual cross-domain connections to threshold."""
        if self.concept_count == 0:
            return 0.0
        return self.cross_domain_edges_total / max(1, self.concept_count)


@dataclass
class PercolationReport:
    """Full percolation status report for the knowledge graph."""
    total_domains: int
    total_concepts: int
    total_relations: int
    threshold_ln_n: float  # ln(n) percolation threshold
    threshold_target: float  # threshold * factor
    avg_concepts_per_domain: float
    avg_cross_domain_edges: float
    domains_absent: int    # 0 concepts
    domains_thin: int      # 1–MIN_DOMAIN_CONCEPTS
    domains_marginal: int  # MIN_DOMAIN_CONCEPTS–MARGINAL_DOMAIN_CONCEPTS
    domains_dense: int     # > MARGINAL_DOMAIN_CONCEPTS
    single_concept_domains: int
    cross_domain_edge_total: int
    cross_domain_edge_deficit: int  # how many more edges needed
    bisociation_readiness: float     # 0–1, how close to percolation
    domain_profiles: list[DomainProfile] = field(default_factory=list)
    bridge_domains: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------

def percolation_threshold(n_domains: int) -> float:
    """
    Compute the percolation threshold for cross-domain connections.

    Based on Erdős–Rényi random graph theory: the giant connected component
    emerges when edge probability exceeds 1/n, translating to each domain
    needing ~ln(n) cross-domain connections on average.

    For scale-free graphs (which knowledge graphs are), this is a conservative
    lower bound — hub domains lower the effective threshold.

    Returns the minimum average cross-domain connections per domain.
    """
    if n_domains < 2:
        return 0.0
    return math.log(n_domains) * PERC_THRESHOLD_FACTOR


def domain_density_report(field: FieldSurface) -> PercolationReport:
    """
    Generate a complete percolation analysis of the knowledge graph.

    Scans all domains, classifies them by density, measures cross-domain
    connectivity, and computes how far the graph is from the percolation
    threshold where bisociation can emerge.
    """
    all_domains = field.domains()
    total_domains = len(all_domains)
    stats = field.stats()
    total_concepts = stats.get("concepts", 0)
    total_relations = stats.get("relations", 0)

    threshold = percolation_threshold(total_domains)

    # Build per-domain profiles
    profiles: list[DomainProfile] = []
    cross_domain_edge_total = 0
    domains_absent = 0
    domains_thin = 0
    domains_marginal = 0
    domains_dense = 0
    single_concept_domains = 0

    # Collect all relations once for cross-domain analysis
    all_relations = field.query_all_relations_with_metadata(include_evidence=False)

    # Build cross-domain edge map: (domain_a, domain_b) -> count
    cross_domain_edges: dict[tuple[str, str], int] = {}

    for r in all_relations:
        src = r.get("src", "")
        tgt = r.get("tgt", "")
        src_domain = field.concept_domain(src) or r.get("src_domain", "")
        tgt_domain = field.concept_domain(tgt) or ""

        if src_domain and tgt_domain and src_domain != tgt_domain:
            key = tuple(sorted([src_domain, tgt_domain]))
            cross_domain_edges[key] = cross_domain_edges.get(key, 0) + 1

    # Domain concept counts
    domain_concept_counts: dict[str, int] = {}
    for d in all_domains:
        concepts = field.concepts(domain=d)
        domain_concept_counts[d] = len(concepts)

    # Build domain connectivity map
    domain_connected_domains: dict[str, set[str]] = {d: set() for d in all_domains}
    domain_cross_edges_in: dict[str, int] = {d: 0 for d in all_domains}
    domain_cross_edges_out: dict[str, int] = {d: 0 for d in all_domains}
    domain_internal_edges: dict[str, int] = {d: 0 for d in all_domains}

    for r in all_relations:
        src = r.get("src", "")
        tgt = r.get("tgt", "")
        src_domain = field.concept_domain(src) or r.get("src_domain", "")
        tgt_domain = field.concept_domain(tgt) or ""

        if src_domain == tgt_domain and src_domain:
            domain_internal_edges[src_domain] = domain_internal_edges.get(src_domain, 0) + 1
        elif src_domain and tgt_domain:
            domain_cross_edges_out[src_domain] = domain_cross_edges_out.get(src_domain, 0) + 1
            domain_cross_edges_in[tgt_domain] = domain_cross_edges_in.get(tgt_domain, 0) + 1
            domain_connected_domains[src_domain].add(tgt_domain)
            domain_connected_domains.setdefault(tgt_domain, set()).add(src_domain)

    # Build profiles
    for d in all_domains:
        n_concepts = domain_concept_counts.get(d, 0)
        if n_concepts == 0:
            density_class = "absent"
            domains_absent += 1
        elif n_concepts < MIN_DOMAIN_CONCEPTS:
            density_class = "thin"
            domains_thin += 1
        elif n_concepts < MARGINAL_DOMAIN_CONCEPTS:
            density_class = "marginal"
            domains_marginal += 1
        else:
            density_class = "dense"
            domains_dense += 1

        if n_concepts == 1:
            single_concept_domains += 1

        profile = DomainProfile(
            name=d,
            concept_count=n_concepts,
            cross_domain_edges_in=domain_cross_edges_in.get(d, 0),
            cross_domain_edges_out=domain_cross_edges_out.get(d, 0),
            internal_edges=domain_internal_edges.get(d, 0),
            connected_domains=domain_connected_domains.get(d, set()),
            density_class=density_class,
        )
        profiles.append(profile)

    # Sort by concept count ascending (thinnest first)
    profiles.sort(key=lambda p: p.concept_count)

    # Aggregate cross-domain edges
    for count in cross_domain_edges.values():
        cross_domain_edge_total += count

    # Compute average cross-domain edges per domain
    n_with_edges = max(1, sum(1 for p in profiles if p.cross_domain_edges_total > 0))
    avg_cross_domain_edges = cross_domain_edge_total / max(1, total_domains)

    # Compute deficit: how many more cross-domain edges needed
    target_edges_per_domain = threshold
    total_target_edges = target_edges_per_domain * total_domains
    cross_domain_edge_deficit = max(0, int(total_target_edges - cross_domain_edge_total))

    # Bisociation readiness: ratio of actual cross-domain edges to threshold
    # Normalized to 0-1 range
    if total_target_edges > 0:
        bisociation_readiness = min(1.0, cross_domain_edge_total / total_target_edges)
    else:
        bisociation_readiness = 1.0

    return PercolationReport(
        total_domains=total_domains,
        total_concepts=total_concepts,
        total_relations=total_relations,
        threshold_ln_n=math.log(total_domains) if total_domains > 1 else 0.0,
        threshold_target=threshold,
        avg_concepts_per_domain=total_concepts / max(1, total_domains),
        avg_cross_domain_edges=avg_cross_domain_edges,
        domains_absent=domains_absent,
        domains_thin=domains_thin,
        domains_marginal=domains_marginal,
        domains_dense=domains_dense,
        single_concept_domains=single_concept_domains,
        cross_domain_edge_total=cross_domain_edge_total,
        cross_domain_edge_deficit=cross_domain_edge_deficit,
        bisociation_readiness=bisociation_readiness,
        domain_profiles=profiles,
    )


def identify_bridge_domains(
    field: FieldSurface,
    report: PercolationReport | None = None,
) -> list[dict[str, Any]]:
    """
    Identify strategic 'bridge domains' that would maximise cross-domain
    connectivity if their concept count were increased.

    A bridge domain is one that:
    1. Already has some concepts (>= BRIDGE_DOMAIN_MIN_CONCEPTS) but is thin
    2. Is connected to many disparate domains (high betweenness potential)
    3. Or is a known strategic domain that enables bisociation
       (creativity theory, network theory, cognitive science, etc.)

    Returns a list of dicts with:
    - domain: domain name
    - concept_count: current concepts
    - connected_domains: number of domains this domain links to
    - bridge_score: how strategic this domain is for percolation
    - reason: why this domain is a bridge candidate
    """
    if report is None:
        report = domain_density_report(field)

    # Known strategic domains that enable bisociation
    STRATEGIC_DOMAINS = {
        # Swedish names (as they appear in the graph)
        "kreativitetsteori": "creativity theory — enables analogical reasoning",
        "nätverksteori": "network theory — enables structural mapping",
        "kognitiv vetenskap": "cognitive science — bridges neuroscience and psychology",
        "neurovetenskap": "neuroscience — bridges biology and cognition",
        "kognitiv_vetenskap": "cognitive_science — bridges neuroscience and psychology",
        "systembiologi": "systems biology — bridges molecular and macro",
        "immunologi": "immunology — bridges biology and network theory",
        "komplexitetsteori": "complexity theory — enables cross-domain analogies",
        "analogi": "analogy — direct enabler of bisociation",
        "metafor": "metaphor — linguistic bridge mechanism",
        # English names
        "creativity theory": "creativity theory — enables analogical reasoning",
        "network theory": "network theory — enables structural mapping",
        "cognitive science": "cognitive science — bridges neuroscience and psychology",
        "neuroscience": "neuroscience — bridges biology and cognition",
        "systems biology": "systems biology — bridges molecular and macro",
        "immunology": "immunology — bridges biology and network theory",
        "complexity theory": "complexity theory — enables cross-domain analogies",
        "analogy": "analogy — direct enabler of bisociation",
        "metaphor": "metaphor — linguistic bridge mechanism",
        "philosophy of mind": "bridges cognition, consciousness, and AI",
        "epistemology": "bridges knowledge theory and reasoning",
        "semiotics": "bridges language, meaning, and cognition",
    }

    bridges: list[dict[str, Any]] = []

    # Build domain connectivity from profiles
    for profile in report.domain_profiles:
        # Skip dense domains — they don't need more concepts
        if profile.density_class == "dense":
            continue

        # Skip absent domains with zero concepts — we can't search for what
        # doesn't exist yet unless it's a known strategic domain
        if profile.concept_count == 0 and profile.name not in STRATEGIC_DOMAINS:
            continue

        n_connected = len(profile.connected_domains)
        is_strategic = profile.name in STRATEGIC_DOMAINS
        strategic_reason = STRATEGIC_DOMAINS.get(profile.name, "")

        # Bridge score: combination of connectivity, thinness, and strategic value
        # Higher = more strategic to fill
        # Strategic domains get a high floor even if they have 0 connections,
        # because they are *needed* bridges that don't exist yet.
        connectivity_score = n_connected / max(1, report.total_domains)
        thinness_score = 1.0 - min(1.0, profile.concept_count / max(1, MARGINAL_DOMAIN_CONCEPTS))
        strategic_boost = BRIDGE_PRIORITY_BOOST if is_strategic else 0.0

        if is_strategic:
            # Strategic domains always score high — they are the *purpose* of
            # the percolation system, not optional
            bridge_score = 0.5 + 0.3 * thinness_score + strategic_boost
        else:
            bridge_score = (
                0.4 * connectivity_score
                + 0.4 * thinness_score
                + strategic_boost
            )

        # Only include domains with potential
        if bridge_score < 0.1:
            continue

        reason = ""
        if is_strategic:
            reason = f"Strategic domain: {strategic_reason}"
        elif n_connected >= 3:
            reason = f"Connects {n_connected} domains — high betweenness"
        elif profile.density_class == "thin":
            reason = f"Thin domain ({profile.concept_count} concepts) — below percolation threshold"
        else:
            reason = f"Marginal domain — approaching threshold"

        bridges.append({
            "domain": profile.name,
            "concept_count": profile.concept_count,
            "connected_domains": n_connected,
            "cross_domain_edges": profile.cross_domain_edges_total,
            "bridge_score": round(bridge_score, 3),
            "reason": reason,
        })

    # Sort by bridge score descending
    bridges.sort(key=lambda b: b["bridge_score"], reverse=True)
    return bridges[:MAX_BRIDGE_DOMAINS]


def generate_ingestion_tasks(
    field: FieldSurface,
    report: PercolationReport | None = None,
    bridges: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Generate targeted research tasks for thin and bridge domains.

    Each task is formatted for the research queue:
    - domain: target domain
    - concepts: sample concepts from the domain (for search queries)
    - kind: "domain_expand" or "bridge_build"
    - priority: 0.0–1.0 (higher = more urgent)
    - rationale: why this domain needs expansion

    Bridge domains get higher priority than thin domains because filling
    them creates cross-domain edges that enable percolation.
    """
    if report is None:
        report = domain_density_report(field)
    if bridges is None:
        bridges = identify_bridge_domains(field, report)

    tasks: list[dict[str, Any]] = []

    # 1. Bridge domain tasks (highest priority)
    for bridge in bridges:
        domain = bridge["domain"]
        concepts = [c["name"] for c in field.concepts(domain=domain)[:4]]
        if not concepts:
            # Strategic domain with zero or few concepts — use domain name as seed
            concepts = [domain]

        # Bridge tasks get high priority — they are the lever that moves percolation
        # Always >= 0.8 because bridge domains are the strategic priority
        priority = min(1.0, max(0.8, 0.7 + bridge["bridge_score"] * 0.25))

        tasks.append({
            "domain": domain,
            "concepts": concepts,
            "kind": "bridge_build",
            "priority": round(priority, 2),
            "rationale": (
                f"Bridge domain ({bridge['reason']}). "
                f"Current: {bridge['concept_count']} concepts, "
                f"{bridge['connected_domains']} cross-domain connections. "
                f"Target: {MIN_DOMAIN_CONCEPTS} concepts for percolation."
            ),
            "query": (
                f"Undersök domänen '{domain}' med fokus på begrepp som "
                f"skapar broar till andra domäner. "
                f"Existerande koncept: {', '.join(concepts)}. "
                f"Sök efter: definitioner, relationer, och analogier som "
                f"kopplar {domain} till andra kunskapsområden."
            ),
        })

    # 2. Thin domain tasks (medium priority)
    thin_profiles = [p for p in report.domain_profiles
                     if p.density_class == "thin" and p.concept_count >= 2]

    # Sort thin domains by connectivity (most connected first = more bridge potential)
    thin_profiles.sort(key=lambda p: len(p.connected_domains), reverse=True)

    for profile in thin_profiles[:MAX_INGESTION_TASKS]:
        # Skip if already covered as a bridge domain
        if any(t["domain"] == profile.name for t in tasks):
            continue

        domain = profile.name
        concepts = [c["name"] for c in field.concepts(domain=domain)[:4]]
        if not concepts:
            continue

        n_connected = len(profile.connected_domains)
        # Cap at 0.75 so bridge_build tasks always outrank domain_expand
        priority = min(0.75, 0.3 + 0.1 * n_connected + 0.05 * profile.concept_count)

        tasks.append({
            "domain": domain,
            "concepts": concepts,
            "kind": "domain_expand",
            "priority": round(priority, 2),
            "rationale": (
                f"Thin domain: {profile.concept_count} concepts "
                f"(threshold: {MIN_DOMAIN_CONCEPTS}). "
                f"Connected to {n_connected} other domains. "
                f"Needs {MIN_DOMAIN_CONCEPTS - profile.concept_count} more concepts."
            ),
            "query": (
                f"Kartlägg domänen '{domain}' — hitta grundläggande begrepp, "
                f"definitioner och relationer. "
                f"Nuvarande koncept: {', '.join(concepts)}. "
                f"Fokusera på att bygga ut förståelsen av denna domän."
            ),
        })

    # Sort all tasks by priority descending, but always reserve at least
    # half the slots for bridge_build tasks (they are the percolation lever)
    tasks.sort(key=lambda t: (0 if t["kind"] == "bridge_build" else 1, -t["priority"]))

    # Ensure bridge_build tasks get at least half the slots
    max_bridge = max(1, MAX_INGESTION_TASKS // 2)
    bridge_tasks = [t for t in tasks if t["kind"] == "bridge_build"][:max_bridge]
    expand_tasks = [t for t in tasks if t["kind"] != "bridge_build"][:MAX_INGESTION_TASKS - len(bridge_tasks)]

    result = bridge_tasks + expand_tasks
    result.sort(key=lambda t: t["priority"], reverse=True)
    return result


def format_report(report: PercolationReport) -> str:
    """Format a percolation report as a human-readable summary."""
    lines = [
        "═══ PERCOLATION REPORT ═══",
        f"Domains: {report.total_domains:,}  |  Concepts: {report.total_concepts:,}  |  Relations: {report.total_relations:,}",
        f"Avg concepts/domain: {report.avg_concepts_per_domain:.1f}  |  Single-concept domains: {report.single_concept_domains}",
        f"Percolation threshold: ln({report.total_domains}) ≈ {report.threshold_ln_n:.1f} connections/domain",
        f"Cross-domain edges: {report.cross_domain_edge_total}  |  Target: {int(report.threshold_target * report.total_domains):,}",
        f"Edge deficit: {report.cross_domain_edge_deficit:,}  |  Bisociation readiness: {report.bisociation_readiness:.1%}",
        "",
        "Domain distribution:",
        f"  Absent (0):    {report.domains_absent}",
        f"  Thin (1-{MIN_DOMAIN_CONCEPTS-1}):     {report.domains_thin}",
        f"  Marginal ({MIN_DOMAIN_CONCEPTS}-{MARGINAL_DOMAIN_CONCEPTS-1}): {report.domains_marginal}",
        f"  Dense ({MARGINAL_DOMAIN_CONCEPTS}+):    {report.domains_dense}",
        "",
        "Thinnest domains (concepts < threshold):",
    ]

    thin_profiles = [p for p in report.domain_profiles if p.concept_count < MIN_DOMAIN_CONCEPTS]
    thin_profiles.sort(key=lambda p: p.concept_count)
    for p in thin_profiles[:15]:
        conn = len(p.connected_domains)
        lines.append(f"  {p.name:30s}  {p.concept_count:3d} concepts  {conn:2d} cross-domains  [{p.density_class}]")

    if len(thin_profiles) > 15:
        lines.append(f"  ... and {len(thin_profiles) - 15} more thin domains")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Nervbana axion density + rigidity + sweet spot
# ---------------------------------------------------------------------------

@dataclass
class NervbanaProfile:
    """Axion density profile for a domain pair (nervbana / neural pathway).

    Mirrors the biological principle that axion count per neural pathway
    determines both signal capacity and rigidity.  Inspired by fMRI
    observations comparing neurotypical and autistic brains: surplus
    axions enhance systematic analysis but also create overstimulation
    and rigidity.  The sweet spot is the percolation optimum.

    Zones:
      "isolated" — too few axions for signal propagation (k < k_min)
      "sweet"    — enough for bisociation, not enough for rigidity
      "rigid"    — over-connected, signals lock up (k > k_rigid)
    """
    domain_a: str
    domain_b: str
    axion_count: int          # cross-domain edges between A and B
    concepts_a: int
    concepts_b: int
    max_possible: int         # theoretical max edges (|A| × |B|)
    density: float            # axion_count / max_possible
    k_min: float              # ln(max(|A|,|B|)) — minimum for propagation
    k_sweet: float            # optimal axion count for bisociation
    k_rigid: float            # above this → rigidity
    rigidity: float           # 0-1, how over-connected (1 = fully rigid)
    isolation: float           # 0-1, how under-connected (1 = fully isolated)
    zone: str                 # "isolated" | "sweet" | "rigid"


def nervbana_profiles(
    field: FieldSurface,
    report: PercolationReport | None = None,
    sweet_multiplier: float = 2.5,
) -> list[NervbanaProfile]:
    """Compute axion density for each domain pair (nervbana).

    Each nervbana is a pair of domains connected by cross-domain edges.
    The "axion count" is the number of such edges — analogous to the
    biological number of axions in a white-matter tract.

    Sweet spot calibration:
      k_min  = ln(max(|A|,|B|))          — minimum for signal propagation
      k_sweet = sweet_multiplier * k_min  — optimal for bisociation
      k_rigid = |A| * |B| / max(|A|,|B|) — overfitting threshold

    The sweet_multiplier controls where creativity peaks on the
    Yerkes-Dodson curve.  Default 2.5 means: 2.5× the minimum
    connections gives optimal bisociation potential.
    """
    if report is None:
        report = domain_density_report(field)

    # Build cross-domain edge map: (domain_a, domain_b) -> count
    nervbana_edges: dict[tuple[str, str], int] = {}
    for r in field.query_all_relations_with_metadata(include_evidence=False):
        src = r.get("src", "")
        tgt = r.get("tgt", "")
        src_domain = field.concept_domain(src) or r.get("src_domain", "")
        tgt_domain = field.concept_domain(tgt) or ""
        if src_domain and tgt_domain and src_domain != tgt_domain:
            key = tuple(sorted([src_domain, tgt_domain]))
            nervbana_edges[key] = nervbana_edges.get(key, 0) + 1

    # Domain concept counts
    domain_sizes: dict[str, int] = {}
    for profile in report.domain_profiles:
        domain_sizes[profile.name] = profile.concept_count

    profiles: list[NervbanaProfile] = []
    for (da, db), axion_count in nervbana_edges.items():
        na = domain_sizes.get(da, 0)
        nb = domain_sizes.get(db, 0)
        if na < 1 or nb < 1:
            continue

        max_possible = na * nb
        density = axion_count / max(1, max_possible)
        k_min = math.log(max(na, nb)) if max(na, nb) > 1 else 1.0
        k_sweet = sweet_multiplier * k_min
        k_rigid = max_possible / max(na, nb)

        # Isolation: how far below k_min (1.0 = fully isolated, 0.0 = at k_min)
        if axion_count < k_min:
            isolation = 1.0 - (axion_count / max(0.01, k_min))
        else:
            isolation = 0.0

        # Rigidity: how far above k_sweet (0.0 = at sweet spot, 1.0 = fully rigid)
        if axion_count > k_sweet and k_rigid > k_sweet:
            rigidity = min(1.0, (axion_count - k_sweet) / max(1, k_rigid - k_sweet))
        else:
            rigidity = 0.0

        # Zone classification
        if axion_count < k_min:
            zone = "isolated"
        elif axion_count <= k_sweet:
            zone = "sweet"
        else:
            zone = "rigid"

        profiles.append(NervbanaProfile(
            domain_a=da,
            domain_b=db,
            axion_count=axion_count,
            concepts_a=na,
            concepts_b=nb,
            max_possible=max_possible,
            density=density,
            k_min=round(k_min, 2),
            k_sweet=round(k_sweet, 2),
            k_rigid=round(k_rigid, 2),
            rigidity=round(rigidity, 3),
            isolation=round(isolation, 3),
            zone=zone,
        ))

    profiles.sort(key=lambda p: (-p.isolation, -p.rigidity))
    return profiles


def domain_rigidity(
    field: FieldSurface,
    report: PercolationReport | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute rigidity score for each domain.

    Rigidity = how over-crystallized a domain is.  High rigidity means:
    - many edges with high w (structural weight) and low u (uncertainty)
    - signals propagate strongly but without exploration
    - the domain is "locked in" — it can analyze systematically but
      cannot adapt or create novel connections

    Formula:
      rigidity(A) = mean(w) * (1 - mean(u))  for cross-domain edges
      where w and u come from edge metadata

    Returns dict: domain_name -> {
        rigidity, mean_w, mean_u, cross_domain_edges,
        crystallized_pct, zone ("fluid" | "balanced" | "rigid")
    }
    """
    if report is None:
        report = domain_density_report(field)

    # Collect edge metadata per domain
    domain_edge_data: dict[str, list[dict]] = {p.name: [] for p in report.domain_profiles}

    for r in field.query_all_relations_with_metadata(include_evidence=True):
        src = r.get("src", "")
        tgt = r.get("tgt", "")
        src_domain = field.concept_domain(src) or r.get("src_domain", "")
        tgt_domain = field.concept_domain(tgt) or ""

        # Only cross-domain edges matter for rigidity
        if not src_domain or not tgt_domain or src_domain == tgt_domain:
            continue

        meta = r.get("metadata", {})
        w = float(meta.get("w", 0.5))
        u = float(meta.get("u", 0.5))

        if src_domain in domain_edge_data:
            domain_edge_data[src_domain].append({"w": w, "u": u})
        if tgt_domain in domain_edge_data:
            domain_edge_data[tgt_domain].append({"w": w, "u": u})

    results: dict[str, dict[str, Any]] = {}
    for domain, edges in domain_edge_data.items():
        if not edges:
            continue

        mean_w = sum(e["w"] for e in edges) / len(edges)
        mean_u = sum(e["u"] for e in edges) / len(edges)
        crystallized = sum(1 for e in edges if e["w"] > 0.55 and e["u"] < 0.35)
        rigidity = mean_w * (1.0 - mean_u)

        if rigidity < 0.3:
            zone = "fluid"
        elif rigidity < 0.6:
            zone = "balanced"
        else:
            zone = "rigid"

        results[domain] = {
            "rigidity": round(rigidity, 3),
            "mean_w": round(mean_w, 3),
            "mean_u": round(mean_u, 3),
            "cross_domain_edges": len(edges),
            "crystallized_pct": round(crystallized / max(1, len(edges)), 3),
            "zone": zone,
        }

    return results


def sweet_spot_report(
    field: FieldSurface,
    report: PercolationReport | None = None,
    max_nervbanor: int = 50,
) -> dict[str, Any]:
    """Generate a sweet spot calibration report.

    Combines nervbana axion density, domain rigidity, and percolation
    data into a unified view of where Nous sits on the Yerkes-Dodson
    curve for knowledge graphs:
      - Isolation (too few axions) → no bisociation
      - Sweet spot (k_sweet axions) → optimal creativity
      - Rigidity (too many axions) → overfitting, lockup

    Returns dict with:
      nervbanor: list of NervbanaProfile
      domain_rigidity: dict of domain -> rigidity data
      summary: {
        n_isolated, n_sweet, n_rigid,
        avg_axion_density, avg_rigidity,
        sweet_spot_fraction  # fraction of nervbanor in sweet zone
      }
    """
    if report is None:
        report = domain_density_report(field)

    nervbanor = nervbana_profiles(field, report)[:max_nervbanor]
    rigidity = domain_rigidity(field, report)

    n_isolated = sum(1 for n in nervbanor if n.zone == "isolated")
    n_sweet = sum(1 for n in nervbanor if n.zone == "sweet")
    n_rigid = sum(1 for n in nervbanor if n.zone == "rigid")
    total = max(1, len(nervbanor))

    avg_density = sum(n.density for n in nervbanor) / total
    avg_rigidity = sum(r.get("rigidity", 0) for r in rigidity.values()) / max(1, len(rigidity))

    log.info(
        f"Sweet spot: {n_isolated} isolerade, {n_sweet} sweet, {n_rigid} rigida "
        f"av {len(nervbanor)} nervbanor. "
        f"Snitt density={avg_density:.4f}, rigidity={avg_rigidity:.3f}"
    )

    return {
        "nervbanor": nervbanor,
        "domain_rigidity": rigidity,
        "summary": {
            "n_isolated": n_isolated,
            "n_sweet": n_sweet,
            "n_rigid": n_rigid,
            "avg_axion_density": round(avg_density, 4),
            "avg_rigidity": round(avg_rigidity, 3),
            "sweet_spot_fraction": round(n_sweet / total, 3),
        },
    }


def format_sweet_spot_report(data: dict[str, Any]) -> str:
    """Format a sweet spot report as a human-readable summary."""
    s = data.get("summary", {})
    nervbanor = data.get("nervbanor", [])
    rigidity = data.get("domain_rigidity", {})

    lines = [
        "═══ SWEET SPOT REPORT ═══",
        f"Nervbanor: {len(nervbanor)} domain pairs",
        f"  Isolerade: {s.get('n_isolated', 0)}  |  Sweet: {s.get('n_sweet', 0)}  |  Rigida: {s.get('n_rigid', 0)}",
        f"  Sweet spot fraction: {s.get('sweet_spot_fraction', 0):.1%}",
        f"  Avg axion density: {s.get('avg_axion_density', 0):.4f}  |  Avg rigidity: {s.get('avg_rigidity', 0):.3f}",
        "",
        "Nervbanor (top 15 by isolation):",
    ]

    # Show most isolated nervbanor first — these are the targets
    sorted_nerv = sorted(nervbanor, key=lambda n: -n.isolation)
    for n in sorted_nerv[:15]:
        lines.append(
            f"  {n.domain_a:25s} ↔ {n.domain_b:25s}  "
            f"k={n.axion_count:3d}  k_min={n.k_min:.1f}  k_sweet={n.k_sweet:.1f}  "
            f"[{n.zone}]  iso={n.isolation:.2f}"
        )

    # Show rigid domains (if any)
    rigid_domains = [(d, r) for d, r in rigidity.items() if r.get("zone") == "rigid"]
    if rigid_domains:
        lines.append("")
        lines.append("Rigida domäner (överstimulerade):")
        for d, r in rigid_domains[:10]:
            lines.append(
                f"  {d:30s}  rigidity={r['rigidity']:.3f}  "
                f"w={r['mean_w']:.2f}  u={r['mean_u']:.2f}  "
                f"crystallized={r['crystallized_pct']:.0%}"
            )

    return "\n".join(lines)


def bridge_bisociation_search(
    field: FieldSurface,
    bridges: list[dict[str, Any]] | None = None,
    tau_threshold: float = 0.55,
    max_epsilon: float = 2.0,
) -> list[dict[str, Any]]:
    """
    Targeted bisociation search among bridge domains only.

    Unlike bisociation_candidates() which operates on the full domain graph
    (and may miss small bridge domains due to max_domains filtering), this
    function ONLY checks pairs of bridge domains against each other.

    This ensures that strategically injected bridge domains — which are
    typically small (1–10 concepts) — are always included in bisociation
    analysis regardless of their size relative to the largest domains.

    Returns a list of bisociation candidate dicts with the same schema as
    bisociation_candidates(), plus a "bridge" field indicating both domains
    are bridge domains.
    """
    if bridges is None:
        bridges = identify_bridge_domains(field, domain_density_report(field))

    if not bridges:
        return []

    bridge_names = [b["domain"] for b in bridges]
    # Use priority_domains to force all bridge domains into the analysis
    return field.bisociation_candidates(
        tau_threshold=tau_threshold,
        max_epsilon=max_epsilon,
        max_domains=0,  # 0 = no limit, include all
        priority_domains=bridge_names,
    )


def identify_loose_nodes(
    field: FieldSurface,
    min_cross_domain_connections: int = 0,
    max_nodes: int = 100,
) -> list[dict[str, Any]]:
    """
    Find concepts with zero or minimal cross-domain connections.

    This is Nous's self-knowledge: "I am not connected to X."
    Concepts with 0 cross-domain connections are *isolated* — they exist
    in a domain but have no bridge to any other domain. Concepts with 1-2
    connections are *loose* — they have a toe-hold but need more.

    The urgency score drives goal-directed curiosity:
      - Isolated (0 connections): urgency 1.0 — "I need to find connections HERE"
      - Loose (1 connection): urgency 0.9 — "One bridge exists, but I'm fragile"
      - Loose (2 connections): urgency 0.7 — "Getting connected, keep going"

    Returns list of dicts sorted by urgency descending:
      node, domain, cross_domain_connections, is_isolated, urgency,
      connected_domains (set of domain names this concept bridges to)
    """
    import logging as _logging
    _log = _logging.getLogger("nouse.percolation.loose_nodes")

    loose: list[dict[str, Any]] = []

    for node_name, node_data in field._G.nodes(data=True):
        domain = node_data.get("domain", "")
        if not domain:
            continue

        # Count cross-domain connections
        cross_domain_connections = 0
        connected_domains: set[str] = set()
        for neighbor in field._G.neighbors(node_name):
            nbr_data = field._G.nodes.get(neighbor, {})
            nbr_domain = nbr_data.get("domain", "")
            if nbr_domain and nbr_domain != domain:
                cross_domain_connections += 1
                connected_domains.add(nbr_domain)

        # Skip concepts that exceed the threshold
        if cross_domain_connections > max(min_cross_domain_connections, 2):
            continue

        # Urgency: 0 connections = isolated (1.0), 1 = fragile (0.9), 2 = loose (0.7)
        if cross_domain_connections == 0:
            urgency = 1.0
            is_isolated = True
        elif cross_domain_connections == 1:
            urgency = 0.9
            is_isolated = False
        else:  # 2 connections
            urgency = 0.7
            is_isolated = False

        loose.append({
            "node": node_name,
            "domain": domain,
            "cross_domain_connections": cross_domain_connections,
            "is_isolated": is_isolated,
            "urgency": urgency,
            "connected_domains": connected_domains,
        })

    # Sort by urgency (highest first), then by cross_domain_connections (lowest first)
    loose.sort(key=lambda n: (-n["urgency"], n["cross_domain_connections"]))
    _log.info(
        f"Loose nodes: {sum(1 for n in loose if n['is_isolated'])} isolated, "
        f"{len(loose)} total (threshold: {min_cross_domain_connections} cross-domain connections)"
    )
    return loose[:max_nodes]