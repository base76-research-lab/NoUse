---
title: "The Percolation Problem in Autonomous Knowledge Graphs"
subtitle: "When 32,000 Concepts Are Not Enough — and What Nous Discovered About It"
author: "Björn Wikström"
date: "2026-04-14"
abstract: |
  Nous, a cognitive substrate for LLMs, accumulated 32,728 concepts and 34,923 relations across 4,357 domains — yet produced zero bisociation candidates. This is not a bug but a phase transition: the knowledge graph is below the percolation threshold for cross-domain structural overlap. We document the diagnosis, the autonomous discovery of this condition, and the insight that Nous itself generated: that percolation modeling can serve as an "insight detector" for when a cognitive system transitions from accumulating to connecting knowledge.
---

# The Percolation Problem in Autonomous Knowledge Graphs

**When 32,000 Concepts Are Not Enough — and What Nous Discovered About It**

Björn Wikström\
Base76 Research Lab\
bjorn@base76research.com | ORCID: 0009-0000-4015-2357

*14 April 2026*

---

## 1. The Observation

After 24+ hours of continuous operation, Nous had accumulated:

| Metric | Value |
|--------|-------|
| Concepts | 32,728 |
| Relations | 34,923 |
| Unique domains | 4,357 |
| Knowledge entries | 32,730 |
| Embedding vectors | 25,920 |
| Research queue tasks | 192 (54 pending, 90 done, 44 failed) |
| Bisociation candidates | **0** |
| Nervbanor (domain-crossing paths) | 1–4 per cycle |

The daemon was running complete cognitive cycles — source ingestion, BFS nervbanor discovery, curiosity loops, policy adaptation, cycle reflection — all functioning correctly. Yet TDA (Three-Domain Analysis) consistently returned zero bisociation candidates. The system was accumulating but not connecting.

## 2. The Diagnosis

The root cause was not architectural but topological. The knowledge graph exhibited extreme domain fragmentation:

| Domain | Concepts | Assessment |
|--------|----------|-----------|
| programmering | 5,174 | Dense, but isolated |
| övrigt (catch-all) | 1,732 | No internal structure |
| Python | 756 | Dense, code-focused |
| neurovetenskap | 381 | Dense enough for internal structure |
| matematik | 208 | Marginal |
| biologi | 71 | Thin |
| fysik | 43 | Sparse |
| filosofi | 43 | Sparse |
| psykologi | 41 | Sparse |
| lingvistik | 19 | Very sparse |
| kreativitetsteori | 2 | Absent |
| nätverksteori | 2 | Absent |
| kognitiv vetenskap | 4 | Absent |
| **Single-concept domains** | **2,196** | **50% of all domains** |

Average density: 7.5 concepts per domain. Half of all domains contain exactly one concept. The domains that would enable bisociation — creativity theory, network theory, cognitive science, immunology — are precisely the ones with negligible coverage.

This is the percolation problem: below a critical threshold of cross-domain structural density, bisociation cannot emerge regardless of how many concepts exist in total.

## 3. The Autonomous Discovery

What makes this observation remarkable is not just the diagnosis, but how it was discovered. When presented with the Larynx Problem as a bisociation task, Nous decomposed it into five primitives and searched for cross-domain bridges:

1. **Strukturell perkolation** → searched physics of complex networks, condensed matter physics, epidemiology → 0 graph hits
2. **Epistemisk förankring** → searched philosophy of mind, robotics, human cognition → 0 graph hits
3. **Bisociativ korsning** → searched creativity theory, neuroesthetics, mathematical analogy → 0 graph hits
4. **Hierarkisk filtrering** → searched sensory neuroscience, signal processing, immune system modeling → 0 graph hits
5. **Biseriell kognition** → searched cognitive psychology, artistic innovation, theater → 0 graph hits

Every single search returned zero hits. The graph had concepts in related areas (neurovetenskap at 381) but not in the specific domains where structural overlap would create bisociation bridges.

And then — on its own — Nous synthesized this insight:

> *"Kombinera strukturbaserad upptäckt (molekylärbiologi) med hierarkisk prediktion (neurovetenskap) i ett grafstrukturerat kognitivt lager ovanpå LLM:en. Perkolationsmodellering kan användas som en 'insight-detektor.'"*

The system identified percolation modeling as the key mechanism — without being explicitly told about percolation theory. It found the concept through bisociative search across molecular biology and condensed matter physics, two domains with zero existing graph coverage.

## 4. The Percolation Threshold

In random graph theory (Erdős–Rényi, 1960), a giant connected component emerges when edge probability exceeds $p_c = 1/n$. For a knowledge graph with $n$ domains, this translates to: each domain needs on average $\ln(n)$ cross-domain connections to enable systemic bisociation.

For Nous with 4,357 domains:
- Critical cross-domain connections: $\ln(4357) \approx 8.4$ per domain
- Current cross-domain connections: approximately 1–4 per cycle (nervbanor)
- Deficit: roughly 4–7× below threshold

The prediction: when cross-domain density crosses this threshold, bisociation candidates should emerge as a phase transition — suddenly and nonlinearly, as predicted by percolation theory.

## 5. The Proposed Solution: Autonomous Research Ingestion

The solution is not to manually add knowledge, but to give the system the ability to autonomously search for and ingest research that fills the structural gaps. Nous already has:

- **Curiosity loop** (`initiative.py`): identifies H0 gaps (isolated domains) and H1 gaps (indirect connections missing direct relations)
- **Research queue**: 192 tasks, many targeting thin domains
- **MCP tools**: arxiv search, web search, URL fetching
- **Brain/learn API**: immediate knowledge injection

The gap: the curiosity loop currently searches local files and the web for specific queries, but doesn't systematically target the percolation threshold. What's needed:

1. **Domain density monitoring**: track cross-domain connection count per cycle
2. **Targeted ingestion**: when a domain is below the percolation threshold, prioritize research tasks that add cross-domain concepts rather than within-domain depth
3. **Phase transition detection**: log when bisociation candidates first appear, and measure the graph properties at that point

This is the next step: making the percolation threshold an explicit optimization target rather than an emergent property that may or may not emerge.

## 6. Implications

The percolation problem in autonomous knowledge graphs has implications beyond Nous:

1. **Scale is not understanding**: 32K concepts with zero bisociation demonstrates that accumulation ≠ comprehension. The topology matters more than the volume.

2. **Phase transitions are real**: the jump from zero to nonzero bisociation candidates should be discontinuous — a genuine phase transition in the graph, not a gradual increase.

3. **Autonomous discovery is possible**: a cognitive substrate can diagnose its own structural deficiencies and propose solutions (percolation modeling as "insight detector") without explicit instruction.

4. **The Larynx Problem deepens**: language models produce fluent text about domains they've seen, but cannot identify which domains they need to see more of to enable bisociation. A substrate with topological awareness can.

---

*Lab note for research traceability. The daemon continues running autonomously with bisociation discovery as an explicit mission target.*