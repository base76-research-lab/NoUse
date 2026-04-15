---
title: "Morning Research Sweep — 2026-04-15"
subtitle: "PubMed + arXiv survey for epistemic grounding, autonomous goals, knowledge graph reasoning, topological plasticity, and bisociation"
author: "Björn Wikström"
date: "2026-04-15"
---

# Morning Research Sweep — 2026-04-15

**Focus areas:** epistemic grounding, cognitive architecture, knowledge graph reasoning, topological plasticity, bisociation, autonomous goals

---

## 1. Paper Summary Table

| # | Paper | Source | Date | Relevance |
|---|-------|--------|------|-----------|
| 1 | Executable Epistemology: The Structured Cognitive Loop | arXiv:2510.15952 | 2025-10 | **Direct** — SCL mirrors Nous cognitive cycle |
| 2 | Coherent Without Grounding, Grounded Without Success | arXiv:2603.28371 | 2026-03 | **Direct** — Epistemic Triangle validates Larynx Problem |
| 3 | Co-Evolution of Policy and Internal Reward (Self-Guide) | arXiv:2604.03098 | 2026-04 | **High** — Self-generated internal reward → Nous drive engine |
| 4 | How Intrinsic Motivation Underlies Embodied Open-Ended Behavior | arXiv:2601.10276 | 2026-01 | **High** — Formalizes intrinsic motivation hierarchy |
| 5 | Reason-Align-Respond: Aligning LLM Reasoning with Knowledge Graphs | PubMed/IEEE TPAMI | 2026-02 | **High** — KG-grounded reasoning validates Nous KG approach |

---

## 2. Detailed Summaries

### Paper 1: Executable Epistemology (arXiv:2510.15952)

**Authors:** Myung Ho Kim (2025)

Introduces the **Structured Cognitive Loop (SCL)** as an "executable epistemological framework" for emergent intelligence. Argues that LLMs lack epistemic architecture — they display "intelligence without genuine epistemic understanding." SCL operationalizes philosophical insights into computationally interpretable structures, treating intelligence as a performed process (judgment → memory → control → action → regulation) rather than a static property. Key finding: functional separation within cognitive architecture yields more coherent behavior than monolithic prompt-based systems.

**Relation to Nous:** **Extension.** SCL mirrors Nous's 18-step cognitive cycle almost exactly. Kim's argument that LLMs need epistemic architecture is precisely the Larynx Problem. SCL validates our architecture — Nous implements the functional separation Kim argues for. Difference: SCL is theoretical; Nous is operational. This paper provides philosophical backing for the cognitive substrate approach.

---

### Paper 2: Coherent Without Grounding, Grounded Without Success (arXiv:2603.28371)

**Authors:** Camilo Chacón Sartori (2026)

Introduces the **Bidirectional Coherence Paradox**: LLMs can succeed while misidentifying mechanisms (low-observability domains) and can generate accurate explanations that fail to translate into effective intervention (high-observability domains). Proposes the **Epistemic Triangle** model (priors, signals, domain knowledge) and a tripartite evaluation: coherence, grounding, and a proper basing relation. Demonstrates experimentally that "neither behavioral success nor explanatory accuracy alone suffices for attributing understanding."

**Relation to Nous:** **Direct support for the Larynx Problem.** The Bidirectional Coherence Paradox is a formalization of exactly what the Larynx Problem describes: LLMs produce coherent text (larynx function) without grounded understanding. The Epistemic Triangle model provides a formal framework Nous could use to evaluate its own epistemic state. The "proper basing relation" is what Nous's knowledge graph + evidence scoring system provides. **Action:** Reference this paper in the Larynx Problem manuscript.

---

### Paper 3: Co-Evolution of Policy and Internal Reward (arXiv:2604.03098)

**Authors:** Wang, Wu, Song et al. (2026-04)

Proposes **Self-Guide**, a self-generated internal reward mechanism for LLM agents. At inference time, a short self-guidance signal steers the next action; at training time, the same signal converts into step-level internal reward. Creates a co-evolutionary loop: better policy → better guidance → better internal reward. Shows 8% improvement over baselines trained with environment reward alone.

**Relation to Nous:** **Extension.** Self-Guide's co-evolution loop parallels Nous's intrinsic drive engine (D1–D6). Where Self-Guide uses self-generated rewards for task performance, Nous uses arousal/curiosity signals for knowledge graph expansion. The key insight — that agents improve by generating their own internal reward — validates Nous's design of autonomous goal-setting via the curiosity loop. **Potential collaboration:** Self-Guide's step-level reward could enhance Nous's relation evidence scoring.

---

### Paper 4: How Intrinsic Motivation Underlies Embodied Open-Ended Behavior (arXiv:2601.10276)

**Authors:** Moreno-Bote, Haefner, Galiano-Landeira, Yang, Maldonado (2026)

Presents a hierarchical framework: **objective → intrinsic reward (motivation) → drives → goals → extrinsic reward**. Reviews formalizations including empowerment, free energy principle, information-gain maximization, and maximum occupancy principle. Key insight: a single intrinsic motivation objective "breaks infinite regress, as drives and goals act only temporarily to serve the objective." Recasts extrinsic rewards as instrumental means to achieve intrinsic objectives, not ultimate ends.

**Relation to Nous:** **Direct support + extension.** This framework maps directly onto Nous's drive hierarchy: Nous's D1 (survival) → D3 (curiosity) → D5 (coherence) mirror the paper's objective → drives → goals hierarchy. The paper's argument that a single intrinsic motivation objective breaks infinite regress validates Nous's design of having a unified mission rather than competing objectives. The empowerment and information-gain formalizations could be adopted as quantitative measures for Nous's curiosity loop. **Action:** Consider adopting empowerment as a drive signal.

---

### Paper 5: Reason-Align-Respond (RAR) (PubMed/IEEE TPAMI, 2026-02)

**Authors:** Shen, Wang, Yang et al. (2026)

Presents a framework that systematically integrates LLM reasoning with knowledge graphs for KGQA. Three components: **Reasoner** (generates reasoning chains), **Aligner** (maps chains to valid KG paths), **Responder** (synthesizes answers). Formulated as a latent variable mixture model optimized with EM. Achieves 93.3% on WebQSP and 91.0% on CWQ. Confirms that aligning LLM reasoning with structured knowledge improves both accuracy and interpretability.

**Relation to Nous:** **Support.** RAR validates the core Nous thesis: LLMs + knowledge graphs produce better, more grounded reasoning than LLMs alone. Where RAR aligns reasoning chains to KG paths for question answering, Nous aligns curiosity-driven exploration to KG topology for bisociation discovery. The Aligner component is analogous to Nous's nervbanor (BFS path finder) + TDA bisociation detector. **Key difference:** RAR works for retrieval; Nous works for discovery.

---

## 3. Additional Papers of Interest

| Paper | Source | Note |
|-------|--------|------|
| Epistemological Fault Lines Between Human and AI (arXiv:2512.19466) | arXiv | Identifies 7 fault lines; "Epistemia" concept = linguistic plausibility substituting for epistemic evaluation |
| PRISM: Pluralistic Reasoning via In-context Structure Modeling (arXiv:2602.21317) | arXiv | Epistemic Evolution paradigm; on-the-fly Epistemic Graphs; closest to "reasoning substrate" concept |
| Explicit Cognitive Allocation (arXiv:2601.13443) | arXiv | Cognitive Universal Agent (CUA) with explicit epistemic function separation |
| KGHaluBench (arXiv:2602.19643) | arXiv/EACL 2026 | KG-based hallucination benchmark; even GPT-5 struggles with factual depth |
| Lie to Me: KGs for Hallucination Self-Detection (arXiv:2512.23547) | arXiv | Converting LLM responses → KGs for hallucination detection; 16% accuracy improvement |
| ProgRAG (AAAI 2026) | AAAI | Progressive KG retrieval; uncertainty-aware pruning for multi-hop reasoning |
| Hemispheric Specialization and Creativity (PMID:3226963) | PubMed | 1988; bisociation as interhemispheric communication; creativity = opposite of alexithymia |
| Semantic Subgroup Discovery (PMID:19964398) | PubMed | 2009; bisociative data analysis in systems biology; Lavrač et al. |

---

## 4. PubMed Searches — No Recent Hits

| Query | Result |
|-------|--------|
| "epistemic grounding" AND "large language model" | 0 (session expired) |
| "cognitive architecture" AND "AI agent" | 0 (session expired) |
| "bisociation" OR "bisociative" | 3 results, all pre-2010 (Hoppe 1988, Lavrač 2009, tracheal cannula 1993) |

The PubMed bisociation results confirm: **bisociation is an established concept in neuroscience (Hoppe 1988: hemispheric bisociation) and data mining (Lavrač 2009: bisociative data analysis), but has zero recent AI/LLM publications.** This is a gap Nous is uniquely positioned to fill.

---

## 5. Implications for Nous

1. **The Larynx Problem has formal backing.** Sartori's Bidirectional Coherence Paradox (arXiv:2603.28371) independently derives the same conclusion: LLM coherence ≠ grounded understanding. The Epistemic Triangle model provides a formal evaluation framework.

2. **Nous's architecture is not isolated.** Kim's SCL (arXiv:2510.15952) independently proposes functional separation in cognitive architecture — validating Nous's 18-step cycle. The concept is circulating under different names: "epistemic architecture," "cognitive substrate," "reasoning substrate."

3. **Intrinsic motivation has formal foundations.** Moreno-Bote et al. (arXiv:2601.10276) provide the theoretical backing Nous's drive engine needs. Empowerment and information-gain maximization are concrete formalizations that could replace our ad hoc curiosity signals.

4. **Bisociation research gap confirmed.** Zero recent AI papers on bisociation. Hoppe 1988 (hemispheric) and Lavrač 2009 (data mining) are the foundational works. Nous's percolation-driven bisociation discovery would be a novel contribution.

5. **Self-generated rewards work.** Self-Guide (arXiv:2604.03098) demonstrates that co-evolving policy and internal reward improves agent performance by 8%. This validates Nous's design where the curiosity loop generates its own research priorities.

---

*Research sweep conducted 2026-04-15. Next sweep in 7 days.*