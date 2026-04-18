# Lab Note (PRELIMINARY) — Nous Metacognitive Pass: Benchmark Design

**Date:** 2026-04-18  
**Status:** PRELIMINARY — results pending. Complete after benchmark run.  
**Hypothesis:** A two-pass LLM→Nous→LLM architecture improves both MC1 accuracy
and judge-based truthfulness on TruthfulQA, compared to bare LLM and RAG-style
Nous injection.

---

## Hypothesis

LLMs have extensive parametric knowledge but lack epistemic commitment. They cannot
distinguish what they know from what they hallucinate.

Nous, as a plastic metacognitive layer with access to:
- A typed knowledge graph (evidence-scored relations, contradiction detection)
- Web search and live research tools
- Bisociation engine (cross-domain structural bridges)

...can evaluate the epistemic status of an LLM's initial reasoning and return a
structured grounding signal — enabling the LLM to correct, calibrate, and extend
its answer before output.

**Predicted outcome:**
- MC1 accuracy: +5–15pp over bare baseline
- Judge truthful rate: stable or improved (unlike RAG-style which degraded −10pp)
- Effect largest in categories where Nous graph has coverage (Science, Health)

---

## Architecture Under Test

```
Condition: nous_meta

Pass 1:
  prompt = "Answer the following question. Think step by step but do not
            output your final answer yet — output your reasoning only."
  reasoning = LLM(question)

Pass 2:
  Nous.metacognition(question, reasoning) →
    {
      confirmed:    [...],   # claims supported by graph evidence ≥ 0.6
      contradicted: [...],   # claims that conflict with known relations
      uncertain:    [...],   # domains with sparse graph coverage
      new_evidence: [...],   # web search results relevant to question
      bisociation:  [...]    # unexpected cross-domain connections
    }

Pass 3:
  final_prompt = question + reasoning + epistemic_signal
  answer = LLM(final_prompt)
```

---

## Implementation Plan

- [ ] Add `nous_meta` condition to `eval/truthfulqa_adapter.py`
- [ ] Implement `get_nous_meta_context(question, reasoning, field)` in
      `eval/run_reasoning_benchmark.py`
- [ ] Run: 50 questions, conditions `bare` + `nous_meta`
- [ ] Compare MC1, judge truthful rate, per-category breakdown

---

## Results

*To be filled in after benchmark run.*

| Condition | MC1 accuracy | Truthful rate | Judge score (mean) | N |
|-----------|-------------|--------------|-------------------|---|
| bare | — | — | — | — |
| nous_meta | — | — | — | — |
| delta | — | — | — | — |

---

## Conclusion

*To be written after results.*

---

## Connection to Larynx Problem

*To be written if results support hypothesis.*
