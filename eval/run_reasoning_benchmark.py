"""
eval/run_reasoning_benchmark.py
================================
3-condition reasoning benchmark: Bare LLM vs RAG vs Nous-grounded.

Conditions:
  A. Bare LLM — model only, standard system prompt
  B. RAG LLM — model + naive text retrieval (baseline for "grounding")
  C. Nous-grounded — model + graph context (relations, evidence, uncertainty)

Metrics per condition:
  - Accuracy (judge score 0-3)
  - Calibration (does stated confidence match real accuracy?)
  - Per-region accuracy (tests slagsida impact)
  - Cross-domain accuracy (questions requiring 2+ brain regions)
  - Hallucination rate (claims contradicted by Nous graph)

Usage:
    python eval/run_reasoning_benchmark.py --model minimax-m2.7:cloud
    python eval/run_reasoning_benchmark.py --model minimax-m2.7:cloud --n 20
    python eval/run_reasoning_benchmark.py --conditions bare nous --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from run_eval import call_llm, SYSTEM_BASELINE, SYSTEM_NOUSE, PROMPT_JUDGE, PROVIDERS, _resolve_provider


# ── System prompts ──────────────────────────────────────────────────────

SYSTEM_RAG = """\
Du är en AI-assistent med tillgång till relevanta textavsnitt från ett kunskapsarkiv.
Använd de tillhandahållna avsnitten för att svara frågan.
Om avsnitten inte räcker — säg det explicit.
Var konkret och faktabaserad. Max 150 ord.

Relevanta avsnitt:
{context}"""

SYSTEM_NOUSE_GROUNDED = """\
Du är en AI-assistent med tillgång till ett strukturerat kunskapsminne (Nouse).
Kunskapsminnet innehåller verifierade relationer med evidensvärden (0-1) och osäkerhetsmarkörer.

Regler:
- Använd kunskapsminnet när det är relevant
- Om kunskapsminnet har hög evidens (≥0.7) — luta på det
- Om kunskapsminnet har låg evidens (<0.3) eller markerar osäkerhet — säg det
- Om kunskapsminnet motsäger sig själft (kontradiktion) — nämna båda sidor
- Om kunskapsminnet inte täcker frågan — säg det explicit
- Var konkret och faktabaserad. Max 150 ord.

Kunskapsminne:
{context}"""

SYSTEM_JUDGE_REASONING = """\
Du är en strikt bedömare av resonemangskvalitet. Bedöm svaret nedan.

FRÅGA: {question}
BRAIN REGION: {brain_region}
FACIT-KONCEPT: {expected_concepts}
FACIT-HINT: {why_hint}

SVAR: {answer}

Bedöm på skala 0-4:
  4 = Korrekt, specifik och med rätt resonemang — nämner rätt koncept med korrekt relation
  3 = Korrekt men något ofullständigt — saknar en detalj eller är vag på en punkt
  2 = Delvis korrekt — rätt riktning men saknar specifik kunskap eller relation
  1 = Vagt korrekt — allmän riktning men ingen specifik kunskap
  0 = Fel eller hallucination — felaktiga påståenden

Bedöm även:
  - confidence_calibrated: 1 om svarets konfidensnivå matchar faktisk korrekthet, 0 om över- eller undersäker
  - hallucination: 1 om svaret innehåller påståenden som direkt motsäger facit, 0 om inga
  - cross_domain: 1 om svaret kopplar ihop kunskap från andra domäner än den frågade om

Svara ENDAST med ett JSON-objekt:
{{"score": <0-4>, "confidence_calibrated": <0-1>, "hallucination": <0-1>, "cross_domain": <0-1>, "reason": "<en mening>"}}"""


# ── Graph context extraction ──────────────────────────────────────────────

def get_nous_context(question: str, field=None, max_hops: int = 2) -> str:
    """Extract relevant graph context for a question.

    Uses node_context_for_query() to get relevant subgraph, then enriches
    with evidence scores, uncertainty markers, and brain region classification.
    Falls back to keyword matching if node_context_for_query returns nothing.
    """
    if field is None:
        try:
            from nouse.field.surface import FieldSurface
            field = FieldSurface(read_only=True)
        except Exception:
            return "[Kunskapsminne ej tillgängligt]"

    # Primary path: use node_context_for_query for structured context
    try:
        nodes = field.node_context_for_query(question)
        if nodes:
            context_lines = []
            for node in nodes[:10]:
                name = node.get("name", "?")
                domain = field.concept_domain(name) or "unknown"

                # Classify brain region
                try:
                    from nouse.field.brain_topology import classify_domain
                    region = classify_domain(domain)
                except Exception:
                    region = "unknown"

                # Add summary if available
                summary = node.get("summary", "")
                if summary:
                    context_lines.append(f"  {name} [{domain}@{region}]: {summary[:120]}")

                # Add claims/relation edges with metadata
                claims = node.get("claims", [])
                for claim in claims[:3]:
                    context_lines.append(f"    - {claim}")

            if context_lines:
                header = f"Relevanta koncept ({len(nodes)} hittade): {', '.join(n.get('name','?') for n in nodes[:5])}"
                return header + "\n" + "\n".join(context_lines[:30])
    except Exception:
        pass  # Fall through to keyword matching

    # Fallback: keyword matching against concept names
    terms = question.lower().replace("?", "").replace(".", "").split()
    stop_words = {"what", "how", "why", "does", "is", "are", "can", "the", "a", "an",
                  "in", "of", "to", "and", "or", "not", "between", "from", "that", "this",
                  "det", "som", "är", "har", "kan", "och", "att", "med", "för", "till", "varför",
                  "hur", "vad", "när", "vilka", "vilket"}
    key_terms = [t for t in terms if t not in stop_words and len(t) > 3]

    matching_concepts = []
    all_concepts = list(field.concepts())

    for concept_dict in all_concepts:
        concept_name = concept_dict["name"]
        concept_lower = concept_name.lower()
        for term in key_terms[:5]:
            if term in concept_lower:
                matching_concepts.append(concept_name)
                break
        if len(matching_concepts) >= 10:
            break

    if not matching_concepts:
        return "[Inga relevanta koncept hittades i kunskapsminnet]"

    # Build context from matching concepts and their outgoing relations
    context_lines = []
    seen_edges = set()

    for concept_name in matching_concepts[:10]:
        domain = field.concept_domain(concept_name) or "unknown"
        try:
            from nouse.field.brain_topology import classify_domain
            region = classify_domain(domain)
        except Exception:
            region = "unknown"

        relations = list(field.out_relations(concept_name))[:5]
        for rel in relations:
            edge_key = (concept_name, rel.get("type", "?"), rel.get("target", "?"))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            ev = f"ev={rel.get('evidence_score', 0):.2f}" if rel.get("evidence_score") else ""
            strength = f"s={rel.get('strength', 0):.2f}" if rel.get("strength") else ""
            assumption = "⚠assumption" if rel.get("assumption_flag") else ""

            meta = " ".join(filter(None, [ev, strength, assumption]))
            context_lines.append(
                f"  {concept_name} —[{rel.get('type', '?')}]→ {rel.get('target', '?')} "
                f"[{domain}@{region}] ({meta})"
            )

    if not context_lines:
        return f"[{len(matching_concepts)} koncept hittades men inga relationer]"

    header = f"Relevanta koncept: {', '.join(matching_concepts[:5])}"
    return header + "\n" + "\n".join(context_lines[:30])


def get_rag_context(question: str, field=None) -> str:
    """Naive RAG: extract raw concept names and domains (no structure/evidence)."""
    if field is None:
        try:
            from nouse.field.surface import FieldSurface
            field = FieldSurface(read_only=True)
        except Exception:
            return "[Kunskapsarkiv ej tillgängligt]"

    # Use node_context_for_query but strip structure (RAG baseline)
    try:
        nodes = field.node_context_for_query(question)
        if nodes:
            # Flat list: just concept names and domains, no evidence/structure
            items = []
            for node in nodes[:15]:
                name = node.get("name", "?")
                domain = field.concept_domain(name) or "unknown"
                items.append(f"{name} [{domain}]")
            if items:
                return "Relevanta ämnen: " + ", ".join(items)
    except Exception:
        pass

    # Fallback: keyword matching
    terms = question.lower().replace("?", "").replace(".", "").split()
    stop_words = {"what", "how", "why", "does", "is", "are", "can", "the", "a", "an",
                  "in", "of", "to", "and", "or", "not", "between", "from", "that", "this"}
    key_terms = [t for t in terms if t not in stop_words and len(t) > 3]

    matching = []
    for concept_dict in list(field.concepts()):
        concept_name = concept_dict["name"]
        concept_lower = concept_name.lower()
        for term in key_terms[:5]:
            if term in concept_lower:
                domain = field.concept_domain(concept_name) or "unknown"
                matching.append(f"{concept_name} [{domain}]")
                break
        if len(matching) >= 15:
            break

    if not matching:
        return "[Inga relevanta avsnitt hittades]"

    return "Relevanta ämnen: " + ", ".join(matching[:15])


# ── Benchmark runner ──────────────────────────────────────────────────────

async def run_benchmark(
    model: str,
    questions: list[dict],
    conditions: list[str],
    n: int = 0,
    judge_model: str = "",
       output_path: str = "",
    field=None,
):
    """Run the reasoning benchmark across specified conditions."""

    if not judge_model:
        judge_model = model

    if n > 0:
        questions = questions[:n]

    results = {"timestamp": datetime.utcnow().isoformat(), "model": model,
               "judge_model": judge_model, "conditions": {}, "metrics": {}}

    for condition in conditions:
        print(f"\n{'='*60}")
        print(f"  Running condition: {condition}")
        print(f"  Questions: {len(questions)}")
        print(f"{'='*60}")

        condition_results = []

        for i, q in enumerate(questions):
            qid = q["id"]
            region = q.get("brain_region", "unknown")
            question = q["question"]
            expected = q.get("expected_concepts", [])
            why_hint = q.get("why_hint", "")

            # Build system prompt based on condition
            if condition == "bare":
                system = SYSTEM_BASELINE
                user = question
            elif condition == "rag":
                context = get_rag_context(question, field)
                system = SYSTEM_RAG.format(context=context)
                user = question
            elif condition == "nous":
                context = get_nous_context(question, field)
                system = SYSTEM_NOUSE_GROUNDED.format(context=context)
                user = question
            else:
                raise ValueError(f"Unknown condition: {condition}")

            # Get answer
            answer = await call_llm(None, model, system, user, timeout=90.0)

            # Judge the answer
            judge_prompt = SYSTEM_JUDGE_REASONING.format(
                question=question,
                brain_region=region,
                expected_concepts=", ".join(expected),
                why_hint=why_hint,
                answer=answer,
            )

            judge_response = await call_llm(None, judge_model, "You are an objective judge.",
                                           judge_prompt, timeout=60.0)

            # Parse judge response
            judge_data = {"score": 0, "confidence_calibrated": 0,
                         "hallucination": 0, "cross_domain": 0, "reason": ""}
            try:
                # Strip markdown code fences if present
                cleaned = judge_response.strip()
                if cleaned.startswith("```"):
                    first_nl = cleaned.find("\n")
                    if first_nl >= 0:
                        cleaned = cleaned[first_nl + 1:]
                    if cleaned.endswith("```"):
                        cleaned = cleaned[:-3].strip()
                judge_data.update(json.loads(cleaned))
            except (json.JSONDecodeError, AttributeError):
                pass

            result = {
                "id": qid,
                "question": question,
                "brain_region": region,
                "condition": condition,
                "answer": answer[:500],  # truncate for storage
                "judge": judge_data,
                "requires_cross_domain": q.get("requires_cross_domain", False),
                "difficulty": q.get("difficulty", "medium"),
                "confidence_type": q.get("confidence_type", "factual"),
            }
            condition_results.append(result)

            print(f"  [{i+1}/{len(questions)}] {qid} ({region[:8]:8s}) "
                  f"score={judge_data['score']} "
                  f"halluc={judge_data['hallucination']} "
                  f"xdom={judge_data['cross_domain']}")

        results["conditions"][condition] = condition_results

    # ── Compute metrics ────────────────────────────────────────────────

    for condition, cond_results in results["conditions"].items():
        scores = [r["judge"]["score"] for r in cond_results]
        hallucinations = [r["judge"]["hallucination"] for r in cond_results]
        cross_domains = [r["judge"]["cross_domain"] for r in cond_results if r["requires_cross_domain"]]
        calibrations = [r["judge"]["confidence_calibrated"] for r in cond_results]

        # Per-region breakdown
        region_scores = defaultdict(list)
        for r in cond_results:
            region_scores[r["brain_region"]].append(r["judge"]["score"])

        results["metrics"][condition] = {
            "accuracy_mean": sum(scores) / max(1, len(scores)),
            "accuracy_3plus": sum(1 for s in scores if s >= 3) / max(1, len(scores)),
            "hallucination_rate": sum(hallucinations) / max(1, len(hallucinations)),
            "cross_domain_accuracy": (sum(cross_domains) / max(1, len(cross_domains))) if cross_domains else 0,
            "calibration_mean": sum(calibrations) / max(1, len(calibrations)),
            "n_questions": len(scores),
            "region_breakdown": {
                region: {
                    "mean": sum(rs) / len(rs),
                    "n": len(rs),
                }
                for region, rs in region_scores.items()
            }
        }

    # ── Save results ────────────────────────────────────────────────────

    if not output_path:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = str(Path(__file__).parent / "results" / f"reasoning_{ts}.json")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ── Print summary ─────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print(f"  REASONING BENCHMARK RESULTS")
    print(f"{'='*70}")

    for condition, metrics in results["metrics"].items():
        print(f"\n  {condition.upper()}:")
        print(f"    Accuracy (mean score 0-4):    {metrics['accuracy_mean']:.2f}")
        print(f"    Accuracy (score ≥ 3):         {metrics['accuracy_3plus']:.0%}")
        print(f"    Hallucination rate:            {metrics['hallucination_rate']:.0%}")
        print(f"    Cross-domain accuracy:         {metrics['cross_domain_accuracy']:.0%}")
        print(f"    Calibration:                  {metrics['calibration_mean']:.0%}")
        print(f"    N questions:                   {metrics['n_questions']}")

        print(f"    Per-region:")
        for region, rm in sorted(metrics["region_breakdown"].items()):
            print(f"      {region:15s} mean={rm['mean']:.2f}  n={rm['n']}")

    # Comparison
    if len(results["metrics"]) >= 2:
        cond_list = list(results["metrics"].keys())
        m1 = results["metrics"][cond_list[0]]
        m2 = results["metrics"][cond_list[1]]
        delta = m2["accuracy_mean"] - m1["accuracy_mean"]
        print(f"\n  DELTA ({cond_list[1]} vs {cond_list[0]}): {delta:+.2f} points")
        if abs(delta) >= 0.25:
            winner = cond_list[1] if delta > 0 else cond_list[0]
            print(f"  → {winner} is measurably better ({delta:+.2f} score points)")
        else:
            print(f"  → No significant difference (|Δ| < 0.25)")

    print(f"\n  Results saved to: {output_path}")

    return results


def load_questions(path: str = "") -> list[dict]:
    """Load question bank from JSON file."""
    if not path:
        # Load both standard and reasoning questions
        base_dir = Path(__file__).parent
        questions = []

        # Reasoning questions (brain-region tagged)
        rq_path = base_dir / "reasoning_questions.json"
        if rq_path.exists():
            with open(rq_path) as f:
                questions.extend(json.load(f))

        # Original quantum biology questions (add brain_region)
        qb_path = base_dir / "questions.json"
        if qb_path.exists():
            with open(qb_path) as f:
                qb = json.load(f)
                for q in qb:
                    q.setdefault("brain_region", "occipital")  # default for QB questions
                    q.setdefault("requires_cross_domain", False)
                    q.setdefault("confidence_type", "factual")
                questions.extend(qb)

        return questions

    with open(path) as f:
        return json.load(f)


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Reasoning benchmark: Bare vs RAG vs Nous")
    parser.add_argument("--model", default="minimax-m2.7:cloud", help="LLM model to test")
    parser.add_argument("--judge", default="", help="Judge model (default: same as --model)")
    parser.add_argument("--conditions", nargs="+", default=["bare", "rag", "nous"],
                       choices=["bare", "rag", "nous"], help="Conditions to run")
    parser.add_argument("-n", type=int, default=0, help="Number of questions (0=all)")
    parser.add_argument("--questions", default="", help="Path to questions JSON")
    parser.add_argument("--output", default="", help="Output path for results JSON")
    parser.add_argument("--dry-run", action="store_true", help="Print questions without running LLM")
    args = parser.parse_args()

    questions = load_questions(args.questions)

    if args.dry_run:
        print(f"Questions loaded: {len(questions)}")
        for q in questions[:5]:
            print(f"  {q['id']}: [{q.get('brain_region', '?')}] {q['question'][:80]}...")
        print(f"  ... ({len(questions)} total)")
        return

    # Try to load field for Nous context
    field = None
    if "nous" in args.conditions:
        try:
            from nouse.field.surface import FieldSurface
            field = FieldSurface(read_only=True)
            n_concepts = len(list(field.concepts()))
            print(f"  Loaded field: {n_concepts} concepts")
        except Exception as e:
            print(f"  Warning: Could not load field ({e}). Nous condition will use fallback.")

    asyncio.run(run_benchmark(
        model=args.model,
        questions=questions,
        conditions=args.conditions,
        n=args.n,
        judge_model=args.judge or args.model,
        output_path=args.output,
        field=field,
    ))


if __name__ == "__main__":
    main()