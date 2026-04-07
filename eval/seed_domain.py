"""
eval/seed_domain.py — Seed NoUse graph with LLM knowledge for a domain.

Seeds the graph before running eval/run_eval.py so that NoUse has
domain knowledge to ground against. Uses the LLM's own trained weights
as a first-order prior (hypothesis-tier).

Usage:
    python eval/seed_domain.py --domain climate_science --model cerebras/llama3.1-8b
    python eval/seed_domain.py --domain medicine --model groq/llama-3.3-70b-versatile
    python eval/seed_domain.py --questions eval/domains/climate_science.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

SEED_SYSTEM = """\
You are a knowledge distiller. Extract structured knowledge about the given domain.
State facts as concrete relations: "X is Y", "X relates to Y", "X causes Z".
Cover key concepts, subdomains, and their connections. Be specific. Max 400 words."""

PROVIDERS = {
    "cerebras/": ("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY"),
    "groq/":     ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "ollama/":   (os.getenv("OLLAMA_HOST", "http://localhost:11434") + "/v1", None),
}


def _resolve_provider(model: str) -> tuple[str, str, dict]:
    for prefix, (base_url, key_env) in PROVIDERS.items():
        if model.startswith(prefix):
            real_model = model[len(prefix):]
            headers = {}
            if key_env:
                api_key = os.getenv(key_env, "")
                headers["Authorization"] = f"Bearer {api_key}"
            return base_url, real_model, headers
    return "http://localhost:11434/v1", model, {}


async def _ask_llm(prompt: str, model: str) -> str:
    import httpx
    base_url, real_model, headers = _resolve_provider(model)
    payload = {
        "model": real_model,
        "messages": [
            {"role": "system", "content": SEED_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 600,
    }
    async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
        r = await client.post(f"{base_url}/chat/completions", json=payload)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


def _domain_prompts(domain: str) -> list[str]:
    """Generate seeding prompts for a domain and its key subdomains."""
    return [
        f"Explain the domain '{domain}': its main concepts, subdomains, and how they connect to each other.",
        f"List the most important facts and relations in '{domain}'. Focus on measurable, verifiable claims.",
        f"What are the key mechanisms and cause-effect relationships in '{domain}'?",
    ]


def _prompts_from_questions(questions: list[dict]) -> list[str]:
    """Build seeding prompts that cover the topics in the question bank."""
    domains = set(q.get("domain", "") for q in questions)
    # Sample answers as learning material
    samples = [f"Q: {q['question']}\nA: {q['answer']}" for q in questions[:20]]
    prompts = []
    for domain in domains:
        if domain:
            prompts.extend(_domain_prompts(domain))
    prompts.append(
        "Here are key facts from this domain. Extract and explain the underlying relations:\n\n"
        + "\n\n".join(samples)
    )
    return prompts


async def seed(
    *,
    domain: str | None,
    questions_path: Path | None,
    model: str,
    brain,
) -> int:
    """Seed brain with LLM knowledge. Returns number of prompts processed."""
    if questions_path:
        questions = json.loads(questions_path.read_text())
        prompts = _prompts_from_questions(questions)
        label = questions_path.stem
    else:
        prompts = _domain_prompts(domain)
        label = domain

    print(f"\n Seeding domain '{label}' using {model}")
    print(f" {len(prompts)} seed prompts\n")

    stats_before = brain._field.stats() if hasattr(brain, "_field") else {}
    processed = 0

    for i, prompt in enumerate(prompts, 1):
        print(f"  [{i}/{len(prompts)}] {prompt[:60]}...")
        t0 = time.monotonic()
        try:
            response = await _ask_llm(prompt, model)
            brain.learn(prompt, response, source=f"domain_seed:{label}")
            elapsed = time.monotonic() - t0
            print(f"         {elapsed:.1f}s — {len(response)} chars learned")
            processed += 1
        except Exception as e:
            print(f"         ERROR: {e}")

    stats_after = brain._field.stats() if hasattr(brain, "_field") else {}

    print(f"\n Done. {processed}/{len(prompts)} prompts processed.")
    if stats_before and stats_after:
        n_before = stats_before.get("nodes", 0)
        e_before = stats_before.get("edges", 0)
        n_after  = stats_after.get("nodes", 0)
        e_after  = stats_after.get("edges", 0)
        print(f" Graph: {n_before}→{n_after} nodes, {e_before}→{e_after} edges")

    return processed


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed NoUse graph with domain knowledge")
    parser.add_argument("--domain", help="Domain name to seed (e.g. climate_science)")
    parser.add_argument("--questions", help="Path to question bank JSON to seed from")
    parser.add_argument("--model", default="cerebras/llama3.1-8b", help="LLM model to use")
    args = parser.parse_args()

    if not args.domain and not args.questions:
        parser.error("Provide --domain or --questions")

    from nouse.ollama_client.client import load_env_files
    import nouse

    load_env_files()
    brain = nouse.attach()  # read-write

    questions_path = Path(args.questions) if args.questions else None

    asyncio.run(seed(
        domain=args.domain,
        questions_path=questions_path,
        model=args.model,
        brain=brain,
    ))


if __name__ == "__main__":
    main()
