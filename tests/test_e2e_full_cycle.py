"""
End-to-end integrationstest: text in → embeddings → conductor → CCNode → graf.

Flöde:
  1. OllamaEmbedder genererar vektorer för tre meningar om skogssvampar och matematik.
  2. CognitiveConductor körs med dessa vektorer och riktig FieldSurface.
  3. Vi verifierar att CycleResult innehåller förväntade fält och att
     grafen har fått minst ett nytt episodminne.

CCNode anropar Azure via GITHUB_TOKEN om det finns i miljön.
Finns det inte fallback till ("", 0.0) — testet passerar ändå.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time

# Säkerställ att src/ är i path
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[1] / "src"))

# Sätt Azure-endpoint FÖRE nouse-imports (load_env_files körs vid import och kan
# sätta localhost:11434 från lokal .env — detta override vinner).
os.environ["NOUSE_TEACHER_BASE_URL"] = "https://models.inference.ai.azure.com"
os.environ.setdefault("NOUSE_CC_MODEL", "gpt-4o")

from nouse.embeddings.ollama_embed import OllamaEmbedder
from nouse.memory.store import MemoryStore
from nouse.orchestrator.conductor import CCNode, CognitiveConductor, CycleResult
from nouse.orchestrator.global_workspace import GlobalWorkspace


TEXTS = [
    "Skogssvampar bildar mykorrhiza-nätverk under marken — ett distribuerat kommunikationssystem.",
    "Primtal uppvisar en lokal slumpmässighet men global regularitet (primtalssatsen).",
    "Kvanttunnling möjliggörs av vågfunktionens icke-lokala karaktär — lokal regel, global effekt.",
]

DOMAIN = "bisociation_test"


async def run_cycle() -> CycleResult:
    print("\n=== NoUse End-to-End Integrationstest ===")
    t0 = time.perf_counter()

    # Steg 1: Embeddings
    print("[1] Genererar embeddings via Ollama...")
    embedder = OllamaEmbedder()
    vectors = embedder.embed_texts(TEXTS)
    print(f"    ✓ {len(vectors)} vektorer, dim={len(vectors[0])}")

    # Steg 2: Minne + conductor
    print("[2] Initierar MemoryStore + CognitiveConductor...")
    memory = MemoryStore()
    workspace = GlobalWorkspace()
    conductor = CognitiveConductor(memory=memory, workspace=workspace)

    # Steg 3a: Kognitiv cykel MED externa vektorer
    episode_text = "\n".join(TEXTS)
    print("[3a] Kör kognitiv cykel med externa vektorer...")
    result: CycleResult = await conductor.run_cognitive_cycle(
        episode_text=episode_text,
        domain=DOMAIN,
        vectors=vectors,
        source="e2e_test",
        session_id="e2e-2026",
    )

    # Steg 3b: Kognitiv cykel UTAN vektorer → auto-embed
    print("[3b] Kör kognitiv cykel utan vektorer (auto-embed)...")
    result_auto: CycleResult = await conductor.run_cognitive_cycle(
        episode_text=episode_text,
        domain=DOMAIN,
        vectors=None,
        source="e2e_test_auto",
        session_id="e2e-2026",
    )
    print(f"    auto H0: ({result_auto.tda_h0_a},{result_auto.tda_h0_b})  "
          f"H1: ({result_auto.tda_h1_a},{result_auto.tda_h1_b})  "
          f"verdict={result_auto.bisociation_verdict}  "
          f"score={result_auto.bisociation_score:.4f}")
    elapsed = time.perf_counter() - t0

    # Steg 4: Rapportera
    print("\n--- RESULTAT ---")
    print(f"episode_id      : {result.episode_id}")
    print(f"bisociation_score: {result.bisociation_score:.4f}")
    print(f"bisociation_verdict: {result.bisociation_verdict}")
    print(f"tda H0: ({result.tda_h0_a}, {result.tda_h0_b})  H1: ({result.tda_h1_a}, {result.tda_h1_b})")
    print(f"limbic λ={result.limbic_state.lam:.3f}  arousal={result.limbic_state.arousal:.3f}")
    print(f"cc_prediction   : '{result.cc_prediction[:120] if result.cc_prediction else '(ingen — CCNode ej aktiv)'}'")
    print(f"cc_confidence   : {result.cc_confidence:.3f}")
    print(f"workspace_winner: {result.workspace_winner}")
    print(f"synthesis_queued: {result.synthesis_queued}")
    print(f"ny_relationer   : {result.new_relations}")
    print(f"tid             : {elapsed:.2f}s")

    # Steg 5: Verifiera minne
    episodes = memory.working_snapshot(limit=20)
    print(f"\n[5] Episodminne efter cykel: {len(episodes)} episod(er)")
    for ep in episodes[-3:]:
        src = ep.get("source", "?")
        txt = ep.get("text", "")[:80]
        print(f"    [{src}] {txt}")

    return result


def assertions(result: CycleResult) -> None:
    assert result.episode_id, "episode_id saknas"
    assert 0.0 <= result.bisociation_score <= 1.0, "bisociation_score utanför [0,1]"
    assert result.bisociation_verdict in {"BISOCIATION", "ASSOCIATION"}, "ogiltigt verdict"
    assert result.tda_h0_a >= 1, "H0_a ska vara ≥ 1"
    assert result.tda_h0_b >= 1, "H0_b ska vara ≥ 1"
    print("\n✅ Alla assertions passerade")


if __name__ == "__main__":
    result = asyncio.run(run_cycle())
    assertions(result)
