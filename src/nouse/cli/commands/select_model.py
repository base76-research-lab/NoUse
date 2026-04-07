"""
select_model.py — CLI för att välja aktiv LLM-modell i NoUse

Användning:
  python -m nouse.cli.commands.select_model
"""
import asyncio
from nouse.llm.autodiscover import detect_providers
from nouse.llm.model_router import _save_state, _load_state

import sys

def print_models(providers):
    print("\nTillgängliga modeller:")
    all_models = []
    for i, p in enumerate(providers):
        print(f"[{i+1}] {p.label()} ({p.kind})")
        for m in p.available_models:
            idx = len(all_models) + 1
            print(f"   {idx}. {m}")
            all_models.append((p, m))
    return all_models

async def main():
    providers = await detect_providers()
    if not providers:
        print("Inga modeller hittades.")
        sys.exit(1)
    all_models = print_models(providers)
    print("\nVälj modellnummer att aktivera (eller 0 för att avbryta): ", end="")
    try:
        choice = int(input().strip())
    except Exception:
        print("Ogiltigt val.")
        sys.exit(1)
    if choice < 1 or choice > len(all_models):
        print("Avbrutet.")
        sys.exit(0)
    provider, model = all_models[choice-1]
    # Spara till router state
    state = _load_state()
    state["active_model"] = {"provider": provider.kind, "model": model}
    _save_state(state)
    print(f"Aktiv modell satt till: {model} ({provider.label()})")

if __name__ == "__main__":
    asyncio.run(main())
