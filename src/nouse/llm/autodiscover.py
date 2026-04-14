"""
LLM Autodiscovery — Hitta och konfigurera LLM-providers automatiskt
====================================================================

Probar vad som finns tillgängligt i denna ordning:

  1. Ollama (lokal)         — http://localhost:11434
  2. LM Studio (lokal)      — http://localhost:1234
  3. GitHub Copilot          — GITHUB_TOKEN + api.githubcopilot.com
  4. Anthropic (Claude)      — ANTHROPIC_API_KEY
  5. OpenAI                  — OPENAI_API_KEY / NOUSE_OPENAI_API_KEY
  6. Groq                    — GROQ_API_KEY
  7. OpenRouter               — OPENROUTER_API_KEY
  8. Anpassad endpoint        — NOUSE_OPENAI_BASE_URL + NOUSE_OPENAI_API_KEY

Resultatet skrivs till model_policy.json och kan användas direkt av
nouse-daemonen utan omstart.

Anropas via:
  b76 llm setup          — interaktiv wizard
  b76 llm detect         — visa vad som hittades (utan att ändra)
  b76 llm status         — visa aktiv konfiguration

Programmatiskt:
  from nouse.llm.autodiscover import detect_providers, apply_best
  providers = detect_providers()   # lista av DiscoveredProvider
  apply_best(providers)            # skriv model_policy.json
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

import httpx

_log = logging.getLogger("nouse.llm.autodiscover")

ProviderKind = Literal[
    "ollama", "lm_studio", "copilot", "anthropic",
    "openai", "groq", "openrouter", "custom",
]


# ── Kända endpoints ───────────────────────────────────────────────────────────

_PROBE_TIMEOUT = 3.0  # sekunder per provider-probe

_PROVIDER_PROBES: list[tuple[ProviderKind, str, str]] = [
    # (kind,          probe_url,                            models_path)
    ("ollama",       "http://localhost:11434",              "/api/tags"),
    ("lm_studio",    "http://localhost:1234",               "/v1/models"),
]

_API_PROVIDERS: dict[ProviderKind, dict] = {
    "copilot": {
        "env_token":  "GITHUB_TOKEN",
        # GitHub Models API (ersätter deprecated copilot_internal/v2/token)
        "base_url":   "https://models.inference.ai.azure.com",
        "models_url": "https://models.inference.ai.azure.com/models",
        "default_models": {
            "chat":       "gpt-4o",
            "agent":      "Meta-Llama-3.1-405B-Instruct",
            "extract":    "gpt-4o-mini",
            "synthesize": "Meta-Llama-3.1-405B-Instruct",
            "curiosity":  "gpt-4o-mini",
            "teacher":    "gpt-4o",
            "decomposition": "gpt-4o",
        },
    },
    "anthropic": {
        "env_token":  "ANTHROPIC_API_KEY",
        "base_url":   "https://api.anthropic.com/v1",
        "models_url": None,
        "default_models": {
            "chat":       "claude-sonnet-4-5",
            "agent":      "claude-sonnet-4-5",
            "extract":    "claude-haiku-4-5",
            "synthesize": "claude-sonnet-4-5",
            "curiosity":  "claude-haiku-4-5",
        },
    },
    "openai": {
        "env_token":  "OPENAI_API_KEY",
        "env_token2": "NOUSE_OPENAI_API_KEY",
        "base_url":   "https://api.openai.com/v1",
        "models_url": "https://api.openai.com/v1/models",
        "default_models": {
            "chat":       "gpt-4o",
            "agent":      "gpt-4o",
            "extract":    "gpt-4o-mini",
            "synthesize": "gpt-4o",
            "curiosity":  "gpt-4o-mini",
        },
    },
    "groq": {
        "env_token":  "GROQ_API_KEY",
        "base_url":   "https://api.groq.com/openai/v1",
        "models_url": "https://api.groq.com/openai/v1/models",
        "default_models": {
            "chat":       "llama-3.3-70b-versatile",
            "agent":      "llama-3.3-70b-versatile",
            "extract":    "llama-3.1-8b-instant",
            "synthesize": "llama-3.3-70b-versatile",
            "curiosity":  "llama-3.1-8b-instant",
        },
    },
    "openrouter": {
        "env_token":  "OPENROUTER_API_KEY",
        "base_url":   "https://openrouter.ai/api/v1",
        "models_url": "https://openrouter.ai/api/v1/models",
        "default_models": {
            "chat":       "anthropic/claude-3.5-sonnet",
            "agent":      "anthropic/claude-3.5-sonnet",
            "extract":    "meta-llama/llama-3.1-8b-instruct",
            "synthesize": "anthropic/claude-3.5-sonnet",
            "curiosity":  "meta-llama/llama-3.1-8b-instruct",
        },
    },
}

# Prioritetsordning: lägre = föredras
_PRIORITY: dict[ProviderKind, int] = {
    "ollama":     1,   # lokal alltid först
    "lm_studio":  2,
    "copilot":    3,
    "anthropic":  4,
    "openai":     5,
    "groq":       6,
    "openrouter": 7,
    "custom":     8,
}


# ── Datamodeller ──────────────────────────────────────────────────────────────

@dataclass
class DiscoveredProvider:
    kind:           ProviderKind
    base_url:       str
    api_key:        str           # tom om lokal
    available_models: list[str]  # faktiska modeller från API
    default_models:   dict[str, str]  # workload → modell
    latency_ms:     float = 0.0
    note:           str   = ""

    @property
    def priority(self) -> int:
        return _PRIORITY.get(self.kind, 99)

    def label(self) -> str:
        names = {
            "ollama":     "Ollama (lokal)",
            "lm_studio":  "LM Studio (lokal)",
            "copilot":    "GitHub Copilot",
            "anthropic":  "Anthropic (Claude)",
            "openai":     "OpenAI",
            "groq":       "Groq (snabb API)",
            "openrouter": "OpenRouter (multi-LLM)",
            "custom":     "Anpassad endpoint",
        }
        return names.get(self.kind, self.kind)


# ── Probe-funktioner ──────────────────────────────────────────────────────────

async def _probe_local(kind: ProviderKind, base: str, path: str) -> DiscoveredProvider | None:
    url = f"{base}{path}"
    try:
        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT) as client:
            r = await client.get(url)
        latency = (time.monotonic() - t0) * 1000

        if r.status_code != 200:
            return None

        models: list[str] = []
        try:
            data = r.json()
            if kind == "ollama":
                models = [m["name"] for m in data.get("models", [])]
            else:  # lm_studio / openai-format
                models = [m["id"] for m in data.get("data", [])]
        except Exception:
            pass

        default_m = models[0] if models else ("llama3" if kind == "ollama" else "local-model")
        defaults  = {wl: default_m for wl in ("chat", "agent", "extract", "synthesize", "curiosity")}

        return DiscoveredProvider(
            kind=kind,
            base_url=base,
            api_key="",
            available_models=models,
            default_models=defaults,
            latency_ms=round(latency, 1),
            note=f"{len(models)} modeller tillgängliga",
        )
    except Exception:
        return None


async def _probe_copilot(token: str) -> DiscoveredProvider | None:
    """Verifiera GitHub Models API-åtkomst via GITHUB_TOKEN."""
    try:
        t0 = time.monotonic()
        cfg = _API_PROVIDERS["copilot"]
        # GitHub Models API — ersätter deprecated copilot_internal/v2/token
        async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT) as client:
            r = await client.get(
                cfg["models_url"],
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/json",
                },
            )
        latency = (time.monotonic() - t0) * 1000
        if r.status_code != 200:
            return None

        # Hämta faktisk modellista från GitHub Models
        try:
            models_data = r.json()
            available = [m.get("name") or m.get("id", "") for m in models_data if m.get("name") or m.get("id")]
        except Exception:
            available = list(cfg["default_models"].values())

        return DiscoveredProvider(
            kind="copilot",
            base_url=cfg["base_url"],
            api_key=token,
            available_models=available or list(cfg["default_models"].values()),
            default_models=cfg["default_models"],
            latency_ms=round(latency, 1),
            note=f"GitHub Models API via GITHUB_TOKEN ({len(available)} modeller)",
        )
    except Exception:
        return None


async def _probe_api(kind: ProviderKind) -> DiscoveredProvider | None:
    cfg = _API_PROVIDERS.get(kind)
    if not cfg:
        return None

    token = (
        os.getenv(cfg.get("env_token", "")) or
        os.getenv(cfg.get("env_token2", ""), "")
    )
    if not token:
        return None

    if kind == "copilot":
        return await _probe_copilot(token)

    # Övriga API-providers: validera token med en enkel models-request
    models_url = cfg.get("models_url")
    models: list[str] = list(cfg["default_models"].values())

    if models_url:
        try:
            t0 = time.monotonic()
            async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT) as client:
                r = await client.get(
                    models_url,
                    headers={"Authorization": f"Bearer {token}"},
                )
            latency = (time.monotonic() - t0) * 1000
            if r.status_code == 200:
                try:
                    data = r.json()
                    fetched = [m["id"] for m in data.get("data", [])]
                    if fetched:
                        models = fetched[:20]
                except Exception:
                    pass
        except Exception:
            return None
    else:
        latency = 0.0

    return DiscoveredProvider(
        kind=kind,
        base_url=cfg["base_url"],
        api_key=token,
        available_models=models,
        default_models=cfg["default_models"],
        latency_ms=round(latency, 1),
        note=f"API-nyckel hittad: {cfg['env_token']}",
    )


async def _probe_custom() -> DiscoveredProvider | None:
    base = os.getenv("NOUSE_OPENAI_BASE_URL", "").rstrip("/")
    key  = os.getenv("NOUSE_OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
    if not base:
        return None

    try:
        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT) as client:
            r = await client.get(
                f"{base}/models",
                headers={"Authorization": f"Bearer {key}"} if key else {},
            )
        latency = (time.monotonic() - t0) * 1000
        models: list[str] = []
        if r.status_code == 200:
            try:
                models = [m["id"] for m in r.json().get("data", [])]
            except Exception:
                pass
        default_m = models[0] if models else "default"
        defaults = {wl: default_m for wl in ("chat", "agent", "extract", "synthesize", "curiosity")}
        return DiscoveredProvider(
            kind="custom",
            base_url=base,
            api_key=key,
            available_models=models,
            default_models=defaults,
            latency_ms=round(latency, 1),
            note=f"NOUSE_OPENAI_BASE_URL={base}",
        )
    except Exception:
        return None


# ── Huvudfunktion ──────────────────────────────────────────────────────────────

async def detect_providers_async() -> list[DiscoveredProvider]:
    """Proba alla providers parallellt. Returnerar sorterad lista (bäst först)."""
    from nouse.config.env import load_env_files
    load_env_files()

    tasks = []
    # Lokala
    for kind, base, path in _PROVIDER_PROBES:
        tasks.append(_probe_local(kind, base, path))
    # API
    for kind in _API_PROVIDERS:
        tasks.append(_probe_api(kind))  # type: ignore[arg-type]
    # Anpassad
    tasks.append(_probe_custom())

    results = await asyncio.gather(*tasks, return_exceptions=True)

    found: list[DiscoveredProvider] = []
    for r in results:
        if isinstance(r, DiscoveredProvider):
            found.append(r)

    found.sort(key=lambda p: (p.priority, p.latency_ms))
    return found


def detect_providers() -> list[DiscoveredProvider]:
    """Synkron wrapper runt detect_providers_async."""
    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop in this thread (normal for FastAPI worker threads).
            return asyncio.run(detect_providers_async())

        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(asyncio.run, detect_providers_async())
                return future.result(timeout=15)

        return loop.run_until_complete(detect_providers_async())
    except Exception as e:
        _log.warning("detect_providers misslyckades: %s", e)
        return []


# ── Applicera bästa provider till model_policy.json ───────────────────────────

def apply_best(
    providers: list[DiscoveredProvider],
    *,
    preferred_kind: ProviderKind | None = None,
) -> DiscoveredProvider | None:
    """
    Välj bästa provider och skriv model_policy.json.
    Om preferred_kind är satt används den om tillgänglig.
    """
    if not providers:
        return None

    chosen = providers[0]
    if preferred_kind:
        for p in providers:
            if p.kind == preferred_kind:
                chosen = p
                break

    _write_policy(chosen)
    _log.info("LLM-provider satt till: %s (%s)", chosen.label(), chosen.base_url)
    return chosen


def _write_policy(provider: DiscoveredProvider) -> None:
    from nouse.llm.policy import MODEL_POLICY_PATH, _normalize_policy

    # Mappa kind → provider-sträng som b76-klienten förstår
    provider_str = {
        "ollama":     "ollama",
        "lm_studio":  "openai_compatible",
        "copilot":    "openai_compatible",
        "anthropic":  "openai_compatible",
        "openai":     "openai_compatible",
        "groq":       "openai_compatible",
        "openrouter": "openai_compatible",
        "custom":     "openai_compatible",
    }.get(provider.kind, "openai_compatible")

    workloads: dict[str, dict] = {}
    for workload, model in provider.default_models.items():
        model_ref = (
            model if provider.kind == "ollama"
            else f"openai_compatible/{model}"
        )
        workloads[workload] = {
            "provider":   provider_str,
            "candidates": [model_ref],
        }

    # Läs ev. befintlig policy och merge
    raw: dict = {}
    if MODEL_POLICY_PATH.exists():
        try:
            raw = json.loads(MODEL_POLICY_PATH.read_text(encoding="utf-8"))
        except Exception:
            raw = {}

    raw["workloads"] = workloads
    raw["_autodiscovered"] = {
        "kind":     provider.kind,
        "base_url": provider.base_url,
        "label":    provider.label(),
    }

    # Sätt env-vars för klienten
    if provider.kind != "ollama":
        _inject_env(provider)

    MODEL_POLICY_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_POLICY_PATH.write_text(
        json.dumps(raw, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _inject_env(provider: DiscoveredProvider) -> None:
    """Sätt env-variabler som b76-klienten läser."""
    os.environ["NOUSE_LLM_PROVIDER"] = "openai_compatible"
    if provider.base_url:
        os.environ["NOUSE_OPENAI_BASE_URL"] = provider.base_url
    if provider.api_key:
        os.environ["NOUSE_OPENAI_API_KEY"] = provider.api_key

    # Spara till ~/.local/share/nouse/.llm_env för nästa session
    env_file = Path.home() / ".local" / "share" / "nouse" / ".llm_env"
    env_file.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f'NOUSE_LLM_PROVIDER=openai_compatible\n',
        f'NOUSE_OPENAI_BASE_URL={provider.base_url}\n',
    ]
    # Skriv INTE api-nyckeln till disk om den kom från env (säkerhet)
    env_file.write_text("".join(lines))
