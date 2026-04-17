"""
nouse.embeddings.model — Enkel embed-API för hela systemet.

Exponerar två funktioner som resten av systemet (conductor, bridge_finder m.fl.)
ska använda direkt — ingen OllamaEmbedder-instansiering utanför denna modul.

    embed(text)         → list[float]
    embed_batch(texts)  → list[list[float]]

Modell och host styrs av miljövariabler (NOUSE_EMBED_MODEL, NOUSE_OLLAMA_HOST).
Vid fel returneras tomma listor — anroparen hanterar fallback.
"""
from __future__ import annotations

import logging
from typing import Sequence

log = logging.getLogger("nouse.embeddings.model")

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from nouse.embeddings.ollama_embed import OllamaEmbedder
        _embedder = OllamaEmbedder()
    return _embedder


def embed(text: str) -> list[float]:
    """Embed en enstaka text. Returnerar [] vid fel."""
    if not text or not text.strip():
        return []
    try:
        vecs = _get_embedder().embed_texts([text])
        return vecs[0] if vecs else []
    except Exception as exc:
        log.warning("embed() misslyckades: %s", exc)
        return []


def embed_batch(texts: Sequence[str]) -> list[list[float]]:
    """Embed en lista texter. Returnerar [] vid fel."""
    clean = [t for t in texts if t and t.strip()]
    if not clean:
        return []
    try:
        return _get_embedder().embed_texts(clean)
    except Exception as exc:
        log.warning("embed_batch() misslyckades: %s", exc)
        return []
