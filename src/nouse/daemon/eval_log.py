"""
daemon.eval_log — Evalving-harness för longitudinell mätning
=============================================================
Loggar baseline-mätvärden varje NightRun-cykel till eval_log.jsonl.

Baseline-mätvärden (P5.1):
  - crystallization_rate:  % av kanter med w > 0.55
  - evidence_quality:       medel-evidence_score på kristalliserade kanter
  - gap_map_shrink_rate:    förändring i antal noder med hög osäkerhet

Evalving-loop (P5.2):
  - Läs senaste 10 eval-poster
  - Beräkna trend per mätvärde (stigande/sjunkande/stabil)
  - Om mätvärde sjunker 3 cykler i rad: generera ändringsförslag
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nouse.config.paths import path_from_env

_log = logging.getLogger("nouse.eval_log")

EVAL_LOG_PATH = path_from_env("NOUSE_EVAL_LOG", "eval_log.jsonl")

# ── Skrivning ──────────────────────────────────────────────────────────────────

def write_eval_entry(
    *,
    cycle: int,
    crystallization_rate: float,
    evidence_quality: float,
    gap_map_shrink_rate: float = 0.0,
    contradiction_catch_rate: float = 0.0,
    contradiction_caught: int = 0,
    contradiction_acted_on: int = 0,
    graph_concepts: int = 0,
    graph_relations: int = 0,
    extra: dict[str, Any] | None = None,
) -> Path:
    """
    Skriv en eval-post till eval_log.jsonl.

    Returnerar sökvägen till loggfilen.
    """
    EVAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "cycle": int(cycle),
        "ts": datetime.now(timezone.utc).isoformat(),
        "crystallization_rate": float(crystallization_rate),
        "evidence_quality": float(evidence_quality),
        "gap_map_shrink_rate": float(gap_map_shrink_rate),
        "contradiction_catch_rate": float(contradiction_catch_rate),
        "contradiction_caught": int(contradiction_caught),
        "contradiction_acted_on": int(contradiction_acted_on),
        "graph_concepts": int(graph_concepts),
        "graph_relations": int(graph_relations),
    }
    if extra:
        entry.update(extra)

    with EVAL_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")

    _log.info(
        "Eval entry: cycle=%d crystallization=%.1f%% evidence_quality=%.3f "
        "contradiction_catch_rate=%.3f gap_shrink=%.3f",
        cycle, crystallization_rate * 100, evidence_quality,
        contradiction_catch_rate, gap_map_shrink_rate,
    )
    return EVAL_LOG_PATH


# ── Läsning ────────────────────────────────────────────────────────────────────

def read_eval_entries(limit: int = 10) -> list[dict[str, Any]]:
    """Läs de senaste N eval-poster från loggfilen."""
    if not EVAL_LOG_PATH.exists():
        return []
    try:
        lines = EVAL_LOG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return []

    entries = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            if isinstance(entry, dict):
                entries.append(entry)
        except json.JSONDecodeError:
            continue
        if len(entries) >= limit:
            break
    return list(reversed(entries))


# ── Trendanalys ────────────────────────────────────────────────────────────────

TREND_STABLE = "stable"
TREND_RISING = "rising"
TREND_FALLING = "falling"


def compute_trend(entries: list[dict[str, Any]], key: str) -> str:
    """
    Beräkna trend för ett mätvärde över de senaste posterna.

    Returnerar:
      - 'rising'  om genomsnittet av de senaste 3 > genomsnittet av de föregående 3
      - 'falling' om genomsnittet av de senaste 3 < genomsnittet av de föregående 3
      - 'stable'  annars
    """
    values = [float(e.get(key, 0.0) or 0.0) for e in entries if key in e]
    if len(values) < 4:
        return TREND_STABLE

    recent = values[-3:]
    previous = values[-6:-3] if len(values) >= 6 else values[:-3]

    if not previous:
        return TREND_STABLE

    recent_avg = sum(recent) / len(recent)
    previous_avg = sum(previous) / len(previous)

    threshold = 0.01  # 1% förändring krävs för trend
    if recent_avg > previous_avg + threshold:
        return TREND_RISING
    if recent_avg < previous_avg - threshold:
        return TREND_FALLING
    return TREND_STABLE


def detect_decline(entries: list[dict[str, Any]], key: str, min_cycles: int = 3) -> bool:
    """
    Detektera om ett mätvärde har sjunkit min_cycles cykler i rad.
    """
    values = [float(e.get(key, 0.0) or 0.0) for e in entries if key in e]
    if len(values) < min_cycles + 1:
        return False

    recent = values[-(min_cycles + 1):]
    declines = 0
    for i in range(1, len(recent)):
        if recent[i] < recent[i - 1]:
            declines += 1
    return declines >= min_cycles


def generate_policy_suggestion(
    entries: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """
    Om något mätvärde sjunker 3 cykler i rad, generera ändringsförslag.

    Returnerar ett förslag-dict eller None.
    """
    metrics = ["crystallization_rate", "evidence_quality", "gap_map_shrink_rate"]
    suggestions: dict[str, Any] = {}

    for key in metrics:
        trend = compute_trend(entries, key)
        if detect_decline(entries, key, min_cycles=3):
            if key == "crystallization_rate":
                suggestions["evidence_floor"] = -0.05   # sänk tröskeln → fler granskas
                suggestions["reason_crystallization"] = "crystallization_rate sjunker"
            elif key == "evidence_quality":
                suggestions["confirmation_delta"] = 0.02  # öka bekräftelse-boost
                suggestions["reason_evidence"] = "evidence_quality sjunker"
            elif key == "gap_map_shrink_rate":
                suggestions["curiosity_priority"] = "high"  # prioritera kunskapsgap
                suggestions["reason_gap"] = "gap_map_shrink_rate sjunker"

    if not suggestions:
        return None

    suggestions["ts"] = datetime.now(timezone.utc).isoformat()
    suggestions["source"] = "eval_log_auto"
    return suggestions