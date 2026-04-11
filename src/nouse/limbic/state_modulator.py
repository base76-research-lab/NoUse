"""
state_modulator.py — Kognitiv tillståndsmodulering
====================================================
Läser cognitive_states.json och klassificerar aktuellt limbiskt tillstånd.
Returnerar SemanticModulation som styr operativa NoUse-parametrar.

Algoritm: Mjuk klassificering via viktad blandning av top-k närmaste tillstånd
(invers euklidisk distans i DA/NA/ACh-rummet). Kontinuerlig interpolation —
inte binär tillståndsmaskining.

Neurobiologisk grund:
  DA  → belöning, exploration, λ-koefficient
  NA  → arousal, surprise, pruning-aggression
  ACh → fokus/diffus-växling, signal/brus (Hasselmo 2006)

Användning:
    from nouse.limbic.state_modulator import modulate
    from nouse.limbic.signals import load_state

    limbic = load_state()
    mod = modulate(limbic)
    # mod.bisociation_propensity_delta, mod.write_back_gate, mod.flags, ...
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nouse.limbic.signals import LimbicState

log = logging.getLogger("nouse.limbic.modulator")

_STATES_FILE = Path(__file__).parent / "cognitive_states.json"

# ACh i LimbicState är β (attention temperature) ∈ [0.1, 2.0].
# cognitive_states.json lagrar normaliserade koncentrationsnivåer ∈ [0, 1].
# Normalisering: ach_norm = ach_beta / ACH_BETA_MAX
_ACH_BETA_MAX = 2.0

# Antal tillstånd att blanda vid klassificering
_TOP_K = 3


# ── SemanticModulation ────────────────────────────────────────────────────────

@dataclass
class SemanticModulation:
    """
    Operativa moduleringsparametrar deriverade från aktuellt kognitivt tillstånd.
    Produceras av classify() och konsumeras av daemon-loopen, injektlagret och
    LLM-wrappern.
    """

    # Deltas mot NoUse-operationer (adderas till baslinjevärden)
    evidence_threshold_delta:    float = 0.0   # neg = mer permissiv, pos = mer strikt
    bisociation_propensity_delta: float = 0.0  # påverkar λ i F_bisoc
    working_memory_slots_delta:  int   = 0     # justering av antal WM-slots
    crystallization_boost:       float = 0.0   # boost till kristalliseringsdrift
    pruning_aggression_delta:    float = 0.0   # NA-styrd pruning-aggressivitet

    # Kvalitativa gate och bearbetningsläge
    write_back_gate: str = "open"       # open|blocked|minimal|cautious|
                                        # revision_only|consolidation_only|priority_open
    response_mode:   str = "balanced"   # exploratory|corrective|defensive|
                                        # goal_directed|consolidating|deep_processing|
                                        # emergency|optimal|insight_capture|
                                        # conservative|strategy_shift

    # Tillståndsidentitet
    dominant_state:  str              = "balanced"
    active_state_ids: list[str]       = field(default_factory=list)
    blend_weights:   dict[str, float] = field(default_factory=dict)
    state_label:     str              = ""

    # Limbisk kontextsnapshot
    arousal:     float = 0.0
    performance: float = 0.0

    # Flaggor aggregerade från aktiva tillstånd (vikt > _FLAG_THRESHOLD)
    # Möjliga nycklar:
    #   hitl_escalation, nightrun_trigger_hint, immediate_crystallize,
    #   promote_to_semantic, all_writes_require_approval, flag_stuck_patterns,
    #   operator_rescue_hint, all_systems_nominal, flag_for_deepdive,
    #   curiosity_burst_priority, gap_map_weight_boost, noise_filter_boost,
    #   mission_weight_boost, u_delta_on_active_edges
    flags: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Kort textrad för logging och HITL-display."""
        flag_str = " ".join(f"[{k}]" for k in self.flags if isinstance(self.flags[k], bool) and self.flags[k])
        return (
            f"state={self.dominant_state!r} "
            f"ev_δ={self.evidence_threshold_delta:+.2f} "
            f"bisoc_δ={self.bisociation_propensity_delta:+.2f} "
            f"wm_δ={self.working_memory_slots_delta:+d} "
            f"xtal={self.crystallization_boost:+.2f} "
            f"gate={self.write_back_gate} "
            f"mode={self.response_mode}"
            + (f" {flag_str}" if flag_str else "")
        )

    @property
    def is_degraded(self) -> bool:
        """True om tillståndet indikerar signifikant prestationsförsämring."""
        return self.response_mode in ("defensive", "emergency", "conservative")

    @property
    def wants_hitl(self) -> bool:
        """True om tillståndet begär human-in-the-loop."""
        return bool(self.flags.get("hitl_escalation") or self.flags.get("all_writes_require_approval"))

    @property
    def wants_nightrun(self) -> bool:
        """True om tillståndet föreslår NightRun-konsolidering."""
        return bool(self.flags.get("nightrun_trigger_hint"))

    @property
    def wants_insight_capture(self) -> bool:
        """True om ett insiktsmoment bör omedelbart kristalliseras."""
        return bool(self.flags.get("immediate_crystallize"))


# ── Laddning ──────────────────────────────────────────────────────────────────

_CACHED_STATES: list[dict] | None = None


def _get_states() -> list[dict]:
    global _CACHED_STATES
    if _CACHED_STATES is None:
        raw = json.loads(_STATES_FILE.read_text(encoding="utf-8"))
        _CACHED_STATES = raw["states"]
        log.debug("Laddade %d kognitiva tillstånd från %s", len(_CACHED_STATES), _STATES_FILE.name)
    return _CACHED_STATES


def reload_states() -> None:
    """Tvinga omladdning av cognitive_states.json (vid runtime-uppdatering)."""
    global _CACHED_STATES
    _CACHED_STATES = None
    _get_states()


# ── Distansberäkning ──────────────────────────────────────────────────────────

def _chemical_distance(da: float, na: float, ach_norm: float, profile: dict) -> float:
    """
    Normaliserad euklidisk distans i DA/NA/ACh-rummet.
    Alla dimensioner ∈ [0, 1].
    """
    d_da  = da       - profile.get("dopamine",      {}).get("level", 0.5)
    d_na  = na       - profile.get("noradrenaline",  {}).get("level", 0.3)
    d_ach = ach_norm - profile.get("acetylcholine",  {}).get("level", 0.5)
    return math.sqrt(d_da**2 + d_na**2 + d_ach**2)


# ── Gate-prioritet ────────────────────────────────────────────────────────────

_GATE_PRIORITY: dict[str, int] = {
    "blocked":           0,
    "emergency":         0,
    "minimal":           1,
    "cautious":          2,
    "revision_only":     3,
    "consolidation_only": 4,
    "open":              5,
    "priority_open":     6,
}

_PRIORITY_TO_GATE: dict[int, str] = {v: k for k, v in _GATE_PRIORITY.items()}


def _blend_gate(gates: list[tuple[str, float]]) -> str:
    """
    Väljer gate baserat på viktat genomsnitt av prioritet.
    Mer restriktiva gates drar ner snittet — systemet är konservativt.
    """
    total_w = sum(w for _, w in gates)
    if total_w == 0:
        return "open"
    weighted_priority = sum(_GATE_PRIORITY.get(g, 5) * w for g, w in gates) / total_w
    # Hitta gate med närmaste prioritet (avrundat)
    rounded = round(weighted_priority)
    rounded = max(0, min(6, rounded))
    # Sök uppåt om exakt nivå saknas
    for delta in range(7):
        for sign in (0, -1, 1):
            candidate = rounded + sign * delta
            if candidate in _PRIORITY_TO_GATE:
                return _PRIORITY_TO_GATE[candidate]
    return "open"


def _blend_mode(modes: list[tuple[str, float]]) -> str:
    """Väljer dominant mode (högst vikt)."""
    if not modes:
        return "balanced"
    return max(modes, key=lambda x: x[1])[0]


# ── Klassificering ────────────────────────────────────────────────────────────

_FLAG_KEYS = frozenset({
    "hitl_escalation",
    "nightrun_trigger_hint",
    "immediate_crystallize",
    "promote_to_semantic",
    "all_writes_require_approval",
    "flag_stuck_patterns",
    "operator_rescue_hint",
    "all_systems_nominal",
    "flag_for_deepdive",
    "curiosity_burst_priority",
    "gap_map_weight_boost",
    "noise_filter_boost",
    "mission_weight_boost",
    "u_delta_on_active_edges",
})

# Minimisvikt för att flaggor ska räknas
_FLAG_THRESHOLD = 0.30


def classify(
    da: float,
    na: float,
    ach_beta: float,
    arousal: float = 0.0,
    performance: float = 0.0,
    top_k: int = _TOP_K,
) -> SemanticModulation:
    """
    Klassificera kemiskt tillstånd och returnera SemanticModulation.

    Parametrar:
        da          LimbicState.dopamine ∈ [0, 1]
        na          LimbicState.noradrenaline ∈ [0, 1]
        ach_beta    LimbicState.acetylcholine (β, attention temp) ∈ [0.1, 2.0]
        arousal     LimbicState.arousal (summerad)
        performance LimbicState.performance (Yerkes-Dodson)
        top_k       Antal tillstånd att blanda

    Algoritm:
        1. Normalisera ach_beta → ach_norm = ach_beta / 2.0
        2. Beräkna euklidisk distans mot alla definierade tillstånd
        3. Välj top_k närmaste
        4. Vikta med invers distans (+ epsilon för stabilitet)
        5. Blanda modulationsparametrar viktat
        6. Samla flags från tillstånd med vikt > _FLAG_THRESHOLD
    """
    ach_norm = ach_beta / _ACH_BETA_MAX

    cognitive_states = _get_states()

    # Distansberäkning
    scored: list[tuple[dict, float]] = []
    for cs in cognitive_states:
        d = _chemical_distance(da, na, ach_norm, cs["chemical_profile"])
        scored.append((cs, d))

    scored.sort(key=lambda x: x[1])
    top = scored[:top_k]

    # Invers distans → normaliserade vikter
    inv_d = [1.0 / (d + 1e-8) for _, d in top]
    total_inv = sum(inv_d)
    weights = [iv / total_inv for iv in inv_d]

    dominant_cs = top[0][0]
    active_ids = [cs["id"] for cs, _ in top]
    blend_weights = {cs["id"]: round(w, 4) for (cs, _), w in zip(top, weights)}

    # Viktat genomsnitt av skalära parametrar
    ev_delta        = 0.0
    bisoc_delta     = 0.0
    wm_slots_raw    = 0.0
    xtal_boost      = 0.0
    pruning_delta   = 0.0

    gate_inputs: list[tuple[str, float]] = []
    mode_inputs: list[tuple[str, float]] = []
    flags: dict[str, Any] = {}

    for (cs, _), w in zip(top, weights):
        mod = cs.get("nouse_modulation", {})

        ev_delta      += mod.get("evidence_threshold_delta",    0.0) * w
        bisoc_delta   += mod.get("bisociation_propensity_delta", 0.0) * w
        wm_slots_raw  += mod.get("working_memory_slots_delta",   0)   * w
        xtal_boost    += mod.get("crystallization_boost",        0.0) * w
        pruning_delta += mod.get("pruning_aggression_delta",     0.0) * w

        gate_inputs.append((mod.get("write_back_gate", "open"), w))
        mode_inputs.append((mod.get("response_mode", "balanced"), w))

        # Flaggor — bara från tillstånd med tillräcklig vikt
        if w >= _FLAG_THRESHOLD:
            for fk in _FLAG_KEYS:
                if fk in mod:
                    flags[fk] = mod[fk]

    return SemanticModulation(
        evidence_threshold_delta    = round(ev_delta, 3),
        bisociation_propensity_delta= round(bisoc_delta, 3),
        working_memory_slots_delta  = round(wm_slots_raw),
        crystallization_boost       = round(xtal_boost, 3),
        pruning_aggression_delta    = round(pruning_delta, 3),
        write_back_gate             = _blend_gate(gate_inputs),
        response_mode               = _blend_mode(mode_inputs),
        dominant_state              = dominant_cs["id"],
        active_state_ids            = active_ids,
        blend_weights               = blend_weights,
        state_label                 = dominant_cs["label"],
        arousal                     = round(arousal, 3),
        performance                 = round(performance, 3),
        flags                       = flags,
    )


def modulate(limbic_state: LimbicState) -> SemanticModulation:
    """
    Huvudingångspunkt — tar ett LimbicState och returnerar SemanticModulation.

    Exempel:
        from nouse.limbic.signals import load_state
        from nouse.limbic.state_modulator import modulate

        limbic = load_state()
        mod = modulate(limbic)
        print(mod.summary())
    """
    mod = classify(
        da=limbic_state.dopamine,
        na=limbic_state.noradrenaline,
        ach_beta=limbic_state.acetylcholine,
        arousal=limbic_state.arousal,
        performance=limbic_state.performance,
    )
    log.info("SemanticModulation: %s", mod.summary())
    return mod
