"""
Brian2Bridge — STDP-timing för Nouse kunskapsgraf
==================================================

Implementerar Spike-Timing-Dependent Plasticity (STDP) ovanpå
LearningCoordinator. Medan LearningCoordinator hanterar Hebbisk
δ-uppdatering, hanterar Brian2Bridge TIMING-aspekten:

  LTP (Long-Term Potentiation):  pre före post  → stärk kanten
  LTD (Long-Term Depression):    post före pre  → försvaga kanten

STDP-fönster (klassisk exponentiell form):
  Δt = t_post - t_pre
  Δw =  A_plus  × exp(-Δt / tau_plus)   om Δt > 0  (LTP)
  Δw = -A_minus × exp( Δt / tau_minus)  om Δt < 0  (LTD)

Mappning till Nouse:
  "Spike" = ett nytt micro-faktum som aktiverar en Concept-nod
  Spike-tid = tidsstämpel när noden senast aktiverades
  Nod A spikades, sedan nod B → A→B förstärks (LTP)
  Nod B spikades, sedan nod A → A→B försvagas (LTD)

Bryggan är OPTIONELL — om Brian2 inte finns degraderar den
tyst till att bara returnera Δw = 0 utan att krascha.
"""
from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

_log = logging.getLogger("nouse.stdp")

# ── STDP-parametrar (konfigurerbara via env) ──────────────────────────────────

def _ef(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default

A_PLUS    = _ef("NOUSE_STDP_A_PLUS",    0.01)   # LTP amplitude
A_MINUS   = _ef("NOUSE_STDP_A_MINUS",   0.012)  # LTD amplitude (asymmetri ger nettopotentiering)
TAU_PLUS  = _ef("NOUSE_STDP_TAU_PLUS",  20.0)   # LTP tidskonstant (sekunder)
TAU_MINUS = _ef("NOUSE_STDP_TAU_MINUS", 20.0)   # LTD tidskonstant (sekunder)
W_MIN     = _ef("NOUSE_STDP_W_MIN",     0.0)    # Minimumvikt
W_MAX     = _ef("NOUSE_STDP_W_MAX",     5.0)    # Maximumvikt


# ── Brian2 (optionell) ────────────────────────────────────────────────────────

_BRIAN2_AVAILABLE = False
try:
    import brian2 as _b2
    _BRIAN2_AVAILABLE = True
    _log.info("Brian2 %s tillgängligt — STDP aktiverat", _b2.__version__)
except (ImportError, AttributeError, Exception) as _brian2_err:
    _log.warning("Brian2 ej tillgängligt (%s) — STDP använder Python-fallback", type(_brian2_err).__name__)


# ── Spiketidregister ──────────────────────────────────────────────────────────

@dataclass
class SpikeRegister:
    """
    Håller reda på när varje Concept-nod senast "spikades" (aktiverades).
    Tidsstämpel i Unix-sekunder (float) för maximal precision.
    """
    _times: dict[str, float] = field(default_factory=dict)
    _history: list[tuple[float, str]] = field(default_factory=list)

    def spike(self, node: str) -> None:
        """Registrera att noden aktiverades nu."""
        t = time.monotonic()
        self._times[node] = t
        self._history.append((t, node))
        # Håll historiken begränsad (senaste 1000 spikes)
        if len(self._history) > 1000:
            self._history = self._history[-500:]

    def last_spike(self, node: str) -> float | None:
        """Returnera tidpunkten för nodens senaste spike, eller None."""
        return self._times.get(node)

    def delta_t(self, pre: str, post: str) -> float | None:
        """
        Δt = t_post - t_pre
        Positivt: pre spikade före post → LTP
        Negativt: post spikade före pre → LTD
        """
        t_pre  = self._times.get(pre)
        t_post = self._times.get(post)
        if t_pre is None or t_post is None:
            return None
        return t_post - t_pre


# ── STDP-beräkning (ren Python, ingen Brian2 krävs) ───────────────────────────

def stdp_delta(delta_t: float) -> float:
    """
    Beräkna viktförändring Δw baserat på spike-timing.

    delta_t = t_post - t_pre (sekunder)
      > 0: pre → post (kausal ordning) → LTP (positiv Δw)
      < 0: post → pre (omvänd ordning) → LTD (negativ Δw)
      = 0: simultana spikes → ingen förändring
    """
    import math
    if delta_t > 0:
        # LTP: pre spikade före post
        return A_PLUS * math.exp(-delta_t / TAU_PLUS)
    elif delta_t < 0:
        # LTD: post spikade före pre
        return -A_MINUS * math.exp(delta_t / TAU_MINUS)
    return 0.0


def clamp_weight(w: float, delta: float) -> float:
    """Applicera Δw med hard bounds [W_MIN, W_MAX]."""
    return max(W_MIN, min(W_MAX, w + delta))


# ── Brian2 SNN-simulator (aktiveras om Brian2 finns) ─────────────────────────

class Brian2STDPNetwork:
    """
    Tunnt Brian2-lager för synaptisk STDP-simulering.

    Noder → LIF-neuroner (Leaky Integrate-and-Fire)
    Kanter → synapser med STDP-inlärningsregel

    Används för att validera att STDP-beräkningarna i stdp_delta()
    överensstämmer med Brian2:s interna simulering.
    """

    def __init__(self, n_neurons: int = 100, sim_dt_ms: float = 0.1):
        if not _BRIAN2_AVAILABLE:
            raise RuntimeError("Brian2 krävs för Brian2STDPNetwork")

        import brian2 as b2
        b2.start_scope()

        self.b2 = b2
        self._dt = sim_dt_ms * b2.ms

        # LIF-neuronmodell
        eqs = """
            dv/dt = (v_rest - v + I_ext) / tau : volt
            I_ext : volt
        """
        self.tau      = 20  * b2.ms
        self.v_rest   = -70 * b2.mV
        self.v_thresh = -55 * b2.mV
        self.v_reset  = -70 * b2.mV

        self.neurons = b2.NeuronGroup(
            n_neurons,
            eqs,
            threshold=f"v > {self.v_thresh/b2.mV}*mV",
            reset=f"v = {self.v_reset/b2.mV}*mV",
            method="euler",
            namespace={"tau": self.tau, "v_rest": self.v_rest},
        )
        self.neurons.v = self.v_rest

        # STDP-synapser
        stdp_eqs = """
            w      : 1
            dApre  /dt = -Apre  / taupre  : 1 (event-driven)
            dApost /dt = -Apost / taupost : 1 (event-driven)
        """
        on_pre = """
            v_post += w * mV
            Apre   += A_plus_param
            w       = clip(w + Apost, w_min, w_max)
        """
        on_post = """
            Apost  += -A_minus_param
            w       = clip(w + Apre,  w_min, w_max)
        """
        self.synapses = b2.Synapses(
            self.neurons, self.neurons,
            model=stdp_eqs,
            on_pre=on_pre,
            on_post=on_post,
            namespace={
                "taupre":        TAU_PLUS  * b2.second,
                "taupost":       TAU_MINUS * b2.second,
                "A_plus_param":  A_PLUS,
                "A_minus_param": A_MINUS,
                "w_min":         W_MIN,
                "w_max":         W_MAX,
            },
        )

        self.spike_monitor = b2.SpikeMonitor(self.neurons)
        self.net = b2.Network(self.neurons, self.synapses, self.spike_monitor)

        # Mappning: Concept-namn → neuron-index
        self._node_to_idx: dict[str, int] = {}
        self._next_idx = 0
        self._n = n_neurons

    def register_node(self, name: str) -> int:
        """Tilldela ett neuron-index till en Concept-nod."""
        if name not in self._node_to_idx:
            if self._next_idx >= self._n:
                raise OverflowError(f"Nätverket är fullt ({self._n} neuroner)")
            self._node_to_idx[name] = self._next_idx
            self._next_idx += 1
        return self._node_to_idx[name]

    def connect(self, src: str, tgt: str, initial_weight: float = 1.0) -> None:
        """Skapa en synaps mellan src och tgt."""
        i = self.register_node(src)
        j = self.register_node(tgt)
        self.synapses.connect(i=i, j=j)
        self.synapses.w[i, j] = initial_weight

    def inject_spike(self, node: str, current_amp: float = 20.0) -> None:
        """Tvinga en spike i noden genom att injicera ström."""
        idx = self._node_to_idx.get(node)
        if idx is None:
            return
        self.neurons.I_ext[idx] = current_amp * self.b2.mV

    def step(self, duration_ms: float = 1.0) -> None:
        """Kör simulatorn ett steg framåt."""
        self.net.run(duration_ms * self.b2.ms)
        # Nollställ injektioner
        self.neurons.I_ext = 0 * self.b2.mV

    def get_weight(self, src: str, tgt: str) -> float | None:
        """Hämta aktuell synaptisk vikt mellan src och tgt."""
        i = self._node_to_idx.get(src)
        j = self._node_to_idx.get(tgt)
        if i is None or j is None:
            return None
        try:
            w = self.synapses.w[i, j]
            return float(w[0]) if hasattr(w, "__len__") else float(w)
        except Exception:
            return None


# ── Huvudbrygga ───────────────────────────────────────────────────────────────

class Brian2Bridge:
    """
    Kopplar SpikeRegister + STDP-beräkning till FieldSurface.

    Flöde för varje nytt micro-faktum:
      1. Registrera spikes för src och tgt
      2. Beräkna Δt = t_post - t_pre
      3. Beräkna STDP Δw via stdp_delta(Δt)
      4. Applicera Δw på FieldSurface-kanten

    Kan använda antingen:
      a) Python-fallback (stdp_delta): alltid tillgänglig
      b) Brian2STDPNetwork: om Brian2 är installerat och nätverk är konfigurerat
    """

    def __init__(
        self,
        field,   # FieldSurface
        *,
        use_brian2: bool = _BRIAN2_AVAILABLE,
        n_neurons: int = 200,
    ):
        self.field = field
        self.register = SpikeRegister()
        self._brian2_net: Brian2STDPNetwork | None = None

        if use_brian2 and _BRIAN2_AVAILABLE:
            try:
                self._brian2_net = Brian2STDPNetwork(n_neurons=n_neurons)
                _log.info("Brian2Bridge: använder Brian2 SNN-simulator")
            except Exception as e:
                _log.warning("Brian2Bridge: kunde inte initiera nätverk (%s) — Python-fallback", e)
        else:
            _log.info("Brian2Bridge: Python STDP-fallback aktiv")

    @property
    def brian2_active(self) -> bool:
        return self._brian2_net is not None

    def on_concept_activated(self, node: str) -> None:
        """
        Anropa när en Concept-nod aktiveras (t.ex. av ny inkommande fakta).
        Registrerar spike-tidpunkt.
        """
        self.register.spike(node)
        if self._brian2_net:
            try:
                self._brian2_net.register_node(node)
                self._brian2_net.inject_spike(node)
                self._brian2_net.step(1.0)
            except Exception as e:
                _log.debug("Brian2 spike-injektion misslyckades: %s", e)

    def on_fact(self, src: str, rel_type: str, tgt: str) -> float:
        """
        Anropa direkt efter add_relation() — EFTER LearningCoordinator.on_fact().

        1. Spika src och tgt
        2. Beräkna Δt
        3. Beräkna STDP Δw
        4. Applicera på kanten i FieldSurface
        5. Returnera Δw (för loggning/test)
        """
        # Spika tgt EFTER src (kausal ordning: src → tgt)
        self.on_concept_activated(src)
        self.on_concept_activated(tgt)

        # Δt = t_tgt - t_src (borde vara strax positivt → LTP)
        delta_t = self.register.delta_t(src, tgt)
        if delta_t is None:
            return 0.0

        # Hämta STDP Δw
        if self._brian2_net:
            w_before = self._brian2_net.get_weight(src, tgt)
            if w_before is not None:
                # Brian2 har redan uppdaterat vikten internt
                # Vi synkar FieldSurface med Brian2-vikten
                w_after = self._brian2_net.get_weight(src, tgt)
                if w_after is not None:
                    dw = w_after - w_before
                    if abs(dw) > 1e-6:
                        self.field.strengthen(src, tgt, dw)
                    return dw

        # Python-fallback
        dw = stdp_delta(delta_t)
        if abs(dw) > 1e-6:
            self.field.strengthen(src, tgt, dw)

        _log.debug(
            "STDP %s─[%s]→%s  Δt=%.3fs  Δw=%.5f  (%s)",
            src, rel_type, tgt, delta_t, dw,
            "LTP" if dw > 0 else "LTD" if dw < 0 else "neutral",
        )
        return dw
