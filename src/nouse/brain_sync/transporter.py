"""
brain_sync — b76 → brain-db-core transport layer
=================================================
Bridges the semantic knowledge graph (b76 FieldSurface)
to the abstract FNC substrate (brain-db-core).

Each significant b76 event is translated to a brain-db-core FieldEvent
and pushed via the /step REST API.

Design principles:
  1. b76 is the "cortex" — rich semantic knowledge, slow to crystallize
  2. brain-db-core is the "brain stem" — abstract FNC dynamics, fast to update
  3. Transport is ONE-WAY b76 → brain, not the reverse
  4. All events are read-only observations from brain's perspective
  5. No blocking: transport failures never stop b76 daemon loop

Usage:
  from nouse.brain_sync.transporter import BrainTransporter
  transporter = BrainTransporter()
  transporter.send_bisociation_event(bridge_strength=0.82, pair=("svampar","matematik"))
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

log = logging.getLogger("nouse.brain_sync")

BRAIN_DB_BASE = (
    os.getenv("NOUSE_BRAIN_DB_BASE") or "http://127.0.0.1:7676"
).rstrip("/")

_TIMEOUT = float(os.getenv("NOUSE_BRAIN_SYNC_TIMEOUT_SEC", "3.0"))
_RETRY_DELAY = float(os.getenv("NOUSE_BRAIN_SYNC_RETRY_DELAY_SEC", "2.0"))
_MAX_RETRIES = int(os.getenv("NOUSE_BRAIN_SYNC_MAX_RETRIES", "2"))


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


# ── Event taxonomy ────────────────────────────────────────────────────────────

class EventType(Enum):
    BISOCIATION = "bisociation"          # creative bridge discovered
    ANALOGY = "analogy"                 # cross-domain structural analogy
    METACOGNITION = "metacognition"      # self-observation / self-model update
    CONCEPT_CRYSTALLIZE = "concept_crystallize"  # high-evidence concept
    LIMBIC_SPIKE = "limbic_spike"       # high neuromodulator signal
    FIELD_SURFACE_UPDATE = "field_surface_update"  # new strong relation in the graph layer


@dataclass
class BrainFieldEvent:
    """b76's canonical representation of a brain-worthy event.

    Maps to brain_db_core.FieldEvent on the brain-db-core side.
    """
    edge_id: str
    src: str
    rel_type: str
    tgt: str
    w_delta: float = 0.02
    r_delta: float = 0.0
    u_delta: float = 0.0
    evidence_score: float | None = None
    provenance: str = "b76_brain_sync"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["_b76_created_at"] = _now_iso()
        return d

    def to_step_payload(self) -> dict[str, Any]:
        return {
            "events": [self.to_dict()]
        }


@dataclass
class TransportStats:
    sent: int = 0
    failed: int = 0
    last_sent_at: str | None = None
    last_event_type: str | None = None
    consecutive_failures: int = 0


# ── Core transporter ─────────────────────────────────────────────────────────

class BrainTransporter:
    """One-way transport from b76 FieldSurface to brain-db-core /step API."""

    def __init__(
        self,
        base_url: str = BRAIN_DB_BASE,
        timeout: float = _TIMEOUT,
        max_retries: int = _MAX_RETRIES,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._stats = TransportStats()
        self._import_client()

    def _import_client(self) -> None:
        """Lazily import httpx only when needed."""
        import httpx
        self._client = httpx

    @property
    def stats(self) -> TransportStats:
        return self._stats

    def _post_step(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        """POST events to brain-db-core /step endpoint."""
        import httpx
        try:
            r = httpx.post(
                f"{self._base}/step",
                json={"events": events},
                timeout=self._timeout,
            )
            r.raise_for_status()
            return r.json()
        except httpx.TimeoutException:
            return {"ok": False, "error": "timeout"}
        except httpx.HTTPStatusError as e:
            return {"ok": False, "error": f"http_{e.response.status_code}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def send(self, event: BrainFieldEvent) -> bool:
        """Send a single event to brain-db-core. Returns True on success."""
        payload = [event.to_dict()]
        for attempt in range(self._max_retries + 1):
            result = self._post_step(payload)
            if result.get("ok") is not False:
                self._stats.sent += 1
                self._stats.last_sent_at = _now_iso()
                self._stats.last_event_type = event.provenance
                self._stats.consecutive_failures = 0
                return True
            if attempt < self._max_retries:
                time.sleep(_RETRY_DELAY)
        self._stats.failed += 1
        self._stats.consecutive_failures += 1
        log.warning(
            f"brain_sync: failed to send event {event.edge_id} "
            f"({self._stats.consecutive_failures} consecutive failures)"
        )
        return False

    def send_batch(self, events: list[BrainFieldEvent]) -> int:
        """Send multiple events. Returns number successfully sent."""
        if not events:
            return 0
        payload = [e.to_dict() for e in events]
        result = self._post_step(payload)
        if result.get("ok") is not False:
            self._stats.sent += len(events)
            self._stats.last_sent_at = _now_iso()
            self._stats.consecutive_failures = 0
            return len(events)
        # Fall back to individual sends
        sent = 0
        for e in events:
            if self.send(e):
                sent += 1
        return sent

    def health_check(self) -> bool:
        """Return True if brain-db-core is reachable."""
        import httpx
        try:
            r = httpx.get(f"{self._base}/health", timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False


# ── Event constructors ───────────────────────────────────────────────────────

def bisociation_event(
    domain_a: str,
    domain_b: str,
    bridge_strength: float,
    bisoc_quality: float,
    path_novelty: float | None = None,
) -> BrainFieldEvent:
    """Fired when the Bisociation Engine discovers a creative bridge."""
    pair_hash = f"{domain_a}_x_{domain_b}"
    return BrainFieldEvent(
        edge_id=f"bisoc:{pair_hash}",
        src=domain_a,
        rel_type="bisociates_with",
        tgt=domain_b,
        w_delta=bridge_strength * 0.15,   # w increments proportional to bridge strength
        r_delta=0.0,
        u_delta=-0.05,                     # uncertainty decreases with confirmed bisoc
        evidence_score=bisoc_quality,
        provenance="bisociation_engine",
    )


def analogy_event(
    concept_a: str,
    concept_b: str,
    relation_strength: float,
    evidence_score: float,
    relation_type: str = "är_analogt_med",
) -> BrainFieldEvent:
    """Fired when field_surface finds a cross-domain analogy."""
    return BrainFieldEvent(
        edge_id=f"analog:{concept_a}:{concept_b}",
        src=concept_a,
        rel_type=relation_type,
        tgt=concept_b,
        w_delta=relation_strength * 0.10,
        r_delta=evidence_score * 0.3,
        u_delta=-0.03,
        evidence_score=evidence_score,
        provenance="field_surface",
    )


def metacognition_event(
    observation_type: str,
    target: str,
    lambda_delta: float = 0.0,
    bisoc_rate_delta: float = 0.0,
) -> BrainFieldEvent:
    """Fired when self_model records a significant self-observation."""
    return BrainFieldEvent(
        edge_id=f"self:{observation_type}:{target[:40]}",
        src="self_model",
        rel_type=observation_type,
        tgt=target,
        w_delta=0.08,
        r_delta=lambda_delta * 0.2 if lambda_delta else 0.0,
        u_delta=0.0,
        evidence_score=0.75,
        provenance="metacognition",
    )


def concept_crystallize_event(
    concept_name: str,
    domain: str,
    evidence_score: float,
    relation_strength: float,
) -> BrainFieldEvent:
    """Fired when a concept crosses the evidence gate (strong fact, high confidence)."""
    safe_name = concept_name.replace(" ", "_")[:50]
    return BrainFieldEvent(
        edge_id=f"concept:{safe_name}",
        src=concept_name,
        rel_type="belongs_to",
        tgt=domain,
        w_delta=evidence_score * 0.12,
        r_delta=0.0,
        u_delta=-0.08,                    # crystallized = low uncertainty
        evidence_score=evidence_score,
        provenance="evidence_gate",
    )


def limbic_spike_event(
    signal_type: str,  # "dopamine" | "noradrenaline" | "acetylcholine"
    magnitude: float,
    target_concept: str | None = None,
) -> BrainFieldEvent:
    """Fired on significant limbic signal spike — routes to the target concept."""
    # Note: brain-db-core doesn't have a direct limbic update API.
    # As a proxy, we emit a FieldEvent that strengthens the most-active concept.
    return BrainFieldEvent(
        edge_id=f"limbic:{signal_type}:{_now_iso()[:19]}",
        src=f"limbic_{signal_type}",
        rel_type=f"amplifies_{signal_type}",
        tgt=target_concept or "prefrontal",  # fallback target
        w_delta=magnitude * 0.06,
        r_delta=0.0,
        u_delta=-0.02,
        evidence_score=None,
        provenance="limbic_spike",
    )


# ── Integration hook ─────────────────────────────────────────────────────────

def integration_hooks() -> dict[str, Any]:
    """
    Returns the set of hook points in b76 daemon where brain_sync should be called.
    Install by adding to the daemon loop at each hook point.

    Hook points:
      1. bisociation_engine.propose() returns BISOCIATION → send bisociation_event
      2. field_surface adds high-strength relation → send analogy_event
      3. self_model.observe_cycle() → send metacognition_event
      4. MemoryStore.consolidate() with high evidence → send concept_crystallize_event
      5. ArousalModel reports spike → send limbic_spike_event
    """
    return {
        "bisociation": {
            "module": "nouse.bisociation.engine",
            "method": "propose",
            "condition": "verdict == BISOCIATION",
            "action": "bisociation_event(domain_a, domain_b, bridge_strength, bisoc_quality)",
        },
        "field_upsert": {
            "module": "nouse.field.surface",
            "method": "add_relation",
            "condition": "strength > 0.7 or evidence_score > 0.65",
            "action": "analogy_event(src, tgt, strength, evidence_score)",
        },
        "self_observation": {
            "module": "nouse.metacognition.self_model",
            "method": "observe_cycle",
            "condition": "lambda_delta != 0 or bisoc_rate_delta != 0",
            "action": "metacognition_event(obs_type, target, lambda_delta, bisoc_rate_delta)",
        },
        "evidence_gate": {
            "module": "nouse.memory.store",
            "method": "consolidate",
            "condition": "evidence_score > 0.85 and relation_strength > 0.7",
            "action": "concept_crystallize_event(concept, domain, evidence_score, relation_strength)",
        },
        "limbic_spike": {
            "module": "nouse.limbic.signals",
            "method": "run_limbic_cycle",
            "condition": "noradrenaline > 0.7 or dopamine > 0.8",
            "action": "limbic_spike_event(signal_type, magnitude)",
        },
    }
