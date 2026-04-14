"""
Live-integrationstest för Nouse plastisitet.

Skapar en riktig SQLite/NetworkX-graf i /tmp, matar in micro-fakta
och verifierar att P1-P5 faktiskt sker i grafen.

Kör med:
    cd /home/bjorn/projects/b76
    python -m tests.test_plasticity_live
"""
from __future__ import annotations

import sys
import time
import tempfile
import shutil
from pathlib import Path

# b76 måste vara i path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nouse.field.surface import FieldSurface
from nouse.limbic.signals import LimbicState
from nouse.learning_coordinator import LearningCoordinator

from nouse.brian2_bridge import Brian2Bridge


# ── Hjälpfunktioner ───────────────────────────────────────────────────────────

def _get_strength(field: FieldSurface, src: str, tgt: str) -> float | None:
    rels = field.out_relations(src)
    for r in rels:
        if r["target"] == tgt:
            return r.get("strength")
    return None

def _get_granularity(field: FieldSurface, name: str) -> int | None:
    try:
        row = field._sql.execute(  # noqa: SLF001 - live smoke helper
            "SELECT granularity AS g FROM concept WHERE name = ?",
            (name,),
        ).fetchone()
        return int(row["g"]) if row and row.get("g") is not None else None
    except Exception:
        return None

def _get_assumption(field: FieldSurface, src: str, tgt: str) -> bool | None:
    rels = field.out_relations(src)
    for r in rels:
        if r["target"] == tgt:
            return r.get("assumption_flag")
    return None


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "✅" if condition else "❌"
    print(f"  {status}  {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        raise AssertionError(f"FAIL: {label}")


# ── Testscenario ──────────────────────────────────────────────────────────────

def run():
    tmpdir = tempfile.mkdtemp(prefix="nouse_plasticity_test_")
    print(f"\n🧠 Nouse Plastisitetstest")
    print(f"   SQLite/NetworkX: {tmpdir}\n")

    try:
        field = FieldSurface(db_path=Path(tmpdir) / "field.sqlite")

        # ── Scenario: havsforsknings-fakta ────────────────────────────────────
        # Simulerar det exakta use-case som triggade genombrottet:
        # WenHai/OceanNet-domän möter Nouse-domän

        facts = [
            # (src, rel_type, tgt, why, evidence_score, support_count)
            ("mesoscale_eddy", "orsakar",    "heat_flux_anomaly",
             "Eddies transporterar värme lateralt", 0.82, 4),
            ("heat_flux_anomaly", "modulerar", "sst_variability",
             "Yttemperatur reagerar på fluxavvikelser", 0.75, 8),
            ("sst_variability",  "reglerar",  "halocline_depth",
             "Temperaturgradienter påverkar språngskikt", 0.40, 1),  # ← hypotes
            ("halocline_depth",  "orsakar",   "oxygen_depletion",
             "", 0.25, 1),  # ← ingen why → assumption_flag=True
        ]

        # Limbic-state: hög noradrenalin (ny domän = överraskning)
        limbic = LimbicState(noradrenaline=0.8, dopamine=0.6)
        coordinator = LearningCoordinator(field, limbic)
        bridge = Brian2Bridge(field, use_brian2=False)

        print("📥 Matar in micro-fakta...\n")
        for src, rel, tgt, why, ev, support in facts:
            field.add_relation(src, rel, tgt, why=why, evidence_score=ev)
            coordinator.on_fact(src, rel, tgt,
                                why=why,
                                evidence_score=ev,
                                support_count=support)
            time.sleep(0.002)  # liten fördröjning för STDP Δt
            bridge.on_fact(src, rel, tgt)
            print(f"   {src} ─[{rel}]→ {tgt}")

        print()

        # ── P1: Hebbisk delta modulerat av noradrenalin ───────────────────────
        print("P1 — Hebbisk delta (NA=0.8 → Δw = 0.05 × 1.8 = 0.09)")
        w = _get_strength(field, "mesoscale_eddy", "heat_flux_anomaly")
        print(f"     strength = {w:.4f}" if w else "     strength = None")
        check("strength > 1.0 (Hebbisk förstärkning skett)", w is not None and w > 1.0,
              f"strength={w}")

        # ── P2: Spreading activation ──────────────────────────────────────────
        print("\nP2 — Spreading activation")
        # heat_flux_anomaly är granne till mesoscale_eddy
        # → sst_variability (granne till heat_flux_anomaly) ska också ha stärkts
        w2 = _get_strength(field, "heat_flux_anomaly", "sst_variability")
        print(f"     heat_flux_anomaly→sst_variability strength = {w2}")
        check("sst_variability stärkt via spreading", w2 is not None and w2 > 1.0,
              f"strength={w2}")

        # ── P3: Assumption flag evolution ─────────────────────────────────────
        print("\nP3 — Assumption flag evolution")
        af_high = _get_assumption(field, "mesoscale_eddy", "heat_flux_anomaly")
        af_low  = _get_assumption(field, "halocline_depth", "oxygen_depletion")
        print(f"     mesoscale_eddy→heat_flux_anomaly  assumption_flag = {af_high}  (ev=0.82, borde vara False)")
        print(f"     halocline_depth→oxygen_depletion  assumption_flag = {af_low}   (ev=0.25, borde vara True)")
        check("Hög evidens → assumption_flag=False", af_high == False, f"got {af_high}")
        check("Låg evidens → assumption_flag kvar True/None", af_low != False,  f"got {af_low}")

        # ── P4: Granularity ───────────────────────────────────────────────────
        print("\nP4 — Granularity update")
        g_eddy   = _get_granularity(field, "mesoscale_eddy")   # support=4 → g=3
        g_halocl = _get_granularity(field, "halocline_depth")  # support=1 → g=1
        print(f"     mesoscale_eddy granularity = {g_eddy}  (support=4, förväntat 3)")
        print(f"     halocline_depth granularity = {g_halocl}  (support=1, förväntat 1)")
        check("mesoscale_eddy granularity = 3", g_eddy == 3,   f"got {g_eddy}")
        check("halocline_depth granularity = 1", g_halocl == 1, f"got {g_halocl}")

        # ── P5: STDP ──────────────────────────────────────────────────────────
        print("\nP5 — STDP (Brian2Bridge Python-fallback)")
        # Spika ett par noder med tidsfördröjning och kontrollera att strength ökar
        field.add_relation("dopamin", "modulerar", "stdp_test_nod", why="test")
        w_before = _get_strength(field, "dopamin", "stdp_test_nod") or 1.0
        bridge.on_concept_activated("dopamin")
        time.sleep(0.01)
        dw = bridge.on_fact("dopamin", "modulerar", "stdp_test_nod")
        w_after = _get_strength(field, "dopamin", "stdp_test_nod") or 1.0
        print(f"     STDP Δw = {dw:.5f}  ({'LTP ✓' if dw > 0 else 'LTD' if dw < 0 else 'neutral'})")
        check("STDP LTP: pre→post ger positiv Δw", dw > 0, f"dw={dw:.5f}")

        # ── Graf-statistik ────────────────────────────────────────────────────
        print("\n📊 Grafstatistik")
        all_concepts = field.concepts()
        print(f"     Noder (Concept): {len(all_concepts)}")
        print(f"     Domäner: {field.domains()}")

        print("\n🎉 Alla tester gröna — plastisiteten fungerar i live-grafen!\n")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    run()
