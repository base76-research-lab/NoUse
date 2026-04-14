from __future__ import annotations

from pathlib import Path

from nouse.field.surface import FieldSurface


def _mk_field(tmp_path: Path) -> FieldSurface:
    return FieldSurface(db_path=tmp_path / "field.sqlite", read_only=False)


def test_add_concept_seeds_minimal_context_and_facts(tmp_path):
    field = _mk_field(tmp_path)
    field.add_concept("Larynx Problem", "ai_forskning", source="unit_test")
    knowledge = field.concept_knowledge("Larynx Problem")

    assert knowledge["summary"]
    assert knowledge["claims"]
    assert knowledge["evidence_refs"]

    audit = field.knowledge_audit(limit=10)
    assert audit["missing_total"] == 0
    assert audit["complete_nodes"] == 1


def test_backfill_repairs_nodes_missing_context_and_facts(tmp_path):
    field = _mk_field(tmp_path)
    field.add_concept("Orphan Node", "system", source="seed", ensure_knowledge=False)

    before = field.knowledge_audit(limit=10)
    assert before["missing_total"] == 1
    assert before["missing"][0]["name"] == "Orphan Node"

    result = field.backfill_missing_concept_knowledge()
    assert result["updated"] == 1

    after = field.knowledge_audit(limit=10)
    assert after["missing_total"] == 0

    knowledge = field.concept_knowledge("Orphan Node")
    assert knowledge["summary"]
    assert knowledge["claims"]
    assert knowledge["evidence_refs"]


def test_strict_gate_tracks_strong_facts_separately(tmp_path):
    field = _mk_field(tmp_path)
    field.add_concept("Isolated", "system", source="seed")

    basic = field.knowledge_audit(limit=10, strict=False)
    strict = field.knowledge_audit(limit=10, strict=True, min_evidence_score=0.65)

    assert basic["missing_total"] == 0
    assert strict["missing_total"] == 1
    assert strict["missing"][0]["name"] == "Isolated"
    assert "missing_strong_facts" in strict["missing"][0]["reasons"]


def test_add_relation_legacy_mode_does_not_pass_unknown_params(tmp_path):
    field = _mk_field(tmp_path)
    field._relation_meta_available = False  # noqa: SLF001 - explicit legacy-mode regression test

    field.add_relation(
        "Legacy Src",
        "beskriver",
        "Legacy Tgt",
        why="legacy path",
        source_tag="unit_test",
    )

    stats = field.stats()
    assert stats["concepts"] >= 2
    assert stats["relations"] >= 1


def test_knowledge_audit_exposes_drift_metrics(tmp_path):
    field = _mk_field(tmp_path)
    field.add_relation(
        "Drift A",
        "beskriver",
        "Drift B",
        why="first edge",
        strength=0.2,
        evidence_score=0.2,
        assumption_flag=True,
    )
    field.add_relation(
        "Drift A",
        "beskriver",
        "Drift B",
        why="second edge",
        strength=0.9,
        evidence_score=0.9,
        assumption_flag=False,
    )
    field.upsert_concept_knowledge(
        "Drift A",
        summary="A",
        claims=["cA"],
        evidence_refs=["source:a"],
        uncertainty=0.1,
    )
    field.upsert_concept_knowledge(
        "Drift B",
        summary="B",
        claims=["cB"],
        evidence_refs=["source:b"],
        uncertainty=0.9,
    )

    audit = field.knowledge_audit(limit=10, strict=False)
    drift = audit["drift_metrics"]

    assert set(drift.keys()) >= {
        "relation_instability_score",
        "confidence_volatility",
        "contradiction_rate",
        "assumption_ratio",
        "relation_count",
        "triple_count",
        "contradictory_triples",
    }
    assert 0.0 <= drift["relation_instability_score"] <= 1.0
    assert 0.0 <= drift["confidence_volatility"] <= 1.0
    assert 0.0 <= drift["contradiction_rate"] <= 1.0
    assert drift["contradictory_triples"] >= 1
