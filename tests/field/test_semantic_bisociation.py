from __future__ import annotations

from pathlib import Path

from nouse.field.surface import FieldSurface


def _mk_field(tmp_path: Path) -> FieldSurface:
    return FieldSurface(db_path=tmp_path / "field_semantic.sqlite", read_only=False)


def test_domain_tda_profile_prefers_semantic_embeddings_when_available(tmp_path, monkeypatch):
    field = _mk_field(tmp_path)
    field.add_concept("Hipocampus", "neuro", source="test", ensure_knowledge=False)
    field.add_concept("Arbetsminne", "neuro", source="test", ensure_knowledge=False)
    field.add_relation("Hipocampus", "modulerar", "Arbetsminne", source_tag="test")

    def _fake_ensure(rows):  # type: ignore[no-untyped-def]
        return {
            "Hipocampus": [1.0, 0.0, 0.5],
            "Arbetsminne": [0.8, 0.2, 0.4],
        }

    monkeypatch.setattr(field, "_ensure_concept_embeddings", _fake_ensure)
    profile = field.domain_tda_profile("neuro", include_centroid=True)
    assert profile["embedding_mode"] == "semantic"
    assert float(profile["embedding_coverage"]) == 1.0
    assert isinstance(profile.get("centroid"), list)
    assert len(profile["centroid"]) == 3


def test_bisociation_candidates_filters_too_semantically_similar_domains(tmp_path, monkeypatch):
    field = _mk_field(tmp_path)

    monkeypatch.setattr(field, "domains", lambda: ["A", "B"])
    monkeypatch.setattr(field, "find_path", lambda *a, **k: None)

    def _fake_profile(domain, max_epsilon=2.0, include_centroid=False):  # type: ignore[no-untyped-def]
        base = {
            "domain": domain,
            "h0": 2,
            "h1": 3,
            "n_concepts": 4,
            "embedding_mode": "semantic",
            "embedding_coverage": 1.0,
        }
        if include_centroid:
            base["centroid"] = [1.0, 0.0] if domain == "A" else [1.0, 0.0]
        return base

    monkeypatch.setattr(field, "domain_tda_profile", _fake_profile)
    none = field.bisociation_candidates(tau_threshold=0.3, semantic_similarity_max=0.95)
    assert none == []

    rows = field.bisociation_candidates(tau_threshold=0.3, semantic_similarity_max=1.0)
    assert len(rows) == 1
    row = rows[0]
    assert "semantic_similarity" in row
    assert "semantic_gap" in row
    assert "score" in row
