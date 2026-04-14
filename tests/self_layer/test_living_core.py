from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

from nouse.persona import assistant_entity_name
from nouse.self_layer.living_core import (
    append_identity_memory,
    ensure_living_core,
    identity_prompt_fragment,
    load_living_core,
    operator_support_prompt_fragment,
    operator_support_snapshot,
    record_self_training_iteration,
    update_identity_profile,
    update_living_core,
)


def test_ensure_living_core_creates_default_state(tmp_path: Path):
    path = tmp_path / "living_core.json"
    state = ensure_living_core(path=path)
    assert path.exists()
    assert state["version"] >= 1
    assert "identity" in state
    assert "mission" in state["identity"]
    assert state["identity"]["name"] == assistant_entity_name()
    assert "self_training" in state
    assert "formula" in (state.get("self_training") or {})


def test_ensure_living_core_personal_mode_uses_operator_support_defaults(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("NOUSE_MODE", "personal")
    path = tmp_path / "living_core.json"
    state = ensure_living_core(path=path)
    identity = state["identity"]
    assert identity["mission"].startswith("Reduce operator overload")
    assert "low_burden_support" in identity["values"]
    assert "smallest viable next step" in identity["personality"]


def test_update_living_core_updates_homeostasis_drives_and_reflection(tmp_path: Path):
    path = tmp_path / "living_core.json"
    ensure_living_core(path=path)
    limbic = SimpleNamespace(
        dopamine=0.7,
        arousal=0.58,
        acetylcholine=1.2,
        performance=0.84,
        lam=0.67,
    )
    state = update_living_core(
        cycle=12,
        limbic=limbic,
        graph_stats={"concepts": 120, "relations": 240},
        queue_stats={"pending": 2, "in_progress": 1, "awaiting_approval": 0, "failed": 0},
        session_stats={"sessions_running": 1},
        new_relations=6,
        discoveries=2,
        bisoc_candidates=1,
        path=path,
    )
    homeo = state["homeostasis"]
    drives = state["drives"]
    reflection = state["last_reflection"]
    assert 0.0 <= float(homeo["energy"]) <= 1.0
    assert 0.0 <= float(homeo["focus"]) <= 1.0
    assert 0.0 <= float(homeo["risk"]) <= 1.0
    assert drives["active"] in {"curiosity", "maintenance", "improvement", "recovery"}
    assert reflection["cycle"] == 12
    assert isinstance(reflection["thought"], str) and reflection["thought"]
    memories = (state.get("identity") or {}).get("memories") or []
    assert memories


def test_append_identity_memory_keeps_recent_limit(tmp_path: Path):
    path = tmp_path / "living_core.json"
    ensure_living_core(path=path)
    for idx in range(260):
        append_identity_memory(
            f"note-{idx}",
            tags=["test"],
            session_id="s1",
            run_id=f"run-{idx}",
            kind="unit",
            path=path,
        )
    state = load_living_core(path=path)
    memories = (state.get("identity") or {}).get("memories") or []
    assert len(memories) == 240
    assert memories[-1]["note"] == "note-259"
    assert memories[0]["note"] == "note-20"


def test_identity_prompt_fragment_contains_identity_and_state(tmp_path: Path):
    path = tmp_path / "living_core.json"
    state = ensure_living_core(path=path)
    prompt = identity_prompt_fragment(state)
    assert "Persistent identity profile" in prompt
    assert "greeting:" in prompt
    assert "mission:" in prompt
    assert "Current regulation" in prompt
    assert "Reflection" in prompt
    assert "Self-training" in prompt


def test_legacy_identity_name_is_migrated_to_current_persona(tmp_path: Path):
    path = tmp_path / "living_core.json"
    path.write_text(
        """
        {
          "identity": {
            "name": "B76",
            "mission": "x",
            "personality": "y",
            "values": ["truth_over_guessing"],
            "boundaries": ["Separate verified facts from assumptions."],
            "memories": []
          }
        }
        """.strip(),
        encoding="utf-8",
    )
    state = load_living_core(path=path)
    assert state["identity"]["name"] == assistant_entity_name()


def test_record_self_training_iteration_updates_state_and_memory(tmp_path: Path):
    path = tmp_path / "living_core.json"
    ensure_living_core(path=path)
    state = record_self_training_iteration(
        known_data_sources=["graph", "web", "conversation"],
        meta_reflection="assumptions=none",
        reflection="Kort reflektion om senaste svar.",
        session_id="s1",
        run_id="r1",
        path=path,
    )
    st = state.get("self_training") or {}
    assert int(st.get("iterations", 0)) >= 1
    assert "graph" in (st.get("source_usage") or {})
    last = st.get("last") or {}
    assert "web" in (last.get("known_data_sources") or [])
    assert "assumptions=" in str(last.get("meta_reflection") or "")
    memories = (state.get("identity") or {}).get("memories") or []
    assert any(str(m.get("kind") or "") == "self_training" for m in memories)


def test_update_identity_profile_can_update_greeting(tmp_path: Path):
    path = tmp_path / "living_core.json"
    ensure_living_core(path=path)
    state = update_identity_profile(
        greeting="Hej, jag är {name}. Hur vill du börja?",
        path=path,
    )
    assert state["identity"]["greeting"] == f"Hej, jag är {assistant_entity_name()}. Hur vill du börja?"


def test_living_core_uses_nouse_home_when_path_not_explicit(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("NOUSE_HOME", str(tmp_path / "profile"))
    state = ensure_living_core()
    assert state["identity"]["name"] == assistant_entity_name()
    assert (tmp_path / "profile" / "self" / "living_core.json").exists()


def test_operator_support_prompt_fragment_switches_to_recovery_mode():
    state = {
        "homeostasis": {"energy": 0.2, "focus": 0.4, "risk": 0.8},
        "drives": {"active": "recovery", "goals": ["reduce load first"]},
    }
    prompt = operator_support_prompt_fragment(state)
    assert "response_mode: recovery" in prompt
    assert "keep replies brief and concrete" in prompt
    assert "reduce load first" in prompt


def test_operator_support_snapshot_detects_stalled_query_and_prefers_restart():
    state = {
        "homeostasis": {"energy": 0.55, "focus": 0.44, "risk": 0.35, "mode": "steady"},
        "drives": {"active": "maintenance", "goals": ["finish the roadmap pass"]},
        "identity": {
            "memories": [
                {
                    "ts": "2026-04-10T00:00:00+00:00",
                    "note": "Roadmap pass about NoUseAI and rescue loop",
                    "tags": ["session_memory"],
                    "kind": "note",
                }
            ]
        },
    }
    support = operator_support_snapshot("jag har fastnat och behöver hjälp mig igång", state)
    assert support["support_state"] == "stalled"
    assert support["response_mode"] == "rescue"
    assert support["intervention"] == "one_step_restart"
    assert "finish the roadmap pass" in support["next_step_hint"]
    assert support["anchors"]


def test_operator_support_prompt_fragment_surfaces_rescue_guidance():
    state = {
        "homeostasis": {"energy": 0.5, "focus": 0.4, "risk": 0.3, "mode": "steady"},
        "drives": {"active": "maintenance", "goals": ["close one open loop"]},
    }
    prompt = operator_support_prompt_fragment(state, query="jag har fastnat")
    assert "response_mode: rescue" in prompt
    assert "intervention: one_step_restart" in prompt
    assert "close one open loop" in prompt
