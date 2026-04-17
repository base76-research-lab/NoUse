from __future__ import annotations

import json
import re
from typing import Any

from nouse.config.paths import path_from_env

_SKILL_ALIASES: dict[str, str] = {
    "rescue": "operator.rescue",
    "operator.rescue": "operator.rescue",
    "capture": "memory.capture",
    "memory.capture": "memory.capture",
    "tri": "research.triangulate",
    "triangulate": "research.triangulate",
    "research.triangulate": "research.triangulate",
    "time": "temporal.grounding",
    "clock": "temporal.grounding",
    "temporal.grounding": "temporal.grounding",
}


def resolve_skill_name(name: str) -> str:
    raw = str(name or "").strip().lower()
    if not raw:
        return ""
    return _SKILL_ALIASES.get(raw, raw)


def _tokens(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[0-9a-zåäö_]+", str(text or "").lower()) if tok}


def _is_time_query(text: str, tokens: set[str]) -> bool:
    return bool(
        "klockan" in tokens
        or "klockslag" in tokens
        or "tid" in tokens
        or "today" in tokens
        or "idag" in tokens
        or "imorgon" in tokens
        or "veckodag" in tokens
        or "date" in tokens
        or "datum" in tokens
        or "just nu" in text
    )


def _is_rescue_query(text: str, tokens: set[str], state: str) -> bool:
    support_markers = {
        "fastnat",
        "stuck",
        "överväldigad",
        "overwhelmed",
        "hjälp",
        "help",
        "igång",
        "restart",
    }
    if any(tok in tokens for tok in support_markers):
        return True
    return state in {"stalled", "overload", "low_energy", "rescue", "recovery"}


def _is_web_query(text: str, tokens: set[str], needs_web: bool) -> bool:
    if needs_web:
        return True
    markers = {
        "web",
        "search",
        "senaste",
        "latest",
        "nyheter",
        "news",
        "internet",
    }
    return any(tok in tokens for tok in markers) or "web search" in text


def _is_memory_capture_query(text: str, tokens: set[str], needs_memory_write: bool) -> bool:
    if needs_memory_write:
        return True
    return bool(
        "kom ihåg" in text
        or "remember" in text
        or "spara" in text
        or "minnet" in text
        or "memory" in tokens
    )


def _toolset_for_skill(skill: str) -> list[str]:
    if skill == "temporal.grounding":
        return ["get_time_context"]
    if skill == "research.triangulate":
        return ["web_search", "fetch_url"]
    if skill == "memory.capture":
        return ["upsert_concept", "add_relation", "kernel_write_episode"]
    if skill == "operator.rescue":
        return []
    return []


def _governance_for_skill(skill: str) -> str:
    if skill == "research.triangulate":
        return "high_precision"
    if skill == "memory.capture":
        return "memory_safe_write"
    if skill == "temporal.grounding":
        return "ground_now"
    if skill == "operator.rescue":
        return "low_burden_support"
    return "default"


def build_route_plan(
    query: str,
    *,
    state: str = "",
    needs_web: bool = False,
    needs_files: bool = False,
    needs_memory_write: bool = False,
    needs_action: bool = False,
    force_tooling: bool = False,
    preferred_skill: str = "",
    probe_models: bool = False,  # noqa: ARG001
) -> dict[str, Any]:
    text = str(query or "").strip().lower()
    tokens = _tokens(text)
    pref = resolve_skill_name(preferred_skill)

    if pref:
        skill = pref
        score = 1.0
        reasons = ["explicit_skill"]
    elif _is_time_query(text, tokens):
        skill = "temporal.grounding"
        score = 0.95
        reasons = ["time_query"]
    elif _is_rescue_query(text, tokens, str(state or "").strip().lower()):
        skill = "operator.rescue"
        score = 0.91
        reasons = ["operator_support"]
    elif _is_memory_capture_query(text, tokens, needs_memory_write):
        skill = "memory.capture"
        score = 0.88
        reasons = ["memory_write"]
    elif _is_web_query(text, tokens, needs_web):
        skill = "research.triangulate"
        score = 0.9
        reasons = ["web_research"]
    else:
        skill = "dialogue.personal"
        score = 0.22
        reasons = ["default_fit"]

    tool_names = _toolset_for_skill(skill)

    if needs_files:
        tool_names.extend(
            [
                "list_local_mounts",
                "find_local_files",
                "search_local_text",
                "read_local_file",
            ]
        )
    if needs_action and skill not in {"memory.capture", "research.triangulate"}:
        tool_names.extend(["upsert_concept", "add_relation", "explore_concept"])

    dedup_tools: list[str] = []
    seen: set[str] = set()
    for tool in tool_names:
        if tool in seen:
            continue
        seen.add(tool)
        dedup_tools.append(tool)

    tool_mode = bool(force_tooling or dedup_tools)
    workload = "agent" if tool_mode else "chat"

    return {
        "skill": skill,
        "skill_score": float(score),
        "skill_reasons": reasons,
        "workload": workload,
        "provider": "auto",
        "governance": _governance_for_skill(skill),
        "tool_names": dedup_tools,
        "tool_mode": tool_mode,
    }


def recommend_capability_route(
    query: str,
    *,
    state: str = "",
    needs_web: bool = False,
    needs_files: bool = False,
    needs_memory_write: bool = False,
    needs_action: bool = False,
    probe_models: bool = False,
) -> dict[str, Any]:
    return build_route_plan(
        query,
        state=state,
        needs_web=needs_web,
        needs_files=needs_files,
        needs_memory_write=needs_memory_write,
        needs_action=needs_action,
        force_tooling=False,
        preferred_skill="",
        probe_models=probe_models,
    )


def filter_tool_schemas(
    tool_schemas: list[dict[str, Any]],
    allowed_tool_names: list[str],
) -> list[dict[str, Any]]:
    allowed = {str(x or "").strip() for x in allowed_tool_names if str(x or "").strip()}
    if not allowed:
        return []
    out: list[dict[str, Any]] = []
    for schema in tool_schemas or []:
        fn = ((schema or {}).get("function") or {}) if isinstance(schema, dict) else {}
        name = str(fn.get("name") or "").strip()
        if name and name in allowed:
            out.append(schema)
    return out


def build_capability_graph(*, probe_models: bool = False) -> dict[str, Any]:  # noqa: ARG001
    skills = sorted(set(_SKILL_ALIASES.values()) | {"dialogue.personal"})
    tools = [
        "web_search",
        "fetch_url",
        "get_time_context",
        "list_local_mounts",
        "find_local_files",
        "search_local_text",
        "read_local_file",
        "upsert_concept",
        "add_relation",
        "kernel_write_episode",
    ]
    planes = {
        "skill_plane": {
            "name": "skills",
            "skills": [{"name": skill} for skill in skills],
        },
        "mcp_plane": {
            "name": "tools",
            "tools": [{"name": tool} for tool in tools],
        },
        "opencode_model_plane": {
            "name": "models",
            "workloads": [
                {"name": "chat", "provider": "auto"},
                {"name": "agent", "provider": "auto"},
            ],
        },
    }
    return {
        "version": 1,
        "planes": planes,
        "counts": {
            "planes": len(planes),
            "bridges": 0,
            "tools": len(tools),
            "skills": len(skills),
            "providers": 1,
            "models": 1,
        },
    }


def save_capability_graph(snapshot: dict[str, Any]) -> dict[str, Any]:
    path = path_from_env("NOUSE_CAPABILITY_GRAPH_PATH", "capability_graph.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"path": str(path), "saved": True}


def index_capability_graph(field: Any, snapshot: dict[str, Any]) -> dict[str, Any]:
    try:
        count = int(((snapshot.get("counts") or {}).get("skills") or 0))
    except Exception:
        count = 0
    return {"indexed": True, "skills": count, "field": type(field).__name__}
