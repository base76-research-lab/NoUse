from __future__ import annotations

from nouse.cli.chat import execute_tool, get_live_tools


class _FakeField:
    def __init__(self):
        self._domains = [f"domain_{i:03d}" for i in range(250)]

    def domains(self):
        return list(self._domains)

    def stats(self):
        return {"concepts": 1000, "relations": 2000}

    def concepts(self, *, domain: str):
        return [{"name": f"{domain}_concept_{i:03d}"} for i in range(300)]

    def add_concept(self, name: str, domain: str, source: str = "auto", ensure_knowledge: bool = True):
        self.last_added = {"name": name, "domain": domain, "source": source, "ensure_knowledge": ensure_knowledge}

    def set_concept_domain(self, name: str, domain: str):
        self.last_domain_update = {"name": name, "domain": domain}
        return True

    def upsert_concept_knowledge(self, name: str, **kwargs):
        self.last_knowledge = {"name": name, **kwargs}


def test_list_domains_is_paginated_by_default():
    field = _FakeField()
    out = execute_tool(field, "list_domains", {})
    assert out["domain_count"] == 250
    assert out["returned"] <= 120
    assert out["truncated"] is True
    assert isinstance(out["next_offset"], int)


def test_list_domains_supports_offset_and_limit():
    field = _FakeField()
    out = execute_tool(field, "list_domains", {"offset": 100, "limit": 30})
    assert out["offset"] == 100
    assert out["limit"] == 30
    assert out["returned"] == 30
    assert out["domains"][0] == "domain_100"


def test_concepts_in_domain_is_paginated():
    field = _FakeField()
    out = execute_tool(field, "concepts_in_domain", {"domain": "ai", "offset": 50, "limit": 40})
    assert out["domain"] == "ai"
    assert out["concept_count"] == 300
    assert out["returned"] == 40
    assert out["concepts"][0] == "ai_concept_050"


def test_upsert_concept_creates_or_updates_node():
    field = _FakeField()
    out = execute_tool(
        field,
        "upsert_concept",
        {"name": "Björn Wikström", "domain": "user", "summary": "profiltext"},
    )
    assert out["ok"] is True
    assert out["concept"] == "Björn Wikström"
    assert out["summary_updated"] is True
    assert field.last_added["name"] == "Björn Wikström"
    assert field.last_domain_update["domain"] == "user"


def test_live_tools_include_web_and_upsert_concept():
    tools = get_live_tools()
    names = {
        ((t or {}).get("function") or {}).get("name")
        for t in tools
        if isinstance(t, dict)
    }
    assert "upsert_concept" in names
    assert "web_search" in names
    assert "fetch_url" in names
    assert "list_local_mounts" in names
    assert "find_local_files" in names
    assert "search_local_text" in names
    assert "read_local_file" in names
