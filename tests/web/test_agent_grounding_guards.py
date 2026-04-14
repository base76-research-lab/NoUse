from __future__ import annotations

import nouse.web.server as ws


def test_identity_queries_do_not_use_simple_fact_shortcut():
    assert ws._is_identity_query("vem är jag")  # noqa: SLF001
    assert ws._is_identity_query("vad minns du om mig")  # noqa: SLF001
    assert not ws._is_identity_query("vad skulle du vilja veta om mig")  # noqa: SLF001
    assert not ws._is_identity_query("vilken fråga skulle hjälpa dig förstå mig bättre")  # noqa: SLF001
    assert not ws._is_simple_fact_query("vem är jag")  # noqa: SLF001
    assert ws._is_simple_fact_query("vem är kung i sverige")  # noqa: SLF001
    assert ws._is_search_info_query("vad skulle du uppdatera i systemet?")  # noqa: SLF001
    assert ws._is_search_info_query("kan du söka över disken och hitta papers?")  # noqa: SLF001
    assert not ws._is_search_info_query("vem är kung i sverige")  # noqa: SLF001
    assert not ws._is_search_info_query(  # noqa: SLF001
        "om du fick drömma, vad skulle du vilja utvecklas till då?"
    )
    assert not ws._is_search_info_query("vad drömmer du om?")  # noqa: SLF001
    assert ws._is_search_info_query("vad säger grafen om senaste kopplingarna?")  # noqa: SLF001


def test_explicit_triangulation_request_detection():
    assert not ws._is_search_info_query("/tri")  # noqa: SLF001
    assert ws._is_explicit_triangulation_request("/tri")  # noqa: SLF001
    assert ws._is_explicit_triangulation_request("/tri visa läget")  # noqa: SLF001
    assert ws._is_explicit_triangulation_request("/triangulate vad vet du om x")  # noqa: SLF001
    assert ws._is_explicit_triangulation_request("kan du triangulera svaret med källor")  # noqa: SLF001
    assert ws._is_explicit_triangulation_request("svara i formatet LLM, System/Graf, Extern, Syntes")  # noqa: SLF001
    assert not ws._is_explicit_triangulation_request("vad drömmer du om")  # noqa: SLF001


def test_explicit_tool_mode_request_detection():
    assert ws._is_explicit_tool_mode_request("/mcp hitta papers om episodic memory")  # noqa: SLF001
    assert ws._is_explicit_tool_mode_request("/skill memory.capture spara detta i minnet")  # noqa: SLF001
    assert ws._is_explicit_tool_mode_request("/skills")  # noqa: SLF001
    assert ws._is_explicit_tool_mode_request("/selfdevelop bygg plan för memory-manager")  # noqa: SLF001
    assert ws._is_explicit_tool_mode_request("använd mcp verktyg och kernel_execute_self_update")  # noqa: SLF001
    assert not ws._is_explicit_tool_mode_request("vad drömmer du om")  # noqa: SLF001


def test_extract_explicit_skill_request_parses_skill_and_payload():
    skill, payload = ws._extract_explicit_skill_request("/skill rescue hjälp mig komma igång")  # noqa: SLF001
    assert skill == "operator.rescue"
    assert payload == "hjälp mig komma igång"


def test_capability_route_prompt_block_prefers_research_for_latest_news():
    text = ws._capability_route_prompt_block("gör en web search och hitta de senaste tech-nyheterna")  # noqa: SLF001
    assert "research.triangulate" in text
    assert "web_search" in text
    assert "high_precision" in text


def test_capability_route_prompt_block_stays_quiet_for_plain_greeting():
    text = ws._capability_route_prompt_block("godmorgon")  # noqa: SLF001
    assert text == ""


def test_capability_route_prompt_block_mentions_rescue_for_stalled_operator():
    text = ws._capability_route_prompt_block("jag har fastnat och behöver hjälp mig igång")  # noqa: SLF001
    assert "operator.rescue" in text
    assert "Operatorstöd: rescue" in text


def test_capability_route_plan_prefers_temporal_grounding_for_time_query():
    route = ws._capability_route_plan("vad är klockan just nu?")  # noqa: SLF001
    assert route["skill"] == "temporal.grounding"
    assert "get_time_context" in route["tool_names"]
    assert route["tool_mode"] is True


def test_capability_route_plan_respects_explicit_skill_request():
    route = ws._capability_route_plan("/skill capture kom ihåg att jag heter Björn")  # noqa: SLF001
    assert route["skill"] == "memory.capture"
    assert route["preferred_skill"] == "memory.capture"
    assert "upsert_concept" in route["tool_names"]


def test_operational_greeting_uses_new_entity_name_and_warmer_tone():
    text = ws._operational_greeting_reply({})  # noqa: SLF001
    assert "NousAi" in text
    assert "B76" not in text
    assert "Vad vill du få ordning på just nu?" in text


def test_capability_route_plan_enables_tool_mode_for_web_search():
    route = ws._capability_route_plan("gör en web search och hitta de senaste tech-nyheterna")  # noqa: SLF001
    assert route["skill"] == "research.triangulate"
    assert route["workload"] == "agent"
    assert route["tool_mode"] is True
    assert "web_search" in route["tool_names"]


def test_capability_route_plan_adds_graph_tools_for_explicit_triangulation():
    route = ws._capability_route_plan(  # noqa: SLF001
        "kan du triangulera svaret med källor",
        explicit_tri_request=True,
    )
    assert route["tool_mode"] is True
    assert "web_search" in route["tool_names"]
    assert any(
        name in route["tool_names"]
        for name in ("list_domains", "concepts_in_domain", "explore_concept", "find_nervbana")
    )


def test_extract_urls_from_text_detects_http_and_www():
    text = "Kolla https://example.com/a?x=1 och www.aftonbladet.se/nyheter/a/KvLBGy."
    urls = ws._extract_urls_from_text(text)  # noqa: SLF001
    assert "https://example.com/a?x=1" in urls
    assert "https://www.aftonbladet.se/nyheter/a/KvLBGy" in urls


def test_auto_mcp_query_detection_for_links_and_link_reference_questions():
    assert ws._is_auto_mcp_query("vem skrev artikeln? https://aftonbladet.se/x")  # noqa: SLF001
    assert ws._is_auto_mcp_query("vem skrev den artikeln som jag länkade?")  # noqa: SLF001
    assert not ws._is_auto_mcp_query("vad drömmer du om?")  # noqa: SLF001
    assert not ws._is_auto_mcp_query("/mcp sök på detta")  # noqa: SLF001


def test_queue_failure_query_detection_from_question_and_orchestrator_block():
    assert ws._is_queue_failure_query("vad är det som failar i queue")  # noqa: SLF001
    assert ws._is_queue_failure_query("Orchestrator Check queue: pending=47 failed=6")  # noqa: SLF001
    assert not ws._is_queue_failure_query("vad drömmer du om")  # noqa: SLF001


def test_queue_failure_answer_lists_failed_tasks(monkeypatch):
    monkeypatch.setattr(
        ws,
        "queue_stats",
        lambda: {
            "pending": 47,
            "awaiting_approval": 0,
            "in_progress": 1,
            "failed": 2,
        },
    )
    monkeypatch.setattr(
        ws,
        "list_tasks",
        lambda **kwargs: [  # noqa: ARG005
            {
                "id": 42,
                "category": "research_task",
                "attempts": 3,
                "last_error": "extract_timeout",
            },
            {
                "id": 41,
                "category": "research_task",
                "attempts": 3,
                "last_error": "LLM timeout efter 45s",
            },
        ],
    )

    text = ws._queue_failure_answer(limit=3)  # noqa: SLF001
    assert "failed=2" in text
    assert "#42" in text
    assert "extract_timeout" in text
    assert "retry" in text.lower()


def test_short_confirmation_reply_detection():
    assert ws._is_short_affirmative_reply("ja")  # noqa: SLF001
    assert ws._is_short_affirmative_reply("kör det")  # noqa: SLF001
    assert ws._is_short_negative_reply("nej tack")  # noqa: SLF001
    assert not ws._is_short_affirmative_reply("kan du visa failed jobs i queue")  # noqa: SLF001
    assert not ws._is_short_negative_reply("vad drömmer du om")  # noqa: SLF001


def test_pending_confirmation_from_session_respects_ttl(monkeypatch):
    stale = "2000-01-01T00:00:00+00:00"
    action, payload = ws._pending_confirmation_from_session(  # noqa: SLF001
        {
            "meta": {
                "pending_confirmation_action": "retry_failed_tasks",
                "pending_confirmation_payload": {"limit": 4},
                "pending_confirmation_set_at": stale,
            }
        }
    )
    assert action == ""
    assert payload == {}

    fresh = ws.datetime.now(ws.timezone.utc).isoformat()  # noqa: SLF001
    action2, payload2 = ws._pending_confirmation_from_session(  # noqa: SLF001
        {
            "meta": {
                "pending_confirmation_action": "retry_failed_tasks",
                "pending_confirmation_payload": {"limit": 4},
                "pending_confirmation_set_at": fresh,
            }
        }
    )
    assert action2 == "retry_failed_tasks"
    assert payload2.get("limit") == 4


def test_queue_retry_failed_answer_reports_retry_and_remaining(monkeypatch):
    monkeypatch.setattr(
        ws,
        "retry_failed_tasks",
        lambda **kwargs: [  # noqa: ARG005
            {"id": 10, "status": "pending"},
            {"id": 9, "status": "pending"},
        ],
    )
    monkeypatch.setattr(
        ws,
        "queue_stats",
        lambda: {
            "pending": 5,
            "awaiting_approval": 0,
            "in_progress": 1,
            "failed": 1,
        },
    )
    monkeypatch.setattr(
        ws,
        "list_tasks",
        lambda **kwargs: [  # noqa: ARG005
            {"id": 8, "category": "research_task", "attempts": 4, "last_error": "timeout"},
        ],
    )
    text = ws._queue_retry_failed_answer(limit=3)  # noqa: SLF001
    assert "Körde retry på 2 failed tasks." in text
    assert "failed=1" in text
    assert "#8" in text


def test_write_scope_guard_allows_chat_and_web_sources(monkeypatch, tmp_path):
    root = tmp_path / "project"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("NOUSE_WRITE_SCOPE_ENFORCE", "1")
    monkeypatch.setenv("NOUSE_WRITE_SCOPE", str(root))

    ok_chat, scope_chat, src_chat = ws._is_source_allowed_by_write_scope("chat:main")  # noqa: SLF001
    assert ok_chat
    assert scope_chat is not None
    assert src_chat is None

    ok_web, scope_web, src_web = ws._is_source_allowed_by_write_scope("web_article:https://example.com")  # noqa: SLF001
    assert ok_web
    assert scope_web is not None
    assert src_web is None


def test_write_scope_guard_blocks_manual_path_outside_scope(monkeypatch, tmp_path):
    root = tmp_path / "project"
    outside = tmp_path / "outside"
    root.mkdir(parents=True, exist_ok=True)
    outside.mkdir(parents=True, exist_ok=True)
    file_outside = outside / "notes.md"
    file_outside.write_text("hej", encoding="utf-8")
    monkeypatch.setenv("NOUSE_WRITE_SCOPE_ENFORCE", "1")
    monkeypatch.setenv("NOUSE_WRITE_SCOPE", str(root))

    allowed, scope_root, source_path = ws._is_source_allowed_by_write_scope(  # noqa: SLF001
        f"manual:{file_outside}"
    )
    assert not allowed
    assert scope_root is not None
    assert source_path is not None


def test_write_scope_guard_allows_manual_path_inside_scope(monkeypatch, tmp_path):
    root = tmp_path / "project"
    root.mkdir(parents=True, exist_ok=True)
    file_inside = root / "doc.md"
    file_inside.write_text("hej", encoding="utf-8")
    monkeypatch.setenv("NOUSE_WRITE_SCOPE_ENFORCE", "1")
    monkeypatch.setenv("NOUSE_WRITE_SCOPE", str(root))

    allowed, scope_root, source_path = ws._is_source_allowed_by_write_scope(  # noqa: SLF001
        f"manual:{file_inside}"
    )
    assert allowed
    assert scope_root is not None
    assert source_path is not None


def test_sensitive_disclosure_detection_and_memory_policy(monkeypatch):
    q = (
        "När jag var 32 år var jag med om en trafikolycka. "
        "Min fru dog och min dotter brann inne."
    )
    assert ws._is_sensitive_disclosure_query(q)  # noqa: SLF001
    assert not ws._allow_persistent_memory_for_query(q)  # noqa: SLF001
    assert ws._allow_persistent_memory_for_query(  # noqa: SLF001
        q,
        session={"meta": {"sensitive_memory_consent": True}},
    )

    monkeypatch.setenv("NOUSE_ALLOW_SENSITIVE_MEMORY_WRITE", "1")
    assert ws._allow_persistent_memory_for_query(q)  # noqa: SLF001

    assert not ws._is_sensitive_disclosure_query(  # noqa: SLF001
        "Hur många dog i trafikolyckan enligt artikeln?"
    )


def test_extract_sensitive_memory_preference_detection():
    assert ws._extract_sensitive_memory_preference(  # noqa: SLF001
        "ja tack, du kan få mer information om du vill"
    ) == "on"
    assert ws._extract_sensitive_memory_preference(  # noqa: SLF001
        "lägg in i den beständiga profilen"
    ) == "on"
    assert ws._extract_sensitive_memory_preference("spara inte detta i minnet") == "off"  # noqa: SLF001
    assert ws._extract_sensitive_memory_preference("vad drömmer du om") is None  # noqa: SLF001


def test_sensitive_empathy_grounding_rewrites_sterile_reply():
    q = "Min fru och dotter dog i en olycka."
    answer = (
        "Jag har noterat informationen och länken du tillhandahöll. "
        "Om du vill prata vidare om det är jag här för dig."
    )
    grounded = ws._ground_sensitive_empathy(answer, q)  # noqa: SLF001
    assert "Tack för att du berättar det" in grounded
    assert "utan att spara detaljer" in grounded
    assert "noterat informationen" not in grounded.lower()


def test_legacy_backend_claims_correction_for_kuzu_migration():
    answer = "Nästa steg är att slutföra Kuzu DB migrationen innan produktion."
    grounded = ws._ground_legacy_backend_claims(answer)  # noqa: SLF001
    assert "KuzuDB är legacy/avvecklat" in grounded
    assert "SQLite WAL + NetworkX" in grounded


def test_legacy_backend_claims_no_duplicate_when_already_correct():
    answer = "Nous kör SQLite WAL + NetworkX. KuzuDB är legacy/avvecklat."
    grounded = ws._ground_legacy_backend_claims(answer)  # noqa: SLF001
    assert grounded == answer


def test_ground_capability_denials_rewrites_local_fs_disclaimer():
    caps = {"local_fs": True, "web": True, "graph_write": True}
    answer = "Jag har ingen filsystemåtkomst och kan inte läsa filer på din dator."
    grounded = ws._ground_capability_denials(answer, caps)  # noqa: SLF001
    assert "list_local_mounts" in grounded
    assert "read_local_file" in grounded


def test_ground_capability_denials_rewrites_local_write_disclaimer():
    caps = {"local_write": True}
    answer = "Jag kan inte skriva filer på disk."
    grounded = ws._ground_capability_denials(answer, caps)  # noqa: SLF001
    assert "write_local_file" in grounded


def test_ground_capability_denials_rewrites_local_exec_disclaimer():
    caps = {"local_exec": True}
    answer = "Jag kan inte köra terminalkommandon eller installera saker."
    grounded = ws._ground_capability_denials(answer, caps)  # noqa: SLF001
    assert "run_local_command" in grounded


def test_ground_unverified_link_claims_rewrites_false_read_statement():
    answer = "Jag läste länken du delade och artikeln beskriver händelsen."
    grounded = ws._ground_unverified_link_claims(answer, web_verified=False)  # noqa: SLF001
    assert "läste länken" not in grounded.lower()
    assert "kan inte verifiera länkens innehåll" in grounded.lower()


def test_ground_unverified_link_claims_keeps_text_when_web_verified():
    answer = "Jag läste länken du delade och artikeln beskriver händelsen."
    grounded = ws._ground_unverified_link_claims(answer, web_verified=True)  # noqa: SLF001
    assert grounded == answer


class _FakeField:
    def domains(self):
        return ["User", "AI"]

    def concepts(self, domain=None):
        if domain is None:
            return [
                {"name": "Björn Wikström", "domain": "User"},
                {"name": "CognOS", "domain": "AI"},
            ]
        if domain == "User":
            return [{"name": "Björn Wikström"}]
        return []

    def out_relations(self, name):
        if name == "Björn Wikström":
            return [
                {"type": "bygger", "target": "FNC"},
                {"type": "bygger", "target": "CognOS"},
            ]
        return []

    def concept_knowledge(self, name):
        if name == "Björn Wikström":
            return {"summary": "Arbetar i skärningspunkten filosofi, AI och systemdesign."}
        return {"summary": ""}

    def node_context_for_query(self, query, limit=5):
        return [
            {
                "name": "Björn Wikström",
                "summary": "Arbetar i skärningspunkten filosofi, AI och systemdesign.",
            }
        ][:limit]


def test_identity_answer_from_graph_uses_user_domain_snapshot():
    answer = ws._identity_answer_from_graph(_FakeField())  # noqa: SLF001
    assert answer is not None
    assert "Björn Wikström" in answer
    assert answer.startswith("Jag känner dig här som Björn Wikström.")
    assert "filosofi, AI och systemdesign" in answer
    assert "FNC" in answer


def test_identity_answer_from_graph_is_more_natural_in_personal_mode(monkeypatch):
    monkeypatch.setenv("NOUSE_MODE", "personal")
    answer = ws._identity_answer_from_graph(_FakeField())  # noqa: SLF001
    assert answer is not None
    assert answer.startswith("Jag känner dig här som Björn Wikström.")
    assert "domän: User" not in answer
    assert "FNC" in answer


class _FakeFieldUsernameFallback:
    def concepts(self, domain=None):
        rows = [
            {"name": "Björn Wikström", "domain": "forskning"},
            {"name": "CognOS", "domain": "AI"},
        ]
        if domain is None:
            return rows
        return [{"name": r["name"]} for r in rows if r.get("domain") == domain]

    def out_relations(self, name):
        if name == "Björn Wikström":
            return [{"type": "bygger", "target": "FNC"}]
        return []

    def concept_knowledge(self, name):
        if name == "Björn Wikström":
            return {"summary": "Forskningsarkitekt inom AI och epistemik."}
        return {"summary": ""}


def test_identity_answer_from_graph_falls_back_to_username(monkeypatch):
    monkeypatch.setenv("USER", "bjorn")
    answer = ws._identity_answer_from_graph(_FakeFieldUsernameFallback())  # noqa: SLF001
    assert answer is not None
    assert "Björn Wikström" in answer
    assert "Forskningsarkitekt inom AI och epistemik." in answer


def test_system_search_info_snapshot_includes_graph_hits():
    caps = {"local_fs": False, "web": True, "graph_write": True}
    text = ws._system_search_info_snapshot(  # noqa: SLF001
        field=_FakeField(),
        query="vad vet du om björn och fnc",
        caps=caps,
    )
    assert "SYSTEM_SEARCH_INFO" in text
    assert "Grafträffar" in text
    assert "Björn Wikström" in text


def test_tool_source_bucket_classifies_graph_and_web_tools():
    assert ws._tool_source_bucket("list_domains") == "graph"  # noqa: SLF001
    assert ws._tool_source_bucket("explore_concept") == "graph"  # noqa: SLF001
    assert ws._tool_source_bucket("web_search") == "web"  # noqa: SLF001
    assert ws._tool_source_bucket("fetch_url") == "web"  # noqa: SLF001
    assert ws._tool_source_bucket("read_local_file") == "local"  # noqa: SLF001
    assert ws._tool_source_bucket("unknown_tool") == "other"  # noqa: SLF001


def test_missing_triangulation_sources_reports_graph_and_web():
    missing = ws._missing_triangulation_sources(  # noqa: SLF001
        {"local"},
        require_graph=True,
        require_web=True,
    )
    assert missing == ["graf", "webb"]
    missing_graph_only = ws._missing_triangulation_sources(  # noqa: SLF001
        {"graph"},
        require_graph=True,
        require_web=False,
    )
    assert missing_graph_only == []


def test_looks_like_triangulated_response_requires_all_sections():
    ok = """
    LLM:
    - intern punkt
    System/Graf:
    - grafpunkt
    Extern:
    - webbkalla
    Syntes:
    - slutsats
    """
    assert ws._looks_like_triangulated_response(ok)  # noqa: SLF001
    bad = "LLM: x\nSystem/Graf: y\nSyntes: z"
    assert not ws._looks_like_triangulated_response(bad)  # noqa: SLF001


def test_graph_action_request_detection():
    assert ws._is_graph_action_request(  # noqa: SLF001
        "lägg till ny nod i grafen och koppla den till AI"
    )
    assert ws._is_graph_action_request(  # noqa: SLF001
        "add relation mellan model och evidence i graph"
    )
    assert ws._is_graph_action_request(  # noqa: SLF001
        "ändra så att jag bara är Björn och står som USER"
    )
    assert ws._is_graph_action_request(  # noqa: SLF001
        "kan du ta bort Claude från mig: jag är bara Björn"
    )
    assert not ws._is_graph_action_request(  # noqa: SLF001
        "vad är skillnaden mellan ai och llm?"
    )


def test_confirmation_prompt_detection():
    assert ws._looks_like_confirmation_prompt(  # noqa: SLF001
        "Vänligen bekräfta: vill du att jag fortsätter?"
    )
    assert ws._looks_like_confirmation_prompt(  # noqa: SLF001
        "Vad ska vi prioritera?"
    )
    assert not ws._looks_like_confirmation_prompt(  # noqa: SLF001
        "Jag har nu lagt till noden och relationerna."
    )


def test_explicit_delegate_intent_detection():
    assert ws._is_explicit_delegate_intent("använd subagents för det här jobbet")  # noqa: SLF001
    assert ws._is_explicit_delegate_intent("/orchestrate bygg en plan")  # noqa: SLF001
    assert ws._is_explicit_delegate_intent("kan du delegera detta till bakgrundsagenterna")  # noqa: SLF001
    assert not ws._is_explicit_delegate_intent(  # noqa: SLF001
        "vill du göra ett tankeexperiment utan att skicka något till subagenter?!"
    )
    assert not ws._is_explicit_delegate_intent(  # noqa: SLF001
        "du behöver delegera när det är flersteg eller större jobb"
    )
    assert not ws._is_explicit_delegate_intent("vem är kung i sverige")  # noqa: SLF001


def test_extract_delegate_preference_command_detection():
    assert ws._extract_delegate_preference_command("sluta deligera") == "off"  # noqa: SLF001
    assert ws._extract_delegate_preference_command("stäng av delegering") == "off"  # noqa: SLF001
    assert ws._extract_delegate_preference_command("aktivera delegering") == "on"  # noqa: SLF001
    assert ws._extract_delegate_preference_command("delegera igen") == "on"  # noqa: SLF001
    assert ws._extract_delegate_preference_command("vem är du") is None  # noqa: SLF001


def test_background_delegate_respects_explicit_agent_intent(monkeypatch):
    monkeypatch.setattr(ws, "FAST_DELEGATE_ENABLED", True)
    monkeypatch.setattr(ws, "FAST_DELEGATE_MIN_WORDS", 99)

    # Kort text med explicit agent-intent ska ändå delegeras.
    assert ws._is_background_delegate_request("använd agent för detta")  # noqa: SLF001

    # Ren faktafråga utan intent ska inte tvångsdelegeras.
    assert not ws._is_background_delegate_request("vem är du")  # noqa: SLF001


def test_background_delegate_respects_session_delegate_disabled(monkeypatch):
    monkeypatch.setattr(ws, "FAST_DELEGATE_ENABLED", True)
    monkeypatch.setattr(ws, "FAST_DELEGATE_IMPLICIT", True)
    monkeypatch.setattr(ws, "FAST_DELEGATE_MIN_WORDS", 3)

    q = "implementera en ny modul för orchestrering och deploy i repo"
    assert ws._is_background_delegate_request(  # noqa: SLF001
        q,
        session={"meta": {"delegate_enabled": True}},
    )
    assert not ws._is_background_delegate_request(  # noqa: SLF001
        q,
        session={"meta": {"delegate_enabled": False}},
    )


def test_background_delegate_keeps_conversational_invite_in_foreground(monkeypatch):
    monkeypatch.setattr(ws, "FAST_DELEGATE_ENABLED", True)
    monkeypatch.setattr(ws, "FAST_DELEGATE_IMPLICIT", True)
    monkeypatch.setattr(ws, "FAST_DELEGATE_MIN_WORDS", 3)

    q = "vill du testa en spännande lek?"
    assert ws._is_conversational_invite_query(q)  # noqa: SLF001
    assert not ws._is_background_delegate_request(q)  # noqa: SLF001


def test_background_delegate_keeps_sensitive_disclosure_in_foreground(monkeypatch):
    monkeypatch.setattr(ws, "FAST_DELEGATE_ENABLED", True)
    monkeypatch.setattr(ws, "FAST_DELEGATE_IMPLICIT", True)
    monkeypatch.setattr(ws, "FAST_DELEGATE_MIN_WORDS", 3)

    q = "Jag förlorade min dotter i en trafikolycka och behöver prata om det."
    assert ws._is_sensitive_disclosure_query(q)  # noqa: SLF001
    assert not ws._is_background_delegate_request(q)  # noqa: SLF001


def test_background_delegate_respects_no_delegate_intent_with_subagents(monkeypatch):
    monkeypatch.setattr(ws, "FAST_DELEGATE_ENABLED", True)
    monkeypatch.setattr(ws, "FAST_DELEGATE_IMPLICIT", True)
    monkeypatch.setattr(ws, "FAST_DELEGATE_MIN_WORDS", 3)

    q = "vill du göra ett tankeexperiment utan att skicka något till subagenter?! bara jag och du"
    assert ws._is_no_delegate_intent(q)  # noqa: SLF001
    assert not ws._is_background_delegate_request(q)  # noqa: SLF001


def test_background_delegate_does_not_override_graph_action(monkeypatch):
    monkeypatch.setattr(ws, "FAST_DELEGATE_ENABLED", True)
    monkeypatch.setattr(ws, "FAST_DELEGATE_MIN_WORDS", 8)

    q = "ta bort Claude från mig som person i grafen och skapa domänen User"
    assert ws._is_graph_action_request(q)  # noqa: SLF001
    assert not ws._is_background_delegate_request(q)  # noqa: SLF001


def test_background_delegate_implicit_off_keeps_strategic_question_in_foreground(monkeypatch):
    monkeypatch.setattr(ws, "FAST_DELEGATE_ENABLED", True)
    monkeypatch.setattr(ws, "FAST_DELEGATE_IMPLICIT", False)
    monkeypatch.setattr(ws, "FAST_DELEGATE_MIN_WORDS", 8)

    q = "hur går vi från nouse i sitt nuvarande tillstånd till nästa steg mellan llm ai agi"
    assert not ws._is_background_delegate_request(q)  # noqa: SLF001


def test_build_all_models_failed_error_adds_timeout_hint():
    msg = ws._build_all_models_failed_error(  # noqa: SLF001
        "agent/tools",
        [
            {
                "model": "glm-5.1:cloud",
                "reason": "service_unavailable",
                "error": "Service Temporarily Unavailable (status code: 503)",
            },
            {
                "model": "gemma4:e2b",
                "reason": "timeout",
                "error": "LLM timeout efter 45.0s",
            },
        ],
    )
    assert "NOUSE_AGENT_LLM_TIMEOUT_SEC" in msg
    assert "/model 2" in msg


def test_evaluate_model_health_status_levels():
    now_ts = 1000.0
    assert ws._evaluate_model_health_status(None, now_ts=now_ts) == "unknown"  # noqa: SLF001
    assert ws._evaluate_model_health_status(  # noqa: SLF001
        {"success": 3, "failure": 0, "timeout": 0, "cooldown_until": 0.0},
        now_ts=now_ts,
    ) == "working"
    assert ws._evaluate_model_health_status(  # noqa: SLF001
        {"success": 1, "failure": 0, "timeout": 1, "cooldown_until": 0.0},
        now_ts=now_ts,
    ) == "degraded"
    assert ws._evaluate_model_health_status(  # noqa: SLF001
        {"success": 0, "failure": 3, "timeout": 1, "cooldown_until": 2000.0},
        now_ts=now_ts,
    ) == "not_working"


def test_model_health_for_workload_marks_degraded_on_primary_failure(monkeypatch):
    monkeypatch.setattr(
        ws,
        "get_workload_policy",
        lambda _w: {"provider": "ollama", "candidates": ["glm-5.1:cloud", "gemma4:e2b"]},
    )
    monkeypatch.setattr(
        ws,
        "resolve_model_candidates",
        lambda _w, defaults: list(defaults),
    )
    monkeypatch.setattr(
        ws,
        "router_status",
        lambda workload=None: {  # noqa: ARG005
            "updated_at": "2026-04-08T00:00:00+00:00",
            "workloads": {
                "agent": [
                    {
                        "model": "glm-5.1:cloud",
                        "success": 0,
                        "failure": 2,
                        "timeout": 1,
                        "consecutive_timeouts": 1,
                        "cooldown_until": 0.0,
                        "updated": "2026-04-08T00:00:00+00:00",
                    },
                    {
                        "model": "gemma4:e2b",
                        "success": 3,
                        "failure": 0,
                        "timeout": 0,
                        "consecutive_timeouts": 0,
                        "cooldown_until": 0.0,
                        "updated": "2026-04-08T00:00:00+00:00",
                    },
                ]
            },
        },
    )

    payload = ws._model_health_for_workload("agent")  # noqa: SLF001
    assert payload["status"] == "degraded"
    assert payload["label"] == "Degraded"
    assert "Fallback aktiv" in payload["detail"]


def test_order_models_with_sticky_primary_keeps_user_choice_first(monkeypatch):
    monkeypatch.setattr(ws, "order_models_for_workload", lambda _w, cands: list(reversed(cands)))
    ordered = ws._order_models_with_sticky_primary(  # noqa: SLF001
        "agent",
        ["ollama/qwen3.5:latest", "ollama/gemma4:e2b", "openai_compatible/gpt-4.1-mini"],
    )
    assert ordered[0] == "ollama/qwen3.5:latest"
    assert set(ordered) == {
        "ollama/qwen3.5:latest",
        "ollama/gemma4:e2b",
        "openai_compatible/gpt-4.1-mini",
    }


def test_identity_action_parsers_extract_name_alias_and_domain():
    q = "ta bort Claude från mig, jag är bara Björn, skapa domän som heter User"
    assert ws._extract_identity_name_from_query(q) == "Björn"  # noqa: SLF001
    assert ws._extract_identity_domain_from_query(q) == "User"  # noqa: SLF001
    aliases = ws._extract_identity_aliases_to_remove(q)  # noqa: SLF001
    assert "Claude" in aliases


def test_identity_graph_action_fallback_applies_direct_writes(monkeypatch):
    calls: list[tuple[str, dict]] = []

    def _fake_execute_tool(_field, name, args):
        calls.append((str(name), dict(args)))
        return {"ok": True}

    monkeypatch.setattr(ws, "execute_tool", _fake_execute_tool)
    monkeypatch.delenv("NOUSE_MODE", raising=False)

    result = ws._apply_identity_graph_action_fallback(  # noqa: SLF001
        field=object(),
        query="ta bort Claude från mig och ändra så att jag är bara Björn och står som USER",
    )
    assert result.get("applied") is True
    assert "Grafen uppdaterad" in str(result.get("answer") or "")

    upserts = [args for name, args in calls if name == "upsert_concept"]
    assert any(str(row.get("name") or "") == "Björn" and str(row.get("domain") or "") == "USER" for row in upserts)
    assert any(str(row.get("name") or "") == "Claude" and str(row.get("domain") or "") == "AI" for row in upserts)
    assert any(name == "add_relation" for name, _args in calls)


def test_identity_graph_action_fallback_uses_human_ack_in_personal_mode(monkeypatch):
    calls: list[tuple[str, dict]] = []

    def _fake_execute_tool(_field, name, args):
        calls.append((str(name), dict(args)))
        return {"ok": True}

    monkeypatch.setattr(ws, "execute_tool", _fake_execute_tool)
    monkeypatch.setenv("NOUSE_MODE", "personal")

    result = ws._apply_identity_graph_action_fallback(  # noqa: SLF001
        field=object(),
        query="ta bort Claude från mig och ändra så att jag är bara Björn och står som USER",
    )
    answer = str(result.get("answer") or "")
    assert result.get("applied") is True
    assert "Jag har uppdaterat det" in answer
    assert "Grafen uppdaterad" not in answer
    assert any(name == "add_relation" for name, _args in calls)


def test_ground_capability_denials_rewrites_read_only_claim_when_trusted_local_enabled():
    answer = (
        "Ja, jag kan skapa noder i grafen. Nej, jag kan inte skriva, skapa eller modifiera lokala filer "
        "på disk — filverktygen är read-only. Jag kan inte köra terminalkommandon eller installera saker."
    )
    rewritten = ws._ground_capability_denials(  # noqa: SLF001
        answer,
        {"local_fs": True, "local_write": True, "local_exec": True, "graph_write": True, "web": False},
    )
    assert "skriva lokala textfiler" in rewritten or "både skriva lokala textfiler och köra lokala shell-kommandon" in rewritten
    assert "köra" in rewritten


def test_ground_capability_denials_rewrites_missing_graph_tools_claim():
    answer = "I den här specifika körningen har jag inga grafverktyg laddade, så jag kan inte skapa noden direkt från min sida."
    rewritten = ws._ground_capability_denials(  # noqa: SLF001
        answer,
        {"local_fs": False, "local_write": False, "local_exec": False, "graph_write": True, "web": False},
    )
    assert "Jag har grafverktyg i denna körning" in rewritten


def test_ground_personal_graph_write_ack_rewrites_technical_diff(monkeypatch):
    monkeypatch.setenv("NOUSE_MODE", "personal")
    answer = (
        "Klart. Här är exakt vad som ändrades:\n\n"
        "Noder (upsert):\n\n"
        " • Björn (domain: person) — uppdaterad summary\n"
        "Relationer (add):\n\n"
        " 1 Björn --[är_del_av]--> User_context\n"
        "Evidenskällor: Samtliga härrör från sessionsinteraktion."
    )
    rewritten = ws._ground_personal_graph_write_ack(answer)  # noqa: SLF001
    assert rewritten.startswith("Jag har uppdaterat det och sparat det som en del av din profil här.")
    assert "Björn" in rewritten
    assert "tekniska ändringen" in rewritten


def test_auth_provider_plan_openai_and_groq():
    default_plan = ws._auth_key_plan_for_provider("codex")  # noqa: SLF001
    assert default_plan["provider"] == "openai_compatible"
    assert "NOUSE_OPENAI_API_KEY" in default_plan["keys"]
    assert str(default_plan["openai_base_url"]).startswith("https://api.openai.com/")

    groq_plan = ws._auth_key_plan_for_provider("groq")  # noqa: SLF001
    assert "GROQ_API_KEY" in groq_plan["keys"]
    assert "OPENAI_API_KEY" in groq_plan["keys"]
    assert str(groq_plan["openai_base_url"]).startswith("https://api.groq.com/")


def test_is_local_request_host_guard():
    assert ws._is_local_request_host("127.0.0.1")  # noqa: SLF001
    assert ws._is_local_request_host("127.0.1.25")  # noqa: SLF001
    assert ws._is_local_request_host("::1")  # noqa: SLF001
    assert ws._is_local_request_host("localhost")  # noqa: SLF001
    assert not ws._is_local_request_host("10.0.0.2")  # noqa: SLF001
    assert not ws._is_local_request_host("")  # noqa: SLF001
