from __future__ import annotations

from pathlib import Path

import httpx

from nouse.mcp_gateway import gateway


def test_get_time_context_returns_local_and_utc_fields():
    out = gateway.get_time_context("UTC")
    assert out["source"] == "system_clock"
    assert out["timezone"] == "UTC"
    assert out["now_local"].endswith("+00:00")
    assert out["now_utc"].endswith("+00:00")
    assert len(str(out["date_local"])) == 10
    assert len(str(out["time_local"])) == 8
    assert out["weekday_local"]
    assert out["weekday_local_en"]
    assert isinstance(out["unix_ts"], int)


def test_execute_mcp_tool_get_time_context_forwards_timezone():
    out = gateway.execute_mcp_tool("get_time_context", {"timezone": "UTC"})
    assert out["timezone"] == "UTC"
    assert out["today_local"] == out["date_local"]


def test_web_search_falls_back_to_duckduckgo_html(monkeypatch):
    class _BrokenDDGS:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def text(self, query: str, max_results: int = 5):  # noqa: ARG002
            raise RuntimeError("ddgs unavailable")

    monkeypatch.setattr(gateway, "DDGS", _BrokenDDGS)
    monkeypatch.setattr(
        gateway,
        "_search_duckduckgo_html",
        lambda query, max_results=5: [  # noqa: ARG005
            {
                "title": "Example",
                "href": "https://example.com",
                "body": "fallback result",
            }
        ],
    )

    out = gateway.web_search("example", max_results=3)
    assert out["provider"] == "duckduckgo_html"
    assert out["results"]
    assert out["results"][0]["href"] == "https://example.com"


def test_web_search_prefers_requested_brave_provider(monkeypatch):
    calls: list[str] = []

    monkeypatch.setenv("BRAVE_API_KEY", "test-key")
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    def _fake_brave(query: str, max_results: int, api_key: str):  # noqa: ARG001
        calls.append("brave")
        return {
            "provider": "brave",
            "query": query,
            "results": [{"title": "x", "href": "https://example.com", "body": "ok"}],
        }

    monkeypatch.setattr(gateway, "_search_brave", _fake_brave)

    out = gateway.web_search("example", max_results=2, provider="brave")
    assert out["provider"] == "brave"
    assert out.get("provider_requested") == "brave"
    assert calls == ["brave"]


def test_execute_mcp_tool_web_search_forwards_provider(monkeypatch):
    seen: dict[str, str] = {}

    def _fake_web_search(query: str, max_results: int = 5, provider: str | None = None):
        seen["query"] = query
        seen["provider"] = str(provider or "")
        return {"provider": "duckduckgo", "query": query, "results": []}

    monkeypatch.setattr(gateway, "web_search", _fake_web_search)

    out = gateway.execute_mcp_tool(
        "web_search",
        {"query": "epistemic grounding", "max_results": 3, "provider": "brave"},
    )
    assert out["query"] == "epistemic grounding"
    assert seen["provider"] == "brave"


def test_execute_mcp_tool_web_search_defaults_to_brave_when_key_present(monkeypatch):
    seen: dict[str, str] = {}

    monkeypatch.setenv("BRAVE_API_KEY", "x-test")
    monkeypatch.delenv("NOUSE_WEB_SEARCH_DEFAULT_PROVIDER", raising=False)

    def _fake_web_search(query: str, max_results: int = 5, provider: str | None = None):
        seen["query"] = query
        seen["provider"] = str(provider or "")
        return {"provider": "brave", "query": query, "results": []}

    monkeypatch.setattr(gateway, "web_search", _fake_web_search)

    out = gateway.execute_mcp_tool(
        "web_search",
        {"query": "epistemic grounding", "max_results": 3},
    )
    assert out["query"] == "epistemic grounding"
    assert seen["provider"] == "brave"


def test_fetch_url_handles_pdf_content(monkeypatch):
    class _DummyResp:
        status_code = 200
        headers = {"content-type": "application/pdf"}
        text = ""
        content = b"%PDF-1.4 dummy"

        def raise_for_status(self) -> None:
            return None

    class _DummyClient:
        def __init__(self, *args, **kwargs):  # noqa: ARG002
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url: str, **kwargs):  # noqa: ARG002
            return _DummyResp()

    monkeypatch.setattr(gateway.httpx, "Client", _DummyClient)
    monkeypatch.setattr(
        gateway,
        "_extract_pdf_from_bytes",
        lambda content, max_chars=4000: {  # noqa: ARG005
            "content": "PDF extracted text",
            "truncated": False,
        },
    )

    out = gateway.fetch_url("https://example.com/paper.pdf")
    assert out["source"] == "direct_fetch_pdf"
    assert out["content"] == "PDF extracted text"
    assert out["truncated"] is False


def test_normalize_duckduckgo_redirect_href():
    href = (
        "//duckduckgo.com/l/?uddg=https%3A%2F%2Fbase76.se%2Fen%2F"
        "&rut=abc"
    )
    out = gateway._normalize_duckduckgo_href(href)
    assert out == "https://base76.se/en/"


def test_read_pdf_text_falls_back_to_extract_text_when_pdftotext_missing(
    tmp_path: Path, monkeypatch
):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    monkeypatch.setattr(gateway.shutil, "which", lambda name: None)
    monkeypatch.setattr(gateway, "extract_text", lambda p: "fallback pdf text")  # noqa: ARG005

    out = gateway._read_pdf_text(pdf, max_chars=1000)
    assert out["content"] == "fallback pdf text"
    assert out["truncated"] is False


def test_write_local_file_requires_trusted_local_runtime(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("NOUSE_TRUSTED_LOCAL_AUTONOMY", raising=False)
    monkeypatch.delenv("NOUSE_LOCAL_FILE_WRITE_ENABLED", raising=False)

    out = gateway.write_local_file(str(tmp_path / "note.txt"), "hej")
    assert "disabled" in str(out.get("error") or "")


def test_write_local_file_writes_when_enabled(monkeypatch, tmp_path: Path):
    target = tmp_path / "notes" / "day.md"
    monkeypatch.setenv("NOUSE_LOCAL_FILE_WRITE_ENABLED", "1")
    monkeypatch.setenv("NOUSE_LOCAL_WRITE_ROOTS", str(tmp_path))

    out = gateway.write_local_file(
        str(target),
        "# logg\n",
        create_dirs=True,
    )
    assert out["ok"] is True
    assert target.read_text(encoding="utf-8") == "# logg\n"


def test_run_local_command_requires_trusted_local_runtime(monkeypatch):
    monkeypatch.delenv("NOUSE_TRUSTED_LOCAL_AUTONOMY", raising=False)
    monkeypatch.delenv("NOUSE_LOCAL_SHELL_ENABLED", raising=False)

    out = gateway.run_local_command("pwd")
    assert "disabled" in str(out.get("error") or "")


def test_run_local_command_executes_when_enabled(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("NOUSE_LOCAL_SHELL_ENABLED", "1")
    monkeypatch.setenv("NOUSE_LOCAL_EXEC_ROOTS", str(tmp_path))

    out = gateway.run_local_command("pwd", workdir=str(tmp_path))
    assert out["ok"] is True
    assert str(tmp_path) in str(out.get("stdout") or "")


def test_mcp_tool_enabled_reflects_trusted_local_env(monkeypatch):
    monkeypatch.delenv("NOUSE_TRUSTED_LOCAL_AUTONOMY", raising=False)
    monkeypatch.delenv("NOUSE_LOCAL_FILE_WRITE_ENABLED", raising=False)
    monkeypatch.delenv("NOUSE_LOCAL_SHELL_ENABLED", raising=False)
    assert gateway.mcp_tool_enabled("write_local_file") is False
    assert gateway.mcp_tool_enabled("run_local_command") is False

    monkeypatch.setenv("NOUSE_TRUSTED_LOCAL_AUTONOMY", "1")
    assert gateway.mcp_tool_enabled("write_local_file") is True
    assert gateway.mcp_tool_enabled("run_local_command") is True
