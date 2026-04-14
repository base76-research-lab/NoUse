from __future__ import annotations

from typer.testing import CliRunner

import nouse.client as client
from nouse.cli.main import app

runner = CliRunner()


def test_start_me_routes_to_chat(monkeypatch):
    called: dict = {}

    monkeypatch.setattr(client, "daemon_running", lambda: True)

    def _fake_chat(*, session_id: str = "main", **_kwargs):  # noqa: ARG001
        called["session_id"] = session_id

    import nouse.cli.main as main_mod

    monkeypatch.setattr(main_mod, "_chat_via_api", _fake_chat)
    result = runner.invoke(app, ["start", "me", "--session-id", "bjorn"])
    assert result.exit_code == 0
    assert called["session_id"] == "bjorn"


def test_start_autonomy_prints_overview(monkeypatch):
    monkeypatch.setattr(client, "daemon_running", lambda: True)
    monkeypatch.setattr(client, "get_status", lambda: {"concepts": 10, "relations": 20, "cycle": 3})
    monkeypatch.setattr(client, "get_system_events", lambda limit=8: {"stats": {"pending_total": 2}})
    monkeypatch.setattr(
        client,
        "brain_clawbot_allowlist",
        lambda channel="ops": {"allowed": ["u1"], "pending": []},  # noqa: ARG005
    )

    result = runner.invoke(app, ["start", "autonomy"])
    assert result.exit_code == 0
    assert "Autonomy Overview" in result.output
    assert "clawbot_ops_allowed=1" in result.output


def test_start_invalid_mode_errors():
    result = runner.invoke(app, ["start", "unknown"])
    assert result.exit_code == 1
    assert "Ogiltigt mode" in result.output
