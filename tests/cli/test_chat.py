import typer
from typer.testing import CliRunner
from nouse.cli.main import app
import os

runner = CliRunner()

def test_chat_command_runs():
    """Smoke-test: Kan starta chat-kommandot och avsluta direkt."""
    result = runner.invoke(app, ["chat"], input="quit\n")
    assert result.exit_code == 0
    assert "Nous Chat" in result.output
    assert "Hejdå" in result.output


def test_chat_loads_env_before_chat_loop(monkeypatch):
    import nouse.cli.main as main_mod
    import nouse.config.env as env_mod

    seen: dict[str, str] = {}

    def _fake_load_env_files(force: bool = False):  # noqa: ARG001
        os.environ["NOUSE_DAEMON_BASE"] = "http://127.0.0.1:8876"

    def _fake_chat_via_api(**kwargs):  # noqa: ARG001
        seen["daemon_base"] = os.environ.get("NOUSE_DAEMON_BASE", "")

    monkeypatch.setattr(env_mod, "load_env_files", _fake_load_env_files)
    monkeypatch.setattr(main_mod, "_chat_via_api", _fake_chat_via_api)

    result = runner.invoke(app, ["chat", "--no-list-models"])
    assert result.exit_code == 0
    assert seen["daemon_base"] == "http://127.0.0.1:8876"
