import typer
from typer.testing import CliRunner
from nouse.cli.main import app

runner = CliRunner()

def test_chat_command_runs():
    """Smoke-test: Kan starta chat-kommandot och avsluta direkt."""
    result = runner.invoke(app, ["chat"], input="quit\n")
    assert result.exit_code == 0
    assert "NoUse Chat" in result.output
    assert "Hejdå" in result.output
