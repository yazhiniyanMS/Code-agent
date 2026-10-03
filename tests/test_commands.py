import pytest

from ycode.app import YCodeApp
from ycode.commands import COMMANDS, handle_command, is_command
from ycode.config import Config

from tests.conftest import FakeProvider, console_text, recording_console, reply
from ycode.ui.console import ConsoleUI


@pytest.fixture
def app(git_workspace):
    ui = ConsoleUI(recording_console())
    return YCodeApp(config=Config(max_steps=50), workspace=git_workspace, provider=FakeProvider(), ui=ui)


def output(app):
    return console_text(app.ui.console)


def test_is_command():
    assert is_command("/help") and is_command("/diff src/a.py")
    assert not is_command("fix /etc/hosts parsing") and not is_command("/etc/hosts is wrong")


def test_help_lists_commands(app):
    assert handle_command(app, "/help")
    out = output(app)
    for name in ("/help", "/clear", "/status", "/diff", "/files", "/model", "/exit"):
        assert name in out


def test_status(app):
    (app.workspace / "app.py").write_text("changed\n")
    handle_command(app, "/status")
    out = output(app)
    assert "Branch" in out and "main" in out
    assert "Modified files" in out and "1" in out
    assert "Steps used" in out and "0/50" in out
    assert "fake-model" in out


def test_diff_and_files(app):
    handle_command(app, "/diff")
    assert "No changes." in output(app)
    (app.workspace / "app.py").write_text("print('bye')\n")
    (app.workspace / "new.txt").write_text("n\n")
    handle_command(app, "/diff")
    out = output(app)
    assert "+print('bye')" in out and "new.txt" in out
    handle_command(app, "/files")
    assert " M app.py" in output(app)


def test_model_switch(app):
    handle_command(app, "/model")
    assert "fake-model" in output(app)
    handle_command(app, "/model claude-sonnet-5-5")
    assert app.provider.model == "claude-sonnet-5-5"


def test_approval_switch(app):
    handle_command(app, "/approval strict")
    assert app.permissions.mode == "strict"
    handle_command(app, "/approval bogus")
    assert app.permissions.mode == "strict" and "Unknown mode" in output(app)


def test_clear_resets_conversation(app):
    app.agent.provider.responses = [reply("hello")]
    app.handle_line("say hi")
    assert app.agent.messages
    handle_command(app, "/clear")
    assert app.agent.messages == []


@pytest.mark.parametrize("cmd", ["/exit", "/quit", "/EXIT"])
def test_exit(app, cmd):
    handle_command(app, cmd)
    assert app.should_exit


def test_unknown_command(app):
    assert handle_command(app, "/frobnicate") is False
    assert "Unknown command /frobnicate" in output(app)


def test_commands_are_not_sent_to_the_model(app):
    app.handle_line("/status")
    assert app.agent.provider.requests == []


def test_task_summary_shows_changes(app):
    app.agent.provider.responses = [
        reply("", [("edit_file", {"path": "app.py", "old_string": "hello", "new_string": "world"})]),
        reply("", [("run_command", {"command": "echo checked"})]),
        reply("Changed the greeting."),
    ]
    app.handle_line("change greeting")
    out = output(app)
    assert "Changes:" in out and "M app.py" in out
    assert "echo checked" in out
    assert "YCode completed the task." in out


def test_every_command_has_help():
    for command in COMMANDS.values():
        assert command.help and command.usage.startswith("/")
