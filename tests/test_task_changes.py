"""The task summary's "Changes:" lists what the task changed, by any tool or command, and not what the user had
already changed before it."""
from tests.conftest import FakeProvider, console_text, recording_console, reply
from ycode.app import YCodeApp
from ycode.config import Config
from ycode.ui.console import ConsoleUI


def _run(workspace, command):
    ui = ConsoleUI(recording_console())
    provider = FakeProvider([reply(calls=[("run_command", {"command": command})]), reply("Done.")])
    app = YCodeApp(config=Config(max_steps=10, approval_mode="auto"), workspace=workspace, provider=provider, ui=ui)
    app.run_task("do it")
    out = console_text(ui.console)
    return out.split("Changes:")[1].split("Commands:")[0] if "Changes:" in out else ""


def test_files_changed_by_a_command_are_listed_and_the_users_own_are_not(git_workspace):
    (git_workspace / "app.py").write_text("print('the user is editing this')\n")  # before the task
    changes = _run(git_workspace, "python3 -c \"open('gen.py', 'w').write('x = 1')\"")
    assert "gen.py" in changes and "app.py" not in changes


def test_a_file_the_user_had_changed_is_listed_once_the_task_changes_it_too(git_workspace):
    (git_workspace / "app.py").write_text("print('the user is editing this')\n")
    changes = _run(git_workspace, "python3 -c \"open('app.py', 'a').write('# and the agent')\"")
    assert "app.py" in changes
