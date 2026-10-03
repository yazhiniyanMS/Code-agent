from ycode.ui.banner import BannerInfo, banner_lines, info_lines, render_banner_text, render_word, supports_unicode
from tests.conftest import console_text, recording_console
from ycode.ui.console import ConsoleUI


def test_banner_rows_are_aligned():
    lines = banner_lines()
    assert len(lines) == 6
    assert len({len(line) for line in lines}) == 1, "every row must have the same width"


def test_each_glyph_has_consistent_width():
    for letter in "YCODE":
        rows = render_word(letter)
        assert len({len(r) for r in rows}) == 1, letter


def test_ascii_fallback_is_pure_ascii():
    for line in banner_lines(unicode=False):
        line.encode("ascii")


def test_banner_text_contains_info_and_no_secrets(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret-value")
    info = BannerInfo(version="0.1.0", model="Claude (claude-opus-5-5)", directory="/tmp/project",
                      branch="main", project_types=("Python",))
    text = render_banner_text(info)
    assert "AI Coding Agent" in text
    assert "v0.1.0" in text and "claude-opus-5-5" in text and "/tmp/project" in text
    assert "sk-ant" not in text
    assert ("Git branch", "main") in info_lines(info)


def test_supports_unicode():
    assert supports_unicode("utf-8")
    assert not supports_unicode("ascii")
    assert not supports_unicode(None)


def test_console_banner_renders():
    console = recording_console()
    ConsoleUI(console).print_banner(BannerInfo(version="0.1.0", model="m", directory="/d"))
    out = console_text(console)
    assert "AI Coding Agent" in out and "Directory" in out
