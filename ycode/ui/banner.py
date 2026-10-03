"""The YCode start-up banner.

The art is assembled from per-letter glyphs (ANSI Shadow style) so every row
has exactly the same width and the letters always line up. A plain-ASCII
fallback is used when the terminal cannot encode box-drawing characters.
"""

from __future__ import annotations

from dataclasses import dataclass

_GLYPHS: dict[str, tuple[str, ...]] = {
    "Y": (
        "██╗   ██╗",
        "╚██╗ ██╔╝",
        " ╚████╔╝ ",
        "  ╚██╔╝  ",
        "   ██║   ",
        "   ╚═╝   ",
    ),
    "C": (
        " ██████╗",
        "██╔════╝",
        "██║     ",
        "██║     ",
        "╚██████╗",
        " ╚═════╝",
    ),
    "O": (
        " ██████╗ ",
        "██╔═══██╗",
        "██║   ██║",
        "██║   ██║",
        "╚██████╔╝",
        " ╚═════╝ ",
    ),
    "D": (
        "██████╗ ",
        "██╔══██╗",
        "██║  ██║",
        "██║  ██║",
        "██████╔╝",
        "╚═════╝ ",
    ),
    "E": (
        "███████╗",
        "██╔════╝",
        "█████╗  ",
        "██╔══╝  ",
        "███████╗",
        "╚══════╝",
    ),
}

_ASCII_FALLBACK = (
    "__   __ ____          _      ",
    "\\ \\ / // ___|___   __| | ___ ",
    " \\ V /| |   / _ \\ / _` |/ _ \\",
    "  | | | |__| (_) | (_| |  __/",
    "  |_|  \\____\\___/ \\__,_|\\___|",
)

TAGLINE = "AI Coding Agent"


def render_word(word: str = "YCODE", gap: str = "") -> list[str]:
    """Render ``word`` with the block glyphs; returns one string per row."""
    letters = [_GLYPHS[ch] for ch in word.upper()]
    height = len(letters[0])
    return [gap.join(letter[row] for letter in letters) for row in range(height)]


def banner_lines(*, unicode: bool = True) -> list[str]:
    return render_word("YCODE") if unicode else list(_ASCII_FALLBACK)


def supports_unicode(encoding: str | None) -> bool:
    if not encoding:
        return False
    try:
        "█╗╔═║╚╝".encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return False
    return True


@dataclass(frozen=True)
class BannerInfo:
    version: str
    model: str
    directory: str
    branch: str | None = None
    project_types: tuple[str, ...] = ()
    approval_mode: str = "normal"
    instructions_loaded: bool = False


def info_lines(info: BannerInfo) -> list[tuple[str, str]]:
    """Key/value rows shown under the banner. Never includes secrets."""
    rows = [
        ("YCode", f"v{info.version}"),
        ("Model", info.model),
        ("Directory", info.directory),
    ]
    if info.branch:
        rows.append(("Git branch", info.branch))
    if info.project_types:
        rows.append(("Project", ", ".join(info.project_types)))
    rows.append(("Approvals", info.approval_mode))
    if info.instructions_loaded:
        rows.append(("Instructions", "YCODE.md loaded"))
    return rows


def render_banner_text(info: BannerInfo, *, unicode: bool = True) -> str:
    """Plain-text banner (used by tests and non-rich output)."""
    lines = banner_lines(unicode=unicode)
    width = max(len(line) for line in lines)
    out = list(lines)
    out.append(TAGLINE.center(width).rstrip())
    out.append("")
    out.extend(f"{key + ':':<13} {value}" for key, value in info_lines(info))
    return "\n".join(out)
