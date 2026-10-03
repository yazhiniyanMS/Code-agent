"""Command classification, path confinement and approval handling.

This is a guardrail, not a sandbox: it stops the agent from casually running
destructive commands or touching files outside the project, and makes the
user explicitly approve risky actions. Code the agent writes and then runs
(``python script.py``) is still arbitrary code, exactly as if you ran it.
"""

from __future__ import annotations

import enum
import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ycode.errors import PathSecurityError


class RiskLevel(enum.IntEnum):
    SAFE = 0
    REQUIRES_APPROVAL = 1
    BLOCKED = 2


@dataclass(frozen=True)
class Decision:
    level: RiskLevel
    reason: str = ""

    @property
    def allowed_without_asking(self) -> bool:
        return self.level is RiskLevel.SAFE


SAFE = Decision(RiskLevel.SAFE)

# --------------------------------------------------------------- vocabulary

# Commands that are always refused.
_BLOCKED_COMMANDS = {
    "mkfs", "shutdown", "reboot", "halt", "poweroff", "fdisk", "parted", "format",
    "diskpart", "wipefs",
}

# Commands that always need explicit approval.
_APPROVAL_COMMANDS = {
    "rm": "deletes files",
    "rmdir": "deletes directories",
    "shred": "destroys files",
    "unlink": "deletes files",
    "sudo": "runs with elevated privileges",
    "su": "switches user",
    "doas": "runs with elevated privileges",
    "chmod": "changes file permissions",
    "chown": "changes file ownership",
    "chgrp": "changes file group",
    "mv": "moves/overwrites files",
    "dd": "low-level disk/file writes",
    "truncate": "truncates files",
    "kill": "terminates processes",
    "killall": "terminates processes",
    "pkill": "terminates processes",
    "crontab": "modifies scheduled jobs",
    "systemctl": "controls system services",
    "service": "controls system services",
    "launchctl": "controls system services",
    "mount": "mounts filesystems",
    "umount": "unmounts filesystems",
    "iptables": "changes firewall rules",
    "ssh": "connects to remote hosts",
    "scp": "copies files to/from remote hosts",
    "rsync": "synchronises/deletes files",
    "eval": "evaluates arbitrary shell code",
    "gh": "acts on GitHub on your behalf",
    "reg": "modifies the Windows registry",
    "del": "deletes files",
    "rd": "deletes directories",
}

# Read-only commands that are fine even in strict mode.
_READ_ONLY = {
    "ls", "dir", "cat", "head", "tail", "wc", "pwd", "echo", "printf", "which", "where",
    "type", "file", "stat", "du", "df", "tree", "grep", "rg", "ag", "find", "fd",
    "sort", "uniq", "cut", "diff", "cmp", "less", "more", "basename", "dirname",
    "realpath", "date", "whoami", "uname", "env", "printenv", "true", "false", "test",
}

_GIT_READ_ONLY = {
    "status", "diff", "log", "show", "rev-parse", "ls-files", "blame", "describe",
    "shortlog", "grep",
}
_GIT_LISTING = {"branch", "tag", "remote", "stash"}
_GIT_LIST_FLAGS = {"-a", "-r", "-v", "-vv", "--list", "list", "--show-current", "--all"}

_SUBCOMMAND_RULES: dict[str, dict[str, str]] = {
    "npm": {"publish": "publishes a package", "unpublish": "unpublishes a package",
            "deprecate": "deprecates a package", "owner": "changes package owners"},
    "yarn": {"publish": "publishes a package"},
    "pnpm": {"publish": "publishes a package"},
    "cargo": {"publish": "publishes a crate", "yank": "yanks a crate"},
    "twine": {"upload": "publishes a package"},
    "pip": {"uninstall": "removes packages"},
    "pip3": {"uninstall": "removes packages"},
    "docker": {"rm": "removes containers", "rmi": "removes images", "prune": "prunes data",
               "kill": "kills containers", "system": "system-level docker operations",
               "volume": "manages volumes", "push": "pushes images"},
    "kubectl": {"delete": "deletes cluster resources", "apply": "changes cluster resources",
                "drain": "drains nodes", "scale": "scales workloads"},
    "terraform": {"apply": "changes infrastructure", "destroy": "destroys infrastructure"},
    "helm": {"uninstall": "removes releases", "delete": "removes releases"},
}

_WRAPPERS = {"nohup", "time", "nice", "command", "builtin", "exec", "xargs", "timeout", "stdbuf"}
_SHELLS = {"sh", "bash", "zsh", "dash", "ksh", "fish"}

_SEPARATOR_RE = re.compile(r"\|\||&&|[;|&\n]")
_SUBSTITUTION_RE = re.compile(r"\$\(([^()]*)\)|`([^`]*)`")
_REDIRECT_RE = re.compile(r"(?:^|[^<>&0-9])(?:[0-9]?>{1,2}|&>)\s*([^\s;&|]+)")
_FORK_BOMB_RE = re.compile(r":\s*\(\s*\)\s*\{.*:\s*\|\s*:")
_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _max(a: Decision, b: Decision) -> Decision:
    return b if b.level > a.level else a


class CommandClassifier:
    """Classifies a shell command line as SAFE, REQUIRES_APPROVAL or BLOCKED."""

    def __init__(self, workspace: Path | None = None) -> None:
        self.workspace = workspace.resolve() if workspace else None

    # Public API -------------------------------------------------------------

    def classify(self, command: str) -> Decision:
        command = command.strip()
        if not command:
            return Decision(RiskLevel.BLOCKED, "empty command")
        if _FORK_BOMB_RE.search(command):
            return Decision(RiskLevel.BLOCKED, "fork bomb")
        decision = SAFE
        # Command substitutions are classified on their own content.
        for match in _SUBSTITUTION_RE.finditer(command):
            inner = match.group(1) if match.group(1) is not None else match.group(2)
            decision = _max(decision, self.classify(inner) if inner.strip() else SAFE)
        if re.search(r"\|\s*(sudo\s+)?(ba|z|da|k)?sh\b", command):
            decision = _max(
                decision, Decision(RiskLevel.REQUIRES_APPROVAL, "pipes data into a shell")
            )
        for target in _REDIRECT_RE.findall(command):
            decision = _max(decision, self._classify_write_target(target))
        for segment in _SEPARATOR_RE.split(command):
            if segment.strip():
                decision = _max(decision, self._classify_segment(segment))
            if decision.level is RiskLevel.BLOCKED:
                break
        return decision

    def is_read_only(self, command: str) -> bool:
        """True when every segment is a known read-only command."""
        if _SUBSTITUTION_RE.search(command) or _REDIRECT_RE.search(command):
            return False
        for segment in _SEPARATOR_RE.split(command):
            tokens = self._tokens(segment)
            if not tokens:
                continue
            name = self._command_name(tokens[0])
            if name == "git":
                rest = [t for t in tokens[1:]]
                sub = next((t for t in rest if not t.startswith("-")), "")
                after = rest[rest.index(sub) + 1:] if sub in rest else []
                if sub in _GIT_READ_ONLY:
                    continue
                if sub in _GIT_LISTING and all(t in _GIT_LIST_FLAGS for t in after):
                    continue
                return False
            if name == "find" and any(t in ("-delete", "-exec", "-execdir", "-ok") for t in tokens):
                return False
            if name not in _READ_ONLY:
                return False
        return True

    # Internals ----------------------------------------------------------------

    @staticmethod
    def _tokens(segment: str) -> list[str]:
        try:
            return shlex.split(segment, posix=True)
        except ValueError:
            return segment.split()

    @staticmethod
    def _command_name(token: str) -> str:
        name = os.path.basename(token).lower()
        return name[:-4] if name.endswith(".exe") else name

    def _classify_write_target(self, target: str) -> Decision:
        if target.startswith("/dev/"):
            if target in ("/dev/null", "/dev/stdout", "/dev/stderr", "/dev/tty"):
                return SAFE
            return Decision(RiskLevel.BLOCKED, f"writes directly to device {target}")
        if self.workspace is None:
            return SAFE
        expanded = os.path.expanduser(os.path.expandvars(target))
        path = Path(expanded)
        if not path.is_absolute():
            path = self.workspace / path
        try:
            resolved = path.resolve()
        except OSError:
            return Decision(RiskLevel.REQUIRES_APPROVAL, f"writes to {target}")
        if not _is_within(resolved, self.workspace):
            return Decision(RiskLevel.REQUIRES_APPROVAL, f"writes outside the project ({target})")
        return SAFE

    def _classify_segment(self, segment: str) -> Decision:
        tokens = self._tokens(segment)
        # Strip leading env assignments and harmless wrappers.
        while tokens and (_ENV_ASSIGN_RE.match(tokens[0]) or self._command_name(tokens[0]) in _WRAPPERS
                          or self._command_name(tokens[0]) == "env"):
            name = self._command_name(tokens[0])
            tokens = tokens[1:]
            if name == "timeout" and tokens and re.match(r"^\d", tokens[0]):
                tokens = tokens[1:]
            while name in ("env", "xargs", "nice", "timeout", "stdbuf") and tokens and tokens[0].startswith("-"):
                tokens = tokens[1:]
        if not tokens:
            return SAFE
        name = self._command_name(tokens[0])
        args = tokens[1:]

        if name in _BLOCKED_COMMANDS or name.startswith("mkfs"):
            return Decision(RiskLevel.BLOCKED, f"`{name}` is never run by YCode")

        if name in _SHELLS:
            if "-c" in args:
                idx = args.index("-c")
                if idx + 1 < len(args):
                    return self.classify(args[idx + 1])
            return SAFE

        if name in ("rm", "del", "rd") or (name == "sudo" and "rm" in args):
            targets = [a for a in args if not a.startswith("-")]
            recursive = any(
                a == "--recursive" or (a.startswith("-") and not a.startswith("--") and "r" in a.lower())
                for a in args
            )
            for target in targets:
                if self._is_catastrophic_target(target) and (recursive or name != "rm"):
                    return Decision(RiskLevel.BLOCKED, f"recursive delete of {target}")

        if name == "dd" and any(a.startswith("of=/dev/") for a in args):
            return Decision(RiskLevel.BLOCKED, "dd onto a device")

        if name == "chmod" and "-R" in args and any(self._is_catastrophic_target(a) for a in args):
            return Decision(RiskLevel.BLOCKED, "recursive chmod of a system path")

        if any(_SENSITIVE_NAMES.search(os.path.basename(a)) for a in args if not a.startswith("-")):
            sensitive = Decision(RiskLevel.REQUIRES_APPROVAL, "may expose secrets (e.g. .env, keys) to the model")
        else:
            sensitive = SAFE

        if name in _APPROVAL_COMMANDS:
            return Decision(RiskLevel.REQUIRES_APPROVAL, f"`{name}` {_APPROVAL_COMMANDS[name]}")

        if name == "git":
            return self._classify_git(args)

        if name == "find" and any(a in ("-delete", "-exec", "-execdir", "-ok") for a in args):
            return Decision(RiskLevel.REQUIRES_APPROVAL, "`find` with -delete/-exec can modify files")

        rules = _SUBCOMMAND_RULES.get(name)
        if rules:
            sub = next((a for a in args if not a.startswith("-")), "")
            if sub in rules:
                return Decision(RiskLevel.REQUIRES_APPROVAL, f"`{name} {sub}` {rules[sub]}")

        if name in ("curl", "wget") and any(
            a in ("-T", "--upload-file", "-d", "--data", "-F", "--form", "-X", "--post-data") for a in args
        ):
            return Decision(RiskLevel.REQUIRES_APPROVAL, f"`{name}` sends data to a remote server")
        return sensitive

    def _classify_git(self, args: list[str]) -> Decision:
        # Skip global options such as `-C dir` or `-c key=value`.
        i = 0
        while i < len(args) and args[i].startswith("-"):
            i += 2 if args[i] in ("-C", "-c", "--git-dir", "--work-tree") else 1
        if i >= len(args):
            return SAFE
        sub, rest = args[i], args[i + 1:]

        def approval(reason: str) -> Decision:
            return Decision(RiskLevel.REQUIRES_APPROVAL, f"`git {sub}` {reason}")

        if sub == "push":
            return approval("publishes commits to a remote")
        if sub == "reset" and any(a in ("--hard", "--merge", "--keep") for a in rest):
            return approval("discards local changes")
        if sub == "clean":
            return approval("deletes untracked files")
        if sub in ("commit", "merge", "rebase", "cherry-pick", "revert", "am"):
            return approval("rewrites or adds history")
        if sub == "checkout" and ("--" in rest or "." in rest or "-f" in rest or "--force" in rest):
            return approval("discards working-tree changes")
        if sub == "restore":
            return approval("discards working-tree changes")
        if sub == "switch" and any(a in ("-f", "--force", "--discard-changes") for a in rest):
            return approval("discards working-tree changes")
        if sub == "branch" and any(a in ("-D", "-d", "--delete", "-f", "--force", "-M", "-m") for a in rest):
            return approval("deletes or rewrites branches")
        if sub == "stash" and rest and rest[0] in ("drop", "clear"):
            return approval("deletes stashed changes")
        if sub == "tag" and any(a in ("-d", "--delete", "-f") for a in rest):
            return approval("deletes or moves tags")
        if sub in ("filter-branch", "filter-repo", "update-ref", "gc", "prune", "rm", "mv"):
            return approval("modifies the repository")
        if sub == "reflog" and rest and rest[0] in ("expire", "delete"):
            return approval("deletes reflog entries")
        if sub == "remote" and rest and rest[0] in ("add", "remove", "rm", "set-url", "rename"):
            return approval("changes remotes")
        if sub == "config" and "--global" in rest and len([a for a in rest if not a.startswith("-")]) > 1:
            return approval("changes global git configuration")
        return SAFE

    @staticmethod
    def _is_catastrophic_target(target: str) -> bool:
        cleaned = target.rstrip("/") or "/"
        return cleaned in ("/", "/*", "~", "~/*", "$HOME", "${HOME}", "$HOME/*", "*", ".", "..",
                           "/usr", "/etc", "/bin", "/var", "/home", "/System", "C:", "C:\\")


# ------------------------------------------------------------------- paths


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


_SENSITIVE_NAMES = re.compile(
    r"(^\.env(\..+)?$|\.pem$|\.key$|^id_(rsa|dsa|ecdsa|ed25519)$|^\.netrc$|^\.npmrc$|^\.pypirc$|credentials)",
    re.IGNORECASE,
)


class PathGuard:
    """Resolves user/model-supplied paths and confines them to the workspace."""

    def __init__(self, workspace: Path, *, allow_outside: bool = False) -> None:
        self.workspace = workspace.resolve()
        self.allow_outside = allow_outside

    def resolve(self, raw: str, *, for_write: bool = False) -> Path:
        if not isinstance(raw, str) or not raw.strip():
            raise PathSecurityError("A non-empty path is required.")
        if "\x00" in raw:
            raise PathSecurityError("Path contains a NUL byte.")
        candidate = Path(os.path.expanduser(raw.strip()))
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        # resolve(strict=False) follows symlinks for the parts that exist,
        # so a symlink pointing outside the project is caught here too.
        resolved = candidate.resolve()
        if not self.allow_outside and not _is_within(resolved, self.workspace):
            raise PathSecurityError(
                f"Path '{raw}' is outside the project directory ({self.workspace}). "
                "Access outside the workspace is disabled."
            )
        if for_write and _is_within(resolved, self.workspace / ".git"):
            raise PathSecurityError("Writing inside the .git directory is not allowed.")
        return resolved

    def relative(self, path: Path) -> str:
        try:
            return path.relative_to(self.workspace).as_posix() or "."
        except ValueError:
            return str(path)

    @staticmethod
    def is_sensitive(path: Path) -> bool:
        return bool(_SENSITIVE_NAMES.search(path.name))


# --------------------------------------------------------------- approvals


@dataclass(frozen=True)
class ApprovalRequest:
    kind: str  # "command" | "write" | "read"
    subject: str  # the command line or file path
    reason: str = ""


# Returns "yes", "no" or "always".
Approver = Callable[[ApprovalRequest], str]


def deny_all(_: ApprovalRequest) -> str:
    return "no"


@dataclass
class PermissionManager:
    """Decides whether an action may proceed, asking the user when needed."""

    workspace: Path
    mode: str = "normal"  # strict | normal | auto
    approver: Approver = deny_all
    allow_outside_workspace: bool = False
    _always: set[tuple[str, str]] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.workspace = self.workspace.resolve()
        self.classifier = CommandClassifier(self.workspace)
        self.paths = PathGuard(self.workspace, allow_outside=self.allow_outside_workspace)

    def check_command(self, command: str) -> tuple[bool, Decision]:
        """Returns (allowed, decision). Asks the user when approval is needed."""
        decision = self.classifier.classify(command)
        if decision.level is RiskLevel.BLOCKED:
            return False, decision
        if decision.level is RiskLevel.SAFE and self.mode == "strict" and not self.classifier.is_read_only(command):
            decision = Decision(RiskLevel.REQUIRES_APPROVAL, "strict mode: commands need approval")
        if decision.level is RiskLevel.SAFE:
            return True, decision
        if self.mode == "auto":
            return True, decision
        return self._ask(ApprovalRequest("command", command, decision.reason)), decision

    def check_write(self, path: Path) -> bool:
        if self.mode != "strict":
            return True
        return self._ask(ApprovalRequest("write", self.paths.relative(path), "strict mode: file writes need approval"))

    def check_read(self, path: Path) -> bool:
        if not PathGuard.is_sensitive(path) or self.mode == "auto":
            return True
        return self._ask(ApprovalRequest(
            "read", self.paths.relative(path),
            "this file may contain secrets that would be sent to the model",
        ))

    def _ask(self, request: ApprovalRequest) -> bool:
        key = (request.kind, request.subject)
        if key in self._always:
            return True
        answer = self.approver(request)
        if answer == "always":
            self._always.add(key)
            return True
        return answer == "yes"
