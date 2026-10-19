"""Shared plumbing for the Claude Code hooks in this directory.

A hook is a process that reads one JSON object on stdin and answers with an
exit code: 0 lets the tool call through, 2 blocks it and hands stderr to the
model as the reason. The `cwd` field, not `CLAUDE_PROJECT_DIR`, is where the
session is: inside a worktree the project dir still points at the main
checkout. An edited file's checkout is found from the file itself.

The hooks run on whatever `python3` is first on PATH, which on macOS can be
the system Python 3.9, outside the project's virtual environment. So they use
the standard library only and no syntax newer than 3.9;
`tests/test_claude_hooks.py` checks both.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, NamedTuple, NoReturn

SESSION_LINK = "claude.ai/code/session"

#: The checkout these hooks were installed from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Payload


def read_input() -> dict[str, Any]:
    """Read the hook's JSON payload; an unreadable one reads as empty, so the hook lets it by."""
    try:
        payload = json.load(sys.stdin)
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _text(mapping: object, key: str) -> str:
    value = mapping.get(key) if isinstance(mapping, dict) else None
    return value if isinstance(value, str) else ""


def session_cwd(payload: dict[str, Any]) -> str:
    """Return the session's working directory from the payload."""
    return _text(payload, "cwd") or str(Path.cwd())


def tool_arg(payload: dict[str, Any], *keys: str) -> str:
    """Return the first non-empty string among `tool_input[key]` for `keys`."""
    tool_input = payload.get("tool_input")
    return next((v for v in (_text(tool_input, k) for k in keys) if v), "")


def deny(reason: str) -> NoReturn:
    """Block the tool call with a reason the model will read."""
    sys.stderr.write(f"{reason}\n")
    sys.exit(2)


# Git


def clean_env() -> dict[str, str]:
    """Return `os.environ` without the `GIT_*` keys a git hook exports.

    A hook that runs git on the session's `cwd` with `GIT_DIR` still set gets
    an answer about the wrong repository.
    """
    return {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}


def git(cwd: str, *args: str) -> str:
    """Return git's trimmed output in `cwd`, or "" when git cannot answer."""
    try:
        result = subprocess.run(  # noqa: S603
            ["git", "-C", cwd, *args],  # noqa: S607
            capture_output=True,
            text=True,
            env=clean_env(),
            check=False,
        )
    except OSError:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def current_branch(cwd: str) -> str:
    """Return the checked-out branch, or "" when detached or outside a repository."""
    return git(cwd, "branch", "--show-current")


def repo_root(cwd: str) -> str:
    """Return the checkout that contains `cwd`, or `cwd` itself outside a repository."""
    return git(cwd, "rev-parse", "--show-toplevel") or cwd


def _common_dir(path: str) -> str:
    common = git(path, "rev-parse", "--git-common-dir")
    return str((Path(path) / common).resolve()) if common else ""


def trusted_root(cwd: str) -> str | None:
    """Return the checkout containing `cwd` if it is this project or one of its worktrees.

    The hooks run the repository's own rule scripts and its `.venv` tools.
    Taking them from any checkout the session has moved into would execute
    code from a repository nobody vetted, so for any other checkout this is
    None and the hooks apply only the rules that need no code from it. A
    worktree shares the project's git common directory, so it checks against
    the code it carries.
    """
    root = repo_root(cwd)
    project = _common_dir(str(PROJECT_ROOT))
    return root if project and _common_dir(root) == project else None


def load_script(cwd: str, name: str) -> ModuleType | None:
    """Load a rule script from the session's checkout, if it is trusted and has one."""
    root = trusted_root(cwd)
    if root is None:
        return None
    path = Path(root) / "scripts" / f"{name}.py"
    if not path.exists():
        return None
    # check_commit_message imports check_prose as a top-level module, the way
    # `python scripts/<name>.py` runs it.
    scripts = str(path.parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    # Registered before it runs: `@dataclass` looks its module up in
    # `sys.modules`, and the second script imports the first by name.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Edited files


class EditedFile(NamedTuple):
    """An edited file as its own checkout sees it."""

    root: str
    path: str


def edited_file(payload: dict[str, Any]) -> EditedFile | None:
    """Return the checkout and repository-relative path of the file an edit tool touches.

    Edit and Write send `file_path`, NotebookEdit `notebook_path`. A relative
    path is resolved against the session's directory, and both sides are
    resolved through symlinks (macOS `/tmp` is `/private/tmp`), so the same
    file always has the same name. The checkout is the one that holds the
    file, which in a worktree is not the session's.
    """
    raw = tool_arg(payload, "file_path", "notebook_path")
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path(session_cwd(payload)) / path
    path = Path(os.path.realpath(path))
    anchor = path.parent
    while not anchor.exists() and anchor != anchor.parent:
        anchor = anchor.parent
    root = os.path.realpath(repo_root(str(anchor)))
    return EditedFile(root, os.path.relpath(path, root).replace("\\", "/"))


# Shell commands


class Segment(NamedTuple):
    """One simple command of a shell line: its text, the same text masked, and its offset."""

    text: str
    masked: str
    start: int


# A heredoc: the operator, an optionally quoted delimiter, the rest of that
# line, the body, and the delimiter alone on its line.
HEREDOC = re.compile(
    r"<<-?[ \t]*([\"']?)([A-Za-z_][A-Za-z0-9_]*)\1[^\n]*\n([\s\S]*?)\n[ \t]*\2(?=\s|$)"
)
QUOTED = re.compile(r"\"(?:[^\"\\]|\\.)*\"|'[^']*'")
SEPARATOR = re.compile(r"&&|\|\||\|&|[;|\n]|(?<![<>&])&(?![&>])")
ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
WRAPPERS = frozenset({"sudo", "env", "command", "time", "exec", "nohup"})


def _blank(text: str, start: int, end: int) -> str:
    return text[:start] + "x" * (end - start) + text[end:]


def mask(command: str) -> str:
    """Return `command` with heredoc bodies and quoted strings replaced by same-length filler.

    The filler holds no separator and no newline, so separators found in the
    masked text sit at the same offsets in the original. Heredocs go first,
    since their bodies may contain quotes.
    """
    masked = command
    for match in HEREDOC.finditer(command):
        masked = _blank(masked, match.start(3), match.end())
    for match in QUOTED.finditer(masked):
        masked = _blank(masked, match.start(), match.end())
    return masked


def segments(command: str) -> list[Segment]:
    """Split a shell line into simple commands at `&&`, `||`, `;`, `|`, `&` and newlines.

    `$(...)`, backticks and `bash -c` are not looked into; lefthook and CI
    are the gate for what they hide.
    """
    masked = mask(command)
    out: list[Segment] = []
    start = 0
    for match in [*SEPARATOR.finditer(masked), None]:
        end = match.start() if match else len(command)
        if command[start:end].strip():
            out.append(Segment(command[start:end], masked[start:end], start))
        start = match.end() if match else end
    return out


class Word(NamedTuple):
    """A whitespace-delimited word: as masked (for flags) and as written (for values)."""

    masked: str
    text: str


def words(segment: Segment) -> list[Word]:
    """Return the simple command's words, without leading assignments, wrappers or a comment."""
    found = [
        Word(m.group(0), segment.text[m.start() : m.end()])
        for m in re.finditer(r"\S+", segment.masked)
    ]
    for i, word in enumerate(found):
        if word.masked.startswith("#"):
            found = found[:i]
            break
    while found and (ASSIGNMENT.match(found[0].masked) or found[0].masked in WRAPPERS):
        found = found[1:]
    return found


def unquote(text: str) -> str:
    """Return a shell word without its surrounding quotes, double-quote escapes undone."""
    if len(text) >= 2 and text[0] == text[-1] == "'":
        return text[1:-1]
    if len(text) >= 2 and text[0] == text[-1] == '"':
        return re.sub(r'\\([\\"$`])', r"\1", text[1:-1])
    return text


def is_command(word: Word, name: str) -> bool:
    """Tell whether a command word runs `name`, bare or by path."""
    return word.masked == name or word.masked.endswith(f"/{name}")


def change_directory(cwd: str, segment_words: list[Word]) -> str:
    """Return the directory a `cd` segment moves to, or `cwd` if it is not one."""
    if len(segment_words) < 2 or segment_words[0].masked != "cd":
        return cwd
    target = Path(unquote(segment_words[1].text)).expanduser()
    return str(target if target.is_absolute() else Path(cwd) / target)
