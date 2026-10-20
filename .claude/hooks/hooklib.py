"""Shared plumbing for the Claude Code hooks in this directory.

A hook is a process that reads one JSON object on stdin and answers with an
exit code: 0 lets the tool call through, 2 blocks it and hands stderr to the
model as the reason. The `cwd` field, not `CLAUDE_PROJECT_DIR`, is where the
session is: inside a worktree the project dir still points at the main
checkout. An edited file's checkout is found from the file itself.

The hooks guard against mistakes, not deliberate evasion: `bash -c`, `eval`
and a variable holding a command are not looked into, and lefthook, CI and
the branch ruleset stay the gate. They run on whatever `python3` is first on
PATH, which on macOS can be the system Python 3.9, outside the project's
virtual environment, so they use the standard library only and no syntax
newer than 3.9; `tests/test_claude_hooks.py` checks both.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any, NamedTuple, NoReturn

# The same string as SESSION_LINK in scripts/check_commit_message.py. The hooks
# keep a copy because they apply this rule in checkouts whose scripts they do
# not load (`trusted_root`).
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


# Paths


def expand(raw: str) -> Path:
    """Return `raw` with `~` expanded, or as written when that fails (`~nouser`)."""
    try:
        return Path(raw).expanduser()
    except (RuntimeError, KeyError, ValueError):
        return Path(raw)


def resolve_dir(base: str, raw: str) -> str:
    """Return the path `raw` names when written in directory `base`."""
    target = expand(raw)
    return str(target if target.is_absolute() else Path(base) / target)


def real(path: str) -> str:
    """Return `path` with symlinks resolved, or as given when it cannot be."""
    try:
        return os.path.realpath(path)
    except (OSError, ValueError):
        return path


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
    except (OSError, ValueError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def current_branch(cwd: str) -> str:
    """Return the checked-out branch, or "" when detached or outside a repository."""
    return git(cwd, "branch", "--show-current")


def repo_root(cwd: str) -> str:
    """Return the checkout that contains `cwd`, or `cwd` itself outside a repository."""
    return git(cwd, "rev-parse", "--show-toplevel") or cwd


def trusted_root(cwd: str) -> str | None:
    """Return the checkout containing `cwd` if it is this project or one of its worktrees.

    The hooks run the repository's own rule scripts and its `.venv` tools.
    Taking them from any checkout the session has moved into would execute
    code from a repository nobody vetted, so for any other checkout this is
    None and the hooks apply only the rules that need no code from it. The
    answer comes from the project's own `git worktree list`, never from the
    candidate, whose `.git` file could point anywhere.
    """
    root = real(repo_root(cwd))
    listed = git(str(PROJECT_ROOT), "worktree", "list", "--porcelain").splitlines()
    trusted = {real(str(PROJECT_ROOT))}
    trusted |= {real(line[len("worktree ") :]) for line in listed if line.startswith("worktree ")}
    return root if root in trusted else None


def load_script(cwd: str, name: str) -> ModuleType | None:
    """Load a rule script from the session's checkout, if it is trusted and has one.

    A script that fails to load counts as absent, so the hook still applies
    the rules that need no script.
    """
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
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        return None
    return module


# Edited files


class EditedFile(NamedTuple):
    """An edited file relative to its checkout, spelled three ways.

    `path` resolves the parent directories but not the file itself, so
    `CLAUDE.md` stays `CLAUDE.md` rather than becoming its target. `written`
    resolves nothing, so a symlinked `data/` still reads as `data/`. `target`
    resolves everything, so a `notes.md` linked to `uv.lock` reads as `uv.lock`.
    """

    root: str
    path: str
    written: str
    target: str

    @property
    def spellings(self) -> tuple[str, str, str]:
        """Return the three spellings, for a rule that must match any of them."""
        return (self.path, self.written, self.target)


def _exists(path: Path) -> bool:
    try:
        return path.exists()
    except (OSError, ValueError):  # permission denied, name too long, NUL byte
        return False


def _relative(path: str, root: str) -> str:
    rel = os.path.relpath(path, root)
    if rel.startswith(".."):
        # A case-insensitive file system: /users/me/repo is /Users/me/repo.
        folded = os.path.relpath(path.casefold(), root.casefold())
        rel = rel if folded.startswith("..") else folded
    return rel.replace("\\", "/")


def edited_file(payload: dict[str, Any]) -> EditedFile | None:
    """Return the checkout and paths of the file an Edit, Write or NotebookEdit touches.

    Edit and Write send `file_path`, NotebookEdit `notebook_path`. A relative
    path is resolved against the session's directory. The checkout is the one
    that holds the file, which in a worktree is not the session's; for a file
    reached through a symlinked directory it is the session's checkout.
    """
    raw = tool_arg(payload, "file_path", "notebook_path")
    if not raw:
        return None
    lexical = Path(os.path.normpath(resolve_dir(session_cwd(payload), raw)))
    resolved = str(Path(real(str(lexical.parent))) / lexical.name)
    anchor = lexical.parent
    while not _exists(anchor) and anchor != anchor.parent:
        anchor = anchor.parent
    roots = [real(repo_root(str(anchor))), real(repo_root(session_cwd(payload)))]
    # The checkout the file was named in comes first, then the one it lives in.
    for root in roots:
        if not _relative(str(lexical), root).startswith(".."):
            break
    else:
        root = next((r for r in roots if not _relative(resolved, r).startswith("..")), roots[0])
    return EditedFile(
        root,
        _relative(resolved, root),
        _relative(str(lexical), root),
        _relative(real(str(lexical)), root),
    )


# Shell commands


class Heredoc(NamedTuple):
    """A heredoc: where its `<<` sits in the command, and its body."""

    operator: int
    body: str


class Scan(NamedTuple):
    """A command with quotes, comments and heredoc bodies masked, and the heredocs found."""

    masked: str
    heredocs: list[Heredoc]


_WORD_END = frozenset(" \t\n;&|<>()")


class _Scanner:
    """One left-to-right pass over a shell command.

    It tracks quotes, escapes, comments, `$(...)` and heredocs, and masks
    everything that is data rather than command structure with filler of the
    same length, so offsets found in the masked text hold in the original.
    """

    def __init__(self, command: str) -> None:
        self.text = command
        self.out = list(command)
        self.heredocs: list[Heredoc] = []
        self.pending: list[tuple[int, str, bool]] = []

    def blank(self, start: int, end: int, fill: str = "x") -> None:
        for k in range(start, min(end, len(self.out))):
            self.out[k] = fill

    def single_end(self, i: int) -> int:
        j = self.text.find("'", i + 1)
        return len(self.text) if j < 0 else j + 1

    def ansi_end(self, i: int) -> int:
        j = i + 2
        while j < len(self.text):
            if self.text[j] == "\\":
                j += 2
            elif self.text[j] == "'":
                return j + 1
            else:
                j += 1
        return len(self.text)

    def double_end(self, i: int) -> int:
        j = i + 1
        while j < len(self.text):
            c = self.text[j]
            if c == "\\":
                j += 2
            elif c == '"':
                return j + 1
            elif self.text.startswith("$((", j) and (end := self.arithmetic_end(j + 3)):
                j = end
            elif self.text.startswith("$(", j):
                j = self.walk(j + 2, nested=True)
            elif c == "`":
                k = self.text.find("`", j + 1)
                j = len(self.text) if k < 0 else k + 1
            else:
                j += 1
        return len(self.text)

    def arithmetic_end(self, i: int) -> int | None:
        """Return the index after the `))` closing arithmetic, where `<<` is a shift.

        Quoted spans and newlines inside are skipped. None when the parentheses
        do not close as `))`, as in `((cd x) && y)`: then it is not arithmetic,
        and the caller scans it as code.
        """
        depth = 0
        j = i
        while j < len(self.text):
            c = self.text[j]
            if c == "'":
                j = self.single_end(j)
                continue
            if c == '"':
                j = self.double_end(j)
                continue
            if c == "(":
                depth += 1
            elif c == ")":
                if depth == 0:
                    return j + 2 if self.text.startswith("))", j) else None
                depth -= 1
            j += 1
        return None

    def operator(self, i: int) -> int:
        """Record the heredoc whose `<<` is at `i`; return where its line continues."""
        j = i + 2
        strip = j < len(self.text) and self.text[j] == "-"
        j += 1 if strip else 0
        while j < len(self.text) and self.text[j] in " \t":
            j += 1
        if j < len(self.text) and self.text[j] in "'\"":
            k = self.text.find(self.text[j], j + 1)
            k = len(self.text) if k < 0 else k
            delimiter, j = self.text[j + 1 : k], min(k + 1, len(self.text))
        else:
            k = j
            while k < len(self.text) and self.text[k] not in _WORD_END:
                k += 1
            delimiter, j = self.text[j:k].replace("\\", ""), k
        if delimiter:
            self.pending.append((i, delimiter, strip))
        return j

    def bodies(self, newline: int) -> int:
        """Read the bodies of the heredocs pending at this newline; return where code resumes."""
        start = newline + 1
        resume = len(self.text)
        for operator, delimiter, strip in self.pending:
            lines: list[str] = []
            j = start
            while j < len(self.text):
                end = self.text.find("\n", j)
                end = len(self.text) if end < 0 else end
                line = self.text[j:end].lstrip("\t") if strip else self.text[j:end]
                if line == delimiter:
                    break
                lines.append(line)
                j = end + 1
            else:
                end = len(self.text)
            self.blank(start, end)
            self.heredocs.append(Heredoc(operator, "\n".join(lines)))
            start, resume = end + 1, end
        self.pending.clear()
        return resume

    def walk(self, i: int, *, nested: bool = False) -> int:
        """Scan unquoted text from `i`; return after the `)` that closes it when nested."""
        while i < len(self.text):
            c = self.text[i]
            if c == "\\":
                continuation = self.text.startswith("\\\n", i)
                self.blank(i, i + 2, " " if continuation else "x")
                i += 2
            elif self.text.startswith("$'", i):
                end = self.ansi_end(i)
                self.blank(i, end)
                i = end
            elif c in "'\"":
                end = self.single_end(i) if c == "'" else self.double_end(i)
                self.blank(i, end)
                i = end
            elif c == "#" and (i == 0 or self.text[i - 1] in " \t\n;&|()"):
                end = self.text.find("\n", i)
                end = len(self.text) if end < 0 else end
                self.blank(i, end, " ")  # a comment holds no words at all
                i = end
            elif self.text.startswith("<<<", i):
                i += 3  # a here-string, not a heredoc
            elif self.text.startswith(("$((", "(("), i) and (
                end := self.arithmetic_end(i + (3 if c == "$" else 2))
            ):
                i = end
            elif self.text.startswith("<<", i):
                i = self.operator(i)
            elif c == "\n" and self.pending:
                i = self.bodies(i)
            elif c == "(":
                i = self.walk(i + 1, nested=True)
            elif c == ")" and nested:
                return i + 1
            else:
                i += 1
        return len(self.text)


def scan(command: str) -> Scan:
    """Mask a command's quoted strings, comments and heredoc bodies, and collect its heredocs."""
    scanner = _Scanner(command)
    scanner.walk(0)
    return Scan("".join(scanner.out), scanner.heredocs)


def mask(command: str) -> str:
    """Return `command` with quoted strings, comments and heredoc bodies masked, same length."""
    return scan(command).masked


class Segment(NamedTuple):
    """One simple command: its text, the same text masked, its offset, and its heredoc body."""

    text: str
    masked: str
    start: int
    heredoc: str | None


SEPARATOR = re.compile(r"&&|\|\||\|&|[;|\n()]|(?<![<>&])&(?![&>])")


def segments(command: str) -> list[Segment]:
    """Split a shell line into simple commands at operators, parentheses and newlines."""
    return [step.segment for step in steps(command, ".")]


_ANSI = {"n": "\n", "t": "\t", "\\": "\\", "'": "'", '"': '"'}


def unquote(text: str) -> str:
    """Return a shell word as the command receives it: quotes removed, escapes undone."""
    out: list[str] = []
    i = 0
    while i < len(text):
        c = text[i]
        if c == "\\" and i + 1 < len(text):
            out.append(text[i + 1])
            i += 2
        elif text.startswith("$'", i):
            j = i + 2
            while j < len(text) and text[j] != "'":
                if text[j] == "\\" and j + 1 < len(text):
                    out.append(_ANSI.get(text[j + 1], text[j : j + 2]))
                    j += 2
                else:
                    out.append(text[j])
                    j += 1
            i = j + 1
        elif c == "'":
            j = text.find("'", i + 1)
            j = len(text) if j < 0 else j
            out.append(text[i + 1 : j])
            i = j + 1
        elif c == '"':
            j = i + 1
            while j < len(text) and text[j] != '"':
                if text[j] == "\\" and j + 1 < len(text) and text[j + 1] in '\\"$`':
                    j += 1
                out.append(text[j])
                j += 1
            i = j + 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


class Word(NamedTuple):
    """A word of a simple command: as masked (for structure) and as written."""

    masked: str
    text: str

    @property
    def value(self) -> str:
        """Return the word as the command receives it."""
        return unquote(self.text)


# Words that start a command line without being the command, and wrappers
# whose own options come before the command they run.
_KEYWORDS = frozenset({"!", "{", "then", "do", "else", "elif", "if", "while", "until", "time"})
# Each wrapper, with the options of its own that take a value. The sets are
# per wrapper on purpose: `-n` takes a value in xargs and nice, not in sudo.
_WRAPPERS: dict[str, frozenset[str]] = {
    "sudo": frozenset({"-u", "-g", "-C", "-D", "-h", "-p", "-r", "-t", "-U"}),
    "env": frozenset({"-u", "--unset", "-C", "--chdir", "-S", "--split-string"}),
    "command": frozenset(),
    "exec": frozenset({"-a"}),
    "nohup": frozenset(),
    "nice": frozenset({"-n", "--adjustment"}),
    "timeout": frozenset({"-s", "--signal", "-k", "--kill-after"}),
    "xargs": frozenset({"-n", "-I", "-L", "-P", "-d", "-E", "-s", "-a"}),
}
# Wrappers whose first plain argument is theirs, not the command's.
_WRAPPER_FIRST_ARG = frozenset({"timeout"})
ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def words(segment: Segment) -> list[Word]:
    """Return the simple command's words, from the command name on."""
    found = [
        Word(m.group(0), segment.text[m.start() : m.end()])
        for m in re.finditer(r"\S+", segment.masked)
    ]
    i = 0
    while i < len(found):
        word = found[i].masked
        if word in _KEYWORDS or ASSIGNMENT.match(word):
            i += 1
        elif word in _WRAPPERS:
            i += 1
            while i < len(found) and found[i].masked.startswith("-"):
                i += 2 if found[i].masked in _WRAPPERS[word] else 1
            if word in _WRAPPER_FIRST_ARG:
                i += 1  # timeout's duration
        else:
            break
    return found[i:]


def is_command(word: Word, name: str) -> bool:
    """Tell whether a command word runs `name`, bare or by path."""
    return word.value == name or word.value.endswith(f"/{name}")


def change_directory(cwd: str, segment_words: list[Word]) -> str:
    """Return the directory a `cd` or `pushd` moves to, or `cwd` if it is not one.

    A target the hook cannot know (`cd -`, `cd "$DIR"`) leaves `cwd` as it was.
    """
    if not segment_words or segment_words[0].value not in {"cd", "pushd"}:
        return cwd
    targets = [w for w in segment_words[1:] if not (w.value.startswith("-") and w.value != "-")]
    if not targets:
        return cwd
    target = targets[0].value
    if target == "-" or "$" in target or "`" in target:
        return cwd
    return resolve_dir(cwd, target)


class Step(NamedTuple):
    """One simple command of a line, its words, and the directory it runs in."""

    segment: Segment
    words: list[Word]
    cwd: str


def steps(command: str, cwd: str) -> Iterator[Step]:
    """Walk a shell line's simple commands, following `cd` from one to the next.

    A `cd` inside `( ... )` stays inside: the directory goes back to what it
    was when the subshell closes.
    """
    scanned = scan(command)
    here = cwd
    outer: list[str] = []
    start = 0
    for match in [*SEPARATOR.finditer(scanned.masked), None]:
        end = match.start() if match else len(command)
        if command[start:end].strip():
            body = next((h.body for h in scanned.heredocs if start <= h.operator < end), None)
            segment = Segment(command[start:end], scanned.masked[start:end], start, body)
            segment_words = words(segment)
            here = change_directory(here, segment_words)
            yield Step(segment, segment_words, here)
        separator = match.group(0) if match else ""
        if separator == "(":
            outer.append(here)
        elif separator == ")" and outer:
            here = outer.pop()
        start = match.end() if match else end
