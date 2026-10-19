"""PreToolUse on Bash: the git rules from AGENTS.md, enforced before the command runs.

- Stage by name: no `git add -A`, `git add .`, `git add :/`, `git commit -a`.
- Never commit on main.
- No force push at main, no `reset --hard`, `clean -f`, `branch -D`, or a
  working-tree wipe through `checkout .` and `restore .`.
- A commit message that would fail `commit-msg` fails here first, so a
  `--no-verify` cannot get it past lefthook.

Everything else, including a plain push, a rebase, a stash and any dry run,
goes through. Exit 2 blocks; stderr is the reason the model reads.

Each simple command on the line is checked on its own (`hooklib.segments`),
with quoted strings and heredoc bodies masked, so a commit message or a file
written by `cat <<EOF` that mentions `git add -A` is not a `git add -A`. A `cd`
earlier on the line and `git -C <dir>` move where the branch is read. `$(...)`,
backticks and `bash -c` are not looked into; lefthook and the ruleset are the
gate for what they hide. The hook has no `if` filter in settings.json on
purpose: `cd x && git add -A` starts with `cd`.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import hooklib
from hooklib import Word

SHORT_CLUSTER = re.compile(r"^-[A-Za-z]+$")
MAIN_REF = re.compile(r"(?:^|[:+])(?:refs/heads/)?main$")
CAT_HEREDOC = re.compile(r"^\$\(\s*cat\s*<<")
# Global options that take a separate argument.
GLOBAL_WITH_ARG = frozenset({"-C", "-c", "--git-dir", "--work-tree", "--exec-path", "--namespace"})
# `git commit` short options whose value follows, in the cluster or as the next word.
COMMIT_VALUE_OPTIONS = frozenset("mFCct")

MAIN_COMMIT = "Never commit on main (AGENTS.md). Fetch, branch from origin/main, then commit."


class GitCall(NamedTuple):
    """One git invocation: the subcommand, its arguments, and the directory it runs in."""

    sub: str
    args: list[Word]
    directory: str


def parse_git(segment_words: list[Word], cwd: str) -> GitCall | None:
    """Return the git invocation a simple command makes, or None if it is not git."""
    if not segment_words or not hooklib.is_command(segment_words[0], "git"):
        return None
    directory = cwd
    i = 1
    while i < len(segment_words) and segment_words[i].masked.startswith("-"):
        flag = segment_words[i].masked
        if flag in GLOBAL_WITH_ARG and i + 1 < len(segment_words):
            if flag == "-C":
                target = Path(hooklib.unquote(segment_words[i + 1].text)).expanduser()
                directory = str(target if target.is_absolute() else Path(directory) / target)
            i += 2
        else:
            i += 1
    sub = segment_words[i].masked if i < len(segment_words) else ""
    return GitCall(sub, segment_words[i + 1 :], directory)


def _short_flag(token: str, letter: str) -> bool:
    """Tell whether `token` is a short flag cluster carrying `letter`, as `-fd` carries `f`."""
    return bool(SHORT_CLUSTER.match(token)) and letter in token[1:]


def _has_flag(args: list[Word], long: str, letter: str = "") -> bool:
    """Tell whether `--long` or, given one, the short `letter` in any cluster is present."""
    return any(a.masked == long or (letter and _short_flag(a.masked, letter)) for a in args)


def _is_dot(token: str) -> bool:
    return token in {".", "./"}


# One rule per subcommand. Each returns the refusal, or None.


def add_rule(call: GitCall) -> str | None:
    """Refuse `git add` of everything."""
    sweeping = any(
        a.masked in {"--all", ":/", ":/."} or _is_dot(a.masked) or _short_flag(a.masked, "A")
        for a in call.args
    )
    if sweeping:
        return (
            "Stage files by name (AGENTS.md): `git add -A`, `git add .` and `git add :/` "
            "sweep up unrelated work in progress. Name the paths."
        )
    return None


def _commit_stages_all(args: list[Word]) -> bool:
    for a in args:
        if a.masked == "--all":
            return True
        if SHORT_CLUSTER.match(a.masked):
            for letter in a.masked[1:]:
                if letter in COMMIT_VALUE_OPTIONS:
                    break  # the rest of the cluster is that option's value: `-ma` is message "a"
                if letter == "a":
                    return True
    return False


def commit_rule(call: GitCall) -> str | None:
    """Refuse `git commit -a`."""
    if _commit_stages_all(call.args):
        return (
            "Stage files by name (AGENTS.md): `git commit -a` commits every modified "
            "file. Stage the paths, then commit."
        )
    return None


def push_rule(call: GitCall) -> str | None:
    """Refuse a force push that lands on main."""
    if _has_flag(call.args, "--dry-run", "n"):
        return None
    positional = [a.masked for a in call.args if not a.masked.startswith("-")]
    refspecs = positional[1:]
    forced = (
        _has_flag(call.args, "--force", "f")
        or any(a.masked.startswith("--force-with-lease") for a in call.args)
        or any(r.startswith("+") for r in refspecs)
    )
    if not forced:
        return None
    if refspecs:
        on_head = any(r.lstrip("+") == "HEAD" for r in refspecs)
        at_main = any(MAIN_REF.search(r) for r in refspecs) or (
            on_head and hooklib.current_branch(call.directory) == "main"
        )
    else:
        at_main = hooklib.current_branch(call.directory) == "main"
    return "No force push at main. The ruleset would reject it anyway." if at_main else None


def reset_rule(call: GitCall) -> str | None:
    """Refuse `git reset --hard`."""
    if _has_flag(call.args, "--hard"):
        return (
            "No `git reset --hard` from an agent: it discards uncommitted work the "
            "user may want. Ask them to run it."
        )
    return None


def clean_rule(call: GitCall) -> str | None:
    """Refuse `git clean -f` unless it is a dry run."""
    if _has_flag(call.args, "--force", "f") and not _has_flag(call.args, "--dry-run", "n"):
        return "No `git clean -f` from an agent. Ask the user to run it."
    return None


def branch_rule(call: GitCall) -> str | None:
    """Refuse a forced branch delete."""
    if any(_short_flag(a.masked, "D") for a in call.args) or (
        _has_flag(call.args, "--delete", "d") and _has_flag(call.args, "--force", "f")
    ):
        return "No `git branch -D` from an agent. Use `-d`, or ask the user to force-delete."
    return None


def checkout_rule(call: GitCall) -> str | None:
    """Refuse `git checkout .`."""
    if any(_is_dot(a.masked) for a in call.args):
        return "No working-tree wipe (`git checkout .`) from an agent. Name the paths to restore."
    return None


def restore_rule(call: GitCall) -> str | None:
    """Refuse `git restore .` on the working tree."""
    index_only = _has_flag(call.args, "--staged", "S") and not _has_flag(
        call.args, "--worktree", "W"
    )
    if any(_is_dot(a.masked) for a in call.args) and not index_only:
        return "No working-tree wipe (`git restore .`) from an agent. Name the paths to restore."
    return None


RULES: dict[str, Callable[[GitCall], str | None]] = {
    "add": add_rule,
    "commit": commit_rule,
    "push": push_rule,
    "reset": reset_rule,
    "clean": clean_rule,
    "branch": branch_rule,
    "checkout": checkout_rule,
    "restore": restore_rule,
}


def _creates_branch(call: GitCall) -> bool:
    flags = {a.masked for a in call.args}
    return (
        (call.sub == "checkout" and bool(flags & {"-b", "-B"}))
        or (call.sub == "switch" and bool(flags & {"-c", "-C"}))
        or (call.sub == "worktree" and bool(call.args) and call.args[0].masked == "add")
    )


def _heredoc_after(command: str, offset: int) -> str | None:
    body = hooklib.HEREDOC.search(command, offset)
    return body.group(3) if body else None


def _message_part(value: str) -> str | None:
    """Return a `-m` value as git records it, or None when it is a substitution not read here."""
    if CAT_HEREDOC.match(value):
        body = hooklib.HEREDOC.search(value)
        return body.group(3) if body else None
    return None if "$(" in value or "`" in value else value


class _UnreadableError(Exception):
    """The message comes from somewhere this hook does not read: a file, an editor, a commit."""


Found = tuple["str | None", int]


def _long_option(args: list[Word], i: int, command: str, offset: int) -> Found | None:
    """Read `--message` and `--file -`; None when `args[i]` is neither."""
    word = args[i]
    following = args[i + 1] if i + 1 < len(args) else None
    if word.masked.startswith("--message="):
        return hooklib.unquote(word.text[len("--message=") :]), i
    if word.masked == "--message" and following is not None:
        return hooklib.unquote(following.text), i + 1
    if word.masked in {"-F", "--file"} or word.masked.startswith("--file="):
        inline = word.masked[len("--file=") :] if "=" in word.masked else None
        source = inline if inline is not None else (following.masked if following else "")
        if source != "-":
            raise _UnreadableError
        return _heredoc_after(command, offset), i if inline is not None else i + 1
    return None


def _short_cluster(args: list[Word], i: int) -> Found:
    """Read `-m` in a short cluster, its value attached (`-mWIP`) or the next word."""
    word = args[i]
    if not SHORT_CLUSTER.match(word.masked):
        return None, i
    for j, letter in enumerate(word.text[1:], start=1):
        if letter not in COMMIT_VALUE_OPTIONS:
            continue
        if letter != "m":
            raise _UnreadableError  # -F file, -C or -c reuse a message this hook cannot read
        rest = word.text[j + 1 :]
        if rest or i + 1 >= len(args):
            return hooklib.unquote(rest), i
        return hooklib.unquote(args[i + 1].text), i + 1
    return None, i


def _option(args: list[Word], i: int, command: str, offset: int) -> Found:
    """Return the message value option `args[i]` gives, if any, and the index of its last word."""
    found = _long_option(args, i, command, offset)
    return found if found is not None else _short_cluster(args, i)


def commit_message(segment: hooklib.Segment, call: GitCall, command: str) -> str | None:
    """Return the message this `git commit` would record, or None when it cannot be read here.

    That covers `-m`, `-mVALUE`, a cluster ending in `m`, `--message`,
    `--message=`, the harness's `-m "$(cat <<'EOF' ... EOF)"`, and `-F -` fed
    by a heredoc. A message from a file or an editor is left to lefthook's
    `commit-msg`.
    """
    parts: list[str] = []
    i = 0
    try:
        while i < len(call.args):
            value, i = _option(call.args, i, command, segment.start)
            if value is not None:
                part = _message_part(value)
                if part is None:
                    return None
                parts.append(part)
            i += 1
    except _UnreadableError:
        return None
    return "\n\n".join(parts) if parts else None


def refusal(command: str, cwd: str) -> str | None:
    """Return the reason to refuse `command`, or None to let it run."""
    here = cwd
    branched = False
    for segment in hooklib.segments(command):
        segment_words = hooklib.words(segment)
        here = hooklib.change_directory(here, segment_words)
        call = parse_git(segment_words, here)
        if call is None:
            continue
        rule = RULES.get(call.sub)
        reason = rule(call) if rule else None
        if reason:
            return reason
        if call.sub == "commit" and not _has_flag(call.args, "--dry-run"):
            if not branched and hooklib.current_branch(call.directory) == "main":
                return MAIN_COMMIT
            message = commit_message(segment, call, command)
            rules = hooklib.load_script(call.directory, "check_commit_message")
            if message is not None and rules is not None:
                problems = rules.check_commit_message(message)
                if problems:
                    listed = "\n".join(f"- {p}" for p in problems)
                    return f"Commit message fails the rule (AGENTS.md):\n{listed}"
        branched = branched or _creates_branch(call)
    return None


def main() -> int:
    """Read the PreToolUse payload and block a command that breaks a git rule."""
    payload = hooklib.read_input()
    reason = refusal(hooklib.tool_arg(payload, "command"), hooklib.session_cwd(payload))
    if reason:
        hooklib.deny(reason)
    return 0


if __name__ == "__main__":
    sys.exit(main())
