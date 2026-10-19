"""PreToolUse on Bash: the git rules from AGENTS.md, enforced before the command runs.

- Stage by name: no `git add -A`, `git add .`, `git commit -a`.
- Never commit on main.
- No force push at main, no `reset --hard`, `clean -f`, `branch -D`, or a
  working-tree wipe through `checkout .` and `restore .`.
- A commit message that would fail `commit-msg` fails here first, so a
  `--no-verify` cannot get it past lefthook.

Everything else, including a plain push, a rebase and a stash, goes through.
Exit 2 blocks; stderr is the reason the model reads.

The command line is tokenized rather than matched as text: git's global
options (`git -C dir add`, `git -c k=v commit`) sit between `git` and the
subcommand, flags come in short, long and clustered forms (`-f`, `--force`,
`-fd`), and a refspec can carry the force as a `+`. Quoted strings and heredoc
bodies are blanked before parsing, so a commit message that mentions
`git add -A` is not a `git add -A`. The hook has no `if` filter in
settings.json on purpose: `cd x && git add -A` starts with `cd`.
"""

from __future__ import annotations

import re
import sys
from typing import NamedTuple

import hooklib

HEREDOC = re.compile(r"<<-?\s*([\"']?)([A-Za-z_][A-Za-z0-9_]*)\1\n([\s\S]*?)\n\2\b")
CAT_HEREDOC = re.compile(r"(?:-m|--message)(?:=|\s+)\"?\$\(\s*cat\s*<<-?\s*[\"']?[A-Za-z_]")
QUOTED = re.compile(r"\"(?:[^\"\\]|\\.)*\"|'[^']*'")
MESSAGE_FLAG = re.compile(r"(?:^|\s)(?:-m|--message)(?:=|\s+)(?:\"((?:[^\"\\]|\\.)*)\"|'([^']*)')")
SEGMENT = re.compile(r"\s*(?:&&|\|\||;|\|)\s*|\n")
SHORT_CLUSTER = re.compile(r"^-[A-Za-z]+$")
MAIN_REF = re.compile(r"(?:^|[:+])(?:refs/heads/)?main$")
GIT_WORD = re.compile(r"\bgit\b")
# Global options that take a separate argument.
GLOBAL_WITH_ARG = frozenset({"-C", "-c", "--git-dir", "--work-tree", "--exec-path", "--namespace"})


class GitCall(NamedTuple):
    """The subcommand and its arguments in one shell segment."""

    sub: str
    args: list[str]


def parse_git(segment: str) -> GitCall | None:
    """Return the git invocation in one shell segment, or None."""
    tokens = segment.split()
    at = next((i for i, t in enumerate(tokens) if t == "git" or t.endswith("/git")), -1)
    if at == -1:
        return None
    i = at + 1
    while i < len(tokens) and tokens[i].startswith("-"):
        i += 2 if tokens[i] in GLOBAL_WITH_ARG else 1
    sub = tokens[i] if i < len(tokens) else ""
    return GitCall(sub, tokens[i + 1 :])


def _is_dot(token: str) -> bool:
    return token in {".", "./"}


def _short_flag(token: str, letter: str) -> bool:
    """Tell whether `token` is a short flag cluster carrying `letter`, as `-fd` carries `f`."""
    return bool(SHORT_CLUSTER.match(token)) and letter in token[1:]


def _has_flag(args: list[str], long: str, letter: str) -> bool:
    """`--long` or the short `letter` in any cluster, as `--force` or `-fd`."""
    return any(t == long or _short_flag(t, letter) for t in args)


def staging_rule(sub: str, args: list[str]) -> str | None:
    """Refuse the commands that stage everything."""
    if sub == "add" and any(t in {"-A", "--all"} or _is_dot(t) for t in args):
        return (
            "Stage files by name (AGENTS.md): `git add -A` and `git add .` sweep up "
            "unrelated work in progress. Name the paths."
        )
    if sub == "commit" and _has_flag(args, "--all", "a"):
        return (
            "Stage files by name (AGENTS.md): `git commit -a` commits every modified "
            "file. Stage the paths, then commit."
        )
    return None


def push_rule(args: list[str], cwd: str) -> str | None:
    """Refuse a force push at main."""
    forced = _has_flag(args, "--force", "f") or any(
        t.startswith(("--force-with-lease", "+")) or t == "--force-if-includes" for t in args
    )
    if not forced:
        return None
    at_main = any(MAIN_REF.search(t) for t in args) or hooklib.current_branch(cwd) == "main"
    return "No force push at main. The ruleset would reject it anyway." if at_main else None


def destructive_rule(sub: str, args: list[str]) -> str | None:
    """Refuse the commands that throw away work the user may want."""
    if sub == "reset" and "--hard" in args:
        return (
            "No `git reset --hard` from an agent: it discards uncommitted work the "
            "user may want. Ask them to run it."
        )
    if sub == "clean" and _has_flag(args, "--force", "f"):
        return "No `git clean -f` from an agent. Ask the user to run it."
    if sub == "branch":
        force_delete = any(_short_flag(t, "D") for t in args) or (
            _has_flag(args, "--delete", "d") and _has_flag(args, "--force", "f")
        )
        if force_delete:
            return "No `git branch -D` from an agent. Use `-d`, or ask the user to force-delete."
    if sub == "checkout" and any(_is_dot(t) for t in args):
        return "No working-tree wipe (`git checkout .`) from an agent. Name the paths to restore."
    if sub == "restore":
        index_only = _has_flag(args, "--staged", "S") and not _has_flag(args, "--worktree", "W")
        if any(_is_dot(t) for t in args) and not index_only:
            return (
                "No working-tree wipe (`git restore .`) from an agent. Name the paths to restore."
            )
    return None


def _branches(call: GitCall) -> bool:
    sub, args = call
    return (
        (sub == "checkout" and any(t in {"-b", "-B"} for t in args))
        or (sub == "switch" and any(t in {"-c", "-C"} for t in args))
        or (sub == "worktree" and bool(args) and args[0] == "add")
    )


def extract_commit_message(text: str) -> str | None:
    """Return the message a `git commit` would record, or None when it cannot be read here.

    That is a file, an editor, or a substitution other than the one shape the
    harness writes, `-m "$(cat <<'EOF' ... EOF)"`, whose heredoc body is the
    message. The heredoc is searched from that `-m`, so a heredoc earlier on
    the line is not a message.
    """
    shape = CAT_HEREDOC.search(text)
    if shape:
        body = HEREDOC.search(text[shape.start() :])
        return body.group(3) if body else None
    parts = [
        m.group(1) if m.group(1) is not None else m.group(2) for m in MESSAGE_FLAG.finditer(text)
    ]
    if not parts or any("$(" in p for p in parts):
        return None
    return "\n\n".join(parts).replace("\\n", "\n")


def check(command: str, cwd: str) -> str | None:
    """Return the reason to refuse `command`, or None to let it run."""
    if not GIT_WORD.search(command):
        return None
    detectable = QUOTED.sub('""', HEREDOC.sub("<<HEREDOC", command, count=1))
    branched = False
    commits = False
    for segment in SEGMENT.split(detectable):
        call = parse_git(segment)
        if call is None:
            continue
        on_main = (
            "Never commit on main (AGENTS.md). Fetch, branch from origin/main, then commit."
            if call.sub == "commit" and not branched and hooklib.current_branch(cwd) == "main"
            else None
        )
        reason = (
            staging_rule(*call)
            or on_main
            or (push_rule(call.args, cwd) if call.sub == "push" else destructive_rule(*call))
        )
        if reason:
            return reason
        branched = branched or _branches(call)
        commits = commits or call.sub == "commit"

    if commits:
        message = extract_commit_message(command)
        rules = hooklib.load_script(hooklib.repo_root(cwd), "check_commit_message")
        if message is not None and rules is not None:
            problems = rules.check_commit_message(message)
            if problems:
                listed = "\n".join(f"- {p}" for p in problems)
                return f"Commit message fails the rule (AGENTS.md):\n{listed}"
    return None


def main() -> int:
    """Read the PreToolUse payload and block a command that breaks a git rule."""
    payload = hooklib.read_input()
    command = (payload.get("tool_input") or {}).get("command") or ""
    reason = check(command, payload.get("cwd") or ".")
    if reason:
        hooklib.deny(reason)
    return 0


if __name__ == "__main__":
    sys.exit(main())
