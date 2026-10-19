"""PreToolUse on Bash: the text a `gh` command would publish gets the checks a commit message gets.

This is the one place the session-link rule can be enforced for pull request
and issue text: it lands on a public repository, and lefthook never sees it.
A pull request title is also the squash-merge subject, so `pr create`,
`pr edit` and `pr merge --subject` get the Conventional Commits check.

Exit 2 blocks; stderr is the reason the model reads. A `--body-file` is read
and checked too, since that is how the harness writes long bodies.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import hooklib

PUBLISHES = re.compile(
    r"\bgh\s+(?:(?:pr|issue)\s+(?:create|edit|comment|review|close|merge)|release\s+(?:create|edit))\b"
)
API_WITH_BODY = re.compile(r"\bgh\s+api\b[\s\S]*\bbody\b")
TITLED = re.compile(r"\bgh\s+pr\s+(?:create|edit|merge)\b")
TITLE_FLAG = re.compile(
    r"(?:^|\s)(?:-t|--title|--subject)(?:=|\s+)(?:\"((?:[^\"\\]|\\.)*)\"|'([^']*)')"
)
BODY_FILE_FLAG = re.compile(r"(?:^|\s)(?:-F|--body-file)(?:=|\s+)(?:\"([^\"]+)\"|'([^']+)'|(\S+))")
SESSION_LINK = "claude.ai/code/session"


def _first(match: re.Match[str]) -> str:
    return next(g for g in match.groups() if g is not None)


def body_file_text(command: str, cwd: str) -> str:
    """Return a `--body-file`'s contents, or "" when there is none or it does not exist yet."""
    match = BODY_FILE_FLAG.search(command)
    if not match:
        return ""
    path = Path(_first(match))
    if not path.is_absolute():
        path = Path(cwd) / path
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def check(command: str, cwd: str) -> list[str]:
    """Every rule the published text breaks; empty to let the command run."""
    if not (PUBLISHES.search(command) or API_WITH_BODY.search(command)):
        return []
    root = hooklib.repo_root(cwd)
    prose = hooklib.load_script(root, "check_prose")
    commit = hooklib.load_script(root, "check_commit_message")
    if prose is None or commit is None:
        return []

    text = f"{command}\n{body_file_text(command, cwd)}"
    problems: list[str] = []
    if SESSION_LINK in text:
        problems.append(f"contains a {SESSION_LINK} link; never publish one (AGENTS.md)")
    problems.extend(
        f"line {v.line} has an {v.kind}: {v.snippet}" for v in prose.find_prose_violations(text)
    )
    title = TITLE_FLAG.search(command) if TITLED.search(command) else None
    if title:
        problems.extend(
            f"PR title is the squash-merge subject: {p}"
            for p in commit.check_commit_message(_first(title))
            if p.startswith("subject")
        )
    return problems


def main() -> int:
    """Read the PreToolUse payload and block a gh command that would publish a broken rule."""
    payload = hooklib.read_input()
    command = (payload.get("tool_input") or {}).get("command") or ""
    problems = check(command, payload.get("cwd") or ".")
    if problems:
        listed = "\n".join(f"- {p}" for p in problems)
        hooklib.deny(
            f"This gh command would publish text that breaks a rule (AGENTS.md):\n{listed}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
