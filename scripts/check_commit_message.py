"""The commit-message rules from AGENTS.md, as a check.

A Conventional Commits subject with a lowercase imperative, no emdash or emoji
anywhere, and never a `claude.ai/code/session` link.

One implementation, three callers. lefthook runs it at `commit-msg`; CI runs it
over every commit a pull request adds (`--range`) and over the pull request
title and body (`--stdin`), which a squash merge turns into the `main` subject
and body.

The session-link rule is the hard one: the repository is public, so a link
that lands on `main` costs a history rewrite. It is checked first.

Usage::

    python scripts/check_commit_message.py <file>          the COMMIT_EDITMSG path
    python scripts/check_commit_message.py --text <str>    a message as a string
    python scripts/check_commit_message.py --stdin         a message on stdin
    python scripts/check_commit_message.py --range <a..b>  every commit in a range

Only the file form strips git's `#` comment lines and the scissors block,
because only COMMIT_EDITMSG carries them. A pull request body arrives on stdin
and its `## Heading` lines are prose to check, not comments to drop.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from check_prose import find_prose_violations, git_lines

TYPES = ("feat", "fix", "docs", "test", "refactor", "style", "perf", "build", "ci", "chore")

# `type(scope)!: subject`, scope optional. The lowercase-imperative half is
# checked below as "does not start with a capital letter", so a subject may
# lead with a digit, a quote, or an identifier in backticks.
SUBJECT = re.compile(rf"^(?:{'|'.join(TYPES)})(?:\([a-z0-9-]+\))?!?: (?P<rest>\S.*)$")

# Subjects written by git, GitHub and Dependabot. Dependabot capitalizes
# "Bump" and offers no setting to change it.
EXEMPT_SUBJECT = re.compile(
    r"^(?:Merge |Revert |fixup! |squash! |(?:build|chore|ci)\(deps(?:-dev)?\): Bump )"
)

SESSION_LINK = "claude.ai/code/session"
SCISSORS = re.compile(r"^# -{8,} >8 -{8,}$", re.MULTILINE)

RULE = (
    "Commit rule: type(scope): lowercase imperative subject, no emdash, no emoji, "
    "no session link. See AGENTS.md.\n"
)


def strip_commentary(message: str) -> str:
    """Drop git's comment lines and everything after the `--verbose` scissors line."""
    scissors = SCISSORS.search(message)
    body = message[: scissors.start()] if scissors else message
    return "\n".join(line for line in body.split("\n") if not line.startswith("#"))


def subject_ok(subject: str) -> bool:
    """Whether a subject line is exempt or a Conventional lowercase subject."""
    if EXEMPT_SUBJECT.match(subject):
        return True
    match = SUBJECT.match(subject)
    return match is not None and not match["rest"][0].isupper()


def check_commit_message(message: str) -> list[str]:
    """Return every problem with `message`; an empty list means it passes."""
    problems: list[str] = []
    if SESSION_LINK in message:
        problems.append(f"contains a {SESSION_LINK} link; drop it, keep the Co-Authored-By trailer")

    subject = next((line for line in message.split("\n") if line.strip()), "")
    if not subject:
        problems.append("the message is empty")
    elif not subject_ok(subject):
        problems.append(
            f'subject "{subject}" is not "type(scope): lowercase imperative" '
            f"with type one of {', '.join(TYPES)}"
        )

    problems.extend(
        f"line {v.line} has an {v.kind}: {v.snippet}" for v in find_prose_violations(message)
    )
    return problems


def _report(label: str, problems: list[str]) -> None:
    for problem in problems:
        sys.stderr.write(f"{label}: {problem}\n")


def main(argv: list[str]) -> int:
    """Run the check; return the process exit code."""
    if not argv:
        sys.stderr.write(__doc__ or "")
        return 2
    mode, rest = argv[0], argv[1:]

    if mode == "--range":
        # Merge commits are walked too: their subject is exempt, their body is
        # where a stray session trailer would sit.
        failed = False
        for sha in git_lines("log", "--format=%H", rest[0]):
            message = subprocess.run(
                ["git", "log", "-1", "--format=%B", sha],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            problems = check_commit_message(message)
            if problems:
                failed = True
                _report(sha[:7], problems)
        if failed:
            sys.stderr.write(RULE)
        return int(failed)

    if mode == "--text":
        message = " ".join(rest)
    elif mode == "--stdin":
        message = sys.stdin.read()
    else:
        message = strip_commentary(Path(mode).read_text(encoding="utf-8"))

    problems = check_commit_message(message)
    if problems:
        _report("commit message", problems)
        sys.stderr.write(RULE)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
