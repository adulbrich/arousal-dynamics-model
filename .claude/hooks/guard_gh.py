"""PreToolUse on Bash: the text a `gh` command would publish gets the checks a commit message gets.

This is the one place the session-link rule can be enforced for pull request
and issue text: it lands on a public repository, and lefthook never sees it.
A pull request title is also the squash-merge subject, so `pr create`,
`pr edit` and `pr merge --subject` get the Conventional Commits check.

Only the `gh` commands that publish are checked, each on its own
(`hooklib.segments`), with the files they read the text from: `--body-file`
for pull requests and issues, `--notes-file` for releases, `--field k=@file`
and `--input` for `gh api`. Exit 2 blocks; stderr is the reason the model
reads. The session-link check needs no code from the checkout and runs
anywhere; the prose and title checks load the rule scripts, so they run in
this project and its worktrees only (`hooklib.trusted_root`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import hooklib
from hooklib import Word

PUBLISHING = frozenset(
    {
        ("pr", "create"),
        ("pr", "edit"),
        ("pr", "comment"),
        ("pr", "review"),
        ("pr", "close"),
        ("pr", "merge"),
        ("issue", "create"),
        ("issue", "edit"),
        ("issue", "comment"),
        ("issue", "close"),
        ("release", "create"),
        ("release", "edit"),
    }
)
TITLED = frozenset({("pr", "create"), ("pr", "edit"), ("pr", "merge")})
TITLE_FLAGS = ("-t", "--title", "--subject")
# Flags whose value is a file of published text, by command group. In
# `gh api`, -F is --field, so its files are the `key=@path` values.
TEXT_FILE_FLAGS = {
    "pr": ("-F", "--body-file"),
    "issue": ("-F", "--body-file"),
    "release": ("-F", "--notes-file"),
    "api": ("--input",),
}
API_WRITE_FLAGS = ("-f", "-F", "--field", "--raw-field", "--input")


def flag_values(args: list[Word], names: tuple[str, ...]) -> list[str]:
    """Return the values given to any of `names`, as `--flag value`, `--flag=value`, `-f value`."""
    out: list[str] = []
    for i, word in enumerate(args):
        for name in names:
            if word.masked == name and i + 1 < len(args):
                out.append(hooklib.unquote(args[i + 1].text))
            elif name.startswith("--") and word.masked.startswith(f"{name}="):
                out.append(hooklib.unquote(word.text[len(name) + 1 :]))
    return out


def text_files(group: str, args: list[Word], cwd: str) -> list[Path]:
    """Return the files a publishing command reads its text from, resolved against `cwd`."""
    paths = flag_values(args, TEXT_FILE_FLAGS.get(group, ()))
    if group == "api":
        paths += [v.split("=@", 1)[1] for v in flag_values(args, ("-F", "--field")) if "=@" in v]
    out = []
    for raw in paths:
        if raw and raw != "-":
            path = Path(raw).expanduser()
            out.append(path if path.is_absolute() else Path(cwd) / path)
    return out


def read_text(path: Path) -> str:
    """Return a file's text, or "" when it does not exist yet or is not UTF-8."""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return ""


def _publishes(group: str, verb: str, args: list[Word]) -> bool:
    if group == "api":
        return any(a.masked.split("=", 1)[0] in API_WRITE_FLAGS for a in args)
    return (group, verb) in PUBLISHING


def problems(command: str, cwd: str) -> list[str]:
    """Return every rule the published text breaks; empty to let the command run."""
    found: list[str] = []
    here = cwd
    for segment in hooklib.segments(command):
        segment_words = hooklib.words(segment)
        here = hooklib.change_directory(here, segment_words)
        if len(segment_words) < 2 or not hooklib.is_command(segment_words[0], "gh"):
            continue
        group = segment_words[1].masked
        verb = segment_words[2].masked if len(segment_words) > 2 else ""
        args = segment_words[2:]
        if not _publishes(group, verb, args):
            continue
        text = "\n".join([segment.text, *(read_text(p) for p in text_files(group, args, here))])
        if hooklib.SESSION_LINK in text:
            found.append(f"contains a {hooklib.SESSION_LINK} link; never publish one (AGENTS.md)")
        prose = hooklib.load_script(here, "check_prose")
        commit = hooklib.load_script(here, "check_commit_message")
        if prose is None or commit is None:
            continue
        found.extend(
            f"line {v.line} has an {v.kind}: {v.snippet}" for v in prose.find_prose_violations(text)
        )
        if (group, verb) in TITLED:
            for title in flag_values(args, TITLE_FLAGS):
                found.extend(
                    f"PR title is the squash-merge subject: {p}"
                    for p in commit.check_commit_message(title)
                    if p.startswith("subject")
                )
    return found


def main() -> int:
    """Read the PreToolUse payload and block a gh command that would publish a broken rule."""
    payload = hooklib.read_input()
    found = problems(hooklib.tool_arg(payload, "command"), hooklib.session_cwd(payload))
    if found:
        listed = "\n".join(f"- {p}" for p in found)
        hooklib.deny(
            f"This gh command would publish text that breaks a rule (AGENTS.md):\n{listed}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
