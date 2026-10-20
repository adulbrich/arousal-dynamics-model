"""PreToolUse on Bash: the text a `gh` command would publish gets the checks a commit message gets.

This is the one place the session-link rule can be enforced for pull request
and issue text: it lands on a public repository, and lefthook never sees it.
A pull request title is also the squash-merge subject, so `pr create`,
`pr edit` and `pr merge --subject` get the Conventional Commits check.

Only the `gh` commands that publish are checked, each on its own
(`hooklib.steps`), with the text they read from elsewhere: `--body-file` for
pull requests and issues, `--notes-file` for releases, `--field k=@file` and
`--input` for `gh api`, and a heredoc when any of those is `-`. Exit 2 blocks;
stderr is the reason the model reads. The session-link check needs no code
from the checkout and runs anywhere; the prose and title checks load the rule
scripts, so they run in this project and its worktrees only
(`hooklib.trusted_root`).
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import hooklib
from hooklib import Step, Word

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
API_FIELD_FLAGS = ("-f", "-F", "--field", "--raw-field")
API_WRITE_FLAGS = (*API_FIELD_FLAGS, "--input")
# Flags whose value is published text, for pr, issue and release.
TEXT_FLAGS = ("-t", "--title", "-b", "--body", "--subject", "-c", "--comment", "-n", "--notes")
REPO_FLAGS = frozenset({"-R", "--repo"})
# gh flags that take a value in every command that has them, so the word after
# one is that value. Flags that are switches in some commands (`-c`, `-r`,
# `-a`, `-m`, `-p`, `-f` in `pr review`, `pr merge`, `pr create`, `release`)
# are left out, and a following word that is itself a known flag is never
# taken as a value (`flag_values`).
VALUE_FLAGS = frozenset(
    {"-t", "--title", "-b", "--body", "-F", "--body-file", "--subject", "--notes"}
    | {"--notes-file", "-l", "--label", "--assignee", "--reviewer", "-B", "--base", "-H"}
    | {"--head", "--milestone", "--project", "-T", "--template", "--field", "--raw-field"}
    | {"--input", "-X", "--method", "-q", "--jq", "--header", "--target", "--repo", "-R"}
)
KNOWN_FLAGS = VALUE_FLAGS | frozenset(TEXT_FLAGS) | frozenset(API_FIELD_FLAGS)
# Flags whose value shapes the output and is never sent anywhere.
LOCAL_VALUE_FLAGS = frozenset({"-q", "--jq", "--template", "--header"})


def sent_words(group: str, args: list[Word]) -> str:
    """Return every word of a command except the values of flags that stay local.

    Comments are already gone (`hooklib.words`). This is the session-link
    backstop's text, in case a flag swallowed a published value.
    """
    local = LOCAL_VALUE_FLAGS | ({"-t", "-H"} if group == "api" else set())
    kept: list[str] = []
    skip = False
    for word in args:
        if skip:
            skip = False
        elif word.value in local:
            skip = True
        elif not any(word.value.startswith(f"{flag}=") for flag in local):
            kept.append(word.value)
    return "\n".join(kept)


def flag_values(args: list[Word], names: tuple[str, ...]) -> list[str]:
    """Return the values given to any of `names`: `--flag v`, `--flag=v`, `-f v` or `-fv`.

    A word that is the value of the flag before it is never read as a flag.
    """
    out: list[str] = []
    i = 0
    while i < len(args):
        value = args[i].value
        if value in names and i + 1 < len(args):
            out.append(args[i + 1].value)
        else:
            for name in names:
                if name.startswith("--") and value.startswith(f"{name}="):
                    out.append(value[len(name) + 1 :])
                elif (
                    not name.startswith("--")
                    and not value.startswith("--")
                    and value.startswith(name)
                    and len(value) > len(name)
                ):
                    out.append(value[len(name) :])
        following = args[i + 1].value if i + 1 < len(args) else ""
        i += 2 if value in VALUE_FLAGS and following not in KNOWN_FLAGS else 1
    return out


def published_text(group: str, args: list[Word], step: Step) -> str:
    """Return the text a publishing command sends: flag values, files, and a heredoc."""
    if group == "api":
        fields = flag_values(args, API_FIELD_FLAGS)
        texts = [v.split("=", 1)[-1] for v in fields if "=@" not in v]
        sources = [v.split("=@", 1)[1] for v in fields if "=@" in v]
    else:
        texts = flag_values(args, TEXT_FLAGS)
        sources = []
    sources += flag_values(args, TEXT_FILE_FLAGS.get(group, ()))
    for raw in sources:
        if raw == "-":
            texts.append(step.segment.heredoc or "")
        elif raw:
            # A file not written yet, or not UTF-8, is left to pr-text on the remote.
            with contextlib.suppress(OSError, ValueError):
                texts.append(Path(hooklib.resolve_dir(step.cwd, raw)).read_text(encoding="utf-8"))
    return "\n".join(texts)


def _command(step: Step) -> tuple[str, str, list[Word]] | None:
    """Return the group, verb and arguments of a gh call, skipping `-R/--repo`."""
    if not step.words or not hooklib.is_command(step.words[0], "gh"):
        return None
    rest: list[Word] = []
    skip = False
    for word in step.words[1:]:
        if skip:
            skip = False
        elif word.value in REPO_FLAGS:
            skip = True
        elif not word.value.startswith(("--repo=", "-R")) or rest:
            rest.append(word)
    if not rest:
        return None
    group = rest[0].value
    verb = rest[1].value if len(rest) > 1 and group != "api" else ""
    return group, verb, rest[1:]


def _publishes(group: str, verb: str, args: list[Word]) -> bool:
    if group == "api":
        methods = flag_values(args, ("-X", "--method"))
        if methods and methods[-1].upper() in {"GET", "HEAD"}:  # the last one wins
            return False
        return any(a.value.startswith(API_WRITE_FLAGS) for a in args)
    return (group, verb) in PUBLISHING


def problems(command: str, cwd: str) -> list[str]:
    """Return every rule the published text breaks; empty to let the command run."""
    found: list[str] = []
    for step in hooklib.steps(command, cwd):
        call = _command(step)
        if call is None or not _publishes(*call):
            continue
        group, verb, args = call
        text = published_text(group, args, step)
        # The session link is the rule that matters most, so it is also looked
        # for in every word the command sends, in case a flag was misread as
        # a value.
        if hooklib.SESSION_LINK in f"{text}\n{sent_words(group, args)}":
            found.append(f"contains a {hooklib.SESSION_LINK} link; never publish one (AGENTS.md)")
        prose = hooklib.load_script(step.cwd, "check_prose")
        commit = hooklib.load_script(step.cwd, "check_commit_message")
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
