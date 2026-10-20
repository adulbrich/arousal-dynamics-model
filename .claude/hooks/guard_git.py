"""PreToolUse on Bash: the git rules from AGENTS.md, enforced before the command runs.

- Stage by name: no `git add -A`, `git add .`, `git add :/`, `git commit -a`.
- Never commit on main.
- No force push at main, no `reset --hard`, `clean -f`, `branch -D`, or a
  working-tree wipe through `checkout .` and `restore .`.
- A commit message that would fail `commit-msg` fails here first, so a
  `--no-verify` cannot get it past lefthook.

Everything else, including a plain push, a rebase, a stash and any dry run,
goes through. Exit 2 blocks; stderr is the reason the model reads.

Each simple command on the line is checked on its own (`hooklib.steps`), with
quoted strings, comments and heredoc bodies masked, so a commit message or a
file written by `cat <<EOF` that mentions `git add -A` is not a `git add -A`.
Options are read as git receives them, quotes removed. A `cd` earlier on the
line and `git -C <dir>` move where the branch is read. The hook has no `if`
filter in settings.json on purpose: `cd x && git add -A` starts with `cd`.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable
from typing import NamedTuple

import hooklib
from hooklib import Segment, Word

SHORT_CLUSTER = re.compile(r"^-[A-Za-z]+")
MAIN_REF = re.compile(r"(?:^|[:+])(?:refs/heads/)?main$")
CAT_SUBSTITUTION = re.compile(r"^\$\(\s*cat\s*<<")
# Global options that take a separate argument.
GLOBAL_WITH_ARG = frozenset({"-C", "-c", "--git-dir", "--work-tree", "--exec-path", "--namespace"})
# `git commit` short options that take a value: in the rest of the cluster, or
# for those in COMMIT_NEXT_WORD, the next word when the cluster ends with them.
COMMIT_VALUE_OPTIONS = frozenset("mFCctuS")
COMMIT_NEXT_WORD = frozenset("mFCct")
# `git push` options whose value is the next word.
PUSH_WITH_ARG = frozenset({"-o", "--push-option", "--repo", "--receive-pack", "--exec"})
WIPE_PATHSPECS = frozenset({".", "./", ":/", ":/."})

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
    while i < len(segment_words) and segment_words[i].value.startswith("-"):
        flag = segment_words[i].value
        if flag in GLOBAL_WITH_ARG and i + 1 < len(segment_words):
            if flag == "-C":
                directory = hooklib.resolve_dir(directory, segment_words[i + 1].value)
            i += 2
        else:
            i += 1
    sub = segment_words[i].value if i < len(segment_words) else ""
    return GitCall(sub, segment_words[i + 1 :], directory)


def _short_flag(token: str, letter: str) -> bool:
    """Tell whether `token` is a short flag cluster carrying `letter`, as `-fd` carries `f`."""
    return bool(re.fullmatch(r"-[A-Za-z]+", token)) and letter in token[1:]


def _has_flag(args: list[Word], long: str, letter: str = "") -> bool:
    """Tell whether `--long` or, given one, the short `letter` in any cluster is present."""
    return any(a.value == long or (letter and _short_flag(a.value, letter)) for a in args)


def _dry_run(args: list[Word], short_with_arg: frozenset[str], booleans: frozenset[str]) -> bool:
    """Tell whether the options, read as git reads them, leave a dry run in force.

    Options are read in order up to `--`; the last word wins, so a
    `--no-dry-run` (or any abbreviation of it) after `-n` cancels it. The
    value of a short option in `short_with_arg` is skipped, and so is the word
    after any long option not known to be a plain switch, since git accepts
    abbreviations (`--excl` for `--exclude`) and the hook must not read a
    value as `-n`. Unknown input therefore leans towards refusing. A cluster
    is not read here: in one, `n` can be another option's value (`-on`).
    """
    dry = False
    skip = False
    for a in args:
        value = a.value
        if skip:
            skip = False
        elif value == "--":
            break
        elif value.startswith("--no-d"):
            dry = False
        elif value in {"--dry-run", "-n"}:
            dry = True
        elif value in short_with_arg or (
            value.startswith("--") and "=" not in value and value not in booleans
        ):
            skip = True
    return dry


ADD_WITH_ARG = frozenset()
ADD_BOOLEANS = frozenset(
    {"--all", "--update", "--force", "--verbose", "--intent-to-add", "--patch", "--edit"}
)
CLEAN_WITH_ARG = frozenset({"-e"})
CLEAN_BOOLEANS = frozenset({"--force", "--quiet", "--interactive"})
PUSH_SHORT_WITH_ARG = frozenset({"-o"})
PUSH_BOOLEANS = frozenset(
    {"--force", "--all", "--mirror", "--tags", "--follow-tags", "--set-upstream", "--verbose"}
    | {"--quiet", "--no-verify", "--atomic", "--porcelain", "--delete", "--prune"}
    | {"--force-with-lease", "--force-if-includes", "--progress", "--thin", "--ipv4", "--ipv6"}
)


# One rule per subcommand. Each returns the refusal, or None.


def add_rule(call: GitCall) -> str | None:
    """Refuse `git add` of everything, unless it is a dry run."""
    if _dry_run(call.args, ADD_WITH_ARG, ADD_BOOLEANS):
        return None
    sweeping = any(
        a.value == "--all" or a.value in WIPE_PATHSPECS or _short_flag(a.value, "A")
        for a in call.args
    )
    if sweeping:
        return (
            "Stage files by name (AGENTS.md): `git add -A`, `git add .` and `git add :/` "
            "sweep up unrelated work in progress. Name the paths."
        )
    return None


def _commit_stages_all(args: list[Word]) -> bool:
    skip = False
    for a in args:
        if skip:
            skip = False
            continue
        if a.value == "--all":
            return True
        if not SHORT_CLUSTER.match(a.value) or a.value.startswith("--"):
            continue
        for k, letter in enumerate(a.value[1:], start=1):
            if letter in COMMIT_VALUE_OPTIONS:
                # The rest is that option's value (`-ma` is the message "a");
                # a cluster that ends on it takes the next word instead.
                skip = letter in COMMIT_NEXT_WORD and k == len(a.value) - 1
                break
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


def _head(ref: str) -> bool:
    return ref.startswith(("HEAD", "@"))


def push_rule(call: GitCall) -> str | None:
    """Refuse a force push that lands on main."""
    if _dry_run(call.args, PUSH_SHORT_WITH_ARG, PUSH_BOOLEANS):
        return None
    positional: list[str] = []
    skip = False
    for a in call.args:
        if skip:
            skip = False
        elif a.value in PUSH_WITH_ARG:
            skip = True
        elif not a.value.startswith("-"):
            positional.append(a.value)
    refspecs = positional[1:]
    forced = (
        _has_flag(call.args, "--force", "f")
        or any(a.value.startswith("--force-with-lease") for a in call.args)
        or any(r.startswith("+") for r in refspecs)
    )
    if not forced:
        return None
    on_main = hooklib.current_branch(call.directory) == "main"
    if refspecs:
        targets = [r.lstrip("+") for r in refspecs]
        at_main = any(MAIN_REF.search(t) for t in targets) or (
            on_main and any(_head(t) and ":" not in t for t in targets)
        )
    else:
        at_main = on_main
    return "No force push at main. The ruleset would reject it anyway." if at_main else None


def reset_rule(call: GitCall) -> str | None:
    """Refuse `git reset --hard`."""
    if _has_flag(call.args, "--hard"):
        return (
            "No `git reset --hard` from an agent: it discards uncommitted work the "
            "user may want. Ask them to run it."
        )
    return None


def _clean_dry_run(args: list[Word]) -> bool:
    """Tell whether a `git clean` is a dry run, clusters included.

    Read as `_dry_run` reads options, last word winning. In clean only `-e`
    takes a value, so an `n` before any `e` in a cluster is `-n`, and a
    cluster ending in `e` takes the next word.
    """
    dry = False
    skip = False
    for a in args:
        value = a.value
        if skip:
            skip = False
        elif value == "--":
            break
        elif value.startswith("--no-d"):
            dry = False
        elif value in {"--dry-run", "-n"}:
            dry = True
        elif value in CLEAN_WITH_ARG or (
            value.startswith("--") and "=" not in value and value not in CLEAN_BOOLEANS
        ):
            skip = True
        elif _short_flag(value, "n") and "e" not in value.split("n", 1)[0]:
            dry, skip = True, value.endswith("e")
        elif _short_flag(value, "e") and value.endswith("e"):
            skip = True
    return dry


def clean_rule(call: GitCall) -> str | None:
    """Refuse `git clean -f` unless it is a dry run."""
    if _has_flag(call.args, "--force", "f") and not _clean_dry_run(call.args):
        return "No `git clean -f` from an agent. Ask the user to run it."
    return None


def branch_rule(call: GitCall) -> str | None:
    """Refuse a forced branch delete."""
    if any(_short_flag(a.value, "D") for a in call.args) or (
        _has_flag(call.args, "--delete", "d") and _has_flag(call.args, "--force", "f")
    ):
        return "No `git branch -D` from an agent. Use `-d`, or ask the user to force-delete."
    return None


def checkout_rule(call: GitCall) -> str | None:
    """Refuse `git checkout .` and `git checkout :/`."""
    if any(a.value in WIPE_PATHSPECS for a in call.args):
        return "No working-tree wipe (`git checkout .`) from an agent. Name the paths to restore."
    return None


def restore_rule(call: GitCall) -> str | None:
    """Refuse `git restore .` and `git restore :/` on the working tree."""
    index_only = _has_flag(call.args, "--staged", "S") and not _has_flag(
        call.args, "--worktree", "W"
    )
    if any(a.value in WIPE_PATHSPECS for a in call.args) and not index_only:
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


def _switches_to_new_branch(call: GitCall) -> bool:
    flags = {a.value.split("=", 1)[0] for a in call.args}
    return (call.sub == "checkout" and bool(flags & {"-b", "-B"})) or (
        call.sub == "switch" and bool(flags & {"-c", "-C", "--create", "--force-create"})
    )


# Subcommands that record a commit on the current branch, and the options
# under which they do not.
RECORDS_COMMIT = {
    "commit": frozenset({"--dry-run"}),
    "merge": frozenset({"--ff-only", "--squash", "--abort", "--quit"}),
    "cherry-pick": frozenset({"-n", "--no-commit", "--abort", "--quit", "--skip"}),
    "revert": frozenset({"-n", "--no-commit", "--abort", "--quit", "--skip"}),
    "am": frozenset({"--abort", "--quit", "--skip", "--show-current-patch"}),
}


# Updating main from its upstream fast-forwards when main carries no commits
# of its own, which the never-commit-on-main rule guarantees.
UPSTREAM_REFS = frozenset({"origin/main", "@{u}", "@{upstream}", "FETCH_HEAD"})


def _records_commit(call: GitCall) -> bool:
    exempt = RECORDS_COMMIT.get(call.sub)
    if exempt is None or any(a.value in exempt for a in call.args):
        return False
    return not (call.sub == "merge" and any(a.value in UPSTREAM_REFS for a in call.args))


class Taken(NamedTuple):
    """A message value an option gives (None for none), and the index of its last word.

    `literal` marks a value written in single quotes, where `$(` and backticks
    are plain text.
    """

    value: str | None
    last: int
    literal: bool = False


def _single_quoted(word: Word) -> bool:
    return len(word.text) >= 2 and word.text[0] == word.text[-1] == "'"


class _UnreadableError(Exception):
    """The message comes from somewhere this hook does not read: a file, an editor, a commit."""


def _long_option(args: list[Word], i: int, heredoc: str | None) -> Taken | None:
    """Read `--message` and `--file -`; None when `args[i]` is neither."""
    word = args[i].value
    following = args[i + 1].value if i + 1 < len(args) else None
    if word.startswith("--message="):
        return Taken(word[len("--message=") :], i)
    if word == "--message" and following is not None:
        return Taken(following, i + 1, _single_quoted(args[i + 1]))
    if word in {"-F", "--file"} or word.startswith("--file="):
        inline = word.split("=", 1)[1] if "=" in word else None
        source = inline if inline is not None else following
        if source != "-" or heredoc is None:
            raise _UnreadableError
        return Taken(heredoc, i if inline is not None else i + 1)
    return None


def _short_cluster(args: list[Word], i: int) -> Taken:
    """Read `-m` in a short cluster, its value attached (`-mWIP`) or the next word."""
    word = args[i].value
    if not SHORT_CLUSTER.match(word) or word.startswith("--"):
        return Taken(None, i)
    for k, letter in enumerate(word[1:], start=1):
        if letter not in COMMIT_VALUE_OPTIONS:
            continue
        rest = word[k + 1 :]
        takes_next = letter in COMMIT_NEXT_WORD and not rest and i + 1 < len(args)
        if letter != "m":
            if letter in {"F", "C", "c"}:
                raise _UnreadableError  # a file, or a message reused from a commit
            return Taken(None, i + 1 if takes_next else i)
        if takes_next:
            return Taken(args[i + 1].value, i + 1, _single_quoted(args[i + 1]))
        return Taken(rest, i)
    return Taken(None, i)


def _message_part(taken: Taken, heredoc: str | None) -> str | None:
    """Return a message value as git records it, or None when it is a substitution not read."""
    value = taken.value or ""
    if taken.literal:
        return value
    if CAT_SUBSTITUTION.match(value) and value.rstrip().endswith(")"):
        return heredoc  # the harness's -m "$(cat <<'EOF' ... EOF\n)", nothing after it
    return None if "$(" in value or "`" in value else value


def commit_message(segment: Segment, call: GitCall) -> str | None:
    """Return the message this `git commit` would record, or None when it cannot be read here.

    That covers `-m`, `-mVALUE`, a cluster ending in `m`, `--message`,
    `--message=`, the harness's `-m "$(cat <<'EOF' ... EOF)"`, and `-F -` fed
    by a heredoc on the same command. A message from a file or an editor is
    left to lefthook's `commit-msg`.
    """
    parts: list[str] = []
    i = 0
    try:
        while i < len(call.args):
            taken = _long_option(call.args, i, segment.heredoc) or _short_cluster(call.args, i)
            if taken.value is not None:
                part = _message_part(taken, segment.heredoc)
                if part is None:
                    return None
                parts.append(part)
            i = taken.last + 1
    except _UnreadableError:
        return None
    return "\n\n".join(parts) if parts else None


def refusal(command: str, cwd: str) -> str | None:
    """Return the reason to refuse `command`, or None to let it run."""
    branched: set[str] = set()
    for step in hooklib.steps(command, cwd):
        call = parse_git(step.words, step.cwd)
        if call is None:
            continue
        rule = RULES.get(call.sub)
        reason = rule(call) if rule else None
        if reason:
            return reason
        checkout = hooklib.real(hooklib.repo_root(call.directory))
        on_main = hooklib.current_branch(call.directory) == "main"
        if _records_commit(call) and checkout not in branched and on_main:
            return MAIN_COMMIT
        if call.sub == "commit" and not _has_flag(call.args, "--dry-run"):
            published = f"{step.segment.text}\n{step.segment.heredoc or ''}"
            if hooklib.SESSION_LINK in published:
                return (
                    f"This commit carries a {hooklib.SESSION_LINK} link, in its message or a "
                    "trailer. Never publish one (AGENTS.md)."
                )
            message = commit_message(step.segment, call)
            rules = hooklib.load_script(call.directory, "check_commit_message")
            if message is not None and rules is not None:
                problems = rules.check_commit_message(message)
                if problems:
                    listed = "\n".join(f"- {p}" for p in problems)
                    return f"Commit message fails the rule (AGENTS.md):\n{listed}"
        if _switches_to_new_branch(call):
            branched.add(checkout)
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
