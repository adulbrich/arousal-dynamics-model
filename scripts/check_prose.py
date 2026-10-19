"""The prose rule from AGENTS.md, as a check: no emdash and no emoji in anything tracked.

That covers docs, comments, docstrings and string literals alike. A `--`
standing in for a sentence dash breaks the same rule but is not checked here,
because every command-line flag would trip it.

One implementation, three callers. lefthook runs it on staged files at
pre-commit, CI runs it over the whole tree (`--all`), and
`check_commit_message.py` imports `find_prose_violations` for commit messages
and pull request text.

Usage::

    python scripts/check_prose.py <file>...     check the named files
    python scripts/check_prose.py --all         check every tracked text file
    python scripts/check_prose.py --text <str>  check a string
    python scripts/check_prose.py --stdin       check stdin as one string
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

EMDASH = "\u2014"

# The blocks emoji are drawn from: the astral range from Mahjong tiles through
# the extended pictographs, Miscellaneous Symbols and Dingbats, the emoji
# subset of Miscellaneous Technical, the information source, and the arrows,
# squares and circles keyboards offer as emoji. Deliberately absent: the
# arrows block (U+2190 to U+21FF), box drawing, Greek letters and the math
# operators an equation in a docstring needs.
EMOJI = re.compile(
    "["
    + "".join(
        [
            "\U0001f000-\U0001faff",  # Mahjong tiles through the extended pictographs
            "\u231a\u231b\u2328\u23cf",  # watch, hourglass, keyboard, eject
            "\u23e9-\u23f3\u23f8-\u23fa",  # media buttons, alarm clock, timers
            "\u2600-\u27bf",  # Miscellaneous Symbols and Dingbats
            "\u2139",  # information source
            "\u2b05-\u2b07\u2b1b\u2b1c\u2b50\u2b55",  # emoji arrows, squares, star, circle
        ]
    )
    + "]"
)

TEXT_SUFFIXES = frozenset(
    {".cfg", ".ini", ".ipynb", ".json", ".md", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml"}
)

# Paths the rule does not reach: notebooks from before the rule, kept as
# history, and machine output.
EXCLUDED_PREFIXES = ("notebooks/archive/",)
EXCLUDED_FILES = frozenset({"uv.lock"})

# The one line the harness appends to every pull request body. Stripped before
# checking so that the footer is not what fails the PR.
HARNESS_FOOTER = re.compile("\U0001f916 Generated with \\[Claude Code\\]\\([^)]*\\)")

RULE = (
    "Prose rule: no emdash (U+2014) and no emoji. Use a comma, a colon, or a new "
    "sentence; use words for a status mark. See AGENTS.md.\n"
)


@dataclass(frozen=True)
class Violation:
    """One rule hit: the 1-based line, which rule, and the offending line."""

    line: int
    kind: str
    snippet: str


def find_prose_violations(text: str) -> list[Violation]:
    """Return every emdash and emoji in `text`, one entry per rule per line."""
    violations: list[Violation] = []
    for number, line in enumerate(HARNESS_FOOTER.sub("", text).split("\n"), start=1):
        if EMDASH in line:
            violations.append(Violation(number, "emdash", line.strip()))
        if EMOJI.search(line):
            violations.append(Violation(number, "emoji", line.strip()))
    return violations


def is_checked_path(path: str) -> bool:
    """Whether the prose rule applies to a repository-relative path."""
    if path in EXCLUDED_FILES or path.startswith(EXCLUDED_PREFIXES):
        return False
    return Path(path).suffix in TEXT_SUFFIXES


def git_lines(*args: str) -> list[str]:
    """Run git and return its non-empty output lines."""
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    return [line for line in result.stdout.split("\n") if line]


def report(label: str, violations: list[Violation]) -> None:
    """Write one stderr line per violation."""
    for v in violations:
        excerpt = v.snippet if len(v.snippet) <= 100 else f"{v.snippet[:100]}..."
        sys.stderr.write(f"{label}:{v.line}: {v.kind}: {excerpt}\n")


def check_files(paths: list[str]) -> bool:
    """Check each file; return True when any of them breaks the rule."""
    failed = False
    for path in paths:
        try:
            text = Path(path).read_text(encoding="utf-8")
        except (FileNotFoundError, UnicodeDecodeError):
            # Deleted in the index but still listed, or not text after all.
            continue
        violations = find_prose_violations(text)
        if violations:
            failed = True
            report(path, violations)
    return failed


def main(argv: list[str]) -> int:
    """Run the check; return the process exit code."""
    mode, rest = (argv[0], argv[1:]) if argv else ("", [])
    if mode in {"--text", "--stdin"}:
        text = " ".join(rest) if mode == "--text" else sys.stdin.read()
        violations = find_prose_violations(text)
        report("text", violations)
        failed = bool(violations)
    elif mode == "--all":
        failed = check_files([p for p in git_lines("ls-files") if is_checked_path(p)])
    else:
        failed = check_files([p for p in argv if is_checked_path(p)])

    if failed:
        sys.stderr.write(RULE)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
