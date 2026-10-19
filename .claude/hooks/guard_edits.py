"""PreToolUse on Edit, Write and NotebookEdit: the files nobody hand-edits, with the reason.

Answers with the JSON deny shape rather than exit 2, so the reason reaches the
model verbatim and the turn continues. The edited path is resolved through
symlinks and against the file's own checkout (`hooklib.edited_file`), and
compared without case, since macOS file systems ignore it. Only this project
and its worktrees are protected.
"""

from __future__ import annotations

import json
import sys

import hooklib

PROTECTED_FILES = {
    "claude.md": "a symlink to AGENTS.md; edit AGENTS.md",
    "uv.lock": "written by uv; change pyproject.toml and run `uv lock`",
}
PROTECTED_PREFIXES = {
    "tests/test_golden/": (
        "a golden array, written only by `uv run pytest tests/test_golden.py --force-regen` "
        "in a commit of its own (CONTRIBUTING.md, Golden outputs)"
    ),
    "notebooks/archive/": "history from before the package and is not edited",
    "data/": "the owner's unpublished data, outside version control; ask before changing it",
}


def reason_for(path: str) -> str | None:
    """Return why a repository-relative path must not be hand-edited, or None."""
    key = path.casefold()
    if key in PROTECTED_FILES:
        return PROTECTED_FILES[key]
    return next((r for p, r in PROTECTED_PREFIXES.items() if key.startswith(p)), None)


def main() -> int:
    """Deny an edit to a protected path, with the reason."""
    payload = hooklib.read_input()
    edited = hooklib.edited_file(payload)
    if edited is None or hooklib.trusted_root(edited.root) is None:
        return 0
    reason = reason_for(edited.path)
    if reason:
        decision = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": f"{edited.path} is {reason}.",
            }
        }
        sys.stdout.write(json.dumps(decision))
    return 0


if __name__ == "__main__":
    sys.exit(main())
