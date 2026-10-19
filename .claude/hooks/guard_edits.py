"""PreToolUse on Edit and Write: the files nobody hand-edits, with the reason each time.

Answers with the JSON deny shape rather than exit 2, so the reason reaches the
model verbatim and the turn continues.
"""

from __future__ import annotations

import json
import sys

import hooklib

PROTECTED_FILES = {
    "CLAUDE.md": "a symlink to AGENTS.md; edit AGENTS.md",
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
    """Why a repository-relative path must not be hand-edited, or None."""
    if path in PROTECTED_FILES:
        return PROTECTED_FILES[path]
    return next((r for p, r in PROTECTED_PREFIXES.items() if path.startswith(p)), None)


def main() -> int:
    """Deny an edit to a protected path, with the reason."""
    payload = hooklib.read_input()
    root = hooklib.repo_root(payload.get("cwd") or ".")
    path = hooklib.repo_relative(root, (payload.get("tool_input") or {}).get("file_path") or "")
    reason = reason_for(path)
    if reason:
        decision = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": f"{path} is {reason}.",
            }
        }
        sys.stdout.write(json.dumps(decision))
    return 0


if __name__ == "__main__":
    sys.exit(main())
