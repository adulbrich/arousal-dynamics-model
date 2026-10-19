"""SessionStart: the facts every session used to have to be told, printed into context.

The branch and whether it is main, uncommitted changes, and whether the uv on
PATH is new enough for the lockfile. No network.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import hooklib

REQUIRED_UV = re.compile(r'^required-version\s*=\s*">=\s*([0-9.]+)"', re.MULTILINE)
UV_VERSION = re.compile(r"\buv ([0-9]+(?:\.[0-9]+)*)")


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


def uv_line(root: str) -> str:
    """Compare the uv on PATH with the minimum `pyproject.toml` requires."""
    try:
        found = subprocess.run(
            ["uv", "--version"],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        found = ""
    have = UV_VERSION.search(found)
    if not have:
        return "uv: not found on PATH. Install it before `uv sync` (README.md)."
    try:
        need = REQUIRED_UV.search((Path(root) / "pyproject.toml").read_text(encoding="utf-8"))
    except OSError:
        need = None
    if need and _version(have.group(1)) < _version(need.group(1)):
        return (
            f"uv: {have.group(1)} on PATH, pyproject.toml requires {need.group(1)} or newer. "
            "Ask the user to run `uv self update`."
        )
    return f"uv: {have.group(1)}."


def lines(cwd: str) -> list[str]:
    """Collect the session facts, one line each."""
    out = []
    branch = hooklib.current_branch(cwd) or "(detached)"
    out.append(
        "Branch: main. Do not commit here: fetch, then branch from origin/main first (AGENTS.md)."
        if branch == "main"
        else f"Branch: {branch}."
    )
    dirty = [line for line in hooklib.git(cwd, "status", "--porcelain").split("\n") if line]
    out.append(
        "Working tree: clean."
        if not dirty
        else (
            f"Working tree: {len(dirty)} uncommitted path(s). Stage by name; "
            "they may be someone else's work in progress."
        )
    )
    out.append(uv_line(hooklib.repo_root(cwd)))
    out.append(
        "Gates: lefthook.yml at commit and push, the hooks under .claude/hooks in this "
        "session. CONTRIBUTING.md has the table."
    )
    return out


def main() -> int:
    """Print the session facts for the model's context."""
    payload = hooklib.read_input()
    sys.stdout.write("\n".join(lines(payload.get("cwd") or ".")) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
