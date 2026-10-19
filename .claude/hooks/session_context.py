"""SessionStart: the facts every session used to have to be told, printed into context.

The branch and whether it is main, uncommitted changes, other worktrees,
local branches whose remote is gone, and whether the uv on PATH is new enough
for the lockfile. No network.
"""

from __future__ import annotations

import os
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


def workspace_lines(cwd: str) -> list[str]:
    """Report other worktrees and branches whose remote is gone; nothing when there are none."""
    root = os.path.realpath(hooklib.repo_root(cwd))
    trees = [
        line[len("worktree ") :]
        for line in hooklib.git(cwd, "worktree", "list", "--porcelain").splitlines()
        if line.startswith("worktree ")
    ]
    others = [t for t in trees if os.path.realpath(t) != root]
    refs = hooklib.git(
        cwd, "for-each-ref", "--format=%(refname:short) %(upstream:track)", "refs/heads"
    )
    gone = [line.split()[0] for line in refs.splitlines() if "[gone]" in line]
    out = []
    if others:
        out.append(
            f"Other worktrees: {', '.join(others)}. Remove one with "
            "`git worktree remove <path>` once its branch has merged."
        )
    if gone:
        out.append(
            f"Branches whose remote is gone: {', '.join(gone)}. A squash merge leaves them "
            "behind; deleting them with `-D` is the user's call."
        )
    return out


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
    out.extend(workspace_lines(cwd))
    out.append(uv_line(hooklib.repo_root(cwd)))
    out.append(
        "Gates: lefthook.yml at commit and push, the hooks under .claude/hooks in this "
        "session. CONTRIBUTING.md has the table."
    )
    return out


def main() -> int:
    """Print the session facts for the model's context."""
    sys.stdout.write("\n".join(lines(hooklib.session_cwd(hooklib.read_input()))) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
