"""Shared plumbing for the Claude Code hooks in this directory.

A hook is a process that reads one JSON object on stdin and answers with an
exit code: 0 lets the tool call through, 2 blocks it and hands stderr to the
model as the reason. The `cwd` field, not `CLAUDE_PROJECT_DIR`, is where the
session is: inside a worktree the project dir still points at the main
checkout, so the repository root is resolved from `cwd`.

The hooks run on whatever `python3` is first on PATH, which on macOS can be
the system Python 3.9, outside the project's virtual environment. So they use
the standard library only and no syntax newer than 3.9;
`tests/test_claude_hooks.py` checks both.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import IO, Any, NoReturn


def clean_env() -> dict[str, str]:
    """`os.environ` without the `GIT_*` keys a git hook exports.

    A hook that runs git on the session's `cwd` with `GIT_DIR` still set gets
    an answer about the wrong repository.
    """
    return {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}


def read_input(stream: IO[str] | None = None) -> dict[str, Any]:
    """Read the hook's JSON payload."""
    return json.load(stream if stream is not None else sys.stdin)


def deny(reason: str) -> NoReturn:
    """Block the tool call with a reason the model will read."""
    sys.stderr.write(f"{reason}\n")
    sys.exit(2)


def git(cwd: str, *args: str) -> str:
    """Git's trimmed output in `cwd`, or "" when git cannot answer."""
    try:
        result = subprocess.run(  # noqa: S603
            ["git", "-C", cwd, *args],  # noqa: S607
            capture_output=True,
            text=True,
            env=clean_env(),
            check=False,
        )
    except OSError:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def current_branch(cwd: str) -> str:
    """Return the checked-out branch, or "" when detached or outside a repository."""
    return git(cwd, "branch", "--show-current")


def repo_root(cwd: str) -> str:
    """Return the checkout that contains `cwd`, or `cwd` itself outside a repository."""
    return git(cwd, "rev-parse", "--show-toplevel") or cwd


def repo_relative(root: str, path: str) -> str:
    """`path` relative to the checkout root, with forward slashes; "" for no path."""
    if not path:
        return ""
    return os.path.relpath(path, root).replace("\\", "/")


def load_script(root: str, name: str) -> ModuleType | None:
    """Load a rule script from the session's checkout, so a worktree checks its own code.

    None when the checkout has no such script: a session outside this
    repository has nothing to enforce.
    """
    path = Path(root) / "scripts" / f"{name}.py"
    if not path.exists():
        return None
    # check_commit_message imports check_prose as a top-level module, the way
    # `python scripts/<name>.py` runs it.
    scripts = str(path.parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    # Registered before it runs: `@dataclass` looks its module up in
    # `sys.modules`, and the second script imports the first by name.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
