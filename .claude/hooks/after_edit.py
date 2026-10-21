"""PostToolUse on Edit, Write and NotebookEdit: ruff and the prose rule on the file just written.

PostToolUse cannot block, but stderr on exit 2 is shown to the model, which is
the point: the same finding lefthook would make at pre-commit, minutes earlier
and one file at a time. Ruff runs from the project's `.venv` when it exists;
`--force-exclude` keeps an excluded path from being linted. The file is found
in its own checkout (`hooklib.edited_file`), and nothing runs for a file
outside this project and its worktrees (`hooklib.trusted_root`).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import hooklib

RUFF_SUFFIXES = frozenset({".py", ".pyi", ".ipynb", ".md"})


def _run(args: list[str], root: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=root, capture_output=True, text=True, check=False)  # noqa: S603


def findings(root: str, path: str) -> list[str]:
    """Return what the gates would report on `path`, one block per tool; empty when clean.

    `path` is relative to `root`. Its scripts and `.venv` tools run, so an
    untrusted `root` gets nothing.
    """
    if hooklib.trusted_root(root) is None:
        return []
    if not path or Path(path).is_absolute() or path.startswith("../"):
        return []
    if not (Path(root) / path).is_file():
        return []
    out: list[str] = []
    prose = hooklib.load_script(root, "check_prose")
    if prose is not None and prose.is_checked_path(path):
        result = _run([sys.executable, "scripts/check_prose.py", path], root)
        if result.returncode != 0:
            out.append(result.stderr)
    ruff = Path(root) / ".venv" / "bin" / "ruff"
    if Path(path).suffix in RUFF_SUFFIXES and ruff.exists():
        for args, fix in (
            (["check", "--force-exclude"], "uv run ruff check --fix"),
            (["format", "--check", "--force-exclude"], "uv run ruff format"),
        ):
            result = _run([str(ruff), *args, path], root)
            if result.returncode != 0:
                report = f"{result.stdout}{result.stderr}"
                out.append(f"ruff reports on {path} (`{fix}` fixes most):\n{report}")
    return out


def main() -> int:
    """Report what the commit gates would say about the edited file."""
    edited = hooklib.edited_file(hooklib.read_input())
    if edited is None:
        return 0
    found = findings(edited.root, edited.path)
    for block in found:
        sys.stderr.write(block if block.endswith("\n") else f"{block}\n")
    return 2 if found else 0


if __name__ == "__main__":
    sys.exit(main())
