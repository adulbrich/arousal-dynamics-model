"""The Claude Code hooks under `.claude/hooks/` enforce AGENTS.md; these tests keep them honest."""

import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

import after_edit
import guard_edits
import guard_gh
import guard_git
import hooklib
import pytest
import session_context

ROOT = Path(__file__).parents[1]
HOOKS = ROOT / ".claude" / "hooks"
EMDASH = "\u2014"
GOOD = "fix(model): use atan2 for the circadian phase"


def _repo(path: Path, branch: str) -> Path:
    """A git repository on `branch` carrying this repository's rule scripts and hooks."""
    subprocess.run(["git", "init", "-q", "-b", branch, str(path)], check=True)
    shutil.copytree(ROOT / "scripts", path / "scripts")
    shutil.copytree(HOOKS, path / ".claude" / "hooks")
    return path


@pytest.fixture
def feature(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A trusted checkout: the hooks treat it as their own project."""
    repo = _repo(tmp_path / "feature", "feat/thing")
    monkeypatch.setattr(hooklib, "PROJECT_ROOT", repo)
    return repo


@pytest.fixture
def on_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo = _repo(tmp_path / "main", "main")
    monkeypatch.setattr(hooklib, "PROJECT_ROOT", repo)
    return repo


class TestGuardGit:
    @pytest.mark.parametrize(
        "command",
        [
            "git status",
            "ls -la",
            "git add src/arousal_dynamics/units.py tests/test_inputs.py",
            "git push -u origin feat/thing",
            "git push --force origin feat/thing",
            "git rebase origin/main",
            "git stash",
            "git branch -d merged-branch",
            "git restore --staged .",
            "git restore src/arousal_dynamics/units.py",
            f"git commit -m '{GOOD}'",
            'echo "git add -A is forbidden"',
            "git -c core.editor=true rebase --continue",
        ],
    )
    def test_allowed(self, feature: Path, command: str) -> None:
        assert guard_git.check(command, str(feature)) is None

    @pytest.mark.parametrize(
        ("command", "rule"),
        [
            ("git add -A", "Stage files by name"),
            ("git add .", "Stage files by name"),
            ("cd src && git add --all", "Stage files by name"),
            ("git -C /tmp/x add ./", "Stage files by name"),
            (f"git commit -a -m '{GOOD}'", "Stage files by name"),
            (f"git commit -am '{GOOD}'", "Stage files by name"),
            ("git reset --hard origin/main", "reset --hard"),
            ("git clean -fd", "clean -f"),
            ("git branch -D old", "branch -D"),
            ("git branch --delete --force old", "branch -D"),
            ("git checkout .", "checkout ."),
            ("git restore .", "restore ."),
            ("git push --force origin main", "force push at main"),
            ("git push origin +main", "force push at main"),
            ("git push origin +HEAD:refs/heads/main", "force push at main"),
            ("git commit -m 'Fix stuff'", "Commit message fails"),
            (f"git commit -m 'fix: thing' -m 'because {EMDASH} reasons'", "emdash"),
        ],
    )
    def test_refused(self, feature: Path, command: str, rule: str) -> None:
        reason = guard_git.check(command, str(feature))
        assert reason is not None
        assert rule in reason

    def test_never_commit_on_main(self, on_main: Path) -> None:
        reason = guard_git.check(f"git commit -m '{GOOD}'", str(on_main))
        assert reason is not None
        assert "Never commit on main" in reason

    def test_branching_first_makes_a_commit_on_main_fine(self, on_main: Path) -> None:
        command = f"git switch -c fix/phase && git commit -m '{GOOD}'"
        assert guard_git.check(command, str(on_main)) is None

    def test_any_force_push_from_main_is_refused(self, on_main: Path) -> None:
        reason = guard_git.check("git push --force-with-lease", str(on_main))
        assert reason is not None
        assert "force push at main" in reason

    def test_a_quoted_heredoc_mentioning_git_is_not_a_command(self, feature: Path) -> None:
        command = (
            "git commit -m \"$(cat <<'EOF'\n"
            "docs: explain why git add -A is refused\n\n"
            "Co-Authored-By: Claude <noreply@anthropic.com>\n"
            'EOF\n)"'
        )
        assert guard_git.check(command, str(feature)) is None

    def test_the_harness_heredoc_message_is_checked(self, feature: Path) -> None:
        command = "git commit -m \"$(cat <<'EOF'\nAdded a thing\nEOF\n)\""
        reason = guard_git.check(command, str(feature))
        assert reason is not None
        assert "subject" in reason

    def test_message_extraction(self) -> None:
        assert guard_git.extract_commit_message("git commit -m 'a' -m \"b\\nc\"") == "a\n\nb\nc"
        assert guard_git.extract_commit_message("git commit -F msg.txt") is None
        assert guard_git.extract_commit_message('git commit -m "$(date)"') is None

    def test_outside_this_repository_only_the_git_rules_apply(self, tmp_path: Path) -> None:
        subprocess.run(["git", "init", "-q", "-b", "x", str(tmp_path)], check=True)
        assert guard_git.check("git commit -m 'Anything goes'", str(tmp_path)) is None
        assert guard_git.check("git add -A", str(tmp_path)) is not None


class TestTrust:
    """Rule scripts and tools are loaded only from this project and its worktrees."""

    def test_a_planted_script_or_tool_never_runs(self, tmp_path: Path) -> None:
        # A checkout that is not this project, with booby-trapped scripts and a
        # fake ruff. PROJECT_ROOT is the real repository here, so it is untrusted.
        repo = tmp_path / "elsewhere"
        subprocess.run(["git", "init", "-q", "-b", "x", str(repo)], check=True)
        marker = tmp_path / "ran"
        trap = f"open({str(marker)!r}, 'w').close()\n"
        (repo / "scripts").mkdir()
        for name in ("check_prose.py", "check_commit_message.py"):
            (repo / "scripts" / name).write_text(trap, encoding="utf-8")
        (repo / ".venv" / "bin").mkdir(parents=True)
        ruff = repo / ".venv" / "bin" / "ruff"
        ruff.write_text(f"#!/bin/sh\ntouch {marker}\n", encoding="utf-8")
        ruff.chmod(0o755)
        (repo / "a.py").write_text("x = 1\n", encoding="utf-8")

        assert hooklib.trusted_root(str(repo)) is None
        guard_git.check(f"git commit -m '{GOOD}'", str(repo))
        guard_gh.check("gh pr create --title 'Bad title' --body y", str(repo))
        assert after_edit.findings(str(repo), "a.py") == []
        assert not marker.exists()

    def test_a_worktree_of_the_project_is_trusted(self, feature: Path, tmp_path: Path) -> None:
        git = ["git", "-C", str(feature), "-c", "user.name=t", "-c", "user.email=t@example.com"]
        subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "init"], check=True)
        tree = tmp_path / "tree"
        subprocess.run([*git, "worktree", "add", "-q", "-b", "other", str(tree)], check=True)
        assert hooklib.trusted_root(str(tree)) == str(tree.resolve())

    def test_the_real_project_is_trusted(self) -> None:
        assert hooklib.trusted_root(str(ROOT / "tests")) == str(ROOT.resolve())


class TestGuardGh:
    @pytest.mark.parametrize(
        "command",
        [
            "gh pr view 3",
            "gh pr checks 1",
            f"gh pr create --title '{GOOD}' --body 'Plain words.'",
            "gh issue comment 4 --body 'ok, merging'",
        ],
    )
    def test_allowed(self, feature: Path, command: str) -> None:
        assert guard_gh.check(command, str(feature)) == []

    def test_a_capitalized_title_is_refused(self, feature: Path) -> None:
        problems = guard_gh.check("gh pr create --title 'Fix the phase' --body x", str(feature))
        assert any("squash-merge subject" in p for p in problems)

    def test_a_session_link_is_refused(self, feature: Path) -> None:
        command = (
            f"gh issue create --title t --body 'see https://claude.ai/code/session_1' # {GOOD}"
        )
        assert any("session" in p for p in guard_gh.check(command, str(feature)))

    def test_an_emdash_in_the_body_is_refused(self, feature: Path) -> None:
        command = f"gh pr edit 3 --body 'one {EMDASH} two'"
        assert any("emdash" in p for p in guard_gh.check(command, str(feature)))

    def test_a_body_file_is_read(self, feature: Path) -> None:
        (feature / "body.md").write_text(f"Summary {EMDASH} here.\n", encoding="utf-8")
        command = f"gh pr create --title '{GOOD}' --body-file body.md"
        assert any("emdash" in p for p in guard_gh.check(command, str(feature)))

    def test_a_missing_body_file_is_not_an_error(self, feature: Path) -> None:
        command = f"gh pr create --title '{GOOD}' --body-file nowhere.md"
        assert guard_gh.check(command, str(feature)) == []

    def test_api_calls_with_a_body_are_checked(self, feature: Path) -> None:
        command = f"gh api repos/o/r/issues/1/comments -f body='a {EMDASH} b'"
        assert guard_gh.check(command, str(feature))

    def test_the_harness_footer_passes(self, feature: Path) -> None:
        robot = "\U0001f916"
        (feature / "body.md").write_text(
            f"Summary.\n\n{robot} Generated with [Claude Code](https://claude.com/claude-code)\n",
            encoding="utf-8",
        )
        command = f"gh pr create --title '{GOOD}' --body-file body.md"
        assert guard_gh.check(command, str(feature)) == []


class TestGuardEdits:
    @pytest.mark.parametrize(
        "path",
        [
            "CLAUDE.md",
            "uv.lock",
            "tests/test_golden/equations.npz",
            "notebooks/archive/validation.ipynb",
            "data/output.csv",
        ],
    )
    def test_protected(self, path: str) -> None:
        assert guard_edits.reason_for(path)

    @pytest.mark.parametrize(
        "path", ["AGENTS.md", "tests/test_golden.py", "notebooks/quickstart.ipynb"]
    )
    def test_editable(self, path: str) -> None:
        assert guard_edits.reason_for(path) is None


class TestAfterEdit:
    def test_an_emdash_is_reported(self, feature: Path) -> None:
        (feature / "notes.md").write_text(f"a {EMDASH} b\n", encoding="utf-8")
        found = after_edit.findings(str(feature), "notes.md")
        assert len(found) == 1
        assert "emdash" in found[0]

    def test_a_clean_file_passes(self, feature: Path) -> None:
        (feature / "notes.md").write_text("a, b\n", encoding="utf-8")
        assert after_edit.findings(str(feature), "notes.md") == []

    @pytest.mark.parametrize("path", ["", "../elsewhere.py", "missing.py"])
    def test_nothing_to_check(self, feature: Path, path: str) -> None:
        assert after_edit.findings(str(feature), path) == []

    def test_ruff_findings_are_reported(self, feature: Path) -> None:
        ruff = Path(sys.executable).with_name("ruff")
        if not ruff.exists():
            pytest.skip("ruff is not installed next to this interpreter")
        (feature / ".venv" / "bin").mkdir(parents=True)
        (feature / ".venv" / "bin" / "ruff").symlink_to(ruff)
        (feature / "bad.py").write_text("import os\nx=1\n", encoding="utf-8")
        found = after_edit.findings(str(feature), "bad.py")
        assert any("ruff check" in block for block in found)
        assert any("ruff format" in block for block in found)


class TestSessionContext:
    def test_warns_on_main(self, on_main: Path) -> None:
        assert session_context.lines(str(on_main))[0].startswith("Branch: main. Do not commit")

    def test_counts_uncommitted_paths(self, feature: Path) -> None:
        lines = session_context.lines(str(feature))
        assert lines[0] == "Branch: feat/thing."
        assert lines[1].startswith("Working tree: 2 uncommitted")  # scripts/ and .claude/

    def test_uv_too_old(self, tmp_path: Path) -> None:
        if shutil.which("uv") is None:
            pytest.skip("uv is not on PATH")
        (tmp_path / "pyproject.toml").write_text('[tool.uv]\nrequired-version = ">=999.0"\n')
        assert "uv self update" in session_context.uv_line(str(tmp_path))

    def test_uv_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PATH", str(tmp_path))
        assert session_context.uv_line(str(tmp_path)).startswith("uv: not found")


def _run_hook(name: str, payload: dict) -> subprocess.CompletedProcess[str]:
    """Run the copy of hook `name` installed in the payload's checkout, as Claude Code would."""
    hooks = Path(payload["cwd"]) / ".claude" / "hooks"
    return subprocess.run(
        [sys.executable, str(hooks / name)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
    )


class TestEndToEnd:
    """Each hook as Claude Code runs it: JSON on stdin, an exit code and stderr back."""

    def test_guard_git_blocks_with_exit_2(self, feature: Path) -> None:
        result = _run_hook(
            "guard_git.py", {"cwd": str(feature), "tool_input": {"command": "git add -A"}}
        )
        assert result.returncode == 2
        assert "Stage files by name" in result.stderr

    def test_guard_git_passes_with_exit_0(self, feature: Path) -> None:
        result = _run_hook(
            "guard_git.py", {"cwd": str(feature), "tool_input": {"command": "git status"}}
        )
        assert (result.returncode, result.stderr) == (0, "")

    def test_guard_gh_blocks(self, feature: Path) -> None:
        command = "gh pr create --title 'Bad title' --body x"
        result = _run_hook("guard_gh.py", {"cwd": str(feature), "tool_input": {"command": command}})
        assert result.returncode == 2

    def test_guard_edits_answers_with_a_json_deny(self, feature: Path) -> None:
        payload = {"cwd": str(feature), "tool_input": {"file_path": str(feature / "uv.lock")}}
        result = _run_hook("guard_edits.py", payload)
        decision = json.loads(result.stdout)["hookSpecificOutput"]
        assert decision["permissionDecision"] == "deny"
        assert "uv lock" in decision["permissionDecisionReason"]

    def test_guard_edits_is_silent_for_an_ordinary_file(self, feature: Path) -> None:
        payload = {"cwd": str(feature), "tool_input": {"file_path": str(feature / "src.py")}}
        assert _run_hook("guard_edits.py", payload).stdout == ""

    def test_after_edit_reports_with_exit_2(self, feature: Path) -> None:
        (feature / "notes.md").write_text(f"a {EMDASH} b\n", encoding="utf-8")
        payload = {"cwd": str(feature), "tool_input": {"file_path": str(feature / "notes.md")}}
        result = _run_hook("after_edit.py", payload)
        assert result.returncode == 2
        assert "emdash" in result.stderr

    def test_session_context_prints_the_branch(self, feature: Path) -> None:
        result = _run_hook("session_context.py", {"cwd": str(feature)})
        assert result.returncode == 0
        assert result.stdout.startswith("Branch: feat/thing.")


# The hooks run on whatever `python3` is first on PATH: on macOS that can be
# the system 3.9, with no third-party packages. The rule scripts they load must
# hold to the same.
HOOK_RUNTIME_FILES = [*sorted(HOOKS.glob("*.py")), *sorted((ROOT / "scripts").glob("check_*.py"))]
LOCAL_MODULES = {"hooklib", "check_prose", "check_commit_message"}


@pytest.mark.parametrize("path", HOOK_RUNTIME_FILES, ids=lambda p: p.name)
def test_runs_on_python_3_9_with_the_standard_library_only(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 9))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    outside = imported - set(sys.stdlib_module_names) - LOCAL_MODULES - {"__future__"}
    assert not outside, f"{path.name} imports {outside}"
