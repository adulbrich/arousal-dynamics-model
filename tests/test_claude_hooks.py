"""The Claude Code hooks under `.claude/hooks/` enforce AGENTS.md; these tests keep them honest."""

import ast
import json
import os
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
EMDASH = chr(0x2014)
ROBOT = chr(0x1F916)
GOOD = "fix(model): use atan2 for the circadian phase"
GIT_ID = ["-c", "user.name=t", "-c", "user.email=t@example.com"]


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *GIT_ID, *args], check=True, capture_output=True)


def _repo(path: Path, branch: str) -> Path:
    """A git repository on `branch` carrying this repository's rule scripts and hooks."""
    subprocess.run(["git", "init", "-q", "-b", branch, str(path)], check=True)
    shutil.copytree(ROOT / "scripts", path / "scripts")
    shutil.copytree(HOOKS, path / ".claude" / "hooks")
    return path.resolve()


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


class TestSegments:
    def test_splits_at_every_separator(self) -> None:
        parts = [s.text.strip() for s in hooklib.segments("a && b; c | d & e || f\ng")]
        assert parts == ["a", "b", "c", "d", "e", "f", "g"]

    def test_quoted_separators_stay_inside(self) -> None:
        assert len(hooklib.segments("git commit -m 'a && b; c'")) == 1

    def test_a_redirect_is_not_a_background_job(self) -> None:
        assert [s.text.strip() for s in hooklib.segments("cmd 2>&1 | tee log")] == [
            "cmd 2>&1",
            "tee log",
        ]

    def test_heredoc_bodies_are_masked_wherever_the_operator_sits(self) -> None:
        command = "cat <<EOF > notes\ngit add -A\nEOF\ncat > b <<'B'\ngit reset --hard\nB"
        for segment in hooklib.segments(command):
            words = hooklib.words(segment)
            assert not (words and words[0].masked == "git"), segment.text

    def test_words_skip_assignments_wrappers_and_comments(self) -> None:
        (segment,) = hooklib.segments("FOO=1 sudo git add x.py # never git add -A")
        assert [w.masked for w in hooklib.words(segment)] == ["git", "add", "x.py"]

    def test_unquote(self) -> None:
        assert hooklib.unquote("'a b'") == "a b"
        assert hooklib.unquote('"say \\"hi\\""') == 'say "hi"'
        assert hooklib.unquote("plain") == "plain"


class TestGuardGit:
    @pytest.mark.parametrize(
        "command",
        [
            "git status",
            "git status 2>&1 | head -5",
            "ls -la",
            "git add src/arousal_dynamics/units.py tests/test_inputs.py",
            "git push -u origin feat/thing",
            "git push --force origin feat/thing",
            "git push --force-if-includes origin feat/thing",
            "git rebase origin/main",
            "git stash",
            "git branch -d merged-branch",
            "git restore --staged .",
            "git restore src/arousal_dynamics/units.py",
            "git clean -n -f",
            "git clean --dry-run --force",
            f"git commit -m '{GOOD}'",
            'git commit -m "$(date)"',
            'echo "git add -A is forbidden"',
            "echo git add -A",
            "git add notes.md  # never git add -A",
            "cat <<EOF > notes.txt\ngit add -A\nEOF",
            "cat > a.sh <<'A'\necho hi\nA\ncat > b.sh <<'B'\ngit reset --hard\nB",
            f"git stash push -m 'wip' && git commit -m '{GOOD}'",
            f"git tag -a v1 -m 'Release 1' && git commit -m '{GOOD}'",
            "git -c core.editor=true rebase --continue",
            "cat <<EOF > notes\nEOF handling is tricky\ngit add -A\nEOF",
            "git commit -F - <<-EOF\n\tfeat: add x\n\tEOF",
            "git commit -F msg.txt && cat <<'EOF' > notes.md\nFix: Bad\nEOF",
            f"git commit -uall -m '{GOOD}'",
            f"git commit -Smykey -m '{GOOD}'",
            "git commit -m $'feat: x\\n\\nBody'",
            "cd ~nouser-xyz && git status",
            "git -C ~nouser-xyz status",
            "git status\x00",
            "cat <<< 'some text'\ngit status",
            "echo $((1<<2))\ngit log -1",
            "(( x <<= 1 ))\ngit status",
            "git add -n .",
            "git add -A --dry-run",
            "git clean -fdn",
            "git clean -n -f --quiet",
            "git push --dry-run --force origin main --verbose",
            "git commit -m 'fix: rename `foo` to bar'",
            "git merge --ff-only origin/main",
            "nice -n 5 git status",
            "xargs -n 1 echo < files.txt",
        ],
    )
    def test_allowed(self, feature: Path, command: str) -> None:
        assert guard_git.refusal(command, str(feature)) is None

    @pytest.mark.parametrize(
        ("command", "rule"),
        [
            ("git add -A", "Stage files by name"),
            ("git add .", "Stage files by name"),
            ("git add -Av", "Stage files by name"),
            ("git add -vA", "Stage files by name"),
            ("git add :/", "Stage files by name"),
            ("cd src && git add --all", "Stage files by name"),
            ("git -C /tmp/x add ./", "Stage files by name"),
            ("git status & git add -A", "Stage files by name"),
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
            ("git commit -m 'Fix stuff'", "subject"),
            (f"git commit -m 'fix: thing' -m 'because {EMDASH} reasons'", "emdash"),
            ("git commit -mWIP", "subject"),
            ("git commit -ma", "subject"),
            ("git commit --message WIP", "subject"),
            ("git commit --message='WIP'", "subject"),
            ('git commit -m"WIP"', "subject"),
            (f"git commit -m '{GOOD}' && git commit -m 'WIP'", "subject"),
            ("git commit --no-verify -m '\\nfix: x'", "subject"),
            ("git commit -F - <<EOF\nWIP\nEOF", "subject"),
            ("git commit -F - <<'MSG-END'\nWIP\nMSG-END", "subject"),
            ("git commit -m='feat: x'", "subject"),
            ("git reset '--hard'", "reset --hard"),
            ('git add "-A"', "Stage files by name"),
            ('git checkout -- "."', "checkout ."),
            ("git checkout -- :/", "checkout ."),
            ("git restore :/", "restore ."),
            ("git push -f origin 'main'", "force push at main"),
            ("git push --force \\\n  origin main", "force push at main"),
            (f"git commit \\\n  -a -m '{GOOD}'", "Stage files by name"),
            ("echo hi # don't\ngit add -A\necho 'x'", "Stage files by name"),
            ("echo Don\\'t && git add -A && echo 'x'", "Stage files by name"),
            ("echo 'x <<EOF'\ngit reset --hard\nEOF", "reset --hard"),
            ("(cd src && git add -A)", "Stage files by name"),
            ("{ git add -A; }", "Stage files by name"),
            ("if true; then git add -A; fi", "Stage files by name"),
            ("! git add -A", "Stage files by name"),
            ("env -i git add -A", "Stage files by name"),
            ("sudo -E git add -A", "Stage files by name"),
            ("sudo -n git add -A", "Stage files by name"),
            ("sudo -u alex git add -A", "Stage files by name"),
            ("sudo --user alex git add -A", "Stage files by name"),
            ("sudo --user git git add -A", "Stage files by name"),
            ("sudo --unknown-option value git add -A", "Stage files by name"),
            ("timeout --foreground 60 git push -f origin main", "force push at main"),
            ("sudo -E timeout 60 git add -A", "Stage files by name"),
            ("sudo --weird env FOO=1 git add -A", "Stage files by name"),
            ("echo $(( $(git reset --hard) ))", "reset --hard"),
            ('echo "$((1 << 3))"\ngit add -A', "Stage files by name"),
            ("cat <<< 'x'\ngit reset --hard", "reset --hard"),
            ("echo $((1<<2))\ngit clean -fd", "clean -f"),
            ("(( x <<= 1 ))\ngit checkout .", "checkout ."),
            ("((cd src) && git add -A)", "Stage files by name"),
            ("echo ((x\ngit reset --hard", "reset --hard"),
            ("git clean -e -n -f", "clean -f"),
            ("git clean -f -- -n", "clean -f"),
            ("git clean -fe -n", "clean -f"),
            ("git clean -n --no-dry-run -f", "clean -f"),
            ("git clean --excl -n -f", "clean -f"),
            ("git push --force --repo -n origin main", "force push at main"),
            ("git add --pathspec-from -n -A", "Stage files by name"),
            ('echo $((1<<"0"))\ngit reset --hard', "reset --hard"),
            ("echo $((1<<2\n))\ngit reset --hard", "reset --hard"),
            ("git branch --merged | grep -v main | xargs git branch -D", "branch -D"),
            ("timeout 60 git push -f origin main", "force push at main"),
            ("git commit -m 'fix: x' --trailer 'Session: claude.ai/code/session_1'", "session"),
        ],
    )
    def test_refused(self, feature: Path, command: str, rule: str) -> None:
        reason = guard_git.refusal(command, str(feature))
        assert reason is not None
        assert rule in reason

    def test_a_message_flag_after_a_cluster_is_a_message_not_all(self, feature: Path) -> None:
        reason = guard_git.refusal("git commit -ma", str(feature))
        assert reason is not None
        assert "Stage" not in reason

    def test_the_harness_heredoc_message_is_checked(self, feature: Path) -> None:
        good = (
            "git commit -m \"$(cat <<'EOF'\n"
            "docs: explain why git add -A is refused\n\n"
            "Co-Authored-By: Claude <noreply@anthropic.com>\n"
            'EOF\n)"'
        )
        bad = "git commit -m \"$(cat <<'EOF'\nAdded a thing\nEOF\n)\""
        assert guard_git.refusal(good, str(feature)) is None
        assert "subject" in (guard_git.refusal(bad, str(feature)) or "")

    def test_never_commit_on_main(self, on_main: Path) -> None:
        reason = guard_git.refusal(f"git commit -m '{GOOD}'", str(on_main))
        assert reason == guard_git.MAIN_COMMIT

    @pytest.mark.parametrize(
        "first",
        ["git switch -c fix/phase", "git switch --create fix/phase", "git checkout -b fix/phase"],
    )
    def test_branching_first_makes_a_commit_on_main_fine(self, on_main: Path, first: str) -> None:
        assert guard_git.refusal(f"{first} && git commit -m '{GOOD}'", str(on_main)) is None

    def test_a_new_worktree_does_not_move_this_checkout(self, on_main: Path) -> None:
        command = f"git worktree add ../w && git commit -m '{GOOD}'"
        assert guard_git.refusal(command, str(on_main)) == guard_git.MAIN_COMMIT

    def test_a_branch_made_elsewhere_does_not_count(self, feature: Path, on_main: Path) -> None:
        command = f"git checkout -b z && cd {on_main} && git commit -m '{GOOD}'"
        assert guard_git.refusal(command, str(feature)) == guard_git.MAIN_COMMIT

    @pytest.mark.parametrize(
        "command",
        [
            "git merge --no-ff feat/x",
            "git merge origin/main",
            "git merge --no-ff origin/main",
            "git cherry-pick abc123",
            "git revert --no-edit HEAD",
            "git am p",
        ],
    )
    def test_other_commands_that_record_a_commit(self, on_main: Path, command: str) -> None:
        assert guard_git.refusal(command, str(on_main)) == guard_git.MAIN_COMMIT

    @pytest.mark.parametrize(
        "command",
        ["git merge --abort", "git cherry-pick --no-commit abc", "git revert -n HEAD"],
    )
    def test_their_no_commit_forms_are_fine(self, on_main: Path, command: str) -> None:
        assert guard_git.refusal(command, str(on_main)) is None

    @pytest.mark.parametrize(
        "command", ["git merge --squash feat/x", "git merge --ff-only origin/main"]
    )
    def test_merges_that_record_nothing_on_main(self, on_main: Path, command: str) -> None:
        assert guard_git.refusal(command, str(on_main)) is None

    def test_a_new_branch_covers_the_whole_checkout(self, on_main: Path) -> None:
        (on_main / "docs").mkdir()
        command = f"git switch --create=fix/x && cd docs && git commit -m '{GOOD}' README.md"
        assert guard_git.refusal(command, str(on_main)) is None

    def test_a_subshell_cd_does_not_leak(self, feature: Path, on_main: Path) -> None:
        command = f"(cd {on_main} && git fetch); git commit -m '{GOOD}'"
        assert guard_git.refusal(command, str(feature)) is None
        command = f"(cd {feature} && true); git commit -m '{GOOD}'"
        assert guard_git.refusal(command, str(on_main)) == guard_git.MAIN_COMMIT

    def test_a_dry_run_commit_on_main_is_fine(self, on_main: Path) -> None:
        assert guard_git.refusal(f"git commit --dry-run -m '{GOOD}'", str(on_main)) is None

    def test_the_branch_is_read_where_git_runs(self, feature: Path, on_main: Path) -> None:
        for command in (
            f"git -C {on_main} commit -m '{GOOD}'",
            f"cd {on_main} && git commit -m '{GOOD}'",
        ):
            assert guard_git.refusal(command, str(feature)) == guard_git.MAIN_COMMIT
        assert guard_git.refusal(f"git -C {feature} commit -m '{GOOD}'", str(on_main)) is None

    @pytest.mark.parametrize(
        ("command", "refused"),
        [
            ("git push --force-with-lease", True),
            ("git push --force origin HEAD", True),
            ("git push -f origin @", True),
            ("git push -f -o ci.skip origin", True),
            ("git push -f origin feat/thing", False),
            ("git push -f origin HEAD:maintenance", False),
            ("git push --dry-run --force origin main", False),
        ],
    )
    def test_force_push_from_main(self, on_main: Path, command: str, refused: bool) -> None:
        assert (guard_git.refusal(command, str(on_main)) is not None) is refused

    def test_outside_this_project_only_the_git_rules_apply(self, tmp_path: Path) -> None:
        subprocess.run(["git", "init", "-q", "-b", "x", str(tmp_path)], check=True)
        assert guard_git.refusal("git commit -m 'Anything goes'", str(tmp_path)) is None
        assert guard_git.refusal("git add -A", str(tmp_path)) is not None


class TestGuardGh:
    @pytest.mark.parametrize(
        "command",
        [
            "gh pr view 3",
            "gh pr checks 1",
            "gh api repos/o/r/pulls --jq '.[].title'",
            f"gh pr create --title '{GOOD}' --body 'Plain words.'",
            "gh issue comment 4 --body 'ok, merging'",
            "echo \"gh pr create --title 'Bad title'\"",
            "cat > t.py <<'EOF'\ngh pr create --title 'Bad title' --body x\nEOF",
            f"gh pr create --title '{GOOD}' --body x && echo 'a {EMDASH} b'",
            f"gh pr create --title '{GOOD}' --body x --label 'bug {ROBOT}'",
            f"gh pr create --title '{GOOD}' --body x  # {EMDASH} note",
            "gh api -X GET search/issues -f q='claude.ai/code/session repo:o/r'",
            f"gh pr create --title '{GOOD}' --body '-thing to do'",
            "gh api -X PATCH repos/o/r/issues/comments/1 -f body=ok "
            "--jq '.body | contains(\"claude.ai/code/session\")'",
            "gh pr comment 5 --body ok # drop the claude.ai/code/session link first",
        ],
    )
    def test_allowed(self, feature: Path, command: str) -> None:
        assert guard_gh.problems(command, str(feature)) == []

    @pytest.mark.parametrize(
        ("command", "rule"),
        [
            ("gh pr create --title 'Fix the phase' --body x", "squash-merge subject"),
            ("gh pr merge 3 --squash --subject 'Bad subject'", "squash-merge subject"),
            ("gh issue create --title t --body 'see https://claude.ai/code/session_1'", "session"),
            (f"gh pr edit 3 --body 'one {EMDASH} two'", "emdash"),
            (f"gh api repos/o/r/issues/1/comments -f body='a {EMDASH} b'", "emdash"),
            ("gh api repos/o/r/issues -fbody='see claude.ai/code/session_1'", "session"),
            ("gh -R o/r pr create --title 'Bad title' --body x", "squash-merge subject"),
            ("gh pr --repo o/r create --title 'Bad title' --body x", "squash-merge subject"),
            ("gh pr create -t'Bad title' --body x", "squash-merge subject"),
            (f"gh api -X GET -X POST repos/o/r/issues -f body='a {EMDASH} b'", "emdash"),
            (f"gh pr create --title '{GOOD}' -l --body 'claude.ai/code/session_1'", "session"),
            (f"gh pr review 12 --comment --body 'a {EMDASH} b'", "emdash"),
            (f"gh pr review 12 -c -b 'a {EMDASH} b'", "emdash"),
            (f"gh pr review 12 -r -b 'a {EMDASH} b'", "emdash"),
            (f"gh release create v1 -p --notes 'a {EMDASH} b'", "emdash"),
            ("gh pr merge 12 -m --subject 'Bad subject' --body x", "squash-merge subject"),
            (f"gh pr create -f --body 'a {EMDASH} b'", "emdash"),
            (
                "BODY='see claude.ai/code/session_1' gh issue create --title t --body \"$BODY\"",
                "session",
            ),
            (
                f"gh pr create --title '{GOOD}' --body-file - <<'EOF'\n"
                "see claude.ai/code/session_1\nEOF",
                "session",
            ),
        ],
    )
    def test_refused(self, feature: Path, command: str, rule: str) -> None:
        assert any(rule in p for p in guard_gh.problems(command, str(feature)))

    def test_a_broken_rule_script_still_leaves_the_session_link_rule(self, feature: Path) -> None:
        (feature / "scripts" / "check_prose.py").write_text("this is not python(\n")
        link = "gh issue create --title t --body 'https://claude.ai/code/session_1'"
        assert any("session" in p for p in guard_gh.problems(link, str(feature)))

    @pytest.mark.parametrize(
        ("command", "where"),
        [
            (f"gh pr create --title '{GOOD}' --body-file body.md", "."),
            (f"cd docs && gh pr create --title '{GOOD}' --body-file body.md", "docs"),
            ("gh release create v1 --notes-file body.md", "."),
            ("gh api repos/o/r/issues -F body=@body.md", "."),
            ("gh api repos/o/r/issues --input body.md", "."),
        ],
    )
    def test_text_files_are_read(self, feature: Path, command: str, where: str) -> None:
        (feature / where).mkdir(exist_ok=True)
        (feature / where / "body.md").write_text(f"Summary {EMDASH} here.\n", encoding="utf-8")
        assert any("emdash" in p for p in guard_gh.problems(command, str(feature)))

    def test_an_absolute_body_file_is_read(self, feature: Path, tmp_path: Path) -> None:
        body = tmp_path / "abs.md"
        body.write_text(f"a {EMDASH} b\n", encoding="utf-8")
        command = f"gh pr create --title '{GOOD}' --body-file {body}"
        assert guard_gh.problems(command, str(feature))

    @pytest.mark.parametrize("content", [None, b"\xff\xfe not utf-8"])
    def test_an_unreadable_body_file_is_not_an_error(
        self, feature: Path, content: bytes | None
    ) -> None:
        if content is not None:
            (feature / "body.md").write_bytes(content)
        command = f"gh pr create --title '{GOOD}' --body-file body.md"
        assert guard_gh.problems(command, str(feature)) == []

    def test_the_harness_footer_passes(self, feature: Path) -> None:
        (feature / "body.md").write_text(
            f"Summary.\n\n{ROBOT} Generated with [Claude Code](https://claude.com/claude-code)\n",
            encoding="utf-8",
        )
        command = f"gh pr create --title '{GOOD}' --body-file body.md"
        assert guard_gh.problems(command, str(feature)) == []

    def test_an_untrusted_checkout_still_gets_the_session_link_rule(self, tmp_path: Path) -> None:
        subprocess.run(["git", "init", "-q", "-b", "x", str(tmp_path)], check=True)
        link = "gh issue create --title t --body 'https://claude.ai/code/session_1'"
        assert guard_gh.problems(link, str(tmp_path))
        emdash = f"gh issue create --title t --body 'a {EMDASH} b'"
        assert guard_gh.problems(emdash, str(tmp_path)) == []


class TestGuardEdits:
    @pytest.mark.parametrize(
        "path",
        [
            "CLAUDE.md",
            "uv.lock",
            "UV.lock",
            "tests/test_golden/equations.npz",
            "notebooks/archive/validation.ipynb",
            "data/output.csv",
            "Data/output.csv",
        ],
    )
    def test_protected(self, path: str) -> None:
        assert guard_edits.reason_for(path)

    @pytest.mark.parametrize(
        "path", ["AGENTS.md", "tests/test_golden.py", "notebooks/quickstart.ipynb"]
    )
    def test_editable(self, path: str) -> None:
        assert guard_edits.reason_for(path) is None


class TestEditedFile:
    def test_a_relative_path_is_resolved_against_the_session(self, feature: Path) -> None:
        payload = {"cwd": str(feature / "scripts"), "tool_input": {"file_path": "../uv.lock"}}
        edited = hooklib.edited_file(payload)
        assert edited is not None
        assert (edited.root, edited.path, edited.written) == (str(feature), "uv.lock", "uv.lock")

    def test_a_symlinked_spelling_names_the_same_file(self, feature: Path, tmp_path: Path) -> None:
        link = tmp_path / "link"
        link.symlink_to(feature)
        payload = {"cwd": str(link), "tool_input": {"file_path": str(link / "data" / "x.csv")}}
        edited = hooklib.edited_file(payload)
        assert edited is not None
        assert (edited.root, edited.path) == (str(feature), "data/x.csv")

    def test_a_notebook_path_counts(self, feature: Path) -> None:
        target = feature / "notebooks" / "archive" / "a.ipynb"
        payload = {"cwd": str(feature), "tool_input": {"notebook_path": str(target)}}
        edited = hooklib.edited_file(payload)
        assert edited is not None
        assert edited.path == "notebooks/archive/a.ipynb"

    def test_a_worktree_file_belongs_to_the_worktree(self, feature: Path, tmp_path: Path) -> None:
        _git(feature, "commit", "-q", "--allow-empty", "-m", "init")
        tree = tmp_path / "tree"
        _git(feature, "worktree", "add", "-q", "-b", "other", str(tree))
        payload = {"cwd": str(feature), "tool_input": {"file_path": str(tree / "uv.lock")}}
        edited = hooklib.edited_file(payload)
        assert edited is not None
        assert (edited.root, edited.path) == (str(tree.resolve()), "uv.lock")
        assert hooklib.trusted_root(edited.root) == edited.root

    def test_claude_md_keeps_its_own_name(self, feature: Path) -> None:
        (feature / "AGENTS.md").write_text("rules\n")
        (feature / "CLAUDE.md").symlink_to("AGENTS.md")
        payload = {"cwd": str(feature), "tool_input": {"file_path": str(feature / "CLAUDE.md")}}
        edited = hooklib.edited_file(payload)
        assert edited is not None
        assert edited.path == "CLAUDE.md"

    def test_a_symlinked_directory_keeps_its_written_name(
        self, feature: Path, tmp_path: Path
    ) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        (feature / "data").symlink_to(outside)
        payload = {
            "cwd": str(feature),
            "tool_input": {"file_path": str(feature / "data" / "x.csv")},
        }
        edited = hooklib.edited_file(payload)
        assert edited is not None
        assert edited.written == "data/x.csv"

    def test_a_link_to_a_protected_file_is_named_by_its_target(self, feature: Path) -> None:
        (feature / "uv.lock").write_text("lock\n")
        (feature / "notes.md").symlink_to("uv.lock")
        payload = {"cwd": str(feature), "tool_input": {"file_path": str(feature / "notes.md")}}
        edited = hooklib.edited_file(payload)
        assert edited is not None
        assert "uv.lock" in edited.spellings

    @pytest.mark.parametrize("kind", ["unreadable", "overlong"])
    def test_an_unreadable_or_overlong_path_does_not_crash(self, feature: Path, kind: str) -> None:
        raw = (
            "/private/var/root/x/uv.lock"
            if kind == "unreadable"
            else str(feature / ("a" * 300) / "x")
        )
        payload = {"cwd": str(feature), "tool_input": {"file_path": raw}}
        assert hooklib.edited_file(payload) is not None

    def test_an_unexpandable_home_does_not_crash(self, feature: Path) -> None:
        payload = {"cwd": str(feature), "tool_input": {"file_path": "~nouser-xyz/y"}}
        assert hooklib.edited_file(payload) is not None

    def test_no_path(self) -> None:
        assert hooklib.edited_file({"tool_input": {}}) is None


class TestAfterEdit:
    def test_an_emdash_is_reported(self, feature: Path) -> None:
        (feature / "notes.md").write_text(f"a {EMDASH} b\n", encoding="utf-8")
        found = after_edit.findings(str(feature), "notes.md")
        assert len(found) == 1
        assert "emdash" in found[0]

    def test_a_clean_file_passes(self, feature: Path) -> None:
        (feature / "notes.md").write_text("a, b\n", encoding="utf-8")
        assert after_edit.findings(str(feature), "notes.md") == []

    @pytest.mark.parametrize("path", ["", "../elsewhere.py", "missing.py", "/etc/hosts"])
    def test_nothing_to_check(self, feature: Path, path: str) -> None:
        assert after_edit.findings(str(feature), path) == []

    def test_a_path_outside_the_prose_rule_is_not_prose_checked(self, feature: Path) -> None:
        (feature / "data.csv").write_text(f"a {EMDASH} b\n", encoding="utf-8")
        assert after_edit.findings(str(feature), "data.csv") == []

    @pytest.fixture
    def with_ruff(self, feature: Path) -> Path:
        ruff = Path(sys.executable).with_name("ruff")
        if not ruff.exists():
            pytest.skip("ruff is not installed next to this interpreter")
        (feature / ".venv" / "bin").mkdir(parents=True)
        (feature / ".venv" / "bin" / "ruff").symlink_to(ruff)
        return feature

    def test_ruff_findings_are_reported(self, with_ruff: Path) -> None:
        (with_ruff / "bad.py").write_text("import os\nx=1\n", encoding="utf-8")
        found = after_edit.findings(str(with_ruff), "bad.py")
        assert any("ruff check" in block for block in found)
        assert any("ruff format" in block for block in found)

    def test_an_excluded_path_is_not_linted(self, with_ruff: Path) -> None:
        (with_ruff / "pyproject.toml").write_text(
            '[tool.ruff]\nforce-exclude = true\nextend-exclude = ["legacy/"]\n', encoding="utf-8"
        )
        (with_ruff / "legacy").mkdir()
        (with_ruff / "legacy" / "bad.py").write_text("import os\nx=1\n", encoding="utf-8")
        assert after_edit.findings(str(with_ruff), "legacy/bad.py") == []


class TestSessionContext:
    def test_warns_on_main(self, on_main: Path) -> None:
        assert session_context.lines(str(on_main))[0].startswith("Branch: main. Do not commit")

    def test_counts_uncommitted_paths(self, feature: Path) -> None:
        lines = session_context.lines(str(feature))
        assert lines[0] == "Branch: feat/thing."
        assert lines[1].startswith("Working tree: 2 uncommitted")  # scripts/ and .claude/

    def test_a_clean_detached_checkout(self, feature: Path) -> None:
        _git(feature, "add", "scripts", ".claude")
        _git(feature, "commit", "-q", "-m", "init")
        _git(feature, "checkout", "-q", "--detach")
        lines = session_context.lines(str(feature))
        assert lines[:2] == ["Branch: (detached).", "Working tree: clean."]

    def test_reports_other_worktrees_and_gone_branches(self, feature: Path, tmp_path: Path) -> None:
        _git(feature, "commit", "-q", "--allow-empty", "-m", "init")
        _git(feature, "worktree", "add", "-q", "-b", "other", str(tmp_path / "tree"))
        _git(feature, "remote", "add", "origin", str(tmp_path / "nowhere"))
        _git(feature, "config", "branch.feat/thing.remote", "origin")
        _git(feature, "config", "branch.feat/thing.merge", "refs/heads/feat/thing")
        text = "\n".join(session_context.workspace_lines(str(feature)))
        assert "Other worktrees:" in text
        assert str((tmp_path / "tree").resolve()) in text
        assert "Branches whose remote is gone: feat/thing." in text

    def test_nothing_to_report_in_a_fresh_checkout(self, feature: Path) -> None:
        assert session_context.workspace_lines(str(feature)) == []

    def test_uv_too_old(self, tmp_path: Path) -> None:
        if shutil.which("uv") is None:
            pytest.skip("uv is not on PATH")
        (tmp_path / "pyproject.toml").write_text('[tool.uv]\nrequired-version = ">=999.0"\n')
        assert "uv self update" in session_context.uv_line(str(tmp_path))

    def test_uv_without_a_pyproject(self, tmp_path: Path) -> None:
        if shutil.which("uv") is None:
            pytest.skip("uv is not on PATH")
        line = session_context.uv_line(str(tmp_path))
        assert line.startswith("uv: ")
        assert "requires" not in line

    def test_uv_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PATH", str(tmp_path))
        assert session_context.uv_line(str(tmp_path)).startswith("uv: not found")


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
        guard_git.refusal(f"git commit -m '{GOOD}'", str(repo))
        guard_gh.problems("gh pr create --title 'Bad title' --body y", str(repo))
        assert after_edit.findings(str(repo), "a.py") == []
        assert not marker.exists()

    def test_a_git_file_pointing_at_the_project_is_not_trusted(
        self, feature: Path, tmp_path: Path
    ) -> None:
        impostor = tmp_path / "impostor"
        impostor.mkdir()
        (impostor / ".git").write_text(f"gitdir: {feature / '.git'}\n")
        assert hooklib.trusted_root(str(impostor)) is None

    def test_the_real_project_is_trusted(self) -> None:
        assert hooklib.trusted_root(str(ROOT / "tests")) == str(ROOT.resolve())

    def test_clean_env_drops_git_variables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GIT_DIR", "/elsewhere/.git")
        assert "GIT_DIR" not in hooklib.clean_env()


class TestSettings:
    """The wiring in `.claude/settings.json`; whether Claude Code honours it is shown live."""

    settings = json.loads((ROOT / ".claude" / "settings.json").read_text(encoding="utf-8"))

    def _entries(self, event: str) -> dict[str, list[str]]:
        return {
            group["matcher"]: [Path(h["args"][0]).name for h in group["hooks"]]
            for group in self.settings["hooks"][event]
        }

    def test_every_hook_runs_an_existing_script_on_python3(self) -> None:
        for groups in self.settings["hooks"].values():
            for group in groups:
                for hook in group["hooks"]:
                    assert hook["command"] == "python3"
                    (arg,) = hook["args"]
                    assert arg.startswith("${CLAUDE_PROJECT_DIR}/.claude/hooks/")
                    assert (HOOKS / Path(arg).name).is_file()

    def test_matchers(self) -> None:
        assert self._entries("PreToolUse") == {
            "Bash": ["guard_git.py", "guard_gh.py"],
            "Edit|Write|NotebookEdit": ["guard_edits.py"],
        }
        assert self._entries("PostToolUse") == {"Edit|Write|NotebookEdit": ["after_edit.py"]}
        (matcher,) = self._entries("SessionStart")
        assert {"startup", "resume", "compact"} <= set(matcher.split("|"))

    def test_the_review_plugin_is_enabled(self) -> None:
        assert self.settings["enabledPlugins"]["mattpocock-skills@claude-plugins-official"]


def _interpreters() -> dict[str, str]:
    """The current Python, and a Python 3.9 to run the hooks the way macOS's system one would."""
    found = {"current": sys.executable}
    wanted = os.environ.get("HOOK_PYTHON")
    candidate = wanted or "/usr/bin/python3"
    if wanted or Path(candidate).exists():
        probe = subprocess.run(
            [candidate, "-c", "import sys; print(sys.version_info[:2])"],
            capture_output=True,
            text=True,
            check=False,
        )
        if wanted or probe.stdout.strip() == "(3, 9)":
            found["python3.9"] = candidate
    return found


INTERPRETERS = _interpreters()


@pytest.fixture(params=list(INTERPRETERS.values()), ids=list(INTERPRETERS))
def python(request: pytest.FixtureRequest) -> str:
    return request.param


def _run_hook(
    python: str, repo: Path, name: str, payload: object
) -> subprocess.CompletedProcess[str]:
    """Run the copy of hook `name` installed in `repo`, as Claude Code would."""
    return subprocess.run(
        [python, str(repo / ".claude" / "hooks" / name)],
        input=payload if isinstance(payload, str) else json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
    )


class TestEndToEnd:
    """Each hook as Claude Code runs it: JSON on stdin, an exit code and stderr back."""

    def test_guard_git_blocks_with_exit_2(self, python: str, feature: Path) -> None:
        payload = {"cwd": str(feature), "tool_input": {"command": "git add -A"}}
        result = _run_hook(python, feature, "guard_git.py", payload)
        assert result.returncode == 2
        assert "Stage files by name" in result.stderr

    def test_guard_git_passes_with_exit_0(self, python: str, feature: Path) -> None:
        payload = {"cwd": str(feature), "tool_input": {"command": f"git commit -m '{GOOD}'"}}
        result = _run_hook(python, feature, "guard_git.py", payload)
        assert (result.returncode, result.stderr) == (0, "")

    def test_guard_gh_blocks(self, python: str, feature: Path) -> None:
        command = "gh pr create --title 'Bad title' --body x"
        payload = {"cwd": str(feature), "tool_input": {"command": command}}
        assert _run_hook(python, feature, "guard_gh.py", payload).returncode == 2

    def test_guard_edits_answers_with_a_json_deny(self, python: str, feature: Path) -> None:
        payload = {"cwd": str(feature), "tool_input": {"file_path": "uv.lock"}}
        result = _run_hook(python, feature, "guard_edits.py", payload)
        decision = json.loads(result.stdout)["hookSpecificOutput"]
        assert decision["permissionDecision"] == "deny"
        assert "uv lock" in decision["permissionDecisionReason"]

    @pytest.mark.parametrize("name", ["CLAUDE.md", "data"])
    def test_guard_edits_sees_through_symlinks(
        self, python: str, feature: Path, tmp_path: Path, name: str
    ) -> None:
        if name == "CLAUDE.md":
            (feature / "AGENTS.md").write_text("rules\n")
            (feature / "CLAUDE.md").symlink_to("AGENTS.md")
            target = feature / "CLAUDE.md"
        else:
            (tmp_path / "outside").mkdir()
            (feature / "data").symlink_to(tmp_path / "outside")
            target = feature / "data" / "x.csv"
        payload = {"cwd": str(feature), "tool_input": {"file_path": str(target)}}
        result = _run_hook(python, feature, "guard_edits.py", payload)
        assert json.loads(result.stdout)["hookSpecificOutput"]["permissionDecision"] == "deny"

    def test_guard_edits_is_silent_for_an_ordinary_file(self, python: str, feature: Path) -> None:
        payload = {"cwd": str(feature), "tool_input": {"file_path": str(feature / "src.py")}}
        assert _run_hook(python, feature, "guard_edits.py", payload).stdout == ""

    def test_after_edit_reports_with_exit_2(self, python: str, feature: Path) -> None:
        (feature / "notes.md").write_text(f"a {EMDASH} b\n", encoding="utf-8")
        payload = {"cwd": str(feature), "tool_input": {"file_path": str(feature / "notes.md")}}
        result = _run_hook(python, feature, "after_edit.py", payload)
        assert result.returncode == 2
        assert "emdash" in result.stderr

    def test_session_context_prints_the_branch(self, python: str, feature: Path) -> None:
        result = _run_hook(python, feature, "session_context.py", {"cwd": str(feature)})
        assert result.returncode == 0
        assert result.stdout.startswith("Branch: feat/thing.")

    @pytest.mark.parametrize(
        "payload", ["", "not json", "[]", "null", '{"tool_input": {"command": 3}}']
    )
    @pytest.mark.parametrize(
        "name", ["guard_git.py", "guard_gh.py", "guard_edits.py", "after_edit.py"]
    )
    def test_a_malformed_payload_lets_the_call_through(
        self, python: str, feature: Path, name: str, payload: str
    ) -> None:
        result = _run_hook(python, feature, name, payload)
        assert (result.returncode, result.stderr) == (0, "")


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
