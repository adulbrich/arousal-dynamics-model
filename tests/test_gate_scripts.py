"""The gate scripts under `scripts/` enforce AGENTS.md; these tests keep them honest."""

from pathlib import Path

import check_commit_message as ccm
import check_prose as cp
import pytest
from hypothesis import given
from hypothesis import strategies as st

EMDASH = "\u2014"
ROBOT = "\U0001f916"

# Text that can never trip the rule: printable ASCII plus the Greek letters,
# arrows and operators that docstrings for this model use.
SAFE_TEXT = st.text(
    alphabet=st.characters(min_codepoint=0x20, max_codepoint=0x7E)
    | st.sampled_from("ψφθραβγΔ→≤≥±·"),
    max_size=200,
)


class TestProse:
    def test_emdash_is_reported_with_its_line(self) -> None:
        hits = cp.find_prose_violations(f"fine\nnot {EMDASH} fine\n")
        assert hits == [cp.Violation(2, "emdash", f"not {EMDASH} fine")]

    def test_emoji_is_reported(self) -> None:
        assert [v.kind for v in cp.find_prose_violations("done \u2705")] == ["emoji"]

    def test_both_rules_on_one_line_give_two_hits(self) -> None:
        kinds = {v.kind for v in cp.find_prose_violations(f"a {EMDASH} b \u2b50")}
        assert kinds == {"emdash", "emoji"}

    @pytest.mark.parametrize("text", ["ψ = atan2(Y, X)", "H → S", "x ≤ 1", "en dash – is fine"])  # noqa: RUF001
    def test_scientific_notation_and_arrows_pass(self, text: str) -> None:
        assert cp.find_prose_violations(text) == []

    def test_harness_footer_is_exempt(self) -> None:
        body = f"Summary.\n\n{ROBOT} Generated with [Claude Code](https://claude.com/claude-code)\n"
        assert cp.find_prose_violations(body) == []

    def test_a_robot_elsewhere_is_not_exempt(self) -> None:
        assert cp.find_prose_violations(f"{ROBOT} ship it")

    @given(SAFE_TEXT)
    def test_safe_text_never_trips(self, text: str) -> None:
        assert cp.find_prose_violations(text) == []

    @given(SAFE_TEXT, SAFE_TEXT)
    def test_an_inserted_emdash_is_always_found(self, before: str, after: str) -> None:
        text = f"{before}{EMDASH}{after}"
        line = text[: text.index(EMDASH)].count("\n") + 1
        assert any(v.kind == "emdash" and v.line == line for v in cp.find_prose_violations(text))

    @pytest.mark.parametrize(
        ("path", "checked"),
        [
            ("README.md", True),
            ("scripts/check_prose.py", True),
            ("pyproject.toml", True),
            (".github/workflows/ci.yml", True),
            ("tests/test_golden/equations.npz", False),
            ("uv.lock", False),
            ("notebooks/archive/scenarios_v1.ipynb", False),
            ("notebooks/quickstart.ipynb", True),
            ("data/output.csv", False),
            ("LICENSE", False),
        ],
    )
    def test_checked_paths(self, path: str, checked: bool) -> None:
        assert cp.is_checked_path(path) is checked

    def test_cli_fails_on_a_bad_file_and_passes_a_good_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bad, good = tmp_path / "bad.md", tmp_path / "good.md"
        bad.write_text(f"one {EMDASH} two\n", encoding="utf-8")
        good.write_text("one, two\n", encoding="utf-8")
        assert cp.main([str(good)]) == 0
        assert cp.main([str(bad), str(good)]) == 1
        assert "bad.md:1: emdash" in capsys.readouterr().err

    def test_cli_text_mode(self) -> None:
        assert cp.main(["--text", "plain words"]) == 0
        assert cp.main(["--text", "a", EMDASH, "b"]) == 1


class TestCommitMessage:
    @pytest.mark.parametrize(
        "subject",
        [
            "feat(model): add a parameter set for Tekieh 2020",
            "fix: use atan2 for the circadian phase",
            "refactor(model)!: rename model_run to simulate",
            "build(deps): Bump numpy from 2.4.0 to 2.5.4",
            "ci(deps): Bump actions/checkout from 6 to 7",
            "Merge pull request #3 from adulbrich/build/uv-tooling-and-gates",
            'Revert "feat: something"',
            "docs: `simulate` returns named states",
            "test: 3 new golden scenarios",
        ],
    )
    def test_accepted_subjects(self, subject: str) -> None:
        assert ccm.check_commit_message(subject) == []

    @pytest.mark.parametrize(
        "subject",
        [
            "Fix the phase",
            "fix: Use atan2",
            "feature: add a thing",
            "fix(Model): scope must be lowercase",
            "fix:missing space",
            "fix: ",
            "added additional plots",
        ],
    )
    def test_rejected_subjects(self, subject: str) -> None:
        assert ccm.check_commit_message(subject)

    def test_session_link_is_rejected_anywhere(self) -> None:
        message = "fix: thing\n\nhttps://claude.ai/code/session_123\n"
        assert any("session" in p for p in ccm.check_commit_message(message))

    def test_emdash_in_body_is_rejected(self) -> None:
        problems = ccm.check_commit_message(f"fix: thing\n\nbecause {EMDASH} reasons\n")
        assert problems == [f"line 3 has an emdash: because {EMDASH} reasons"]

    def test_co_authored_by_trailer_passes(self) -> None:
        message = "fix: thing\n\nWhy.\n\nCo-Authored-By: Claude <noreply@anthropic.com>\n"
        assert ccm.check_commit_message(message) == []

    def test_empty_message(self) -> None:
        assert ccm.check_commit_message("\n\n") == ["the message is empty"]

    def test_strip_commentary_drops_comments_and_the_verbose_diff(self) -> None:
        raw = (
            "fix: thing\n"
            "# Please enter the commit message\n"
            "body\n"
            "# ------------------------ >8 ------------------------\n"
            f"diff with an {EMDASH} that must not count\n"
        )
        assert ccm.strip_commentary(raw) == "fix: thing\nbody\n"

    def test_cli_file_mode(self, tmp_path: Path) -> None:
        msg = tmp_path / "COMMIT_EDITMSG"
        msg.write_text("Fix stuff\n# comment\n", encoding="utf-8")
        assert ccm.main([str(msg)]) == 1
        msg.write_text("fix: stuff\n# Comment With Capitals is ignored\n", encoding="utf-8")
        assert ccm.main([str(msg)]) == 0

    def test_cli_without_arguments_is_a_usage_error(self) -> None:
        assert ccm.main([]) == 2
