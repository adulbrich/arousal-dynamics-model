# Contributing

How a change moves through this repository. `AGENTS.md` holds the rules that bind
an agent on every turn; `README.md` covers what the model is and how to run it.

## Setup

```bash
uv sync
uv run lefthook install
```

`uv sync` creates `.venv` with the Python in `.python-version` and the locked
dependencies, including the `dev` group. `lefthook install` wires the git hooks
below into this clone. The notebook tooling is a separate group:
`uv sync --group notebooks`.

## The process

```
branch             commit                 push               pull request          merge
------             ------                 ----               ------------          -----
from origin/main   lefthook: ruff,        lefthook: ty,      CI: verify, test,     the owner
fix/ feat/ ...     prose, branch, lock,   fast tests         lowest, pr-text;      merges
                   commit message                            review loop
```

1. **Branch from a fresh `origin/main`.** `git fetch origin main` first. Prefix
   the branch with the commit type: `fix/`, `feat/`, `test/`, `docs/`, `build/`,
   `ci/`, `refactor/`, `chore/`.
2. **Commit by name.** Stage paths, never `git add -A`. The subject is
   Conventional Commits with a lowercase imperative:
   `fix(model): use atan2 for the circadian phase`.
3. **Push and open a pull request.** Every change reaches `main` through one; the
   owner merges. The template asks what changed, whether any number moved, what
   ran locally, the review passes, and which docs changed.
4. **Run the review loop.** `mattpocock-skills:code-review` and, on any pull request
   that changes behaviour, `correctness-review` beside it, until a pass raises
   nothing unanswered. Answered means fixed, or declined in one line in the PR.
   Record the pass count for each. GitHub requires no approving review, so this
   loop is the review the owner merges on. `docs/agents/code-review.md` has the
   local rules; `AGENTS.md` says what exempts a PR from the correctness pass.
5. **Clean up** the local branch after the merge.

## Golden outputs

`tests/test_golden.py` pins the model's output: every equation function on an
input grid, and four three-day trajectories with all states and derived outputs
(KSS, circadian drive, phase, aMT6s). The arrays in `tests/test_golden/` were
written by the code as it stood at commit 55eeb3e, known discrepancies with the
papers included ([issue #7](https://github.com/adulbrich/arousal-dynamics-model/issues/7)).

Two kinds of pull request touch the model, and they never mix:

- **A refactor or tooling change moves no number.** The golden tests pass
  unchanged. If one fails, the refactor changed the science; find out why.
- **A science change moves numbers on purpose.** One discrepancy per pull
  request. Regenerate in a commit of its own,

  ```bash
  uv run pytest tests/test_golden.py --force-regen
  ```

  and name in the PR body which arrays moved, citing the paper, equation and
  page. A reviewer reads the regenerated `.npz` diff as the change's effect.

The trajectory tolerance is 1e-5 of each variable's range: the worst drift
measured between SciPy 1.13 and 1.18 was 3e-6, and any change to an equation or
constant moves far more than that. `tests/test_golden.py` explains the numbers.

## Tests

`uv run pytest` runs everything; `uv run pytest -m "not slow"` is what the pre-push hook
runs. CI adds branch coverage (at least 95 %) and the `ci` Hypothesis profile, which
draws more examples with a fixed seed so a failure reproduces from the log.

| File | What it pins |
| --- | --- |
| `test_golden.py` | Model output against the arrays the legacy code wrote. A refactor must leave it green. |
| `test_equations.py` | Each equation against a value worked out by hand from the paper, and scalar and array calls agreeing exactly. |
| `test_properties.py` | Hypothesis invariants: bounds, monotonicity, gating, periodicity, unit round trips. |
| `test_invariants.py` | Physiology over simulated days: sleep at night, pressure building by day, entrainment. Ends with strict expected failures, one per audited code bug. |
| `test_simulation.py`, `test_inputs.py` | The public API, its errors, and the `Model` interface through a toy model with a closed-form answer. |
| `test_validation.py` | KSS against `data/output.csv`, the authors' reference; skipped when `data/` is absent. |
| `test_notebooks.py` | `notebooks/*.ipynb` run, and are committed without outputs. |
| `test_plotting.py`, `test_gate_scripts.py` | The plotting helpers and the scripts behind the gates. |

A discrepancy with a paper that is not fixed yet is a test marked
`xfail(strict=True)` with its issue number in its reason. The pull request that fixes it
removes the marker; strict mode fails the suite if the test passes while still marked.

## The gates

One script per rule, run at the earliest point that can see the violation. The
first column is what stops you locally, the middle one what stops an agent before
the command runs, the last what stops the merge.

| Rule | Local (lefthook) | Claude Code hook | CI |
| --- | --- | --- | --- |
| Conventional subject; no emdash, emoji or session link in the message | `commit-msg` | `guard_git.py` reads the `-m` text first | `verify` walks the PR's commits; `pr-text` checks the title and body |
| No emdash or emoji in tracked prose and code | `pre-commit`, staged files | `after_edit.py` on the edited file | `verify`: `check_prose.py --all` |
| No session link, emdash or emoji in PR or issue text | (never sees it) | `guard_gh.py` refuses the command | `pr-text`, for the PR title and body |
| Ruff lint and format clean | `pre-commit`, staged files | `after_edit.py` | `verify` |
| `uv.lock` matches `pyproject.toml` | `pre-commit`, when either is staged | `guard_edits.py` refuses a hand edit of `uv.lock` | `verify`: `uv lock --check`, `uv sync --locked` |
| Stage by name; never commit on `main` | `pre-commit` branch check | `guard_git.py` refuses `add -A`, `commit -a`, a commit on `main` | a ruleset, once the owner adds one (below) |
| No force push at `main`, `reset --hard`, `clean -f`, `branch -D`, `checkout .` | | `guard_git.py` | the ruleset (force push) |
| Golden arrays, archived notebooks, `data/` and `CLAUDE.md` are not hand-edited | | `guard_edits.py` | review |
| ty clean | `pre-push` | | `verify` |
| Test suite green | `pre-push`, without `slow` | | `test`: Ubuntu and macOS, Python 3.12 to 3.14, branch coverage at least 95 % |
| The declared dependency floors work | | | `lowest`: `--resolution lowest-direct` on 3.12 |

Skipping locally: `LEFTHOOK=0 git commit` or `--no-verify`. The hooks and CI catch
what was skipped, so skipping moves the failure rather than removing it.

Dependabot opens a weekly grouped pull request for `uv.lock` and one for the
SHA-pinned GitHub Actions. A dependency bump that moves a golden trajectory
fails the test matrix; that is intended, since a numpy or SciPy release changing
the model's output deserves a look before it merges.

### Making the checks required

Nothing blocks a merge until `main` has a ruleset. The owner can add one with the
checks this repository runs:

```bash
gh api repos/adulbrich/arousal-dynamics-model/rulesets --method POST --input .github/ruleset-main.json
```

`.github/ruleset-main.json` requires a pull request and two checks, `ci` and
`pr-text`, and blocks force pushes and deletion. `ci` is an aggregate job in
`ci.yml` that fails unless `verify`, every `test` cell and `lowest` passed, so
the matrix can change without touching the ruleset.

## Working with Claude Code

`CLAUDE.md` is a symlink to `AGENTS.md`, which Claude Code reads at the start of
every session. The rules there are the ones the gates above enforce, plus the
golden-output rule, which only review can enforce.

`.claude/settings.json` enables the `mattpocock-skills` plugin and installs the
hooks under `.claude/hooks/`, the middle column above. A fresh clone asks to trust
the project settings once. The hooks are standard-library Python run by the
`python3` on PATH, which on macOS may be the system 3.9; `tests/test_claude_hooks.py`
keeps them to that. `SessionStart` prints the branch, uncommitted paths, and whether
the uv on PATH is new enough. `.claude/skills/correctness-review/` is the
correctness half of the review loop.
