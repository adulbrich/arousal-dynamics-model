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
fix/ feat/ ...     prose, branch, lock,   fast tests         lowest, pr-text       merges
                   commit message
```

1. **Branch from a fresh `origin/main`.** `git fetch origin main` first. Prefix
   the branch with the commit type: `fix/`, `feat/`, `test/`, `docs/`, `build/`,
   `ci/`, `refactor/`, `chore/`.
2. **Commit by name.** Stage paths, never `git add -A`. The subject is
   Conventional Commits with a lowercase imperative:
   `fix(model): use atan2 for the circadian phase`.
3. **Push and open a pull request.** Every change reaches `main` through one; the
   owner merges. The template asks what changed, whether any number moved, what
   ran locally, and which docs changed.
4. **Clean up** the local branch after the merge.

## Golden outputs

`tests/test_golden.py` pins the model's output: every equation function on an
input grid, and four three-day trajectories with all states and derived outputs
(KSS, circadian drive, phase, aMT6s). The arrays in `tests/test_golden/` were
written by the code as it stood at commit 55eeb3e, known discrepancies with the
papers included (`docs/research/`).

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
`xfail(strict=True)` with the audit ID in its reason. The pull request that fixes it
removes the marker; strict mode fails the suite if the test passes while still marked.

## The gates

One script per rule, run at the earliest point that can see the violation.

| Rule | Local (lefthook) | CI |
| --- | --- | --- |
| Conventional subject; no emdash, emoji or session link in the message | `commit-msg` | `verify` walks the PR's commits; `pr-text` checks the title and body |
| No emdash or emoji in tracked prose and code | `pre-commit`, staged files | `verify`: `check_prose.py --all` |
| Ruff lint and format clean | `pre-commit`, staged files | `verify` |
| `uv.lock` matches `pyproject.toml` | `pre-commit`, when either is staged | `verify`: `uv lock --check`, `uv sync --locked` |
| Never commit on `main` | `pre-commit` branch check | a ruleset, once the owner adds one (below) |
| ty clean | `pre-push` | `verify` |
| Test suite green | `pre-push`, without `slow` | `test`: Ubuntu and macOS, Python 3.12 to 3.14, branch coverage at least 95 % |
| The declared dependency floors work | | `lowest`: `--resolution lowest-direct` on 3.12 |

Skipping locally: `LEFTHOOK=0 git commit` or `--no-verify`. CI runs the same
commands, so skipping moves the failure to the pull request rather than removing
it.

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
