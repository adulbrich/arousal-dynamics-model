# AI Agent Instructions

A Python implementation of the light-driven arousal dynamics model: Postnova et al.
2016 and 2018 for sleep/wake and alertness, Abeysuriya et al. 2018 for melatonin and
aMT6s, Tekieh et al. 2020 for melanopic light input. Citations are in `README.md`.

`CLAUDE.md` is a symlink to this file. Claude Code reads `CLAUDE.md`, so keep the
symlink and never keep a second copy.

## Layout

`src/arousal_dynamics/`: the package. `models/arousal.py` holds the model, its
parameters and its equations; `simulation.py` holds `simulate` and the `Model`
interface. `tests/` mirrors it (CONTRIBUTING.md, "Tests"). `notebooks/archive/` is
history and is excluded from every gate.

## Before you commit

```bash
uv run ruff check && uv run ruff format --check
uv run ty check
uv run pytest
```

All clean. lefthook runs these at commit and push once `uv run lefthook install`
has run in the clone.

## Always

- **No emdash and no emoji** in commits, comments, docstrings, string literals, docs
  or pull request text. `scripts/check_prose.py` enforces it at commit and in CI. A
  `--` used as a sentence dash breaks the same rule. Greek letters, arrows and math
  symbols in docstrings are fine. The one line the harness appends to a PR body is
  exempt (`HARNESS_FOOTER` in the script).
- **Conventional Commits, lowercase imperative subject, area in parens when one
  fits:** `fix(model): use atan2 for the circadian phase`. Types: `feat`, `fix`,
  `docs`, `test`, `refactor`, `style`, `perf`, `build`, `ci`, `chore`; `!` before the
  colon for a breaking change. `scripts/check_commit_message.py`.
- **Never publish a `claude.ai/code/session` link** in a commit, pull request or
  issue. The repository is public.
- **Stage files by name**, never `git add -A`, `git add .` or `git commit -a`.
  `data/` holds unpublished data that must never be committed.
- **Never commit to `main`.** Branch from `origin/main`, push, open a pull request.
  The owner merges.
- **A refactor moves no number; a science fix moves numbers on purpose.** Never mix
  them in one pull request. A refactor that fails `tests/test_golden.py` changed
  the science. A science fix regenerates the golden arrays in a commit of its own
  and names the paper, equation and page. CONTRIBUTING.md, "Golden outputs".
- **Cite the source of every equation and constant** in its docstring or comment:
  paper, equation or table number. Where the code deliberately departs from a paper
  (a typo in the paper, a refit), say so at that line and why.
- **Review loop on every PR, Dependabot's included.** GitHub requires no approving
  review, and green CI is not one, so this is the review. Run
  `mattpocock-skills:code-review` (conformance) and, on a PR that changes behaviour,
  `correctness-review` (what breaks) beside it. Repeat until a pass raises nothing you
  have not answered; answered means fixed, or declined in one line in the PR. Later
  passes review the code earlier ones made you write. Verify a finding before acting:
  reviewers are sometimes confidently wrong. Record each pass count and every decline
  in the PR. Correctness is skipped, saying so in the PR, for a diff of only docs,
  comments or templates, or a dependency bump with no source change. There is no
  security pass. `docs/agents/code-review.md` has the local rules.
- **A known discrepancy is a strict xfail.** Add a test that states what the paper
  requires, marked `xfail(strict=True)` with its issue number; the fix removes the marker.
- **The papers can be wrong.** A discrepancy between code and paper is not
  automatically a code bug. Check the git history and the comments for a
  documented deviation before changing anything, and file a Model discrepancy
  issue when unsure.

## Agent skills

- Issue tracker: GitHub issues via `gh`. `docs/agents/issue-tracker.md`.
- Code review: the standards sources and the golden-array rule.
  `docs/agents/code-review.md`.

## Reference docs

- `CONTRIBUTING.md`: setup, the process, the golden-output rule, the gates.
- `docs/adr/`: the decisions. Research, audits and open work live in GitHub issues, not
  in the repository or a To Do list: the equation audit against the papers is issue #7,
  the model landscape #15, the evidence for a daytime model #16. The pull request that
  finishes an item closes its issue.
- `README.md`: what the model is, the papers, how to run it.
