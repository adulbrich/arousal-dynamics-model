---
name: correctness-review
description: Sweep the changes since a fixed point (commit, branch, tag, or merge-base) for the defect classes a numerical model is prone to, the way code-review sweeps for smells, and report each finding with the input that triggers it. Use when a diff touches behaviour and the question is "is it wrong", not "does it conform"; with `mutate`, also removes each guard in the diff and reports the ones no test catches.
---

Correctness sweep of the diff between `HEAD` and a fixed point the user supplies.
Three read-only sub-agents run in parallel, each with one way of finding things:
reading the diff for what it does to numbers, executing pure functions at their
boundaries, and checking what the diff takes on trust from a paper. An optional
fourth step mutates guards in a scratch worktree.

It answers a different question from `mattpocock-skills:code-review`, which checks
conformance; see _Why_.

## Process

### 1. Pin the fixed point

Whatever the user said is the fixed point. If they did not name one, ask. Capture the
diff command once, `git diff <fixed-point>...HEAD` (three-dot, against the
merge-base), and the commit list, `git log <fixed-point>..HEAD --oneline`.

Confirm `git rev-parse <fixed-point>` resolves and the diff is non-empty before
spawning anything. Note whether the user passed `mutate`.

### 2. Find the spec

Look in this order: issue references in the commit messages, fetched with the
workflow in `docs/agents/issue-tracker.md`; a path the user passed; the ADRs under
`docs/adr/` that the diff touches. If none exists, the spec-dependent entries report
"no spec".

### 3. The checklist

Each entry reads _what it is_, then _how to find it_. Paste the whole list into every
sub-agent prompt: the sub-agent has no other access to it.

- **Unit at a boundary.** A value that leaves one function in one unit and is read
  by another in a different one: seconds and hours, melanopic EDI in lux and
  melanopic irradiance in W/m^2, radians and degrees. Find it: for every number the
  diff passes between functions, name its unit on both sides from the parameter's
  field metadata or the docstring. A number whose unit cannot be named is a finding.
- **Threshold and window edges.** A comparison at a switch: `V_m` against `v_th`,
  `phi` against `[phi_on, phi_off]`, the clock hour against `waketime` and
  `bedtime`, and the wraps at 24 h and at plus or minus pi. Find it: call the
  function exactly at the edge, one ulp either side (`numpy.nextafter`), and across
  the wrap, from a scratch script, and say which side each lands on.
- **Boundary conditions in pure functions.** Empty arrays, a single sample, zero
  light, negative input, NaN, a zero denominator (`X = 0` in `Y / X`), and a scalar
  where an array is usual. Find it: call the function with each from a scratch
  script. Reading the function is not finding it.
- **Silent dtype or shape change.** An integer where a float belongs
  (`np.vectorize` takes its dtype from the first result, which once truncated
  `wake_effort` to integers here; `np.where` with integer branches), broadcasting
  that succeeds with the wrong shape, a 0-d array where a float is expected. Find
  it: print the dtype and shape of every array the diff produces, for a scalar
  input and for an array input.
- **Solver sensitivity.** A change to the right-hand side that adds a discontinuity,
  reorders floating-point operations, or changes tolerances. The golden tests allow
  1e-5 of a variable's range, so a reordering can pass on one platform and drift on
  another. Find it: run `tests/test_golden.py` scenarios with `rtol` and `atol`
  tightened a hundredfold in a scratch script and compare against the arrays; a new
  discontinuity without a note on how the solver meets it is a finding.
- **Unvalidated claim about a paper.** An equation, constant, unit or sign taken
  from a paper without the equation or table number and page, or with one that does
  not match the paper. The papers are not always right, so a deliberate departure
  is fine when the code says so at that line and why. Find it: for each value or
  form the diff adds or changes, check the cited equation in the paper; one that
  cannot be checked from here is reported as unverified, with the citation.
- **Never-executed code.** A branch, parameter combination or error path no run
  reaches: `photic_version="2018"` in a full simulation, an interpolation kind, a
  `raise`. Find it: say what input makes the line run for the first time, and
  whether a test supplies it (`uv run pytest --cov` reports the lines).
- **A comment that claims correctness.** Every comment or docstring saying
  something matches a paper, cannot be negative, is identical to the original, or
  always holds is a hypothesis. Find it: construct the case that disproves it, and
  report the claims that survive as well as the ones that fall.

A finding carries the concrete input that triggers it and what comes out. A
description of the risk is not a finding.

### 4. Spawn three sub-agents in parallel

Every prompt opens with: "You are read only. Write nothing under the repository,
stage nothing, change no branch. A scratch script goes under `$TMPDIR` and runs with
`uv run --project <repository root> python <script>`, importing `arousal_dynamics`
from the checkout." Every prompt carries the diff command, the commit list, the full
checklist from step 3, and a 400-word limit on the report.

- **Numbers.** Entries: unit at a boundary, silent dtype or shape change, solver
  sensitivity, and comment claims. Brief: "For each entry, walk every hunk it
  applies to. Report each finding under the entry's name with the input that
  triggers it and the output you got."
- **Boundaries.** Entries: threshold and window edges, boundary conditions. Brief:
  "List every pure function the diff adds or changes. For each, run it with the
  inputs from both entries from a scratch script, and report the output for each
  input. Read nothing you can run."
- **Sources.** Entries: unvalidated claim about a paper, never-executed code. Also
  carries the spec from step 2. Brief: "For each value, unit or equation form the
  diff takes from a paper, give the citation it carries and whether it matches. For
  each new branch and error path, say what would run it for the first time and
  whether a test does."

### 5. Mutate the guards (only with `mutate`)

Guard mutation runs tests against changed code, so it happens in a scratch worktree
and never in the checkout the user is working in.

1. `git worktree add --detach "$TMPDIR/mutate-<sha>" HEAD`, then `uv sync` inside
   it. Run `uv run pytest -q -m "not slow"` once, unchanged, to prove the worktree
   works before trusting a result from it.
2. List every conditional in the diff that protects something: an input validation
   that raises, a clamp (`np.maximum(..., 0)`), a threshold, a unit conversion.
3. For each guard, remove it (or invert it), run `uv run pytest -q -m "not slow"`
   against the test files for its module, and restore it before the next.
   `git checkout -- <file>` in the worktree restores it.
4. Report each guard with the tests that failed. A guard whose removal fails no
   related test is **unverified here**: say so in those words, and add whether the
   slow tests (`-m slow`), which this step does not run, cover it.
5. `git worktree remove --force "$TMPDIR/mutate-<sha>"`.

### 6. Aggregate

Present the reports under `## Numbers`, `## Boundaries`, `## Sources` and, if it
ran, `## Mutation`, verbatim or lightly cleaned, findings under the checklist entry
names. Keep the sections separate and their order fixed. End with one line per
section: finding count and the worst finding in that section. Name no winner across
sections.

## Why

`mattpocock-skills:code-review` catches code that conforms to the standards but not
the spec, and code that does what the spec says but breaks a convention. A third
case passes both: code that conforms, implements the issue, and is wrong. In a model
that case is usually a number: a unit read on the wrong scale, a threshold on the
wrong side of an equality, a dtype that quietly truncates, a constant transcribed
from the wrong table. Comparing the code with its spec cannot find those, because
both came from the same reading of the paper. This skill asks the diff what breaks
it instead, which is why the sections stay separate: a unit error and a missed
branch are not the same kind of finding, and ranking them against each other hides
one.
