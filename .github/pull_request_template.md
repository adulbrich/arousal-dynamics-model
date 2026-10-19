Closes #

<!-- What changed and why, from the reader's side. Two to five bullets. -->

-

## Numbers

<!-- Does this change any model output? Keep one line, delete the other. A
     science fix regenerates the golden arrays in a commit of its own
     (CONTRIBUTING.md, "Golden outputs"). -->

- No golden array moved.
- Golden arrays regenerated in <commit>: <files>, because <paper, equation, page>.

## Ran locally

<!-- Keep the lines that ran. lefthook runs ruff and the prose check at
     commit, ty and the fast tests at push. -->

- `uv run ruff check`, `uv run ty check`, `uv run pytest`

## Review loop

<!-- AGENTS.md: mattpocock-skills:code-review, and correctness-review beside it
     when the PR changes behaviour, until a pass raises nothing unanswered. State
     each pass count, or why the correctness pass does not apply. A declined
     finding gets one line. -->

- Passes:
- Correctness passes:
- Declined:

## Docs

<!-- README, CONTRIBUTING, AGENTS, an ADR, or "none, because ...". -->

-
