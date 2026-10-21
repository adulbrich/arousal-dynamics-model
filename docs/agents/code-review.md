# Code review

`AGENTS.md` holds the review rule: `mattpocock-skills:code-review` for conformance,
`correctness-review` under `.claude/skills/` for whether a change is wrong, and no
security pass, since the package has no server, no credentials and no untrusted
input. Two local deltas from the code-review skill's default brief:

- **The Standards axis reads these sources and no others:** `AGENTS.md`, the
  "Golden outputs" and "Tests" sections of `CONTRIBUTING.md`, and the ADRs under
  `docs/adr/` that touch the diff. Grep them for the sections the diff reaches. The
  rest of `CONTRIBUTING.md` and `README.md` describe the project, not how its code is
  written.
- **A pull request that moves a golden array carries its reason.** The regenerated
  arrays sit in a commit of their own, and the PR body names which arrays moved and
  cites the paper, equation and page behind the change. A refactor moves none. The
  Spec axis reports a moved array without that line as a finding.
