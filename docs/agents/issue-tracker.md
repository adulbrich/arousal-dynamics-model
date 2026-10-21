# Issue tracker: GitHub

Issues for this repository live as GitHub issues on
`adulbrich/arousal-dynamics-model`. Use the `gh` CLI for all of it.

- **Create**: `gh issue create --title "..." --body "..."`, with a heredoc for a
  multi-line body, or through one of the forms under `.github/ISSUE_TEMPLATE/`. A
  disagreement between the code and a paper uses the Model discrepancy form: one
  discrepancy per issue, with the paper, equation and page. The hook
  `.claude/hooks/guard_gh.py` refuses a title or body with an emdash, an emoji or a
  `claude.ai/code/session` link.
- **Read**: `gh issue view <number> --comments`.
- **List**: `gh issue list --state open --json number,title,labels`.
- **Comment**: `gh issue comment <number> --body "..."`.
- **Close**: a pull request closes its issue with `Closes #N` in the body.
