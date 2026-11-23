---
name: codoxear-report-issue
description: File a Codoxear bug report against the canonical upstream yiwenlu66/codoxear. Use when the user asks to report a bug, file an issue, open an issue, or report broken Codoxear behavior. Checks existing issues for duplicates first, then gathers version, environment, and reproduction evidence before creating the issue with gh.
---

# Report a Codoxear issue

Issues are filed against the canonical upstream **yiwenlu66/codoxear**. Never
assume a remote nickname (`origin`, `upstream`) names it: use
`--repo yiwenlu66/codoxear` on issue/PR commands. Repository commands take an
explicit repository argument; auth commands are account-scoped. For git reads,
identify the canonical remote by its HTTPS or SSH URL. Run repository commands
from the editable checkout's root (`cd "$(git rev-parse --show-toplevel)"`).

## 1. Check duplicates first

```sh
gh auth status          # run `gh auth login` when unauthenticated
gh issue list --repo yiwenlu66/codoxear --state all \
  --search "<two or three key words from the symptom>" --limit 20
```

- Open duplicate: add relevant new evidence with
  `gh issue comment <number> --repo yiwenlu66/codoxear --body "<new evidence>"`
  instead of filing another issue. The request to report the bug covers this
  contribution; no extra confirmation is needed. If there is nothing new to
  add, return the existing issue URL instead of posting a redundant comment.
- Closed duplicate: check whether the same symptom is a recurrence; link the
  previous issue in a new report when there is a current bug to report.

## 2. Gather version, environment, reproduction

- Codoxear version: source checkout `git log -1 --format='%H %cs'`; a deployed
  server: `git -C ~/.local/share/codoxear/deploy log -1 --format='%H %cs'`.
- Backend CLIs involved: `codex --version`, `pi --version`, `claude --version`.
- Environment: `uname -sr`, `python3 -V`, and how the server runs (systemd
  service, `scripts/codoxear-docker-sandbox`, or manual start; default port 8743).
- Reproduction: exact steps, expected vs observed. When runtime proof is
  needed, reproduce inside the Docker sandbox (`scripts/codoxear-docker-sandbox`)
  — never against the live server and never on the host runtime state.
- Use user-supplied affected-session diagnostics and log excerpts when
  available. For a Docker reproduction, the authenticated
  `GET /api/sessions/<id>/diagnostics` response provides session details.
  Do not probe the live deployment to gather evidence.
- Relevant excerpt of the backend session log under the reproduction server's HOME
  (Codex `~/.codex/sessions/rollout-*.jsonl`, Pi `~/.pi/agent/sessions/*.jsonl`,
  Claude Code `~/.claude/projects/**/*.jsonl`).

Redact passwords, tokens, cookies, and API keys before anything leaves the
machine. Ask before attaching logs that contain private prompt content.

## 3. File the issue

```sh
gh issue create --repo yiwenlu66/codoxear \
  --title "<observed symptom, not a guessed diagnosis>" \
  --body "<summary / environment / reproduction steps / evidence>"
```

Quote back only the URL `gh` prints. If `gh` is unavailable or unauthenticated,
hand the prepared title and body to the user instead of inventing a URL.

## Boundaries

- Never create an issue as part of testing this skill; validation means
  discovery checks and dry-run text only.
- A code change that fixes the bug belongs to the `codoxear-contribute-fix`
  skill, not the bug report.
