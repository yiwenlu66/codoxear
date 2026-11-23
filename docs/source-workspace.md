# Work on Codoxear with an agent

A **source workspace** is an editable Codoxear clone or worktree. Use it to ask
how Codoxear works, investigate a bug, or make changes — not to edit the serving
installation or runtime data.

## Open a source session

1. In **New Session**, choose **Use Codoxear source** beside the directory field.
   It fills that field and keeps your backend/model choices; it does not launch.
   **Help → Ask an agent about Codoxear** opens the same dialog with the source
   directory prefilled.
2. If no workspace is available, choose **Set up source workspace** explicitly.
   Opening the dialog or checking availability never clones a repository.
3. Review the settings and press **Start** to launch the agent session.

Source discovery excludes the deploy snapshot, installed package files, and live
runtime state. The authenticated `GET /api/source-workspace` reports
`{available, path, can_setup, reason}` without creating anything; the explicit
setup action uses `POST /api/source-workspace`.

## Report a bug or contribute a fix

Ask the agent to “report this bug” or “submit this fix,” or invoke a skill:

| Skill | Pi | Codex | Claude Code |
|---|---|---|---|
| `codoxear-report-issue` | `/skill:codoxear-report-issue` | `$codoxear-report-issue` | `/codoxear-report-issue` |
| `codoxear-contribute-fix` | `/skill:codoxear-contribute-fix` | `$codoxear-contribute-fix` | `/codoxear-contribute-fix` |

Both target **`yiwenlu66/codoxear`**, even when your checkout's `origin` is a
fork. They check for prior issues/PRs first. A bug report adds new evidence to
an open duplicate instead of creating another issue. A contribution preserves
existing commits and local edits, isolates only the requested fix, verifies it
in Docker, and opens a PR against upstream `main`. They require GitHub CLI
(`gh`) authentication to post; without it, the agent can return a prepared
report or explain the remaining submission step. No issue/PR is posted merely
to test a skill.

Issue/PR commands specify `--repo yiwenlu66/codoxear`; repository commands use
an explicit repository argument. Auth commands and `gh api user` act on the
GitHub account and do not take that repository flag. Only URLs actually
returned by GitHub/git are reported as created submissions.

### Skill discovery and maintenance

Each skill has one canonical body at `.agents/skills/<name>/SKILL.md`, with
optional Codex UI metadata at `agents/openai.yaml`.

| Backend | Project discovery | Repository entry |
|---|---|---|
| Pi | `.agents/skills/` from cwd through ancestors, stopping at the repository root | canonical body |
| Codex | `.agents/skills/` from cwd through ancestors to the repository root | canonical body |
| Claude Code | `.claude/skills/` from the start directory through ancestors to the repository root | relative symlink to `../../.agents/skills/<name>` |

- Edit the canonical body, never copies under backend-specific directories.
  Preserve the Claude Code symlinks.
- Keep frontmatter `name` equal to the directory name; `description` states
  when the skill applies. Keep guidance portable: no local hosts, accounts,
  credentials, or machine-specific paths.
- Validate frontmatter and real CLI discovery from both the repository root
  and a nested directory **inside Docker**, not through source-string tests.
  Keep validation logs outside reader documentation.
- `.codex/skills/codoxear-docker-test/` is historical, not a current automatic
  discovery path. Use the referenced Docker verification scripts directly.

Discovery references: [Pi skills](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/skills.md),
[Codex skills](https://developers.openai.com/codex/skills), and
[Claude Code skills](https://code.claude.com/docs/en/skills).

## When changes reach the server

The supported service deployment uses a **committed snapshot**, not the editable
workspace. After review and commit, the maintainer deploys with:

```sh
scripts/deploy.sh <commit>
```

This installs the selected commit from the detached worktree at
`~/.local/share/codoxear/deploy` and restarts only `codoxear-server.service`.
Workspace edits do not change that deployed service. A manually started
**development server can read an editable checkout**; that is not the supported
service deployment. Agent-run tests still belong in Docker, never on the live
server (port 8743) or host runtime state.

Deploys preserve brokers, backend CLIs, sockets, uploads, queues, and session
stores. Conversations remain in backend logs; never kill a broker or agent CLI
to deploy. Roll back with `scripts/deploy.sh <previous-commit>`.

When setting up or modifying a deployment, configure `CODOXEAR_UPDATE_COMMAND`
in the service environment to use that same deployment mechanism (for example
`/path/to/codoxear/scripts/deploy.sh origin/main`). Once configured successfully,
enable auto-update in `~/.local/share/codoxear/update_settings.json` with
`{"auto_update": true}`. See [deployment and operations](../README.md#deployment-and-operations)
for prerequisites and service behavior. Contribution sessions do not deploy.
