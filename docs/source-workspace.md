# Source workspace, contribution skills, and maintenance

This document covers three connected things: what a Codoxear *source workspace*
is, the repository agent skills for reporting issues and contributing fixes,
and the maintenance rule that decides when source changes reach a running
server.

## Editable source is not the serving snapshot

A source workspace is an editable checkout of this repository (a normal clone
or worktree on the user's machine). It is development state. The deployed
server never imports code or static assets from it: a deployment installs a
**committed snapshot** — `scripts/deploy.sh <commit>` resolves the commit,
updates a detached snapshot worktree at `~/.local/share/codoxear/deploy`, and
installs that snapshot with pipx. Consequences:

- Edits in a source workspace change nothing on a running server. They apply
  only after the change is committed and `scripts/deploy.sh <commit>` runs.
- Deploys restart **only** `codoxear-server.service`. Live brokers, backend
  CLIs, sockets, uploads, queues, and session stores are untouched — sessions
  live in backend log files on disk, and the server only reads them. Never kill
  a broker or a backend CLI to deploy or restart the server.
- When an agent sets up or modifies a deployment, it must also configure
  `CODOXEAR_UPDATE_COMMAND` in the service environment (for example
  `/path/to/codoxear/scripts/deploy.sh origin/main`) and, once it works,
  enable auto-update via `update_settings.json` (`{"auto_update": true}`), so
  updates flow through the same committed-snapshot path.

The New Session dialog's "Use Codoxear source" shortcut opens an agent session
in a source workspace. It fills the existing cwd field without launching, and
the context line switches to source-workspace prompts ("Ask how Codoxear works,
investigate a bug, or make changes."). The browser learns the workspace lazily
through the authenticated `GET /api/source-workspace`
(`{available, path, can_setup, reason}`); creating one is an explicit setup
gesture via `POST /api/source-workspace` — no clone on GET. A source workspace
is never the deploy snapshot worktree, the installed package files, or the
live runtime state directory.

## Contribution skills

Two project skills ship in this repository for agent sessions (Pi, Codex, and
Claude Code) working inside a source workspace:

| Skill | Invoke | Use when |
|---|---|---|
| `codoxear-report-issue` | `/skill:codoxear-report-issue` (Pi), `$codoxear-report-issue` (Codex), `/codoxear-report-issue` (Claude Code) | report a bug, file an issue |
| `codoxear-contribute-fix` | `/skill:codoxear-contribute-fix`, `$codoxear-contribute-fix`, `/codoxear-contribute-fix` | submit a fix, open a PR |

Both target the canonical upstream **`yiwenlu66/codoxear`**, stated explicitly
and independently of the checkout's fork `origin`: every `gh` call passes
`--repo yiwenlu66/codoxear`, and a remote whose URL is `yiwenlu66/codoxear`
(HTTPS or SSH) is treated as canonical regardless of its nickname. Both check
for duplicate issues/PRs first, gather version/environment/reproduction
evidence, verify only inside Docker, never create issues or PRs as part of
testing, and quote back only URLs that `gh`/`git` actually printed.

## Skill discovery: one canonical body, documented per-backend paths

The skill bodies live **once**, canonically, in
`.agents/skills/<name>/SKILL.md` (+ optional `agents/openai.yaml`). No
backend-specific copy of the text exists anywhere else in the repository, so
the three backends cannot drift.

| Backend | Documented discovery | Loads |
|---|---|---|
| Pi | Project `.agents/skills/` (Agent Skills location), found from the working directory through ancestors, stopping at the repository root; also project `.pi/skills/` | canonical body directly |
| Codex | `.agents/skills` in every directory from the cwd up to the repository root; supports symlinked skill folders | canonical body directly |
| Claude Code | Project `.claude/skills/<name>/SKILL.md`, from the start directory up to the repository root; a skill entry may be a **symlink** to a directory elsewhere on disk | symlink `.claude/skills/<name>` → `../../.agents/skills/<name>` |

Discovery evidence (no source-string tests):

- **Pi (real CLI metadata)**: `pi --mode rpc` + `{"type":"get_commands"}`
  lists both skills as `skill:codoxear-report-issue` /
  `skill:codoxear-contribute-fix` with `sourceInfo.scope: "project"`,
  `sourceInfo.source: "auto"`, and `path` inside `.agents/skills/` — verified
  from the repository root and from a subdirectory. Documented in Pi's
  `docs/skills.md` and `docs/rpc-commands.md`.
- **Codex (official documentation)**: developers.openai.com/codex/skills
  documents the repo-scope `.agents/skills` scan (cwd → repository root),
  `$skill-name` invocation, `/skills` listing, optional `agents/openai.yaml`,
  and symlinked skill folders.
- **Claude Code (official documentation)**: code.claude.com/docs/en/skills
  documents project `.claude/skills/` discovery from the start directory up to
  the repository root and that a skill entry may be a symlink to a directory
  elsewhere on disk.

Rules that keep this true:

- Edit only the canonical `.agents/skills/<name>/SKILL.md`. Never copy skill
  text into `.claude/`, `.pi/`, or `.codex/`; add a relative symlink there if a
  backend needs one.
- Skills are general project guidance: no machine-specific paths, hosts,
  accounts, or credentials.
- `.codex/skills/codoxear-docker-test/` is historical and preserved in place.
  Its directory is not on Codex's documented scan list, so its content reaches
  agents through references to `scripts/codoxear-docker-sandbox` and the
  Docker verification scripts, not through automatic discovery.
- Frontmatter `name` must match the skill directory name, and `description`
  must state the real trigger conditions — it is the model-invocation surface.

Note for interactive terminal sessions: adding `.agents/skills/` triggers Pi's
project-trust dialog once per checkout. Codoxear launches Pi with `--approve`,
so web-owned sessions are unaffected.
