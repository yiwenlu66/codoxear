# Development

This document describes the development flow and documentation sync rules.

## Development environment runtime
Requires Python 3.10+.

Install in editable mode:
`python3 -m pip install -e .`

Start server (fixed working directory, standard):
`scripts/codoxear-server-dev`

Daemonized server control (this host runtime):
`supervisorctl status codoxear`
`supervisorctl restart codoxear`

Start broker from a terminal:
`CODEX_WEB_CLI=codex codoxear-broker -- <codex args>`
`CODEX_WEB_CLI=claude codoxear-broker -- <claude args>`

Optional shell wrapper:
`codex() { CODEX_WEB_CLI=codex codoxear-broker -- "$@"; }`
`claude() { CODEX_WEB_CLI=claude codoxear-broker -- "$@"; }`

Default bind:
`CODEX_WEB_HOST=::`

Default port:
`CODEX_WEB_PORT=8743` (the daemon on this host is currently configured to `13780`)

Required env:
`CODEX_WEB_PASSWORD` must be set (via `.env` or environment).

Env file location (this host):
`/root/code/codoxear/.env`

Runtime state directory:
`~/.local/share/codoxear`

Status helper (SSH-friendly):
`scripts/codoxear-status --web --last`

## Branch model (this fork)
- `upstream/main`: source-of-truth mainline.
- `origin/main`: mirror of `upstream/main`; keep aligned.
- `dev` / `origin/dev`: integration branch for ongoing work.
- `feature/*`: short-lived branches based on `dev`.

## Normal branch workflow
1. Sync remotes first:
   - `git fetch origin --prune`
   - `git fetch upstream --prune`
2. Start from `dev`:
   - `git checkout dev`
   - `git pull --ff-only origin dev`
   - `git checkout -b feature/<topic>`
3. Merge finished work back into `dev` and push:
   - Prefer fast-forward when possible.
   - `git push origin dev`
4. Keep `main` aligned with `upstream/main` (do not merge feature/dev directly into `main`).

## Recovery when `main` is accidentally advanced
1. Preserve work on `dev`:
   - `git checkout dev`
   - `git merge --ff-only <main-or-feature-tip>`
   - `git push origin dev`
2. Realign `main` to upstream:
   - `git checkout main`
   - `git fetch upstream --prune`
   - `git reset --hard upstream/main`
   - `git push --force-with-lease origin main`
3. Verify:
   - `git rev-list --left-right --count upstream/main...main`
   - Expected output: `0 0`

## Documentation sync rules
- When changing or adding a feature, update the corresponding feature doc.
- When changing API or auth rules, update `docs/features/ROUTES.md`.
- When changing key data structures or runtime paths, update the relevant feature doc.
- When work is completed, update `docs/records/WORK_RECORDS.md` and the relevant session record.

## Development constraints
- Keep changes minimal and avoid unrelated modifications.
- Read related feature or flow docs before editing implementation code.
- `docs/records/WORK_RECORDS.md` is mandatory to read before any work.

## Notes
- Editing without reading related files increases regression risk.
- Missing doc updates raise handoff cost.
