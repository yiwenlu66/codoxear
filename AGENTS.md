# AGENTS.md

## Purpose
Codoxear is a Linux-first companion UI for continuing Codex CLI TUI sessions on a phone or laptop browser. The web UI is a view/controller while all files, tools, and credentials stay on the host machine.

## Standard startup (this host)
- Start the server with `scripts/codoxear-server-dev` to pin the working directory to the repo root.
- The server loads `.env` from `/root/code/codoxear/.env` when started from the repo root.

## Structure at a glance
- `codoxear/server.py`  HTTP server, JSON API, auth cookie, session discovery, Harness scheduler
- `codoxear/broker.py`  PTY wrapper for Codex CLI, socket control channel, log discovery, busy/idle heuristics
- `codoxear/sessiond.py`  Headless session launcher that mimics broker metadata output
- `codoxear/rollout_log.py`  Rollout JSONL parsing, chat event extraction, idle detection, token stats
- `codoxear/util.py`  Shared helpers for app dir, log scanning, JSONL reads
- `codoxear/static/`  Browser UI (`index.html`, `app.js`, `app.css`)
- `tests/`  Pytest coverage for log parsing, idle heuristics, and server behavior
- `README.md`  User-facing overview and quick start

## Documentation taxonomy
Docs are split into Feature, Flow, and Work Records. Feature docs describe purpose, implementation, key files, and call stacks. Flow docs describe repeatable dev/deploy/test workflows. Work records summarize requests and actions by session.

## Documentation index
`docs/features/server-and-api.md`: Server architecture, auth, discovery, and API behavior.
`docs/features/broker.md`: Broker PTY lifecycle, socket protocol, and busy/idle state.
`docs/features/sessiond.md`: Headless session runner and metadata behavior.
`docs/features/ui.md`: Browser UI polling, local echo, queueing, and user actions.
`docs/features/rollout-log-parsing.md`: Rollout JSONL parsing and idle heuristics.
`docs/features/ROUTES.md`: API routes and auth overview.
`docs/flows/DEVELOPMENT.md`: Dev environment, run commands, and doc sync rules.
`docs/flows/DEPLOYMENT.md`: Deployment considerations and runtime paths.
`docs/flows/TESTING.md`: Tests and verification steps.
`docs/records/WORK_RECORDS.md`: Work records summary of current focus with per-session headings and links.
`docs/records/sessions/2026-02-22-docs-bootstrap.md`: Work record for initial documentation and AGENTS structure.

## Development or debug workflow
Every dev task must follow these four steps in order and report results.

1. Read new files
Read related files and relevant feature or flow docs. `docs/records/WORK_RECORDS.md` is mandatory to read for every task, plus the related session record. If anything is unclear, consult docs first. When using Codex CLI to validate doc reading, use read-only prompts and require file path references.
2. Code
Only modify files relevant to the request.
3. Test
All code changes must be tested. Run existing tests. If there are no tests, explain why and provide required manual verification steps. If existing tests cannot validate the new functionality, add new tests and write the new test commands into the relevant feature doc(s) and the testing doc.
4. Add detailed notes
Add clear comments or documentation, and update relevant feature and flow docs to keep them concise and consistent with the code. Update the work records summary and session records as needed so the collaboration context is complete.

At the end of each conversation, if this round is debug or development, provide a short four-step summary: step 1 lists files read; steps 2/3/4 each list files changed.

## Definition of Done
- Requested change implemented and verifiable
- Self-review and risks noted
- Tests run or an explicit reason for skipping
- Relevant comments or docs updated

## Concurrent collaboration
This repository is actively edited by multiple people/agents at the same time. Treat concurrent changes as normal and collaborate accordingly.

- Check `git status` before and after edits to understand what changed during your task.
- Do not revert or overwrite unrelated in-flight changes from other contributors.
- Keep edits minimal and scoped to the request so merges are easier.
- If a new change appears in files you are also editing, pause and coordinate before proceeding.
- Document assumptions and handoff notes clearly so parallel work can continue safely.
