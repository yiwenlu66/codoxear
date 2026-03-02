# Session: 2026-03-02 Gemini CLI Support

## Focus
Add first-class Gemini CLI support as a major feature while preserving Codex/Claude behavior.

## Requests
- Create a new feature branch.
- Add Gemini support end-to-end.
- Do not reuse the older streamd-based Gemini implementation.

## Actions Taken
- Created feature branch `feature/gemini-support` in a clean worktree.
- Extended CLI support helpers with `gemini` normalization, env/bin/home resolution, log-path detection, and Gemini session-file parsing helpers.
- Added Gemini log support in util/log parsing:
  - detect `~/.gemini/tmp/**/chats/session-*.json`,
  - map Gemini session JSON messages into unified chat events,
  - support incremental `/messages` offsets via virtual JSONL byte offsets.
- Extended broker runtime:
  - Gemini fallback log discovery by `cwd` and mtime,
  - Gemini turn-end handling on assistant events to avoid queue stalls,
  - Gemini env propagation (`GEMINI_HOME`) into child process.
- Extended server session spawn env wiring (`GEMINI_HOME`/`GEMINI_BIN`) and CLI validation messaging.
- Updated UI and helper script:
  - New-session CLI toggle cycles `Codex -> Claude -> Gemini`,
  - session tools resume command supports `gemini --resume <id>`,
  - `scripts/codoxear-resume` supports Gemini sessions.
- Added/updated tests for Gemini across CLI helpers, broker discovery, server spawn env, chat flags, idle heuristics, busy-state transitions, and offset reading.
- Updated README and feature/flow docs for Gemini operations and configuration.

## Outcomes
- Codoxear now supports Codex, Claude, and Gemini sessions in one runtime.
- Web-owned session creation and resume tools are Gemini-aware.
- Gemini chat logs are parsed and surfaced in the existing message polling pipeline with queue-safe turn-end behavior.

## Tests
- `node --check codoxear/static/app.js`
- `python3 -m unittest discover -s tests`
