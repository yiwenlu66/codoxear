# Session: 2026-02-27 Claude Code Support

## Focus
Add first-class Claude Code support as a major feature while keeping existing Codex behavior.

## Requests
- Create a dedicated feature branch for the change.
- Add Claude Code support end-to-end (broker/server/log parsing/UI/resume helper).
- Keep daemon/runtime workflow intact and commit the change.

## Actions Taken
- Continued work on branch `feature/claude-code-support`.
- Added `codoxear/cli_support.py` to normalize CLI selection (`codex`/`claude`), resolve CLI homes/binaries/log roots, infer CLI from log path, and parse Claude message content.
- Extended broker discovery/runtime to support both Codex rollout logs and Claude project logs, including metadata sidecar field `cli` and Claude turn-end handling (`system.subtype=turn_duration|api_error`).
- Extended server session discovery/listing to track `cli`, infer it from log path when metadata is missing, and avoid Codex-only main-thread coercion for Claude logs.
- Updated web session spawn API to accept `cli` (`POST /api/sessions`), validate values, propagate CLI env to broker, and return `cli` in spawn response.
- Extended rollout parsing/idle heuristics for Claude logs (`user`/`assistant`/`system` records) while preserving Codex parsing.
- Updated UI session tools resume command to match session CLI and updated new-session flow to prompt for CLI.
- Updated `scripts/codoxear-resume` to pick resume command per-session (`codex resume` vs `claude --resume`) for `--list`, `--last`, `--id`, and interactive selection.
- Added/updated tests for Claude log discovery, busy/idle state, chat flags, idle heuristics, and spawn CLI env handling.
- Updated docs and env example to document multi-CLI operation.

## Outcomes
- Codoxear now supports both Codex and Claude Code sessions in the same runtime.
- New web sessions can be created explicitly for either CLI.
- Session metadata/listing now includes CLI identity for correct resume command behavior.

## Tests
- `python3 -m unittest discover -s tests`

## Notes
- Existing sessions without a `cli` field are backfilled by log-path inference (`rollout-*.jsonl` => codex, Claude UUID project logs => claude).
