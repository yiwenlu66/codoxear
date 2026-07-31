## Objective
Improve Codoxear's transcript search path so large logs do not require materializing every record/event before search limits can take effect, while preserving the transcript projection invariant: search, tail, history, live, export, and recovered-session search must agree on normalized visible chat rows, dedupe, no-response injection, and history/load cursors.

Done when the implementation is committed, locally validated, Docker/browser proof shows the user-visible search path still works on a large-log session, clean-room review accepts the slice, and task/project memory records the accepted invariant.

## Workbench
1. Replace or refactor `iter_positioned_chat_events_forward` / `search_chat_log_bounded` so first-order count-limited searches can stop after the bounded match count instead of reading the whole log, and latest-order searches keep bounded match memory without all-record materialization.
2. Preserve positioned event semantics: assistant dedupe, synthetic no-response/error injection, `_before_byte` / `_after_byte`, `history_cursor` / `load_cursor`, `before` history boundary behavior, oversized-line truncation signaling, and CC pending-tool state.
3. Add tests that would fail if count-limited search consumes the whole record stream before returning.
4. Validate focused and full local tests; prove in Docker/browser with a container-only large synthetic transcript search session.
5. Run clean-room adversarial review and record durable memory.

## Context
Active checkout: `/home/yiwen/codex-web-product-recovery` on branch `recovery/product-gaps`.
Protected checkout: `/home/yiwen/codex-web` on `main`; do not touch.
Project memory: `.memory/project/ARCHITECTURE.md`, `.memory/project/VALIDATION.md`.
Current code: `codoxear/transcript_search.py`, `codoxear/message_routes.py`, `codoxear/static/app_chat_search.js`.
Relevant tests: `tests/test_transcript_export.py`, `tests/test_message_routes.py`, source tests around transcript/search modules.
Docker skill: `.codex/skills/codoxear-docker-test/SKILL.md`.

## Task specifications
Current mechanism: `iter_positioned_chat_events_forward()` reads all bounded JSONL records into `records`, extracts all visible events into `events`, then applies `_dedupe_assistant_chat_events()` and `_inject_no_response_events()` before yielding. `search_chat_log_bounded()` applies `limit`, `order`, and `count_limit` only after this full materialization. Browser loaded-chat search schedules `/messages/search?...&count_max=1000`; the server still parses the whole log before that count bound can stop work.

Target mechanism: search consumes positioned events incrementally. Adjacent assistant dedupe should use constant state (`last_assistant_key`, reset on user). Synthetic no-response injection should use constant turn state (`open user byte`, `turn has visible assistant`, close row detection) while yielding events in byte order. Same-row close/error handling must emit any visible assistant/error event before deciding whether a synthetic no-response row is needed. Regular event `_after_byte` should be the source record end; synthetic close rows should use the close record end. `before_byte` must stop at records whose start is at/after the boundary.

The slice must not weaken transcript truth for Codex, Pi, or Claude Code logs. Existing no-response search/cursor tests are part of acceptance, not incidental tests.

If a pure streaming refactor proves too risky, escalate with evidence before substituting a file-size guard. A low search cap that makes copy-too-large's “Use search” guidance false is not acceptable.

## Constraints
Do not edit/promote/merge protected `/home/yiwen/codex-web` or `main`.
Do not touch live runtime dirs: `~/.local/share/codoxear`, `~/.claude`, `~/.codex`, host Pi logs/sockets, systemd/tailscale.
Docker-only for broker/server/session/tmux/browser verification; avoid port `8743`.
Cleanup must be exact-PID/container scoped; no `pkill -f`, `killall`, or broad kills.
Keep functional, proof/evidence, review, and memory commits separate.
Browser + Docker evidence required for browser/product usability claims.
Delegate concrete implementation/validation work to executor subagents where possible.
Run clean-room adversarial review before yielding.
Do not copy secrets into committed artifacts; exclude cookies, auth headers, credential values, private file contents, bulky logs.
Monaco remains required; no plain textarea/diff fallback certification.
