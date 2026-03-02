# Session: 2026-02-27 Claude Busy Stop Detection

## Focus
Fix Claude running-state stop detection when no explicit turn-end marker is emitted.

## Requests
- Improve Claude reply-stop detection so the session does not remain stuck in running state.

## Actions Taken
- Reproduced the issue on the live daemon (`13780`): Claude replies appeared, but `/messages` kept returning `busy=true` with no `turn_end`.
- Identified root cause in broker idle fallback logic:
  - non-queue idle fallback had been restricted to queue-only release paths,
  - Claude turns without explicit `system.subtype=turn_duration|api_error` could remain busy.
- Added Claude-only non-queue idle fallback:
  - `_should_idle_fallback_clear_busy_without_queue`,
  - enabled in `_log_watcher` idle check,
  - uses conservative quiet-window gating (`IDLE_TURN_END_QUIET_SECONDS`) and existing completion-candidate guardrails.
- Added tests covering:
  - Claude non-queue idle fallback clears busy,
  - non-Claude keeps previous behavior.

## Outcomes
- Claude sessions now return from running to idle after reply output settles, even when explicit turn-end markers are absent.
- Codex queue/turn-end gating behavior remains unchanged.

## Tests
- `python3 -m unittest tests.test_broker_busy_state`
- `python3 -m unittest discover -s tests`

## Manual Verification
- Live API check against `http://127.0.0.1:13780`:
  - Create Claude session, send prompt, poll `/messages`.
  - Confirmed `busy` transitions from `true` to `false` after assistant output and quiet window.
