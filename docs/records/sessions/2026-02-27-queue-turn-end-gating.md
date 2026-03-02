# Session: 2026-02-27 Queue Turn-End Gating

## Focus
Reduce premature queue release by making broker queue drain turn-boundary-driven, with a safer idle fallback.

## Requests
- Keep showing intermediate assistant outputs.
- Prevent queued messages from being injected early when intermediate commentary appears.
- Keep fallback responsive (not overly long) when explicit turn-end markers are missing.

## Actions Taken
- Updated broker queue release to tag queued items with a required turn-end gate and release one item only when that gate is satisfied.
- Counted explicit turn-end markers in broker state and used them as the primary queue-release trigger.
- Added a guarded idle turn-end fallback window via `CODEX_WEB_IDLE_TURN_END_QUIET_SECONDS` (default max of `BUSY_QUIET_SECONDS` and 8s).
- Added phase-aware completion-candidate logic so assistant `phase=commentary` does not mark a turn as ready for idle close.
- Limited quiet idle fallback to cases where queue/key input is pending, so normal multi-stage replies do not flip to idle mid-turn.
- Extended broker busy-state tests for:
  - duplicate turn-end dedupe,
  - commentary vs final-answer completion candidates,
  - queue gate readiness via explicit end and idle fallback,
  - queue-only idle fallback gating.
- Updated feature/testing docs and work records for the new behavior.

## Outcomes
- Queue release now prefers explicit turn-end markers and is less likely to trigger on intermediate commentary-only output.
- Idle fallback remains available for sessions that miss explicit turn-end markers, with a safer quiet window.
- Busy/idle status is less likely to flicker to idle between intermediate assistant updates when no queue is waiting.

## Tests
- `python3 -m unittest tests.test_broker_busy_state`
- `python3 -m unittest discover -s tests`
