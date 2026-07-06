# Final clean-room review blockers — HEAD 3427fef

Verdict: BLOCKER.

The critic found two product-contract contradictions not covered by prior artifacts.

## Blocker 1: Claude Code split live polling can still end with no visible result

Mechanism:
- Public `/api/sessions/<id>/messages/live` uses `_codex_prior_open_turn_context()` when a poll starts after byte 0.
- The earlier CC transcript fix added CC-aware split-context logic in `_read_chat_live_delta()` / `_prior_open_turn_context()`, but the public route bypasses it.
- A CC user row delivered in one poll and a later `system/subtype:turn_duration` close delivered in the next poll can produce no assistant/error/no-response event.

User-visible failure:
- Browser can show the user prompt, go idle, and render no visible result.

## Blocker 2: fresh discovery drops interrupted-idle truth

Mechanism:
- New-session discovery constructs a `Session` with `registration.interrupted_idle`, then immediately calls `reset_log_caches()`, clearing that flag and baseline.
- Existing-session refresh uses the helper after `f5b4710`; fresh registry insertion still bypasses the helper's final state.

User-visible failure:
- After server restart/fresh rediscovery, an interrupted stopped turn can appear busy/spinning even though broker reports idle-interrupted.

Implication:
Prior evidence remains valid for its exact claims, but HEAD is not accepted until these two route/state gaps have regressions and fixes.
