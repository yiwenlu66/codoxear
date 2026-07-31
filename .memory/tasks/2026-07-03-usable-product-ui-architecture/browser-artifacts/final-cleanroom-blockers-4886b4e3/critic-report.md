BLOCKER.

Current HEAD is not acceptable against the product contract.

## Blocker 1 — Claude Code live polling can still end with no visible result

Mechanism: `codoxear/message_routes.py:372-379` uses `_codex_prior_open_turn_context()` in the public `/messages/live` route. The CC fix added CC-aware split-live support in `_read_chat_live_delta()`, but the route bypasses that helper.

If a Claude Code `user` row is delivered in one poll and a later `system/subtype:turn_duration` close arrives in the next poll, the live route has `prior_user_byte=None`, so `_extract_positioned_chat_events()` sees only the close and injects no no-response event.

User-visible failure: the browser can show the user prompt, go idle, and show no assistant/error/no-response message during normal live polling.

Reproduction observed with a temp CC log:

- `_read_chat_live_delta()` helper: emits no-response.
- route-equivalent logic: returns `[]`.

Minimal fix target: make `handle_messages_live()` use the CC-aware prior context (`_prior_open_turn_context`) or delegate to `_read_chat_live_delta()`. Add a route-level live-split regression.

## Blocker 2 — fresh discovery drops interrupted-idle truth

Mechanism: `codoxear/session_discovery_registry.py:91-103` constructs a new `Session` with `registration.interrupted_idle`, then immediately calls `reset_log_caches()`, which clears `interrupted_idle`.

For a fresh server/discovery path where the broker reports `busy=false, interrupted_idle=true` over a non-final log, the stored session loses the override and `/api/sessions` projects `busy=true`.

Observed reproduction:

```text
after_new_discovery interrupted_idle= False baseline= 0 meta_log_off= 312
listing busy= True status= None interrupted_idle_after_list= False
```

User-visible failure: after restart/fresh rediscovery, an interrupted stopped turn can appear busy/spinning even though the broker reports idle-interrupted.

Minimal fix target: in the new-registration branch, reset caches first, then route `registration.interrupted_idle` through `set_session_interrupted_idle(session, ...)`; add an empty-registry discovery/listing regression.

## Evidence boundaries

The preserved artifacts are good for their exact claims:

- stale discovery-refresh fix: existing-session refresh now projects phase-2 busy true.
- CC outcome proof: completed-log/tail DOM rendering works.
- mobile file dpad: 44×44 CSS target proof holds.

They do not cover the two blockers above.

`git status --short`: no output.
No staged files.
