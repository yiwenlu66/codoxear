All artifacts are in place, the container is removed, and git status shows only the new untracked task-artifacts directory. The verification is complete.

## Summary

I re-ran the same live Docker discriminator as the defect report (`readiness-stale-interrupted-idle-19260`), now against the fixed HEAD `206fb6c Use session interrupt authority for readiness`, and it **PASSES**.

**The fix under test:** `SessionReadinessCoordinator.runtime_status_from_state_and_log` now routes its `BrokerRuntimeState` through `broker_runtime_state_with_session_idle_authority(...)`, which takes the interrupted-idle override from the stored, suppression-aware `Session.interrupted_idle` instead of the raw broker value — closing the divergence where the sidebar projected busy while `/send` and `/enqueue` still delivered.

**Decisive evidence** (Docker `codoxear-sandbox-19264`, port 19264; isolation preflight passed; host runtime and 8743 untouched):

1. **Listing/sidebar busy** while direct broker socket still reports raw `interrupted_idle:true` — `busy:true` on every poll (`phase2-polls.json`, `phaseQ2-polls.json`).
2. **Direct send → HTTP 409** `{"error":"session is busy; wait before sending"}`, broker1 call log **zero `cmd:send`** (195 calls, all `cmd:state`). This inverts the defect (which was HTTP 200 + a confirmed `cmd:send`).
3. **Clean queue (fresh session, before any send) → HTTP 200** `{"queued":true,"queue_len":1}`, item retained `sending:false`; broker2 call log **zero `cmd:send`** (375 calls, all `cmd:state`). This inverts the defect (which promoted + sent).
4. **Browser proof:** both sidebar `stateDot` = `busy` (2/2); selected session's attachment button disabled with title "Wait for the current response to finish before attaching a file"; queue badge `1`; screenshot saved.
5. **Focused validation:** `139 passed, 26 subtests passed`.

**No source edits, no staging, no commits.** Container removed via the exact sandbox `stop`; port 19264 free; no `pkill`/pattern cleanup.
