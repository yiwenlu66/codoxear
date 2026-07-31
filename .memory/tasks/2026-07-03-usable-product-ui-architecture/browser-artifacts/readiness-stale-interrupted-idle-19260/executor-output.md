## Result: DEFECT

The risk is **confirmed**. Both the direct-send and queue-promotion readiness paths deliver input to the broker while `/api/sessions` and the sidebar project `busy`, because the readiness path builds `BrokerRuntimeState` from **raw** broker state (which still carries the stale `interrupted_idle:true`) and never consults the `interrupted_idle_suppressed` flag that the listing path honours.

**Decisive evidence (real Docker server, port 19260, broker always reports `interrupted_idle:true`):**

| Discriminator | Sidebar at probe | Route result | Broker call log |
|---|---|---|---|
| Direct send (`/send`) | `busy:true` | **HTTP 200** `{"queued":false,"queue_len":0,"busy":true}` | **1× `cmd:send`** `"probe direct send while sidebar busy"` |
| Queue promotion (`/enqueue`, fresh session, queue-before-send) | `busy:true` | **HTTP 200** `{"queued":false,"queue_len":0}` (item promoted+sent, queue empty after) | **1× `cmd:send`** `"queue probe on fresh busy session"` |

Both broker call logs are saved as raw artifacts. Validation suite passes (135 passed, 26 subtests) but does not cover this readiness divergence — it is a test-suite gap.

**Minimal likely fix target (no edits made):** make `runtime_status_from_state_and_log` in `session_readiness.py` build its `BrokerRuntimeState` from the stored, suppression-aware session flags (as `build_runtime_enriched_session_rows` already does for listing), instead of raw broker state — so a suppressed `interrupted_idle` cannot reactivate `allows_interrupted_idle_override`.
