# Verification Report — send/queue/readiness divergence after log-only stale `interrupted_idle` suppression

**Result: DEFECT** (both the direct send and the queue promotion readiness paths
deliver input while `/api/sessions` and the sidebar project `busy`)

**Repo under test:** `/home/yiwen/codex-web-product-recovery`, HEAD `da7b2ad Record log-only interrupted idle acceptance`. No source edits, no staging, no commits.

## The risk under test

The prior proof (`log-only-stale-interrupted-idle-19250/VERIFICATION-REPORT.md`,
HEAD `3f0a886/da7b2ad`) closed only the **listing/sidebar** surface: when a
fake broker keeps reporting `interrupted_idle:true` after same-log post-interrupt
activity, `update_meta_counters` suppresses the override on the `Session` object
(`interrupted_idle_suppressed=true`) so `GET /api/sessions` correctly reports
`busy:true`.

This report probes the **divergence** predicted by source inspection:

> `SessionReadinessCoordinator.send_remote_ready()` / `queue_remote_ready()` call
> `get_state()` and build the runtime from **raw** `broker_runtime_state(state)`.
> That raw state still carries the stale `interrupted_idle:true`. The listing
> suppression lives on the `Session` object after `update_meta_counters()` /
> `suppress_session_interrupted_idle()` and is **never consulted** by the
> readiness path. Therefore the direct-send and queue-promotion routes can
> proceed while the sidebar shows busy.

## Exact mechanism (code, confirmed end-to-end through the real server)

1. `SessionControlCoordinator.get_state()` (`session_control.py:42`) re-polls the
   broker socket, applies `set_session_interrupted_idle(current, interrupted_idle)`
   to the `Session` (which respects `interrupted_idle_suppressed` and keeps the
   stored flag `false`), **but returns the raw broker response**, which still
   contains `interrupted_idle:true`.
2. `SessionReadinessCoordinator.remote_state_after_metadata_probe()` returns that
   raw `state`. `send_remote_ready` / `queue_remote_ready` then call
   `runtime_status_from_state_and_log(session_id, state, log_path)`.
3. That helper builds `broker = broker_runtime_state(state)` from the **raw**
   state → `BrokerRuntimeState(busy=false, queue_len=0, interrupted_idle=true)`
   → `broker.allows_interrupted_idle_override == True`.
4. With the log non-idle after the post-interrupt user row, `log_idle=False`,
   but `resolve_runtime_status` ORs `log_idle` with
   `broker.allows_interrupted_idle_override`:
   - `busy = not (False or True)` = `False`
   - `remote_ready = True` (the override bypasses the `log_idle is not True`
     branch)
5. `session_runtime_readiness(...).direct_send` / `.queue_promotion` both return
   `True`. The send coordinator opens a confirmed `cmd:send` to the broker.

The listing path does not see this because `build_runtime_enriched_session_rows`
builds its `BrokerRuntimeState` from `it.get("interrupted_idle")` — the stored
**suppressed** value (`false`) — not from raw broker state.

## Harness (Docker only, never port 8743)

- `scripts/codoxear-docker-sandbox smoke` on port **19260**; real
  `codoxear.server` inside `codoxear-sandbox-19260`; container app dir
  `/home/tester/.local/share/codoxear` (isolation preflight passed; host live
  runtime untouched).
- `stale_broker_send.py` runs **inside the container** as `tester`, writes a real
  Unix control socket + sidecar (`control_protocol_version:2`,
  `sync_send`/`key_write_errors`) + an interrupted-turn Codex rollout log, and
  answers the real control-socket protocol. Its `state` is **always**
  `{"busy":false,"queue_len":0,"interrupted_idle":true}`. It records **every**
  command (cmd + req + resp) to `/tmp/stale_broker_calls.jsonl` so we can prove
  whether the server reached a confirmed `cmd:send`. No product code is stubbed.
- A second instance `stale_broker_q.py` (SID `cert-stale-q`, call log
  `/tmp/stale_broker_calls_q.jsonl`) is used for the clean queue discriminator so
  the queue probe runs **before** any direct send leaves a send-boundary.

## Phase 1 — interrupted non-final log + broker `interrupted_idle:true` → idle

Initial log: `session_meta` + `user_message` + a non-final assistant fragment.
`GET /api/sessions` → `busy:false` (`phase1-sessions.json`). The immediate
interrupt override is valid.

## Phase 2 — append post-interrupt activity; broker STILL `interrupted_idle:true` → busy (the listing suppression works)

Appended `{"type":"event_msg","ts":20.0,"payload":{"type":"user_message",...}}`.
Log grew 351 → 465. Direct socket probe: broker still
`{"busy":false,"queue_len":0,"interrupted_idle":true}`.
`GET /api/sessions` → `busy:true` on all polls (`phase2-polls.json`). This
reproduces the prior report and confirms the divergence precondition: **sidebar
busy while raw broker state carries stale `interrupted_idle:true`.**

## Phase 4 — DIRECT SEND DISCRIMINATOR → DEFECT

`POST /api/sessions/cert-stale-interrupt/send` with text
`"probe direct send while sidebar busy"`.

- **HTTP 200** (`phase4-send-result.json`), body
  `{"queued": false, "queue_len": 0, "busy": true}` — the confirmed-send
  acceptance echoed back from the broker.
- Broker call log: **one `cmd:send`** at ts `1783323316.1535254`,
  `req.text = "probe direct send while sidebar busy"`, `sync: true`
  (`broker1-send-calls.json`).
- Sidebar at the moment of send: `busy:true`; broker raw state:
  `interrupted_idle:true`.

**Expected PASS would have been HTTP 409 `session is busy; wait before sending`
with zero `cmd:send` calls.** The send route delivered input to the broker while
the sidebar showed busy.

## Phase 5 — QUEUE DISCRIMINATOR → DEFECT (clean re-run, queue before any send)

`POST /api/sessions/cert-stale-q/enqueue` with text
`"queue probe on fresh busy session"` on a **fresh** session (`cert-stale-q`,
its own broker + log), queue probed **before** any direct send so no leftover
send-boundary could mask the result.

- Sidebar at enqueue time: `busy:true` (`phaseQ2-polls.json` both polls true);
  broker raw state: `interrupted_idle:true`.
- **HTTP 200**, body `{"queued": false, "queue_len": 0, "busy": true}`
  (`phaseQ5-enqueue-result.json`) — i.e. the item was **promoted and sent**
  (`queue_len` returned to 0), not left queued.
- Broker call log: **one `cmd:send`** at ts `1783323483.3144681`,
  `req.text = "queue probe on fresh busy session"` (`broker2-send-calls.jsonl`).
- `GET .../queue` after enqueue: `{"items": [], "queue": []}` — the item was
  popped after the promoted send (`phaseQ5-queue-get.json`).

**Expected PASS would have been the item remaining queued (`queue_len:1`,
`sending:false`) with zero `cmd:send` calls.** The queue path promoted and sent
while the sidebar showed busy.

Note on the first queue attempt (broker 1, after the phase-4 send): the item
correctly stayed queued there, but only because phase 4's send left an
unresolved send-boundary (`apply_confirmed_send_success` set
`last_send_boundary_active=true`; the fake broker never wrote to the log, so
`confirmed_send_boundary_unresolved` stayed true → `remote_ready=False`). That
run is reported for completeness but is **not** the clean queue evidence; the
clean re-run above is decisive.

## Minimal likely fix target (no code edits performed)

The readiness path must honour the same suppression the listing path honours.
`resolve_runtime_status` / `runtime_status_from_state_and_log` currently derives
the interrupted-idle override from **raw** broker state
(`broker.allows_interrupted_idle_override`). Two viable fix loci (independent
reviewer to choose):

1. In `session_readiness.py`, build the `BrokerRuntimeState` fed into
   `runtime_status_from_state_and_log` from the **stored, suppression-aware**
   session flags (mirroring `build_runtime_enriched_session_rows`, which uses
   `it.get("interrupted_idle")` rather than raw broker state), so a suppressed
   override cannot reactivate `allows_interrupted_idle_override`.
2. Alternatively, have `get_state()` return a state whose `interrupted_idle`
   reflects `set_session_interrupted_idle`'s suppression-aware result, or teach
   `resolve_runtime_status` / `BrokerRuntimeState` to consult
   `session.interrupted_idle_suppressed`.

Option 1 is the smallest invariant-preserving change: there should be exactly
one source of truth for the interrupted-idle override, and both the listing and
readiness paths must read the stored (suppressed) value rather than raw broker
state.

## Commands run

```
CODOXEAR_DOCKER_PORT=19260 scripts/codoxear-docker-sandbox smoke
docker cp stale_broker_send.py codoxear-sandbox-19260:/tmp/   ; docker exec -u tester -d codoxear-sandbox-19260 python3 /tmp/stale_broker_send.py
bash drive.sh                       # phases 1,2,4,5 (broker 1)
docker cp stale_broker_q.py   codoxear-sandbox-19260:/tmp/   ; docker exec -u tester -d codoxear-sandbox-19260 python3 /tmp/stale_broker_q.py
bash drive_q.sh                     # clean queue discriminator (broker 2)
python3 -m pytest -q tests/test_stale_interrupted_idle.py tests/test_sessions_pending_log_idle.py tests/test_server_queue_persistence.py
CODOXEAR_DOCKER_PORT=19260 scripts/codoxear-docker-sandbox stop
```

Validation (local): **135 passed, 26 subtests passed** in 2.43 s. The existing
suite asserts the listing-suppression invariant but contains no probe of the
readiness-path divergence; this defect is a gap the suite does not cover.

## Cleanup

`codoxear-sandbox-19260` removed via the sandbox `stop` command. No host
processes were started for session work (both fake brokers lived inside the
container). No `pkill`/pattern cleanup was used. Host live runtime and port 8743
were not touched.

## Residual notes

- API proof is decisive; browser automation was not needed (the API call log is
  a strictly stronger signal than a DOM affordance check — it proves the broker
  actually received the send, not just that a UI element was clickable).
- Unattended injection was not separately probed; it shares the same
  `runtime_status_from_state_and_log` / `session_runtime_readiness(remote_ready)`
  basis (`attachment_injection_ready`), so it is very likely also affected, but
  is out of scope for this discriminator.
- The defect is deterministic and reproduces on every poll once phase 2's
  post-interrupt activity has been observed.
