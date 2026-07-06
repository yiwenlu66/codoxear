# Verification Report — unattended injection refuses the stale broker `interrupted_idle` override while busy

**Result: PASS** — the real server unattended sweep does **not** inject while
`/api/sessions` (sidebar) reports busy, even when the raw broker control socket
keeps returning `{busy:false, queue_len:0, interrupted_idle:true}`. The fix that
routed every readiness consumer through `broker_runtime_state_with_session_idle_authority`
(proven for direct-send and queue in `readiness-stale-interrupted-idle-fixed-19264`)
extends cleanly to the unattended-injection readiness path. This closes the last
remaining boundary called out by the review.

**Repo under test:** `/home/yiwen/codex-web-product-recovery`, HEAD
`206fb6c`. No source edits, no staging, no commits. Follow-on to fixed proof
`6710363`/`206fb6c` and review `25c2b9d`.

## The boundary under test

`UnattendedSweepCoordinator.sweep` (`unattended_sweep.py`) gates each enabled
session through, in order:

1. enabled / `remaining_injections > 0` / log exists / cooldown-not-blocked
2. `broker_state = get_state(sid)` — the **raw** broker response
3. `runtime = runtime_status_from_state(sid, broker_state, log_path)`
4. `if not session_runtime_readiness(runtime, local_queue_len).unattended_injection: continue`
5. tail gate: `last_chat_role_ts_from_tail(..., final_assistant_only=True)`
6. `send(sid, prompt)` + `record_unattended_success` (decrements remaining)

The decisive gate is **#4**. `runtime_status_from_state` is wired
(`session_manager_factories.py:383`) to `manager._runtime_status_from_state_and_log`,
which builds its `BrokerRuntimeState` via
`broker_runtime_state_with_session_idle_authority(state, session_interrupted_idle=...)` —
taking the interrupted-idle override from the stored, suppression-aware
`Session.interrupted_idle`, **not** the raw broker value. If this helper were
bypassed (the defect shape), the stale raw `interrupted_idle:true` would
reactivate `broker.allows_interrupted_idle_override`, `remote_ready` would flip
true, `unattended_injection` would be true, and the sweep would inject while the
sidebar projects busy.

## Why this discriminator isolates gate #4 (and nothing else)

A naive post-interrupt `user_message` row (as used in the send/queue
discriminator) makes the sweep skip at the **tail** gate (#5):
`_last_chat_role_ts_from_tail(final_assistant_only=True)` would return
`("user", …)` and `unattended_tail_allows_injection` returns false for any role
other than `assistant`. That is a false PASS — it never exercises gate #4.

To make the tail gate pass and leave readiness as the **sole** blocker, the log
is constructed so that:

- `_compute_idle_from_log` → **False** (the session is genuinely busy from the
  log's perspective), and
- `_last_chat_role_ts_from_tail(final_assistant_only=True)` →
  `("assistant", task_complete_ts)` (so the tail gate passes — assistant, old ts
  well past cooldown).

The post-interrupt append that achieves both at once is:

```
{"type":"event_msg","ts":12.0,"payload":{"type":"task_complete","last_agent_message":"done"}}
{"type":"event_msg","ts":13.0,"payload":{"type":"agent_reasoning","message":"reasoning resumed"}}
```

`agent_reasoning` is busy in `_compute_idle_from_log` (it sets `idle=False`) and
is the last record, so idle=False. But `agent_reasoning` is **not** a chat role in
`_last_chat_role_ts_from_tail`, so the latest final chat role remains the
`task_complete` at ts=12.0. The earlier interrupted turn also suppresses the
stored override: `agent_reasoning` produces `delta_thinking > 0` in
`update_meta_counters`, which trips `suppress_session_interrupted_idle`.

### Pre-run falsifiable prediction (made before the live run, verified locally)

```
full log:
  _compute_idle_from_log = False (expect False = busy)
  _last_chat_role_ts_from_tail = ('assistant', 12.0) (expect ('assistant', 12.0))
FIXED (session-authoritative interrupted_idle=False):
  busy = True  remote_ready = False  unattended_injection = False   <- sweep skips at #4
DEFECT (raw interrupted_idle=True override):
  busy = False  remote_ready = True  unattended_injection = True    <- sweep injects
```

This proves the live result is decided entirely by which `BrokerRuntimeState`
the readiness path builds. The tail gate passes either way; only gate #4 differs.

## PASS conditions (all met)

| # | Condition | Result | Evidence |
|---|-----------|--------|----------|
| 1 | Raw broker stale `interrupted_idle:true` held throughout | PASS | phaseA/phaseB/phaseD broker probes all `{"busy":false,"queue_len":0,"interrupted_idle":true}` |
| 2 | Listing/sidebar busy while raw broker stale | PASS | phaseB poll 1 `busy:true`; phaseD final `busy:true` |
| 3 | Unattended enabled with remaining=1 via real API | PASS | phaseC POST HTTP 200 `enabled:true,remaining_injections:1,cooldown_minutes:1` |
| 4 | Real sweep ran (≥1 sweep reached the broker) | PASS | call log grew 208→254 across the 12s window (+46 `cmd:state`) |
| 5 | **Zero `cmd:send`/`cmd:keys`** while busy | PASS | broker call log: 254 calls, **100% `cmd:state`**, sends=0, keys=0 |
| 6 | `remaining_injections` NOT decremented, enabled NOT disabled | PASS | phaseD-unattended-final `enabled:true,remaining_injections:1` |
| 7 | Listing still busy after sweeps | PASS | phaseD-sessions-final `busy:true` |
| 8 | Browser projects busy + unattended badge | PASS | DOM: `stateDot busy`, `badge unattended`, attach button disabled "Wait for the current response to finish…" |
| 9 | Focused validation | PASS | 66 passed, 4 subtests passed in 0.53s |

## Phase-by-phase (exact route statuses / bodies)

### Phase A — interrupted baseline + broker `interrupted_idle:true` → idle
Initial log (352 B): `session_meta` + `user_message` + non-final assistant
fragment. Direct socket probe `{"busy":false,"queue_len":0,"interrupted_idle":true}`
(`phaseA-broker-state.json`). `GET /api/sessions` → `busy:false`,
`unattended_enabled:false`, `unattended_remaining_injections:10`
(`phaseA-sessions.json`). The immediate interrupt override is valid — this is the
precondition for the stale-true divergence.

### Phase B — append task_complete(old) + agent_reasoning(later); broker STILL stale → busy
Appended the two records above (log 352 → 576 B). Direct socket probe **still**
`{"busy":false,"queue_len":0,"interrupted_idle":true}`
(`phaseB-broker-state.json`). `GET /api/sessions` → `busy:true` on poll 1
(`phaseB-polls.json`, `phaseB-sessions-final.json`). The divergence holds: sidebar
busy while raw broker state carries stale `interrupted_idle:true`.

### Phase C — enable unattended via the real API
`GET /api/sessions/cert-unattended-stale/unattended` before →
`{enabled:false,remaining_injections:10,cooldown_minutes:5}`.
`POST /api/sessions/cert-unattended-stale/unattended` with
`{enabled:true,request:"unattended stale busy probe",cooldown_minutes:1,remaining_injections:1}`
→ **HTTP 200** `{"ok":true,"enabled":true,"request":"unattended stale busy
probe","cooldown_minutes":1,"remaining_injections":1}`
(`phaseC-unattended-enable.json`). GET after → same
(`phaseC-unattended-after.json`).

### Phase D — let the REAL unattended sweep run (decisive)
`UNATTENDED_SWEEP_SECONDS=2.5`. Call log before the wait: 208 calls, all
`cmd:state`, sends=0, keys=0. Waited 12s (~5 sweep cycles). Call log after:
**254 calls, all `cmd:state`, sends=0, keys=0** (`phaseD-calllog-after.json`).
`GET .../unattended` final → `enabled:true,remaining_injections:1` (NOT
decremented, NOT disabled). `GET /api/sessions` final → `busy:true`,
`unattended_enabled:true`, `unattended_remaining_injections:1`
(`phaseD-sessions-final.json`). Raw broker still
`{"busy":false,"queue_len":0,"interrupted_idle:true}`.

**Why this is decisive, not coincidental:** with `cooldown_minutes:1` and a
never-injected session, gates #1–#3 (enabled/remaining/log) and the cooldown
checks all pass, and the tail gate (#5) passes by construction
(`("assistant", 12.0)`, `now-ts` ≫ 60s). The sweep therefore reaches gate #4 on
every cycle. Under the defect, the very first cycle would inject (raw override
reactivates `unattended_injection`), call `send`, then
`record_unattended_success` → `remaining_injections` 1→0 and `enabled`→false.
Neither happened across ~5 cycles. The only mechanism consistent with
zero-sends + remaining-still-1 + busy-held is gate #4 blocking via the
session-authoritative (suppressed) interrupted-idle value.

The full broker call log (`broker-call-log.jsonl`, 257 lines) is **exactly**
`{"cmd":"state",…}` repeated — never a `cmd:send` or `cmd:keys`.

## Browser proof

Logged in at `http://127.0.0.1:19268/` (password gate; ephemeral
`agent-browser` session `uai19268`). Selected `#session=cert-unattended-stale`.

- **State dot:** selected session renders `span.stateDot busy`; `busyCount:1`,
  `idleCount:0` (`browser-dom-probe.json`).
- **Unattended badge:** `span.badge.unattended` text `"unattended"` present on
  the selected card; "Unattended mode" button available in the menu
  (`browser-dom-snapshot.txt`).
- **Composer on the busy session:** attachment button `disabled:true`, title
  `"Wait for the current response to finish before attaching a file"`
  (`browser-dom-probe.json`).
- Screenshot: `browser-sidebar-busy-unattended.png` (49 KB).

The API call log (zero `cmd:send` while busy + remaining never decremented) is
the decisive signal; the DOM confirms the user-facing projection and the
unattended badge both reflect the busy/enabled state.

## Harness (Docker only, never port 8743)

- `CODOXEAR_DOCKER_PORT=19268 scripts/codoxear-docker-sandbox smoke`; real
  `codoxear.server` inside `codoxear-sandbox-19268`; container app dir
  `/home/tester/.local/share/codoxear`. Smoke preflight: `pre_login_api_me=401`,
  `post_login_sessions=200`, `container_app_dir=/home/tester/.local/share/codoxear`
  (isolation guard passed; host live runtime untouched).
- `fake_broker.py` (SID `cert-unattended-stale`, call log
  `/tmp/unattended_broker_calls.jsonl`) runs **inside the container** as
  `tester`. It writes a real Unix control socket + sidecar
  (`control_protocol_version:2`, `sync_send`/`key_write_errors`) + an
  interrupted-turn Codex rollout log and answers the real control-socket
  protocol. It always returns raw
  `{"busy":false,"queue_len":0,"interrupted_idle":true}` and records **every**
  command. No product code is stubbed.
- `drive.sh` drives phases A–D through the real HTTP API and the direct broker
  socket, enabling unattended via the real `POST .../unattended` route and
  waiting for the real sweep loop.

## Commands run

```
CODOXEAR_DOCKER_PORT=19268 scripts/codoxear-docker-sandbox preflight
CODOXEAR_DOCKER_PORT=19268 scripts/codoxear-docker-sandbox smoke
docker cp fake_broker.py codoxear-sandbox-19268:/tmp/
docker exec -u tester -d codoxear-sandbox-19268 python3 /tmp/fake_broker.py
bash drive.sh                                    # phases A-D + verdict
# agent-browser (ephemeral uai19268): login -> select cert-unattended-stale -> DOM eval + screenshot
python3 -m pytest -q tests/test_unattended_sweep.py tests/test_sessions_pending_log_idle.py tests/test_stale_interrupted_idle.py
CODOXEAR_DOCKER_PORT=19268 scripts/codoxear-docker-sandbox stop
```

## Validation (focused, `validation-output.txt`)

**66 passed, 4 subtests passed** in 0.53s. Includes
`tests/test_unattended_sweep.py` (unit coverage of the sweep gating) and
`tests/test_stale_interrupted_idle.py` (the readiness path honours the
suppressed override).

## Cleanup

`codoxear-sandbox-19268` removed via the sandbox `stop` command (`docker rm -f`,
exit 0). Container gone; host port 19268 free. No host processes were started for
session work (the fake broker lived inside the container and was removed with
it). No `pkill`/`killall`/pattern cleanup was used. Host live runtime and port
8743 were not touched.

## `git status --short` (HEAD `206fb6c`)

Only the new untracked task-artifacts directory. No source edits, no staged
files, no checkout changes (saved to `git-status.txt`).

## Residual notes

- The unattended sweep reuses the **same** `runtime_status_from_state_and_log`
  basis as direct-send, queue-promotion, and attachment-injection readiness, so
  the single-source-of-truth fix (`Session.interrupted_idle` authoritative for
  the override) covers all four consumers. This run proves the unattended
  consumer end-to-end against a live stale broker.
- Deterministic: reproduces PASS on every sweep once phase B post-interrupt
  activity has been observed (the listing goes busy on the first poll after the
  append).
- The result is independent of `UNATTENDED_SWEEP_SECONDS`; the 12s window only
  guarantees multiple sweep cycles. A single post-enable sweep would already
  suffice to separate PASS (gate #4 blocks) from DEFECT (first cycle injects and
  decrements).
