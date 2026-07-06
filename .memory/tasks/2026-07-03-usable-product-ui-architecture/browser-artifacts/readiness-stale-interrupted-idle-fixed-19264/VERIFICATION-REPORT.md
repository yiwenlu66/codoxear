# Verification Report — readiness honours session interrupt authority after log-only stale `interrupted_idle`

**Result: PASS** — the direct-send and queue-promotion readiness paths now refuse to
deliver input while `/api/sessions` and the sidebar project `busy`. The previously
confirmed divergence (defect `a48ca8e`) is closed at HEAD `206fb6c`.

**Repo under test:** `/home/yiwen/codex-web-product-recovery`, HEAD
`206fb6c Use session interrupt authority for readiness`. No source edits, no
staging, no commits. This is a re-run of the live Docker discriminator documented
in `readiness-stale-interrupted-idle-19260/VERIFICATION-REPORT.md` (DEFECT at
`da7b2ad`), now expected to PASS.

## What changed (the fix under validation)

`SessionReadinessCoordinator.runtime_status_from_state_and_log`
(`session_readiness.py`) previously built its `BrokerRuntimeState` from the **raw**
broker response via `broker_runtime_state(state)`. A stale broker
`interrupted_idle:true` therefore reactivated
`broker.allows_interrupted_idle_override` on the readiness path even after the
listing/log watcher had suppressed it on the `Session` object — so the sidebar
projected `busy` while `/send` and `/enqueue` still delivered.

The fix routes the readiness `BrokerRuntimeState` through the new
`broker_runtime_state_with_session_idle_authority(state, session_interrupted_idle=...)`
(`session_runtime.py`). That helper still validates raw `busy` / `queue_len` /
`interrupted_idle` as the broker sent them (a malformed reply still raises), but
takes the **interrupted-idle override from the stored, suppression-aware
`Session.interrupted_idle`** rather than the raw broker value. Raw broker busy and
queue_len remain authoritative as-sent. There is now exactly one source of truth
for the interrupted-idle override — the same stored flag both the listing path and
the readiness path read.

## PASS conditions (all met)

| # | Condition | Result | Evidence |
|---|-----------|--------|----------|
| 1 | Listing/sidebar busy while raw broker `interrupted_idle:true` | PASS | `phase2-polls.json`/`phaseQ2-polls.json`: `busy:true` on every poll; direct socket probe `interrupted_idle:true` |
| 2 | Direct send returns not-ready with **zero** `cmd:send` | PASS | `phase4-send-result.json`: HTTP **409** `session is busy; wait before sending`; broker1 call log `sends:0` (195 calls, all `cmd:state`) |
| 3 | Queue retains item with **zero** `cmd:send` | PASS | `phaseQ5-enqueue-result.json`: HTTP 200 `{"queued":true,"queue_len":1}`; queue GET item retained `sending:false`; broker2 `sends:0` (375 calls, all `cmd:state`) |
| 4 | Browser projects busy | PASS | DOM: both sidebar `stateDot` = `busy` (2/2); selected session attach button **disabled** "Wait for the current response to finish before attaching a file"; queue badge `1`; screenshot `browser-sidebar-busy.png` |
| 5 | Focused validation | PASS | `validation-output.txt`: **139 passed, 26 subtests passed** in 1.85s |

## Phase-by-phase (exact route statuses / bodies)

### Phase 1 — interrupted non-final log + broker `interrupted_idle:true` → idle
Initial log: `session_meta` + `user_message` + a non-final assistant fragment
(351 bytes). `GET /api/sessions` → `busy:false` (`phase1-sessions.json`). The
immediate interrupt override is valid.

### Phase 2 — append post-interrupt activity; broker STILL `interrupted_idle:true` → busy (precondition reproduced)
Appended `{"type":"event_msg","ts":20.0,"payload":{"type":"user_message",...}}`
(log 351 → 465 bytes). Direct socket probe still
`{"busy":false,"queue_len":0,"interrupted_idle":true}`. `GET /api/sessions` →
`busy:true` on **all three** polls (`phase2-polls.json`). The divergence
precondition holds: sidebar busy while raw broker state carries stale
`interrupted_idle:true`.

### Phase 4 — DIRECT SEND DISCRIMINATOR → PASS
`POST /api/sessions/cert-stale-interrupt/send` with text
`"probe direct send while sidebar busy FIXED"`.

- **HTTP 409** (`phase4-send-result.json`), body
  `{"error": "session is busy; wait before sending"}`.
- Broker1 call log: **`sends:0`** before and after; total 195 calls, **all
  `cmd:state`** — no `cmd:send`, no `cmd:keys`.
- Sidebar at send time: `busy:true`; raw broker: `interrupted_idle:true`.

This is the exact inversion of the defect (which was HTTP 200 +
`{"queued":false,"busy":true}` + one confirmed `cmd:send`).

### Phase 5 — QUEUE (same session, after the refused send) → PASS
`POST /api/sessions/cert-stale-interrupt/enqueue` with text
`"probe queue promotion while sidebar busy FIXED"`.

- **HTTP 200** (`phase5-enqueue-result.json`), body
  `{"queued": true, "queue_len": 1, "item": {...}}` — item **queued, not promoted**.
- Broker1 call log: **`sends:0`** before (total 189) and after (total 195).
- `GET .../queue` (`phase5-queue-get.json`): item retained, `sending:false`,
  `commit_unknown:false`.

### Phase Q — CLEAN QUEUE DISCRIMINATOR (fresh session, queue before any send) → PASS
Fresh session `cert-stale-q` (own broker + log), queue probed **before** any direct
send so no send-boundary can mask the result.

- Phase Q2: appended post-interrupt activity (log 343 → 457 bytes); direct socket
  probe still `interrupted_idle:true`; `GET /api/sessions` → `busy:true` on both
  polls (`phaseQ2-polls.json`).
- `POST /api/sessions/cert-stale-q/enqueue` with text
  `"queue probe on fresh busy session FIXED"` → **HTTP 200**
  `{"queued": true, "queue_len": 1, "item": {...}}` (`phaseQ5-enqueue-result.json`).
- Broker2 call log: **`sends:0`** before (total 369) and after (total 375); all 375
  calls are `cmd:state`.
- `GET .../queue` (`phaseQ5-queue-get.json`): item retained, `sending:false`.

This is the decisive clean-queue inversion of the defect (which promoted and sent:
`{"queued":false,"queue_len":0,"busy":true}` + one `cmd:send`, queue emptied).

## Browser proof

Logged in at `http://127.0.0.1:19264/` (password gate; `agent-browser` ephemeral
session `cert19264`). Selected `#session=cert-stale-interrupt`.

- **Sidebar state dots:** both fake-session cards render `stateDot busy`
  (`browser-statedot-probe.json` / DOM eval: `{total:2, busy:2, idle:0}`).
- **Composer on the busy session:** attachment button `disabled:true`, title
  `"Wait for the current response to finish before attaching a file"`; queued-
  messages badge text `"1"`, title `"Queued messages"`
  (`browser-composer-state.json`).
- Screenshot: `browser-sidebar-busy.png` (57 KB).

The API call log (zero confirmed `cmd:send` while busy) is the decisive signal; the
DOM confirms the user-facing projection matches.

## Harness (Docker only, never port 8743)

- `CODOXEAR_DOCKER_PORT=19264 scripts/codoxear-docker-sandbox smoke`; real
  `codoxear.server` inside `codoxear-sandbox-19264`; container app dir
  `/home/tester/.local/share/codoxear`. Smoke preflight: `pre_login_api_me=401`,
  `post_login_sessions=200`, `container_app_dir=/home/tester/.local/share/codoxear`
  (isolation guard passed; host live runtime untouched).
- `stale_broker_send.py` (SID `cert-stale-interrupt`, call log
  `/tmp/stale_broker_calls.jsonl`) and `stale_broker_q.py` (SID `cert-stale-q`,
  call log `/tmp/stale_broker_calls_q.jsonl`) run **inside the container** as
  `tester`. Each writes a real Unix control socket + sidecar
  (`control_protocol_version:2`, `sync_send`/`key_write_errors`) + an
  interrupted-turn Codex rollout log and answers the real control-socket protocol.
  Both always return raw
  `{"busy":false,"queue_len":0,"interrupted_idle":true}` and record **every**
  command. No product code is stubbed. Scripts reused verbatim from the defect run
  (they reference only container paths).

## Commands run

```
CODOXEAR_DOCKER_PORT=19264 scripts/codoxear-docker-sandbox preflight
CODOXEAR_DOCKER_PORT=19264 scripts/codoxear-docker-sandbox smoke
docker cp stale_broker_send.py codoxear-sandbox-19264:/tmp/   ; docker exec -u tester -d codoxear-sandbox-19264 python3 /tmp/stale_broker_send.py
docker cp stale_broker_q.py   codoxear-sandbox-19264:/tmp/   ; docker exec -u tester -d codoxear-sandbox-19264 python3 /tmp/stale_broker_q.py
bash drive.sh                       # phases 1,2,4,5 (broker 1)
bash drive_q.sh                     # clean queue discriminator (broker 2)
# agent-browser: login -> select cert-stale-interrupt -> DOM eval + screenshot
python3 -m pytest -q tests/test_sessions_pending_log_idle.py tests/test_stale_interrupted_idle.py tests/test_server_queue_persistence.py
CODOXEAR_DOCKER_PORT=19264 scripts/codoxear-docker-sandbox stop
```

## Validation (focused, `validation-output.txt`)

**139 passed, 26 subtests passed** in 1.85s. Up from 135 in the defect report —
the fix shipped regression coverage
(`tests/test_stale_interrupted_idle.py`) that asserts the readiness path honours
the suppressed interrupted-idle override.

## Cleanup

`codoxear-sandbox-19264` removed via the sandbox `stop` command (`docker rm -f`,
exit 0). Container gone; host port 19264 free. No host processes were started for
session work (both fake brokers lived inside the container and were removed with
it). No `pkill`/`killall`/pattern cleanup was used. Host live runtime and port
8743 were not touched.

## `git status --short` (HEAD `206fb6c`)

```
?? .memory/tasks/2026-07-03-usable-product-ui-architecture/browser-artifacts/readiness-stale-interrupted-idle-fixed-19264/
```

Only the new untracked task-artifacts directory. No source edits, no staged files,
no checkout changes. `git status --short` saved to `git-status.txt`.

## Residual notes

- The attachment-injection readiness path shares the same
  `runtime_status_from_state_and_log` basis, so it inherits the fix; it was not
  separately probed (out of scope for this discriminator) but is no longer exposed
  to the stale-override route.
- The fix preserves the invariant that raw broker `busy` and `queue_len` remain
  authoritative; only the interrupted-idle override authority moved to the stored
  session flag, matching the listing path's single source of truth.
- Deterministic: reproduces PASS on every poll once phase 2 / Q2 post-interrupt
  activity has been observed.
