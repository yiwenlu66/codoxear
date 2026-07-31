# Verification Report — log-only stale `interrupted_idle` suppression

**Result: PASS** (phases 1–3, API + browser DOM)

**Repo under test:** `/home/yiwen/codex-web-product-recovery`, HEAD `d04511d Record transcript search acceptance`, branch `recovery/product-gaps`. No source edits, no staging, no commits.

**Scope closed:** the verification boundary explicitly left open in `attachment-mobile-noresponse-19200/VERIFICATION-REPORT.md` (Claim 5 note): *the broker keeps reporting `interrupted_idle:true` while post-interrupt log activity arrives on the same log.* That report proved only the broker-state path (broker flips to `busy:true`). This report proves the **log-only** path: the broker never changes its stale `interrupted_idle:true` answer, yet `/api/sessions` and the browser sidebar correctly project `busy`.

## Product invariant under test

Busy/idle is binary. `interrupted_idle` is an override valid only for an interrupted **non-final** log tail. Once post-interrupt user/assistant activity proves the same log resumed, the override is stale and must be suppressed; otherwise `/api/sessions` and the browser would show idle for a live turn. The mechanism: `set_session_interrupted_idle` records the log byte offset (`interrupted_idle_log_off`) at the moment the broker confirms an interrupt; `SessionLogRuntimeCoordinator.update_meta_counters` clears+suppresses the override when it observes user/assistant activity past that baseline; `suppress_session_interrupted_idle` prevents a stale broker `true` from reactivating the override until the broker reports `false` or the session/log resets.

## Harness (Docker only, never port 8743)

- `scripts/codoxear-docker-sandbox smoke` on port **19250**; real `codoxear.server` inside `codoxear-sandbox-19250`; container app dir `/home/tester/.local/share/codoxear` (isolation preflight passed; host live runtime untouched).
- `unified_stale_broker.py` (artifact) runs **inside the container** as `tester`. It writes a real Unix control socket + sidecar JSON (`control_protocol_version:2`, `sync_send`/`key_write_errors`) + an initial interrupted-turn Codex rollout log under the container app dir, and answers the real control-socket protocol (`{"cmd":"state"}` newline-JSON). The server discovers and polls it through its **real** discovery → prune → list → runtime-resolution code paths. No product code is stubbed.
- The broker's `state` response is **always** `{"busy": false, "queue_len": 0, "interrupted_idle": <ctrl>}`. A control file (`/tmp/stale_broker_ctrl`) selects `interrupted_idle`: default/`true` for phases 1–2, `false` for phase 3a, `true` again for phase 3b. The broker's PID is a real live process PID, so the server's `pid_alive` checks pass.
- Log appends use `docker exec -i … cat >> $LOG` so stdin reaches the inner shell. (An earlier broken run omitted `-i`; appends silently no-op'd and the log never grew. That run's artifacts were discarded. The `-i` runs are the evidence of record.)

## Phase 1 — initial interrupted non-final log + broker `interrupted_idle:true` → idle (PASS)

Initial log: `session_meta` + `event_msg/user_message` + a non-final assistant `response_item` fragment (no `task_complete`). Broker `state`: `{"busy":false,"queue_len":0,"interrupted_idle":true}`.

`GET /api/sessions` → `busy: false` (`phase1-sessions.json`). Browser sidebar: `stateDot idle`, `dotBg rgba(107,114,128,0.85)` (`browser-dom.json` `phase1`, screenshot `browser-phase1-idle.png`). The immediate interrupt override is valid: the tail is genuinely non-final.

## Phase 2 — append post-interrupt activity; broker STILL `interrupted_idle:true` → busy, stable (PASS — the core discriminator)

Appended one row to the **same** log: `{"type":"event_msg","ts":20.0,"payload":{"type":"user_message","message":"resumed turn after interrupt"}}`. Log grew 351 → 465 bytes. Direct socket probe confirmed the broker **still** returned `interrupted_idle:true` throughout.

`GET /api/sessions`, 5 consecutive polls ~1 s apart (`phase2-polls.json`):

| poll | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| `busy` | **true** | **true** | **true** | **true** | **true** |

Broker state during every poll: `{"busy": false, "queue_len": 0, "interrupted_idle": true}`.

Browser DOM (`browser-dom.json` `phase2`, screenshot `browser-phase2-busy.png`): sidebar `stateDot busy`, `dotBg rgba(29,78,216,0.95)`; in-page `fetch('/api/sessions')` → `busy:true`. Repoll 2 s later (`phase2_repoll`): `stateDot busy`, `busy:true` — stable, not a one-shot clear.

This is exactly the log-only stale-true scenario: the override was suppressed by `update_meta_counters` observing the post-baseline `user_message`, and `interrupted_idle_suppressed=true` kept the stale broker `true` from reactivating it. **DEFECT condition never observed.**

## Phase 3a (optional) — broker clears the interrupt → suppression clears, busy stays true on a non-idle log (PASS)

Control file flipped to `false`. Broker `state`: `interrupted_idle:false`. With the override gone and the log non-idle (the appended user_message has no completion), `GET /api/sessions` → `busy:true` on both polls (`phase3a-polls.json`). This confirms `set_session_interrupted_idle(false)` clears the suppression flag so a later interrupt can record a fresh baseline.

## Phase 3b (optional) — fresh interrupt at a later offset, then post-arm resume → override re-applies, then suppresses again (PASS)

Appended a fresh interrupted turn (`user_message` + non-final assistant fragment) at offset 465 → 733; re-armed broker `interrupted_idle:true`. `GET /api/sessions` → `busy:false` immediately after re-arm (the override re-applies with a fresh baseline at the current tail — `rearm_busy`). Then appended one more post-arm `user_message` (733 → 855); `GET /api/sessions` → `busy:true` on both subsequent polls (`phase3b-polls.json`), broker still `interrupted_idle:true`. The suppression re-triggered against post-baseline activity.

## Commands run

```
CODOXEAR_DOCKER_PORT=19250 scripts/codoxear-docker-sandbox smoke       # real server, 401 pre-login, 200 post-login
docker cp unified_stale_broker.py … ; docker exec -u tester -d … python3 …   # plant stale broker inside container
bash drive.sh                                                          # phases 1–3 via real /api/sessions
NODE_PATH=/tmp/plain-editor-driver/node_modules node browser_proof2.js # browser DOM, real Chromium 146
python3 -m pytest -q tests/test_stale_interrupted_idle.py tests/test_sessions_pending_log_idle.py tests/test_session_discovery.py
CODOXEAR_DOCKER_PORT=19250 scripts/codoxear-docker-sandbox stop        # exact container teardown
```

Targeted tests: **58 passed, 4 subtests passed** (0.60 s).

## Exact mechanism (confirmed end-to-end through real server)

1. Discovery registers the session with `interrupted_idle=true`; `SessionDiscoveryRegistryCoordinator.upsert_registration` routes the flag through `set_session_interrupted_idle`, which records `interrupted_idle_log_off` = current log size (351).
2. Listing calls `prune_dead_sessions` → `refresh_session_state`, which re-polls the broker (still `true`) and calls `set_session_interrupted_idle(true)` — the helper preserves the existing baseline (does not move it forward past appended activity) because the override is already active with a positive offset.
3. Listing calls `update_meta_counters`. With `interrupted_idle_log_off=351 <= size=465`, it sets `post_baseline` and advances the read cursor to 351; the post-baseline chunk contains the `user_message` chat event → `clear_interrupted_idle=true` → `suppress_session_interrupted_idle` sets `interrupted_idle=false`, `interrupted_idle_log_off=0`, `interrupted_idle_suppressed=true`.
4. `build_runtime_enriched_session_rows` builds `BrokerRuntimeState(busy=false, interrupted_idle=false)` (the stored override is now false) over a non-idle log → `resolve_runtime_status` returns `busy=true`. The public row strips `interrupted_idle`; only `busy=true` is visible.
5. On every subsequent poll the broker still reports `true`, but `set_session_interrupted_idle` respects `interrupted_idle_suppressed` and refuses to reactivate the override, so `busy` stays `true`.

## Residual notes

- No source/test code changed. The harness scripts live only under the artifact dir and `/tmp`.
- Browser screenshots are real Chromium captures; the DOM/JSON summaries alongside them are the primary structured evidence (images were not OCR-verified, per the same convention as the prior report).
- The earlier no-op-append run (omitted `docker exec -i`) produced a false DEFECT; it was diagnosed as a harness stdin bug (log size never changed) and discarded before any conclusion was drawn. Only the `-i` runs are evidence of record.

## Cleanup

`codoxear-sandbox-19250` removed via the sandbox `stop` command. No host processes were started for session work (the fake broker lived inside the container). Pre-existing containers (`codoxear-inline-repro`, `codoxear-session-test`) and the live port-8743 server were not touched. No `pkill`/pattern cleanup was used.


## Artifact normalization note

`phase2-polls.raw.txt`, `phase3a-polls.raw.txt`, and `phase3b-polls.raw.txt` preserve the original shell-produced one-line captures. Those captures embedded broker JSON strings without escaping, so the committed `*.json` files were normalized to equivalent valid JSON objects for machine readability. The normalized values match the raw captures and the report: phase 2 has five `busy:true` polls while broker state remains `interrupted_idle:true`; phase 3a has broker `interrupted_idle:false`; phase 3b has re-arm `busy:false` then two post-arm `busy:true` polls while broker returns stale true.


## Clean-room acceptance

Fresh critic `ce77e902-0e96-4364-a21c-3699e45b8ace` returned ACCEPT with no blockers. The critic confirmed the evidence proves the named log-only stale `interrupted_idle` boundary through a real Docker Codoxear server and real `/api/sessions`, with the fake broker limited to the necessary experimental condition. It also accepted the raw-to-normalized JSON artifact handling.

Boundaries preserved by the critic: browser replay is less self-contained than the API proof, but the API phase contains the decisive broker/log/poll evidence; the proof closes `/api/sessions` plus sidebar busy projection for this boundary, not every busy-derived affordance or provider-real interrupt behavior.
