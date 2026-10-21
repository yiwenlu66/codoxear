# OPS — undelivered verdict fires on a delivered first message

## 2026-10-22T04:10Z — mechanism traced in source

`~/.npm-global/lib/node_modules/@earendil-works/pi-coding-agent/dist/core/session-manager.js`:
`newSession()` sets `flushed = false`; `_persist()` writes nothing while
`fileEntries` holds no assistant message; the file is created with
`openSync(sessionFile, "wx")` only when the first assistant entry arrives.
`codoxear/pi_active_session_bridge.ts::materializeNewSession` runs only for
`session_start` reason `new`.
`codoxear/broker_turn_state.py::_update_busy_from_pty_text` detected Pi's
`Working...` / OSC `9;4;3` indicator but spent it only on
`_clear_pi_error_probe`, never on `last_turn_activity_ts`.

## 2026-10-22T04:25Z — pre-fix reproduction (subagent, Docker, HEAD 0c3817f7)

`CODOXEAR_UDV_ARTIFACTS=/tmp/udv-art2 bash scripts/docker_undelivered_send_verify.sh HEAD`
-> exit 1; scenarios 1–4 pass, scenario 5 (new) fails.

| event | t (s after Send) |
| --- | --- |
| slow provider receives request | 0.276 |
| row carries `send_verdict` (payload ts) | 3.238 |
| Pi `.jsonl` first appears, already containing the user row | 6.548 |
| row `log_path` bound | 7.17 |
| composer restored + toast | 8.288 |
| composer at 25 s | still the mark |

Pi `.jsonl` for that session did not exist at send time; `log_path` was null.

## 2026-10-22T04:40Z — fix implemented

- `broker_turn_state`: working indicator bumps `busy` + `last_turn_activity_ts`;
  new `send_latch_window_start_ts` / `mark_backend_ready`; `State.backend_ready_ts`.
- `broker_log_binding`: a bound log marks readiness.
- `broker_launch._pi_bridge_session_is_live`: pid-validated live bridge marker.
- `broker._refresh_pi_active_session_observability` and sessiond's capability
  refresh mark readiness from that marker.

Rejected mid-design: a 30 s pre-readiness deadline. It flipped 15 existing
latch tests (every state with no recorded readiness) and delayed the
swallowed-prompt restore on paths where the backend was already live; the
readiness anchor alone covers the observed cases and the boot-window gap is
recorded as a boundary in EPISTEMIC.md.

## 2026-10-22T04:55Z — unit suite

`.venv/bin/python -m pytest tests/ -q` -> 2293 passed, 112 subtests, 58 s.
New: `tests/test_pi_bridge_readiness.py` (marker reader + broker wiring) and
seven state-machine tests in `tests/test_broker_send_verdict.py`.

## 2026-10-22T05:05Z — committed

`8f3b4877` Count Pi's working indicator and backend readiness as send-latch evidence

## 2026-10-22T05:12Z — post-fix Docker verification (real Pi, real browser)

`CODOXEAR_UDV_ARTIFACTS=/tmp/udv-fix bash scripts/docker_undelivered_send_verify.sh 8f3b4877`
-> exit 0, all five scenarios pass.

Scenario 5: `.jsonl` absent at send (`existed_at_send_time: false`,
`row_log_path_at_send_time: null`), provider request 0.266 s, file + user row
6.489 s, `verdict_payload: null`, `final_composer_value: ""` — the prompt was
delivered and never returned.

Scenarios 1–4 (positive control): gate swallows the prompt, verdict 3.742 s
(3.328 s pre-fix; the readiness anchor adds the post-gate boot), browser
restores the mark with the toast, and the delivered second send publishes no
verdict of its own.
