# Current HEAD browser/API verification — recovery/product-gaps

Branch: `recovery/product-gaps`  HEAD tested: `13718b2` (working tree has unrelated in-flight edits from a concurrent sibling worker in `codoxear/broker.py`, `codoxear/broker_log_binding.py`, `codoxear/cc_log.py`, `codoxear/rollout_idle.py`, `tests/test_cc_backend_error_projection.py`, `tests/test_claude_backend_source.py`, new `tests/test_broker_cc_fallback_log.py` — **not mine**; I made no source edits).

## Sandbox / isolation

- Used my **own** isolated Docker container (sibling worker occupied the shared `codoxear-sandbox-19130` container/port). My container: `codoxear-verify-19131`, host port `127.0.0.1:19131` (not 8743; 1913x range as instructed).
- Preflight guard: `CODOXEAR_DOCKER_PORT=19131 CODOXEAR_DOCKER_NAME=codoxear-verify-19131 scripts/codoxear-docker-sandbox preflight` → `preflight ok: root=/tmp/codoxear-docker-sandbox-19131 home=/tmp/codoxear-docker-sandbox-19131/home`.
- Image: `codoxear-sandbox:latest` (reused; no network rebuild needed).
- `APP_DIR` inside container: `/home/tester/.local/share/codoxear` (verified via `python3 -c 'from codoxear import server; print(server.APP_DIR)'`). Host live `~/.local/share/codoxear` was never touched.
- Synthetic state was created **inside the sandbox app dir only**: bad sidecars, two Codex log fixtures, and a minimal fake broker (Unix-domain socket server replying to `{"cmd":"state"}` with idle state) so listings/prune treat the rows as live without needing real backend credentials.

## Fixtures (disposable, under sandbox app dir only)

- `~/.local/share/codoxear/logs/no_response.jsonl` — Codex `event_msg` rows: `user_message` ("What is 2+2?") then `task_complete` (no assistant output, no error).
- `~/.local/share/codoxear/logs/with_answer.jsonl` — Codex `event_msg` rows: `user_message` ("Say hi"), `agent_message` ("Hi there!", `phase=final_answer`), `task_complete` with `last_agent_message`.
- Bad sidecars in `socks/`: `bad-json.sock.json` (`{not valid json`) and `bad-fields.sock.json` (`{"agent_backend":"codex"}` — missing required fields).
- Valid synthetic sidecars+sockets: `syn-noresp` (→ no_response log) and `syn-answer` (→ with_answer log), each driven by a fake broker.
- Failed launch row: appended one `launch_attempt` record (`launch_id=fail-test-1`, `state=failed`, `stage=spawn`, `error="broker exited before log bind"`) to `session_launches.jsonl`.

## Goal-by-goal evidence

### (1) Server starts with sandbox preflight guard — PASS
- `pre_login_api_me=401` (server up, auth gate active before login).
- Preflight refuses paths that alias/contain the host live runtime; it passed for the throwaway root and would have refused the live app dir.

### (2) Bad sidecar metadata does not crash listing — PASS
- With `bad-json.sock.json` + `bad-fields.sock.json` (plus `.sock` peers) present, `GET /api/sessions` after login returned **200** with the valid synthetic neighbors still listed (0 valid sessions when only bad sidecars existed; 2 once `syn-*` sidecars were added). Invalid sidecars are skipped, not fatal; valid neighbors remain visible.

### (3) Codex user + task_complete, no answer → explicit no-response transcript message — PASS
- `GET /api/sessions/syn-noresp/messages/tail?limit=20` → 200, `transcript_state=bound`, events:
  - `role=user` text `"What is 2+2?"`
  - `role=assistant`, `message_class=error`, text `"The backend completed this turn without producing a response."`
- Browser (`#session=syn-noresp`): the message renders as `<div class="msg assistant error">The backend completed this turn without producing a response.</div>`.

### (4) Codex agent_message/last_agent_message → assistant message, no false no-response — PASS
- `GET /api/sessions/syn-answer/messages/tail?limit=20` → 200, events:
  - `role=user` text `"Say hi"`
  - `role=assistant`, `message_class=final_response`, text `"Hi there!"`
  - `NO_RESPONSE_EVENTS = 0`
- Browser (`#session=syn-answer`): `<div class="msg assistant">Hi there!</div>` (no `error` class), `hasNoResp=false`, `hasAnswer=true`. No false positive.

### (5) Failed launch row uses failed badge + recovery panel, not a failed-colored state dot — PASS
- API row for `fail-test-1`: `launch_state="failed"`, `launch_error="broker exited before log bind"`, `launch_stage="spawn"`, while `busy=false` and `state_busy=false`. There is no `state="failed"` field; failure is carried by `launch_state`/`launch_error`, and the busy/state-dot projection stays idle.
- Browser DOM after login (failed row auto-selected at `#session=fail-test-1`):
  - `badges=[{cls:"badge launchFailed", text:"failed"}]` — the failed badge is rendered.
  - `dots=[{cls:"stateDot idle"},{cls:"stateDot idle"},{cls:"stateDot idle"}]` — **no** failed-colored state dot; all dots idle. `app.js:3051-3057` builds the dot class from `launchPending ? "pending" : snoozed/blocked ? "suppressed" : busy ? "busy" : "idle"` — `launchFailed` is intentionally absent.
  - Recovery panel visible: buttons `Dismiss launch`, `New like this`, disabled composer (`Failed launch cannot receive messages`), disabled file/unattended/queue controls.

### (6) Mobile viewport can read the transcript/error-styled message — PASS
- `set viewport 390 844` → `window.innerWidth=390, innerHeight=844`.
- On `#session=syn-noresp`: `.msg.assistant.error` element `rect={w:303,h:73,x:10,y:269}`, `fontSize=16px`, fully on-screen (`right ≤ 390`, `x=10`), `readable=true`, `hasNoResp=true`.

## Commands run (exact)

```
# Preflight + start my isolated sandbox
CODOXEAR_DOCKER_PORT=19131 CODOXEAR_DOCKER_NAME=codoxear-verify-19131 scripts/codoxear-docker-sandbox preflight
CODOXEAR_DOCKER_PORT=19131 CODOXEAR_DOCKER_NAME=codoxear-verify-19131 CODOXEAR_DOCKER_IMAGE=codoxear-sandbox:latest scripts/codoxear-docker-sandbox start
curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:19131/api/me          # 401

# Auth + listings/API
curl -c cookies.txt -H 'Content-Type: application/json' -d '{"password":"test-password"}' http://127.0.0.1:19131/api/login
curl -b cookies.txt -o sessions.json -w '%{http_code}\n' http://127.0.0.1:19131/api/sessions                                                  # 200
curl -b cookies.txt -o tail.noresp.json  "http://127.0.0.1:19131/api/sessions/syn-noresp/messages/tail?limit=20"                             # 200
curl -b cookies.txt -o tail.answer.json  "http://127.0.0.1:19131/api/sessions/syn-answer/messages/tail?limit=20"                             # 200

# Browser (agent-browser, ephemeral headless sessions)
agent-browser open http://127.0.0.1:19131/ ; fill password ; click Login
agent-browser open http://127.0.0.1:19131/#session=fail-test-1   # failed-row + recovery panel + stateDot idle
agent-browser open http://127.0.0.1:19131/#session=syn-noresp    # .msg.assistant.error no-response
agent-browser open http://127.0.0.1:19131/#session=syn-answer    # .msg assistant "Hi there!", no error
agent-browser set viewport 390 844 ; reopen #session=syn-noresp  # mobile-readable error message
```

## Artifacts (under `.memory/tasks/2026-07-03-usable-product-ui-architecture/browser-artifacts/`)

Mine (this run):
- `ch-failed-launch-recovery.png` — desktop, failed-launch auto-selected: failed badge + recovery panel, stateDot idle.
- `ch-noresp-transcript.png` — desktop, no-response `assistant.error` transcript row.
- `ch-mobile-noresp-transcript.png` — 390×844 mobile, no-response error row readable.

(Other `ch-*.png`/`wb01-*.png` files in that dir belong to a concurrent sibling worker, not this run.)

API JSON evidence retained under `/tmp/codoxear-docker-sandbox-19131/artifacts/` (`sessions.json`, `tail.noresp.json`, `tail.answer.json`, `sessions.failed.json`, cookies).

## Observations / notes

- The synthetic fake broker was needed because the server prunes any registered session whose `.sock` cannot answer `{"cmd":"state"}` and whose broker/codex pids are dead. The real broker names sidecars `<stem>.json` for a `<stem>.sock` (i.e. `sock_path.with_suffix(".json")` → replaces `.sock`), **not** `<stem>.sock.json`; the fake broker was adjusted to match. This is a reusable recipe for future fixture-based verification.
- The messages read route is `/api/sessions/<id>/messages/tail` (and `/messages/live`); a bare `/messages` returns 404 — not a defect, just the route shape.
- HEAD `13718b2` plus the truthfulness fixes (`d73876c` no-response projection, `eba95e9` remove failed state-dot color, `1dcd31f` clear stale interrupted idle) hold up under isolated browser/API evidence for the six requested items.

## Residual risks / blockers

- None for the six requested checks. All pass from the browser/API in an isolated sandbox.
- Untested here (out of scope for this task): Pi and Claude Code transcript projection with real credentials; live-send end-to-end (requires real backend creds); file/git workbench surfaces. The concurrent sibling worker appears to be exercising some of those (its in-flight source edits and `wb-*` artifacts were left untouched).
- A concurrent sibling worker was sharing the original `codoxear-sandbox-19130` container; I avoided collision by spinning up my own `codoxear-verify-19131` container and have since removed it. The shared container and the sibling's uncommitted edits were not touched.

## Acceptance

All six goals verified PASS with browser DOM + API JSON evidence. No source files edited or staged by this run.
