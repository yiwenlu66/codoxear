# Copy Conversation count truthfulness browser proof

Artifact directory: `.memory/tasks/2026-07-07-copy-conversation-count-truth/browser-artifacts/copy-count-19518/`

## Result

PASS. Docker unit/smoke and real-browser proof on port `19518` show the Copy Conversation toast counts copied sections, not raw export events.

## Required assertions

Source: `raw/required-assertions.json`.

- Desktop browser: `/messages/export` returned 4 raw events; clipboard contained exactly 2 sections; toast was `Copied 2 messages`; no horizontal overflow.
- Mobile browser at `390x844`: `/messages/export` returned 4 raw events; clipboard contained exactly 2 sections; toast was `Copied 2 messages`; no horizontal overflow.
- Clipboard exact payload was:

```text
## User (1/1/1970, 8:00:01 AM)

first copied user turn

---

## Assistant (1/1/1970, 8:00:03 AM)

assistant copied answer
```

- Filtered blank user/assistant rows were absent from the clipboard.
- Broker call summary: `send=0`, `keys=0`, `shutdown=0`; only state polls were observed.
- Auth/API: `401` before login and `200` for `/api/sessions` after login.

## Docker validation

- `CODOXEAR_DOCKER_PORT=19518 scripts/codoxear-docker-sandbox preflight` passed: isolation root `/tmp/codoxear-docker-sandbox-19518`, home `/tmp/codoxear-docker-sandbox-19518/home`.
- `CODOXEAR_DOCKER_PORT=19518 scripts/codoxear-docker-sandbox test` passed: `1835 passed, 1 skipped, 134 subtests passed in 49.76s`.
- `CODOXEAR_DOCKER_PORT=19518 scripts/codoxear-docker-sandbox smoke` passed: pre-login `/api/me=401`, post-login `/api/sessions=200`, container app dir `/home/tester/.local/share/codoxear`.

## Raw artifacts

- `raw/browser-desktop-eval.json`
- `raw/browser-mobile-eval.json`
- `raw/required-assertions.json`
- `raw/browser-desktop.png`
- `raw/browser-mobile.png`
- `raw/browser-desktop-snapshot.txt`
- `raw/browser-mobile-snapshot.txt`
- `raw/fake_copy_count_session.py`
- `raw/copy-count-proof-eval.js`
- `raw/fake-broker-start.txt`
- `raw/fake-session-sidecar.txt`
- `raw/broker-calls.jsonl`
- `raw/broker-call-summary.json`
- `raw/api-me-pre-login.status`
- `raw/api-sessions-post-login.status`
- `raw/api-login.json`
- `raw/api-cookie-sanitized.txt`
- `raw/smoke-isolation.txt`
- `raw/server.log`
- `raw/cleanup.log`

## Cleanup

Cleanup used exact browser session names, exact fake broker PID from `/tmp/fake_copy_count.pid`, and the exact sandbox script/container name for port `19518`. The fake broker PID still answered immediately after `kill` in the cleanup probe, then `CODOXEAR_DOCKER_PORT=19518 scripts/codoxear-docker-sandbox stop` removed the exact container. After cleanup, no `codoxear-sandbox-19518` container was listed and port `19518` had no listener. No broad `pkill`, `killall`, or host-runtime cleanup was used.

## Residual concerns

None for the requested proof. The proof uses a deterministic fake Docker broker/session, so it validates Codoxear UI/API counting and clipboard behavior, not a live Codex provider.
