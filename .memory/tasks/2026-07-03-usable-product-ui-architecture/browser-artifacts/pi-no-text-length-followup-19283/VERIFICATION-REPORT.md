# Pi length-continuation follow-up — verification report

Verdict: **PASS** — after `235ca80`, Pi `stopReason:"length"` no-text/thinking rows stay nonterminal and busy, while `stop`/`end_turn` no-text rows still render the explicit no-response outcome.

## Mechanism corrected

Clean-room review found that the first predicate in `32d914b` was a denylist and therefore classified `stopReason:"length"` as terminal. Real Pi logs can use `length` as a context-compaction continuation boundary, so classifying it as no-response would fabricate a false outcome and transient false idle.

`235ca80` changes `pi_assistant_is_terminal_no_visible_response()` to an allowlist:

- terminal no-visible-response: `stopReason in {"stop", "end_turn"}` plus no visible text and no tool call.
- nonterminal: `length`, `toolUse`, missing/empty stopReason, and unknown future stop reasons.

## Focused validation

Focused host command:

```bash
python3 -m pytest -q tests/test_codex_no_response_projection.py tests/test_idle_heuristics.py tests/test_broker_busy_state.py tests/test_server_chat_flags.py tests/test_pi_message_source.py
```

Result: `174 passed in 2.13s`.

A direct discriminator showed:

- `pi_assistant_is_terminal_no_visible_response(length+thinking)` → `False`.
- `PiBackend.chat_event_from_log_row(length+thinking)` → `None`.
- `length+thinking → compaction → continuation(toolUse text+toolCall)` projects only the user row and the continuation narration row; no `_NO_RESPONSE_TEXT` appears.
- A log ending at the `length` row remains busy in both `_compute_idle_from_log` and `pi_current_turn_state_before`.

## Docker/API/browser environment

- Proof sandbox: `codoxear-sandbox-19283`, port `127.0.0.1:19283`.
- Throwaway app dir: `/home/tester/.local/share/codoxear` inside the container.
- No host live runtime, host Pi logs, host sockets, or protected checkout were touched.
- Synthetic fixture: `fake_pi_length_sessions.py` wrote deterministic logs, sidecars, and live broker control sockets inside the container only.
- Cleanup: stopped via `CODOXEAR_DOCKER_PORT=19283 scripts/codoxear-docker-sandbox stop`.

## Synthetic sessions

| session | row shape | expected state |
|---|---|---|
| `pi-stop-empty-regression` | user + assistant `stopReason:"stop", content:[]` | no-response row, idle |
| `pi-length-prefix-control` | user + assistant `stopReason:"length"`, thinking-only content | no no-response, busy |
| `pi-length-continuation-control` | user + length thinking + compaction/custom rows + assistant `toolUse` with text/toolCall | continuation narration, no no-response, busy |

## API observations

`api_probe.py` drove the real Docker server through `/api/me`, login, `/api/sessions`, `/messages/tail`, and `/messages/search`.

| session | tail roles | no-response row | search `completed this turn` | search continuation | `/api/sessions busy` |
|---|---|---:|---:|---:|---:|
| `pi-stop-empty-regression` | `user, assistant` | yes | `1` | `0` | `false` |
| `pi-length-prefix-control` | `user` | no | `0` | `0` | `true` |
| `pi-length-continuation-control` | `user, assistant` | no | `0` | `1` | `true` |

The continuation assistant row is `message_class:"narration"` with text `continuing with a tool`, proving the real continuation appears without an invented no-response row.

Raw API evidence lives under `api/`, with rollup `api/SUMMARY.json`.

## Browser observations

Real Chrome via `agent-browser` selected all three sessions by hash.

Observed DOM classes/texts in `browser/browser-dom-summary.json`:

- `pi-stop-empty-regression`: `['msg user', 'msg assistant error']`, assistant text `The backend completed this turn without producing a response.`
- `pi-length-prefix-control`: `['msg user', 'msg assistant typing']`, no no-response text.
- `pi-length-continuation-control`: `['msg user', 'msg assistant', 'msg assistant typing']`, continuation text `continuing with a tool`, no no-response text.

Screenshots:

- `browser/stop-empty-no-response.png`
- `browser/length-prefix-typing.png`
- `browser/length-continuation-no-false-noresponse.png`

## Canonical validation after the follow-up

- Full local pytest: `1757 passed, 132 subtests passed in 23.61s`.
- Docker unit (`CODOXEAR_DOCKER_PORT=19284 scripts/codoxear-docker-sandbox test`): `1756 passed, 1 skipped, 132 subtests passed in 45.26s`.
- Docker smoke (`CODOXEAR_DOCKER_PORT=19285 scripts/codoxear-docker-sandbox smoke`): pre-login `/api/me` `401`, post-login `/api/sessions` `200`, app dir `/home/tester/.local/share/codoxear`.

## Conclusion

The review concern is resolved. `length` no longer closes a Pi turn or fabricates a no-response row. The original user-visible fix remains intact for `stop`/`end_turn`: completed no-text Pi turns render the existing error-styled no-response outcome and classify idle.
