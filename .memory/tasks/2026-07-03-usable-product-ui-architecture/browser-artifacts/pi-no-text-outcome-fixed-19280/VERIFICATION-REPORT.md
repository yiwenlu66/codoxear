# Pi no-text terminal outcome fix — verification report

Verdict: **PASS** — the fixed code at `32d914b` makes Pi terminal assistant rows with no visible text render a persistent no-response outcome and clears the busy state. The same proof preserves the negative controls: nonterminal thinking and `toolUse` rows remain in-progress.

## Mechanism under test

The fix adds one Pi predicate, `pi_assistant_is_terminal_no_visible_response()`, for assistant `message` rows that have no visible text, no error/aborted/tool-call semantics, and a non-empty terminal `stopReason` other than `toolUse`, `error`, or `aborted`.

That predicate is consumed by:

- `PiBackend.chat_event_from_log_row()` → projects `_NO_RESPONSE_TEXT` with `message_class:"error"`.
- `pi_current_turn_state_before()` and `_compute_idle_from_log()` → classify the terminal rows as idle.
- `broker_turn_state._apply_rollout_obj_to_state()` and `PiBackend.message_keeps_turn_busy()` → close broker/sessiond turn state instead of leaving it busy.
- sidebar/conversation timestamp helpers → treat the no-response row as a visible final outcome.

## Focused validation

Host focused test command:

```bash
python3 -m pytest -q tests/test_codex_no_response_projection.py tests/test_idle_heuristics.py tests/test_broker_busy_state.py tests/test_server_chat_flags.py tests/test_pi_message_source.py
```

Result: `170 passed in 2.02s`.

A direct post-fix discriminator over the formerly failing shapes showed:

| case | transcript roles | search `completed this turn` | `_compute_idle_from_log` | `pi_current_turn_state_before` |
|---|---:|---:|---:|---:|
| `stopReason:"stop", content:[]` | `user, assistant` | `1` | idle | idle |
| `stopReason:"end_turn", content:[]` | `user, assistant` | `1` | idle | idle |
| `stopReason:"stop", thinking-only` | `user, assistant` | `1` | idle | idle |
| nonterminal thinking | `user` | `0` | busy | busy |
| `toolUse` tool call | `user` | `0` | busy | busy with pending tool |

## Docker/API/browser environment

- Proof sandbox: `codoxear-sandbox-19280`, port `127.0.0.1:19280`.
- Throwaway app dir: `/home/tester/.local/share/codoxear` inside the container.
- No host live runtime, host Codoxear service, host Pi logs, or host sockets were used.
- Synthetic fixture: `fake_pi_no_text_sessions.py` wrote deterministic Pi logs, sidecars, and live broker control sockets inside the container only.
- Cleanup: stopped through `CODOXEAR_DOCKER_PORT=19280 scripts/codoxear-docker-sandbox stop`.

## Synthetic sessions

| session | row shape | expected state |
|---|---|---|
| `pi-no-text-stop-empty` | user + assistant `stopReason:"stop", content:[]` | no-response row, idle |
| `pi-no-text-end-turn-empty` | user + assistant `stopReason:"end_turn", content:[]` | no-response row, idle |
| `pi-no-text-stop-thinking` | user + assistant `stopReason:"stop"`, thinking-only content | no-response row, idle |
| `pi-nonterminal-thinking-control` | user + assistant thinking-only, no terminal stopReason | user + typing row, busy |
| `pi-tool-use-control` | user + assistant `stopReason:"toolUse"`, toolCall content | user + typing row, busy |

## API observations

`api_probe.py` drove the real server through `/api/me`, `/api/login`, `/api/sessions`, `/messages/tail`, `/messages/search`, and `/messages/history`.

Terminal rows:

| session | tail roles | assistant no-response row | error class | search matches | history cursor rehydrates row | `/api/sessions busy` |
|---|---|---:|---:|---:|---:|---:|
| `pi-no-text-stop-empty` | `user, assistant` | yes | yes | `1` | yes | `false` |
| `pi-no-text-end-turn-empty` | `user, assistant` | yes | yes | `1` | yes | `false` |
| `pi-no-text-stop-thinking` | `user, assistant` | yes | yes | `1` | yes | `false` |

Controls:

| session | tail roles | no-response search matches | `/api/sessions busy` |
|---|---|---:|---:|
| `pi-nonterminal-thinking-control` | `user` | `0` | `true` |
| `pi-tool-use-control` | `user` | `0` | `true` |

Raw API evidence is under `api/`, with the rollup in `api/SUMMARY.json`.

## Browser observations

Real Chrome via `agent-browser` logged into the Docker server and selected each session by hash.

Terminal sessions rendered exactly the user row plus an assistant error row containing:

```text
The backend completed this turn without producing a response.
```

Classes observed in `browser/browser-dom-summary.json`:

- `pi-no-text-stop-empty`: `['msg user', 'msg assistant error']`
- `pi-no-text-end-turn-empty`: `['msg user', 'msg assistant error']`
- `pi-no-text-stop-thinking`: `['msg user', 'msg assistant error']`

Control sessions rendered a user row plus a typing row, not a no-response row:

- `pi-nonterminal-thinking-control`: `['msg user', 'msg assistant typing']`
- `pi-tool-use-control`: `['msg user', 'msg assistant typing']`

Reload/select persistence proof: `browser/reload-stop-thinking-rows.json` reloaded `#session=pi-no-text-stop-thinking` and re-rendered `['msg user', 'msg assistant error']` with the same no-response text. Screenshots:

- `browser/stop-thinking-no-response.png`
- `browser/tool-use-control-typing.png`

## Canonical validation

- Full local pytest: `1753 passed, 132 subtests passed in 23.72s`.
- Docker unit (`CODOXEAR_DOCKER_PORT=19281 scripts/codoxear-docker-sandbox test`): `1752 passed, 1 skipped, 132 subtests passed in 47.89s`.
- Docker smoke (`CODOXEAR_DOCKER_PORT=19282 scripts/codoxear-docker-sandbox smoke`): pre-login `/api/me` `401`, post-login `/api/sessions` `200`, container app dir `/home/tester/.local/share/codoxear`.

## Boundary

The proof uses deterministic synthetic Pi logs rather than waiting for a live Pi model to produce a rare empty terminal row. That is the decisive layer for this defect: the failure and the fix are entirely in Codoxear's log normalizer and busy reducers. The synthetic rows are the real Pi row shapes proven by the defect scout and are consumed by the actual Docker server, API routes, search/history cursor path, and browser renderer.
