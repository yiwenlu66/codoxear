# Pi visible-text length false-idle fix — verification report

Verdict: **PASS** — after `89e60e8`, Pi `stopReason:"length"` with visible text remains nonfinal/busy during compaction and continuation. Visible-text `stop` remains a normal final response.

## Mechanism corrected

The defect proof showed `pi_assistant_is_final_turn_end()` treated visible-text `length` rows as final, which closed broker turn state, made `_compute_idle_from_log()` idle, and made runtime readiness direct-send true while Pi could still be compacting and continuing.

The fix adds a narrow guard in `pi_assistant_is_final_turn_end()`:

```python
if message.get("stopReason") == "length":
    return False
```

This preserves final handling for visible-text `stop`, while making all `length` rows nonfinal continuation boundaries.

## Focused validation

Focused command:

```bash
python3 -m pytest -q tests/test_codex_no_response_projection.py tests/test_idle_heuristics.py tests/test_broker_busy_state.py tests/test_server_chat_flags.py tests/test_pi_message_source.py
```

Result: `179 passed in 2.04s`.

Direct discriminator after the fix:

- `pi_assistant_is_final_turn_end(length+text)` → `False`.
- `PiBackend.chat_event_from_log_row(length+text)` → assistant `message_class:"narration"`.
- `pi_assistant_is_final_turn_end(stop+text)` → `True` and projection `message_class:"final_response"`.
- `length+text` prefix stays busy in `_compute_idle_from_log()` and `pi_current_turn_state_before()`.
- `length+text -> compaction/custom -> toolUse continuation` stays busy with pending `toolu_1`.

## Docker/API/browser proof

Proof sandbox:

- Container: `codoxear-sandbox-19286`
- Port: `127.0.0.1:19286`
- App dir: `/home/tester/.local/share/codoxear` inside the container
- Fixture: `fake_pi_length_text_sessions.py`
- Cleanup: container stopped by sandbox helper

Synthetic sessions:

| session | row shape | expected state |
|---|---|---|
| `pi-length-text-prefix-fixed` | user + assistant `stopReason:"length"`, visible text | narration row, busy |
| `pi-length-text-continuation-fixed` | prefix + compaction/custom + assistant `toolUse` continuation | two narration rows, busy |
| `pi-stop-text-control` | user + assistant `stopReason:"stop"`, visible text | final_response row, idle |

API observations from the real server:

| session | tail roles | assistant classes | no-response text | search `completed this turn` | `/api/sessions busy` |
|---|---|---|---:|---:|---:|
| `pi-length-text-prefix-fixed` | `user, assistant` | `narration` | no | 0 | true |
| `pi-length-text-continuation-fixed` | `user, assistant, assistant` | `narration, narration` | no | 0 | true |
| `pi-stop-text-control` | `user, assistant` | `final_response` | no | 0 | false |

The continuation session also search-matches `resuming after compaction`, proving the real continuation appears without the prefix being misclassified as final.

Browser observations from real Chrome:

- `pi-length-text-prefix-fixed`: `['msg user', 'msg assistant', 'msg assistant typing']`, visible partial text plus typing row.
- `pi-length-text-continuation-fixed`: `['msg user', 'msg assistant', 'msg assistant', 'msg assistant typing']`, partial text, continuation text, and typing row.
- `pi-stop-text-control`: `['msg user', 'msg assistant']`, visible final answer.

Screenshots:

- `browser/length-text-prefix-busy.png`
- `browser/length-text-continuation-busy.png`
- `browser/stop-text-final.png`

## Broader validation

- Full local pytest: `1762 passed, 132 subtests passed in 24.18s`.
- Docker unit through sandbox helper: `1761 passed, 1 skipped, 132 subtests passed in 55.35s`.
- Docker smoke: the helper smoke command could not start because Docker Hub token/metadata requests returned EOF before server startup. This was an external registry boundary, not a product result. The same smoke contract was then run manually against the already-built `codoxear-sandbox:latest` image with isolated `/tmp/codoxear-docker-sandbox-19288/home`: pre-login `/api/me` returned `401`, post-login `/api/sessions` returned `200`, and container app dir was `/home/tester/.local/share/codoxear`.

## Boundary

This proof uses deterministic synthetic Pi logs. That is the decisive layer: the bug and fix are in Codoxear's Pi log interpretation, transcript projection, and busy/readiness reducers. The synthetic rows match the real Pi `length -> compaction -> continuation` mechanism identified by clean-room review.
