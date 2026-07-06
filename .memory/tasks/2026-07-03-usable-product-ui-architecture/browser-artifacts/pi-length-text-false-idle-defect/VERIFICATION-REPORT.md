# Pi `stopReason:"length"` visible-text false-idle proof

Verdict: **DEFECT**

## Command

```bash
python3 .memory/tasks/2026-07-03-usable-product-ui-architecture/browser-artifacts/pi-length-text-false-idle-defect/prove_pi_length_text_false_idle.py
```

## Observation

### Synthetic prefix: user row then assistant `stopReason:"length"` with visible text and no tool call

- `pi_assistant_is_final_turn_end()` on the length row: `True`
- `PiBackend.chat_event_from_log_row()` projects the length row as role `assistant` with `message_class='final_response'` and text `'partial before compaction'`.
- `_compute_idle_from_log()` returns `True`.
- `pi_current_turn_state_before(..., EOF)` returns pending `[]` and idle `True`.
- Broker reducer after session/user/length rows: `busy=False`, `turn_open=False`, `turn_has_completion_candidate=False`.
- Runtime/readiness projection from that log+broker state: `busy=False`, `direct_send=True`.

### Synthetic continuation: same prefix plus compaction rows and assistant `toolUse` continuation

- Last assistant continuation event: role `assistant`, `message_class='narration'`, text `'resuming after compaction and calling a tool'`.
- `_compute_idle_from_log()` returns `False`.
- `pi_current_turn_state_before(..., EOF)` returns pending `['toolu_1']` and idle `False`.
- Broker reducer after all rows: `busy=True`, `turn_open=True`, pending `['toolu_1']`.
- Runtime/readiness projection after continuation: `busy=True`, `direct_send=False`.

### Control: assistant `stopReason:"stop"` with visible text

- `pi_assistant_is_final_turn_end()` on the stop row: `True`.
- Projection class: `'final_response'`.
- `_compute_idle_from_log()` returns `True`.

## Interpretation

Current HEAD treats a Pi assistant row with visible text and `stopReason:"length"` as a final answer. The same row closes broker turn state, makes `_compute_idle_from_log()` idle, makes `pi_current_turn_state_before()` idle, and makes runtime readiness sendable. The synthetic continuation then reopens/busies the same turn when a later `toolUse` row appears. That is a transient false-idle window at the compaction/continuation boundary, violating the binary busy/idle invariant: the browser can consider the session idle/sendable while Pi is still continuing the turn.

The `stopReason:"stop"` control remains final/idle under the same mechanisms, so the proof isolates `length` rather than visible text itself.

## Files written

- `prove_pi_length_text_false_idle.py`
- `proof-output.json`
- `proof-summary.txt`
- `VERIFICATION-REPORT.md`

No source files or tests are modified by this artifact proof.
