# VERIFICATION REPORT — next-outcome-defect-scout

**Date:** 2026-07-06
**Repo:** `/home/yiwen/codex-web-product-recovery`
**Branch:** `recovery/product-gaps`
**Task:** Deterministic artifact-only discriminator for Pi assistant rows with empty text content.

## Scope

Minimal synthetic Pi log construction and discriminant execution.
No source modifications, no test changes, no staged files, no Docker builds.
Protected directories (`/home/yiwen/codex-web`, live runtime dirs) were not accessed.

## Files Produced

| File | Purpose |
|------|---------|
| `prove_next_outcome_defect.py` | Executable discriminator script |
| `proof-output.json` | Machine-readable results |
| `proof-summary.txt` | Human-readable condensed findings |
| `VERIFICATION-REPORT.md` | This report |

## Candidates and Verdicts

### C1: `stopReason=stop, content=[]` → **DEFECT**

**Root cause chain:**
1. `pi_assistant_text()` returns `None` (no text content parts)
2. `pi_assistant_is_final_turn_end()` returns `False` (requires text first)
3. `_pi_message_keeps_turn_busy()` returns `False` (no thinking/tool calls)
4. `_single_chat_event()` returns `None` — no visible transcript event
5. In `_compute_idle_from_log`: the row falls through ALL checks (not user, not aborted, not text, not error, not keeps_busy). The user row set `idle=False`; the assistant row never updates it.
6. Result: `_compute_idle_from_log` → `False` (BUSY), session appears busy forever despite completed turn.

**Transcript impact:** User-only event. No assistant message rendered. No synthetic no-response injected (only Codex/CC have injector paths).

### C2: `stopReason=end_turn, content=[]` → **DEFECT**

**Root cause chain:** Identical to C1. `stopReason='end_turn'` is not inspected anywhere in the idle computation path. `pi_assistant_is_final_turn_end()` returns `False` because text content is empty — it never reaches the stopReason check.

**Transcript impact:** Same as C1.

### C3: `stopReason=stop, content=[{type:'thinking', thinking:''}]` → **DEFECT**

**Root cause chain:**
1. `pi_assistant_text()` returns `None` (no `type:'text'` part)
2. `pi_assistant_thinking_count()` returns 1
3. `_pi_message_keeps_turn_busy()` returns `True` (thinking_count > 0)
4. But `pi_assistant_is_final_turn_end()` returns `False` (no text)
5. `_single_chat_event()` returns `None`
6. `_compute_idle_from_log`: `_pi_message_keeps_turn_busy` sets `saw_terminal_signal=True, idle=False` — BUSY permanently.

**Transcript impact:** User-only event. Empty-thinking assistant row produces no event and never resolves to idle.

### Task B: Post-log backend death → **SCOUT**

**Evidence:**
- `session_prune.py:prune_dead_sessions()` removes sessions via socket exist/error/pid checks; does not inspect log for terminal events
- `session_control.py:_dead_processes()` checks `broker_pid` and `codex_pid`
- `message_routes.py:handle_messages_tail/live` returns empty response for unbound/missing logs
- No Pi close/turn_end/delivery row format exists for injection

**Rationale:** The prune path handles cleanup normally. Log-driven idle is BUSY (DEFECT) but irrelevant post-death. No deterministic transcript event confirms outcome → SCOUT.

## Residual Risks

- **C1/C2 idle mismatch:** Without a fix, any Pi session receiving an assistant response with empty content but non-error/aborted stopReason will appear busy indefinitely in the UI. This blocks subsequent user input via the remote_ready gate.
- **C3 idle mismatch:** Empty-thinking content keeps sessions busy permanently. Same send-blocking effect.
- **No-response injection gap:** Pi has no equivalent to Codex's `_inject_no_response_events`. If a fix makes these rows terminal (idle→True), the transcript will still show user-only with no visible answer.

## Run Evidence

```
$ python3 prove_next_outcome_defect.py

C1: stopReason=stop, content=[]
  pi_assistant_text:            None
  _single_chat_event:           None
  positioned_chat_events:       1 events (user-only)
  _compute_idle_from_log:       BUSY
  pi_current_turn_state_before: BUSY
  search_backend_completed:     count=0
  search_interrupted:           count=0

C2: stopReason=end_turn, content=[]
  [identical results]

C3: stopReason=stop, content=[{type:'thinking', thinking:''}]
  pi_assistant_thinking_count:  1
  _pi_message_keeps_turn_busy:  True
  _single_chat_event:           None
  positioned_chat_events:       1 events (user-only)
  _compute_idle_from_log:       BUSY
  pi_current_turn_state_before: BUSY
  search_backend_completed:     count=0
  search_interrupted:           count=0
```

## Verification Checklist

- [x] No source files modified (`codoxear/*.py` untouched)
- [x] No test files modified
- [x] No staged git changes
- [x] Protected directories not accessed
- [x] Synthetic logs built in temp dirs (auto-cleaned)
- [x] All three candidates discriminated
- [x] Task B classification complete
- [x] All four artifacts written
