# Clean-room re-review: Pi no-text length follow-up

**Verdict: PASS** — the `length` overreach is fixed correctly, forward-safely, and consistently across every authority; commit separation and tree cleanliness hold. One pre-existing, out-of-scope nonblocking follow-up is documented below.

## Mechanism verified

The predicate `pi_assistant_is_terminal_no_visible_response()` (`codoxear/pi_message.py:103`) is now an **allowlist** — `235ca80` replaced the denylist guard with `if message.get("stopReason") not in {"stop", "end_turn"}: return False` (pi_message.py:110) and removed the now-dead `pi_assistant_error_text/pi_assistant_is_aborted_turn` checks while correctly keeping the independent `errorMessage`/`isError` field guards (pi_message.py:112–115). `pi_assistant_text` (pi_message.py:67) excludes thinking-only content, so thinking-only stops still count as no visible response.

**Q1 — length overreach fixed: YES.** Every `length` shape is nonterminal. I confirmed by import against the shipped code:

| row | no_visible_response | idle at that row (`_compute_idle_from_log`) |
|---|---|---|
| length+text-only / +text+thinking | False | True* (pre-existing, see residual) |
| length+thinking-only / +tool-only / +empty | False | **False (busy)** ✓ |
| stop+empty / end_turn+empty / stop+thinking-only | **True** ✓ | True ✓ |
| toolUse / error / aborted / unknown_future | False ✓ | — |

Real-log grounding (1374 Pi logs under `~/.pi/agent/sessions`): all **83** `length` turns auto-continue (57 `toolResult`, 24 `compaction`, 2 `custom_message`); **zero** end the turn. This empirically validates keeping `length` nonterminal. `end_turn` never occurs in real logs — the allowlist entry is defensive only (harmless).

**Q2 — defect coverage without false positives: YES.** The original defect cases are real and covered: `stop`+empty (1 occurrence) and `stop`+thinking-only (6 occurrences) exist in real logs and both project no-response + idle. `length`/`toolUse`/`error`/`aborted` and unknown future reasons all stay nonterminal (forward-safe). Tests enforce this: `tests/test_pi_message_source.py:83–84` (length→False, unknown→False), `:74–77` (stop/end_turn/thinking-only→True).

**Q3 — authorities synchronized: YES.** The predicate reaches all six authorities with identical branch ordering (after aborted/error, before tool/text-busy):
- `_compute_idle_from_log` — rollout_idle.py:192
- `pi_current_turn_state_before` — pi_log.py:249; `_last_chat_role_ts_from_tail` — rollout_idle.py:358/362
- broker/sessiond shared reducer `_apply_rollout_obj_to_state` — broker_turn_state.py:245
- sidebar/updated timestamp `_sidebar_conversation_ts` — rollout_chat_events.py:53
- Pi busy predicate `message_keeps_turn_busy` → False — agent_backend.py:600
- transcript projection `chat_event_from_log_row` → `_build_no_response_event` — agent_backend.py:654

Transcript/search/history/live/export all flow through the single extractor `_single_chat_event`/`_extract_positioned_chat_events` (rollout_log.py:147, message_routes.py:370–375), so the no-response event and the length continuation are consistent everywhere. `_inject_no_response_events` is Codex-only (`_detect_codex_no_response_closes` gates on `event_msg`), so Pi is not double-injected. The length-fix tests exercise the reducer end-to-end (`test_pi_length_thinking_only_keeps_busy_and_current_turn`: length→compaction→continuation all `busy`) and projection (`test_pi_length_compaction_continuation_does_not_insert_false_no_response`).

**Q4 — evidence sufficient: YES** for the claimed scope. The follow-up proof (`pi-no-text-length-followup-19283`) uses a Docker sandbox (port 19283, containerized HOME per the AGENTS.md isolation rule), real server API, and real Chrome DOM. Raw artifacts corroborate the report exactly: `stop-empty` → `msg assistant error` "The backend completed this turn without producing a response.", `busy:false`; `length-prefix` (thinking-only) → `msg assistant typing`, `busy:true`, `has_no_response_row:false`; `length-continuation` → continuation narration "continuing with a tool", `busy:true`, no false no-response. I independently reproduced the full suite: **1757 passed, 132 subtests passed**.

**Q6 — commit separation + clean tree: YES.** `32d914b` = 6 code + 4 test files, 0 memory; `235ca80` = 1 code + 4 test files, 0 memory; `097d593`/`30c037c`/`14b6266` = memory/evidence only. Working tree clean, nothing staged.

## Q5 — Residual (nonblocking, pre-existing, out of scope)

**Text-bearing `length` turns (with text, no tool call) still transiently classify idle.** This runs through `pi_assistant_is_final_turn_end` (pi_message.py:128), which returns `True` for any `length`+text row — a code path **unchanged by these commits** (confirmed: `32d914b` only inserted the no-visible-response branch *after* the pre-existing `pi_assistant_text → idle = pi_assistant_is_final_turn_end` branch in rollout_idle.py). So `_compute_idle_from_log` returns idle=True while such a row is the log tail.

- Real data: 9/83 `length` turns are text-no-tool; **all 9 auto-continue** via compaction/custom_message — so idle=True there is a genuine (brief) false idle.
- Escalation path the prior review left untraced: `resolve_runtime_status` makes log-derived idle the readiness authority once a log is bound (`remote_ready = log_idle is True or override`, session_runtime.py:552). A false idle → `remote_ready=True`; if a compaction window exceeds `QUEUE_IDLE_GRACE_SECONDS` (default 10s, server_config.py:278) of continuous readiness, a queued prompt could dispatch mid-compaction.
- **Severity is bounded**: no fabricated no-response occurs (length+text renders the real partial text as `final_response`, verified); it self-heals once the continuation row lands (final state is `busy`); and it requires a rare compound condition (text-bearing length ≈ 11% of the 0.08% length rows + non-empty queue + compaction > 10s). It is neither introduced nor claimed-fixed by this work; the VERIFICATION-REPORT correctly scopes its claim to "no-text/thinking rows."

Suggested follow-up (separate change): make `pi_assistant_is_final_turn_end` treat `stopReason == "length"` as nonterminal (same allowlist logic), and add a `length`+text-then-compaction idle-stays-busy test.

## Boundary of this PASS
Verified: the **no-visible-text** `length` path (thinking-only, tool-only, empty) is now nonterminal and busy across all six authorities, defect cases for `stop`/`end_turn` remain covered, and the evidence is real and sufficient. Not in scope and still latent: the **text-bearing** `length` transient false idle via the untouched `pi_assistant_is_final_turn_end`.