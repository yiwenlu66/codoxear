## Clean-room review: Pi length/text busy fix — VERDICT: PASS

I reviewed the three code commits, traced every busy/idle/projection authority, built an empirical truth table with the real functions, ran the suites, and audited the evidence and commit hygiene. No files edited/staged/committed (my only writes were to `/tmp`; tree verified clean).

### Mechanism verified (empirical truth table, real functions, `/tmp/pi_truth_table.py`)

| case | is_final | no_visible | msg_class | idle_log | turn_state | broker_busy | stale_clear |
|---|---|---|---|---|---|---|---|
| stop+text | True | False | final_response | IDLE | IDLE | False | — |
| end_turn+text | True | False | final_response | IDLE | IDLE | False | — |
| no-text stop | False | **True** | error(no-response) | IDLE | IDLE | False | — |
| no-text end_turn | False | **True** | error(no-response) | IDLE | IDLE | False | — |
| **length+text** | **False** | False | **narration** | **busy** | **busy** | **True** | **False** |
| **length+text+FINAL sig** | **False** | False | **narration** | **busy** | **busy** | **True** | **False** |
| length no-text | False | False | (no event) | busy | busy | True | False |
| length thinking-only | False | False | (no event) | busy | busy | True | False |
| toolUse | False | False | narration | busy | busy | True | False |
| error | False | False | error | IDLE | IDLE | False | — |
| aborted | False | False | error(interrupted) | IDLE | IDLE | False | — |

**Q1 — `pi_assistant_is_final_turn_end()` correctly makes `length` nonfinal.** The change (`89e60e8`, `codoxear/pi_message.py:132-133`) is a single top-level short-circuit `if message.get("stopReason") == "length": return False`, placed *before* the text and `textSignature` checks. Consequence: even `length` + a `{"phase":"final_answer"}` textSignature is nonfinal (verified, `tests/test_pi_message_source.py:119-123`). stop/end_turn→final, toolUse→nonfinal (via `tool_use_count>0`), textSignature-final fallback, and no-response are all preserved unchanged.

**Q2 — All authorities synchronized.** The fix lives in the two shared predicates in `pi_message.py`; every authority consumes them:
- Server log idle `_compute_idle_from_log` (`rollout_idle.py:180-198`): `length+text` hits the `pi_assistant_text` branch → `idle = pi_assistant_is_final_turn_end(obj)` = False → busy.
- Server turn state `pi_current_turn_state_before` (`pi_log.py:280-287`): same → busy.
- Broker/sessiond share `_apply_rollout_obj_to_state` (`broker_turn_state.py`) + `_should_clear_busy_state`. `length+text` skips `_close_turn_state` (nonfinal), stays busy; staleness clear is blocked by the `turn_open and not turn_has_completion_candidate` gate (Pi never sets a completion candidate), so busy holds indefinitely until an explicit terminal row.
- Readiness `resolve_runtime_status`/`session_runtime_readiness` (`session_runtime.py:558-585`): `length` → `remote_ready=False` → `direct_send=False` → `unattended_injection=False` (verified at HEAD). This closes the escalation path (queued/unattended dispatch mid-compaction) that the prior review left unverified.
- `_sidebar_conversation_ts`/`_last_chat_role_ts_from_tail`/transcript/search/history/export all route through `chat_event_from_log_row` + the same predicates; `length+text`→narration with a sidebar ts, `length` no-text→no event/no ts, no-text stop→no-response error row + ts.

The `message_keeps_turn_busy=False` for pure `length+text` is benign: every idle authority evaluates the `pi_assistant_text` branch first, and the probe helper returns `None`→keep-previous. No path goes idle for `length+text`.

**Q3 — Proof sufficiency: adequate.** The Docker/API/browser proof (`pi-length-text-fixed-19286`) is decisive for the user-visible claim. The fixture socket returns canned `busy:False`, yet `/api/sessions` reports `busy:true` for both length sessions and `false` for the stop control — proving the reported busy is derived from the server **log-idle authority** (`_compute_idle_from_log`→`resolve_runtime_status`: `busy = not log_idle`), which is the dominant authority for terminal-owned sessions and exactly the code the fix touches. Real Chrome DOM confirms it: length rows render `msg assistant typing`; stop renders none (`browser/browser-dom-summary.json`). The defect proof (`4af9a3f`) captured the pre-fix false state including readiness (`runtime busy/sendable: False/True`), using the real `resolve_runtime_status`/`session_runtime_readiness`. The manual no-build smoke boundary is acceptable: the helper smoke's Docker Hub EOF was an external registry failure *before server startup* (not a product result), and the actual fix proof server (port 19286) **did** start in-container; the manual smoke against the already-built image validated bring-up/auth (`401→200`, correct container app dir).

**Q4 — No residual blocker.** The prior review's CONCERNS (denylist over-reach sweeping `length` into terminal-no-response, causing a transient false idle at compaction) are fully resolved: `235ca80` switched `pi_assistant_is_terminal_no_visible_response` to the allowlist `stopReason in {stop,end_turn}`, flipped the test that had enshrined `length→terminal=True` to `assertFalse`, added `unknown_future_reason→False` (forward-safety for new nonterminal reasons), added length continuation busy tests, and dropped the redundant error/aborted checks.

**Q5 — Commit separation clean.** `32d914b`/`235ca80`/`89e60e8` each touch only `codoxear/` + `tests/` (code ships with its own tests, atomic); all evidence commits touch only `.memory/`. No runtime artifacts (`sock`/`server.log`/`__pycache__`/`hmac`) in code commits. Tree clean, nothing staged. **Claim 4 holds** — no new message_class/busy state/color/category; only existing `narration`/`final_response`/`error` reused.

### Remaining boundary (precise)
1. The browser proof exercises the **server log-idle authority** (dominant for terminal sessions); the **live broker/sessiond reducer** is covered by unit tests (`test_broker_busy_state.py`, incl. new length cases) and my truth table, not the browser proof (fake socket). Both call the identical predicates, so behavior is uniform.
2. Proof uses **deterministic synthetic Pi logs** — the decisive layer, since the fix is entirely Pi-log interpretation; row shapes match real `length→compaction→continuation` (validated against 1372 real logs in the prior no-text review).
3. A hypothetical truly-terminal `length` row (Pi stops at a length boundary and never continues) keeps the session busy until process-death detection. This is **correct** under the "must not go idle while continuing" invariant; permanent process death is handled by the separate broker pid-liveness/stale mechanism.