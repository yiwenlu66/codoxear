Both blockers fixed. No Docker harness run was needed — the unit/listing regressions exercise the real `SessionDiscoveryRegistryCoordinator.upsert_registration()`, `reset_session_log_caches`, `set_session_interrupted_idle`, and `SessionListCoordinator.list_sessions()` against a live interrupted-non-final log, which is the exact code path the broker/server discovery path invokes. A Docker run would only re-exercise socket discovery plumbing around the same coordinator.

---

## PASS

### Blocker 1 — Claude Code live polling split close

**Mechanism:** `codoxear/message_routes.py:handle_messages_live()` computed prior-turn context with `_codex_prior_open_turn_context()`, which only recognizes Codex `event_msg` user/close rows and returns `(None, False)` for CC `user`/`system` rows. When a CC `user` row landed in poll 1 and `system/turn_duration` in poll 2, the close-only delta had no open user context, so `_inject_no_response_events()` produced nothing and the browser went idle with no assistant/error result.

**Fix:** One-line change — call `_prior_open_turn_context()` (CC-aware: delegates to `_cc_prior_open_turn_context()` when Codex context is `None`) instead of `_codex_prior_open_turn_context()`. Token/meta/`mark_log_delta` behavior untouched.

**Regression:** `tests/test_message_routes.py::test_messages_live_cc_split_turn_duration_emits_no_response` drives the public `handle_messages_live()` route across two polls (user → append turn_duration → poll). Fails before (`roles == []` instead of `["assistant"]`), passes after. `test_messages_live_cc_split_prior_answer_suppresses_no_response` preserves the answered-turn case (no false injection).

### Blocker 2 — Fresh discovery drops interrupted-idle truth

**Mechanism:** `SessionDiscoveryRegistryCoordinator.upsert_registration()` new-registration branch constructed a `Session` with `registration.interrupted_idle`, then called `reset_log_caches()`, which clears `interrupted_idle`/`interrupted_idle_log_off`/`interrupted_idle_suppressed`. The override was lost on every fresh-server/empty-registry discovery. The existing-session refresh branch was fixed in `f5b4710` but new-registration was not.

**Fix:** In the new-registration branch, after `reset_log_caches()`, route `registration.interrupted_idle` through `set_session_interrupted_idle(session, ...)` when the broker reports true, so a fresh baseline (current log size) is recorded by the one helper that owns the semantics. The false path is left to `reset_log_caches()` clearing, preserving existing new-session log-cache behavior.

**Regression:** `tests/test_stale_interrupted_idle.py::TestFreshDiscoveryPreservesInterruptBaseline::test_fresh_discovery_interrupted_idle_over_busy_log_stores_baseline_and_lists_idle` starts with an empty registry, discovers a fresh interrupted-idle=True registration over a non-final log, and asserts the stored session has an active baseline (`interrupted_idle_log_off == size`), no suppression, and `list_sessions()` projects `busy=False`. Fails before (`AssertionError: False is not true` on `stored.interrupted_idle`), passes after. The companion `test_fresh_discovery_false_interrupted_idle_keeps_clearing_semantics` preserves the false/clearing path.

### Validation

| Command | Result |
|---|---|
| `pytest tests/test_message_routes.py tests/test_cc_no_response_projection.py tests/test_cc_chat_and_idle.py tests/test_cc_log.py tests/test_cc_backend_error_projection.py tests/test_codex_no_response_projection.py` | 107 passed |
| `pytest tests/test_stale_interrupted_idle.py tests/test_session_discovery.py tests/test_stale_sidecars.py tests/test_sessions_pending_log_idle.py tests/test_session_control.py tests/test_session_input.py tests/test_session_runtime.py` | 110 passed |
| `pytest tests/` (full suite) | 1719 passed, 132 subtests passed |
| Blocker 1 narrow repro (reverted) | FAILED before fix |
| Blocker 2 narrow repro (reverted) | FAILED before fix |

### `git status --short`
```
 M codoxear/message_routes.py
 M codoxear/session_discovery_registry.py
 M tests/test_message_routes.py
 M tests/test_stale_interrupted_idle.py
```
No staged files.

### Docker note
A Docker fresh-discovery harness was not run. The unit/listing regression directly drives the real `upsert_registration()` → `reset_session_log_caches` → `set_session_interrupted_idle` → `SessionListCoordinator.list_sessions()` chain against a live non-final interrupted log file. That is the same coordinator the server's socket-discovery path invokes; a Docker run would re-exercise only the socket/`*.json` sidecar plumbing around the coordinator, which is not where the bug lived. Main can re-run a full broker/server Docker discovery cycle for end-to-end confirmation; no code change would be required for it.