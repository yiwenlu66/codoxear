# Follow-up fix review — post-confirmation prelog tail isolation

Repo: `/home/yiwen/codex-web-product-recovery` (branch `recovery/product-gaps`, tree clean)
Delta audited: `a37ff43..b0a6a09` = exactly one commit `b0a6a09 Isolate prelog tail value errors`
Mode: review only. No edits, no staging, no commits.

## Verdict: ACCEPTED — no blockers

`b0a6a09` closes both nonblockers from the prior clean-room review of `785b3d2`
(`/tmp/post-send-tail-isolation-review.md`): NB1 (prelog projection excluded `ValueError`)
and NB2 (no prelog catch-branch test). The change is the minimal, invariant-preserving fix
the prior review recommended, with a cleaner consolidation: the two divergent tuples collapse
into one shared `_POST_CONFIRMATION_TAIL_ERRORS = (ValueError, OSError, KeyError)` applied
uniformly to all four post-confirmation tail sites. Pre-delivery semantics are structurally
untouched. All acceptance questions resolve affirmatively.

The diff is +52/-6 across two files: `codoxear/session_send.py` (tuple unification + 4 catch
clauses) and `tests/test_server_queue_persistence.py` (one new regression test). No leftover
references to the deleted constant names exist anywhere in the tree.

---

## Acceptance questions

### Q1 — Post-confirmation prelog `ValueError` now returns 200 with `send_state_cleanup_error`? YES

Mechanism, before → after:
- `session_send.py:15` now defines `_POST_CONFIRMATION_TAIL_ERRORS = (ValueError, OSError, KeyError)`,
  replacing the prior split (`_POST_CONFIRMATION_CLEANUP_ERRORS = (ValueError, OSError, KeyError)`
  for cleanup and the narrower `_POST_CONFIRMATION_PROJECTION_ERRORS = (OSError, KeyError)` for prelog).
- `session_send.py:110-116` wraps `record_prelog_user_message` (line 112) in `except
  _POST_CONFIRMATION_TAIL_ERRORS` (line 113). A prelog `ValueError` is now caught, appended via
  `_add_send_warning(response, "send_state_cleanup_error", f"prelog_user_message: {msg}")`, and
  execution continues; `send()` returns the confirmed-send response dict (HTTP 200).
- Before the fix a prelog `ValueError` escaped `send()`. Route mapping confirms the outcome:
  `_handle_send` (`control_routes.py:212-230`) catches only `KeyError`→404, not_ready→409,
  injection→502, commit_unknown→504 — not `ValueError`. It then unwinds to `do_POST`
  (`server_handler.py:285-286`, `except Exception → handle_route_exception`) →
  `server_http.py:43-47` → HTTP 500, after the backend already delivered. So the fix converts a
  false post-delivery 500 into a success response carrying a visible warning.
- The `ValueError` escape is real (low realism): `launch_attempt_store.read_launch_attempts`
  reaches it via `deque(f, ...)` over a UTF-8 stream (`launch_attempt_store.py:137-139`, non-UTF-8
  → `UnicodeDecodeError`, a `ValueError` subclass) and the unguarded
  `out.sort(key=lambda item: float(item.get("updated_ts", ...) or 0.0))`
  (`launch_attempt_store.py:165`, a non-numeric string `updated_ts` skips the isinstance-guarded
  cutoff at line 161 and hits `float()`).
- User-visible as success: `app.js:7028-7035` reads `send_state_cleanup_error` in the **success**
  branch and renders `"sent; send state cleanup failed: prelog_user_message: ..."`; it is not the
  failure `catch`, so the delivered turn is not treated as retryable.

### Q2 — New test proves the previous failure mechanism is closed? YES

`test_prelog_valueerror_after_confirmed_send_is_success_with_visible_error`
(`tests/test_server_queue_persistence.py:511-556`) drives the **real** coordinator through
`SessionManager.send` (which delegates to `_send_coordinator_for_manager().send`, `server.py:970-976`),
overriding `_record_prelog_user_message` to raise `ValueError("launch ledger invalid")`. It asserts
every element the task requires:
- backend send delivered: `seen == [{"cmd": "send", "text": committed_text, "sync": True}]`.
- prelog failed after delivery: `recorded == [committed_text]` plus the error surfaced in the response.
- cleanup still ran despite prelog failure: `cleanup_calls == [sid]`.
- staged/pending state cleared on cleanup success: `mgr._staged_attachments == {}`,
  `pending_attachment` False, `sid not in mgr._pending_attachment_ids`.
- send-boundary busy state applied: `session.busy is True` and `session.last_send_boundary_active is True`
  — this is the key regression proof, because `apply_confirmed_send_success` (`session_send.py:121`)
  sits after the prelog `except` (unwrapped); previously the uncaught `ValueError` unwound past it,
  leaving busy/boundary unset and staged attachments un-cleared.
- success response with visible error: response equals
  `{"queued": False, "queue_len": 0, "busy": True, "send_state_cleanup_error": "prelog_user_message: launch ledger invalid"}`.

The test isolates the prelog failure from cleanup (the `clear_staged` stub succeeds), so it proves
exactly the closed mechanism: delivered + prelog-failed + cleanup-ran + state-cleared + boundary-applied.
Independently re-run: PASSED.

### Q3 — Did it accidentally catch pre-delivery `ValueError`/not-ready/injection/commit_unknown? NO

- The commit only renamed the exception tuple inside the four **existing** post-confirmation tail
  `try/except` blocks (`session_send.py:112-116`, `127-134`, `136-143`, `145-152`). No new `try`
  was introduced around any pre-delivery code.
- All pre-delivery/delivery operations sit structurally before the tail and outside every wrapper:
  `require_send_preconditions` (`:64`, raises not_ready), `send_remote_ready` (`:71`, not_ready),
  `call_confirmed_send` (`:95`, commit_unknown/not_ready/timeout), `parse_confirmed_send_response`
  (`:105`, commit_unknown/injection). None are within a tail `try`.
- Structural guarantee: `SessionNotReadyError`, `SessionCommitUnknownError`, `SessionInjectionError`
  all derive from `RuntimeError` (`session_errors.py:16,20,24`; verified `issubclass(..., (ValueError,
  OSError, KeyError))` is `False` for all three). Even if one were raised inside a tail block (it is
  not), the tuple could not catch it.
- The only tuple-member raised pre-delivery is the initial `raise KeyError("unknown session")`
  (`session_send.py:62`), but it is at the top of `send()`, outside all tail `try` blocks → propagates
  to `_handle_send`→404. Unaffected by the change.

### Q4 — Shared `_POST_CONFIRMATION_TAIL_ERRORS` tuple appropriate for all four sites? YES

All four sites are post-confirmation cleanup/bookkeeping that must not fail an already-confirmed send:
prelog projection (`record_prelog_user_message`, `:112`), staged cleanup (`clear_staged_attachments`,
`:129`), pending clear (`set_pending_attachment`, `:138`), commit_unknown clear
(`set_commit_unknown_send`, `:147`). Each realistically raises `ValueError` (validation/ledger),
`OSError` (persistence/unlink), or `KeyError` (state lookup). Unifying the two prior tuples removes
the exact inconsistency that produced NB1 — prelog was the sole site excluding `ValueError` while the
invariant is literal ("any post-confirmation tail failure is isolated"). The curated tuple (not bare
`Exception`) is a deliberate, now-uniform boundary: unexpected types (`RuntimeError`, `TypeError`)
still surface as 500 to expose genuine bugs rather than being masked.

### Q5 — Blocker or residual requiring code change? NO BLOCKER

Residuals are pre-existing properties of the tail-isolation policy (established in `785b3d2`,
unchanged here), not defects introduced by this commit:
- A tail op raising a type outside `(ValueError, OSError, KeyError)` (e.g. `RuntimeError`) still
  becomes a post-delivery 500. Accepted trade-off: surfacing genuine bugs over blanket suppression.
- Including `ValueError` in the prelog catch can mask a genuine recorder logic bug as a warning field
  (traceback not logged), but only after confirmed delivery, where a visible warning is strictly
  better than a false 500. Consistent with the policy already applied to the other three sites.

Neither warrants a code change before acceptance.

---

## What survived scrutiny (checked, holds)

- `apply_confirmed_send_success` remains correctly unwrapped and runs after the prelog `except`
  (`session_send.py:117-127`); it is pure attribute assignment on the `Session` dataclass reading a
  pre-validated `SendResponseResult` (`session_input.py:69-83`), guarded by `if current:`. The new
  test proves busy/boundary are applied even when prelog raises.
- No dangling references: `grep` for `_POST_CONFIRMATION_CLEANUP_ERRORS` / `_POST_CONFIRMATION_PROJECTION_ERRORS`
  across the repo returns nothing; the new constant appears once as definition plus four catch sites.
- Frontend gating is coherent: on prelog failure the response carries `send_state_cleanup_error` (not
  `attachment_cleanup_error`), so `app.js:7036 if (allowPendingAttachment && !attachmentCleanupError)`
  still clears the staged UI, matching the server truth that staged cleanup succeeded — no browser/server desync.

---

## Validation independently reproduced

- `pytest -k after_confirmed_send` → `6 passed` (prelog ValueError + staged ValueError/OSError/KeyError
  + pending-clear + commit_unknown-clear: full tail matrix).
- `pytest test_prelog_valueerror_after_confirmed_send_is_success_with_visible_error` → `1 passed`.
- Focused suite (`test_server_queue_persistence` + `test_control_routes` + `test_session_control` +
  `test_queue_routes` + `test_send_ack` + `test_session_routes`) → `155 passed, 22 subtests passed`.
- `node --check codoxear/static/app.js` → OK.
- `git diff --check` → clean. `git status --porcelain` → empty (clean tree, no staged files).
