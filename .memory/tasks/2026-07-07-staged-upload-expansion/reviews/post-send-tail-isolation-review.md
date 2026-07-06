# Clean-room adversarial review — post-confirmed-send tail isolation

Repo: `/home/yiwen/codex-web-product-recovery` (branch `recovery/product-gaps`, tree clean)
Commit: `785b3d2 Isolate post-send cleanup tail failures`
Mode: review only. No edits, no staging, no commits.

## Verdict: ACCEPTED — with 2 nonblockers

The commit meets its primary objective. Every high-value target that review `09a9aa9`
flagged is now isolated: post-confirmation `OSError` from unlink/persistence (staged
cleanup, pending clear, commit_unknown clear) and post-confirmation `KeyError` from
mid-send prune (the false-404) return normal send fields plus explicit warning fields
instead of an error status. Pre-delivery semantics are unchanged; the frontend gates
staged-UI clearing correctly; no PTY/key write or new commit boundary is introduced.

Two residuals remain. Neither is triggerable by the application's own behavior, so
neither blocks; both are named against the literal deliverables so the parent can
elevate if strict conformance is required.

---

## Findings

### NB1 (nonblocker) — Prelog projection does not isolate `ValueError`; contradicts literal deliverable #1

`codoxear/session_send.py:16` defines a **separate, narrower** tuple for the prelog wrapper:

```
15  _POST_CONFIRMATION_CLEANUP_ERRORS   = (ValueError, OSError, KeyError)
16  _POST_CONFIRMATION_PROJECTION_ERRORS = (OSError, KeyError)   # <-- no ValueError
```

and applies it at `session_send.py:113-119`:

```
113   self.record_prelog_user_message(session, committed_text)
114 except _POST_CONFIRMATION_PROJECTION_ERRORS as exc:   # OSError, KeyError only
```

Deliverable #1 explicitly names `ValueError` as a covered tail failure **from prelog
projection**. It is not caught there, so a post-confirmation `ValueError` from prelog
escapes.

**Mechanism (empirically confirmed).** The real prelog dependency chain is
`record_prelog_user_message → PrelogUserMessageRecorder.record → latest_launch_attempt →
read_launch_attempts` (`codoxear/launch_attempt_store.py`). Two `ValueError` paths exist:

- `launch_attempt_store.py:142` `deque(f, ...)` iterates a UTF-8 file. A non-UTF-8 ledger
  raises `UnicodeDecodeError`, which **is a `ValueError` subclass**.
- `launch_attempt_store.py:165` `out.sort(key=lambda item: float(item.get("updated_ts", ...) or 0.0))`
  is unguarded. A JSON-valid record with a non-numeric string `updated_ts` reaches this
  line (the cutoff filter at line ~160 is `isinstance`-guarded and skips it) and raises
  `ValueError: could not convert string to float`. (Note: in-content `JSONDecodeError`,
  the plausible ValueError, is already caught inside `read_launch_attempts`, so it does
  not escape.)

Escape path is worse than a plain false failure: because the prelog `try` is inside the
`with self.lock:` block and its `except` does not match, an uncaught prelog `ValueError`
unwinds past `apply_confirmed_send_success` (session.busy not applied), past
`clear_staged_attachments` (staged attachments NOT cleared → re-injected on retry), out of
`send()`, and — since `control_routes._handle_send` (`codoxear/control_routes.py:218-231`)
only catches `KeyError`/not_ready/injection/commit_unknown — becomes an unhandled 500 after
the backend already delivered. Client sees failure → likely double-send.

**Evidence.**
- `/tmp/probe_prelog_scope.py`: prelog `OSError` → `send_state_cleanup_error` (200);
  prelog `KeyError` → `send_state_cleanup_error` (200); prelog `ValueError` → **RAISED**;
  prelog `JSONDecodeError` → **RAISED**.
- `read_launch_attempts` on a `\xff\xfe` file → `UnicodeDecodeError` (ValueError True).
- `read_launch_attempts` on `{"type":"launch_attempt","launch_id":"x","updated_ts":"garbage"}`
  → `ValueError: could not convert string to float`.
- End-to-end probe: prelog `ValueError` during a staged send → RAISED, `staged_cleanup_ran? False`,
  `staged_still_present? True`, `busy_applied? False`.

**Realism: LOW.** The app writes the ledger as ASCII JSON (`json.dumps(..., sort_keys=True)`,
default `ensure_ascii=True`) with float timestamps, so it never produces non-UTF-8 bytes or
string timestamps. Triggering requires an externally corrupted/foreign-written ledger, and
prelog only runs in the narrow pre-log window (`session.owned and session.log_path is None
and session.launch_id` — `PrelogUserMessageRecorder.record`). This is why I score it
nonblocker rather than blocker.

**Possible author rationale.** The narrower prelog tuple may be deliberate: the cleanup path
has an *expected* `ValueError` (path-safety `validate_staged_attachment_file_target`, the
reason `4963ba6` caught it), whereas the recorder has no *expected* ValueError in the
author's model, so excluding it surfaces genuine logic bugs. That rationale is defensible but
does not match deliverable #1's text and misses the `read_launch_attempts` ValueError paths.

**Recommended fix (one line):** wrap prelog with `_POST_CONFIRMATION_CLEANUP_ERRORS`
(i.e., include `ValueError`) at `session_send.py:114`, and delete the now-redundant
`_POST_CONFIRMATION_PROJECTION_ERRORS`. Alternatively, if the exclusion is intentional,
document it and amend deliverable #1.

### NB2 (nonblocker) — The prelog catch branch has zero test coverage

All four new tests and every pre-existing send test stub `_record_prelog_user_message` to a
no-op or a recorder (`tests/test_server_queue_persistence.py:566,603,622,647` for the new
tests; also 429/446/468/492/520/...). None make it raise. The new
`_POST_CONFIRMATION_PROJECTION_ERRORS` handler (`session_send.py:113-119`) — the only
brand-new control path in the commit — is never exercised in its catch state.

**Recommended fix:** add a test that injects `OSError` (and/or `KeyError`) from
`record_prelog_user_message` and asserts 200 with
`send_state_cleanup_error == "prelog_user_message: <msg>"`, staged/pending state intact, and
that `apply_confirmed_send_success` still ran (session.busy set). Adding an
`updated_ts:"garbage"` / non-UTF-8 real-ledger regression for NB1 would double as coverage.

---

## What survived scrutiny (checked, holds)

- **apply_confirmed_send_success left unwrapped is correct** (`session_send.py:121-127`).
  It is pure attribute assignment on `Session`, a plain `@dataclass` (`session_model.py:9`,
  no `__slots__`/property setters/`__setattr__`), reading already-validated
  `SendResponseResult` fields. It cannot raise `ValueError`/`OSError`/`KeyError`. Guarded by
  `if current:` so a mid-send prune skips it rather than faulting.
- **KeyError catch closes the mid-send-prune false 404 without swallowing the initial
  unknown-session** (deliverable #2 / audit Q3). Initial check
  `if not session: raise KeyError("unknown session")` sits at the top of `send()` *before* any
  tail `try` and outside all wrappers → propagates to `control_routes.py:219-221` → 404.
  Probe: unknown session raises `KeyError` with `delivery attempted? False`. The tail
  `KeyError` catch only wraps post-delivery `clear_staged_attachments`, which is the sole real
  mid-send-prune `KeyError` source (`session_pending_state.py:102`); `set_pending_attachment`
  and `set_commit_unknown_send` use `.get()` and never raise `KeyError` on unknown session, so
  their `KeyError` catch is dead-but-harmless. Test
  `test_staged_attachment_keyerror_after_confirmed_send_is_success_with_visible_error` proves
  clear→KeyError→200+`attachment_cleanup_error`.
- **Pre-delivery failures unchanged (deliverable #2).** not-ready (`require_send_preconditions`,
  `send_remote_ready`), injection (`parse_confirmed_send_response`), and commit_unknown
  (`raise_commit_unknown`) all raise before the tail `try` blocks → 409/502/504. The tail that
  clears staged is only reached after successful parse, so staged entries are preserved on every
  pre-delivery failure (`test_staged_attachments_survive_commit_unknown_send`).
- **Frontend warnings (deliverables #3, #5)** — `app.js:7026-7038`. Both warning types are read
  (`attachment_cleanup_error||attachments_cleanup_error`, and `send_state_cleanup_error`), both
  pushed into `cleanupWarnings`, and shown in the toast joined by `; `. Staged-UI clearing
  (`setSelectedSessionPendingAttachment(false); setAttachCount(0)`) is gated on
  `allowPendingAttachment && !attachmentCleanupError` — so an attachment-cleanup warning
  preserves the local staged UI (correct: server still holds them), and a `send_state_cleanup_error`
  does NOT gate clearing and triggers no resend (informational only). Field names match the server
  (singular `attachment_cleanup_error`, `send_state_cleanup_error`).
- **Successful cleanup unchanged (deliverable #4).** No exception → no warning field added →
  `response` identical to pre-commit; `test_..._compose_at_confirmed_send_boundary_and_clear_on_success`
  passes.
- **No hidden PTY/key write or new commit boundary (deliverable #6).** The diff adds only two
  helpers (`_tail_error_message`, `_add_send_warning`) and `try/except` around existing calls;
  call order is preserved; no `inject_keys`/new `call_confirmed_send`; `set_commit_unknown_send(None)`
  is the pre-existing clear, now merely wrapped.
- **Tests exercise the real coordinator (audit Q4).** `SessionManager.send` (`server.py:970-976`)
  delegates to a real `SessionSendCoordinator` (`session_manager_factories.py:598-611`); the new
  tests call `SessionManager.send(mgr, ...)` and inject failures only at leaf boundaries
  (`_sock_call` = delivery, `clear_staged_attachments`/`_set_pending_attachment`/
  `_set_commit_unknown_send` = cleanup, with the pending/commit tests running the *real*
  in-memory method before raising). This is genuine coordinator-path coverage, not monkeypatching
  around it. Gap is only the prelog branch (NB2).
- **Helper correctness.** `_tail_error_message` returns `exc.args[0]` for single-string args
  (so `KeyError("unknown session")` yields `unknown session`, not the quote-wrapped `'unknown session'`
  repr — matches test assertions) and never returns empty. `_add_send_warning` copies the response
  and concatenates multiple messages into one field via `; `, so prelog+pending+commit_unknown can
  all accumulate into `send_state_cleanup_error` safely; `response` is guaranteed a dict at that
  point because `parse_confirmed_send_response` rejects non-dicts.

---

## Evidence checked

- `node --check codoxear/static/app.js` → OK
- `git diff --check` → clean; `git status --porcelain` → empty (no staged/unstaged files)
- `pytest tests/test_server_queue_persistence.py -k "confirmed_send_boundary or after_confirmed_send or cleanup_failure or survive_commit_unknown"` → 8 passed
- New tail tests + both control_routes send-warning tests + `tests/test_attach_button_source.py` → 13 passed
- Focused suite `pytest tests/test_server_queue_persistence.py tests/test_control_routes.py tests/test_attach_button_source.py` → 130 passed, 22 subtests passed
- Probes (ad-hoc, `/tmp`, no repo edits):
  - prelog `OSError`/`KeyError` caught → `send_state_cleanup_error`; prelog `ValueError`/`JSONDecodeError` escape (raise)
  - `read_launch_attempts` raises `UnicodeDecodeError` (non-UTF-8 ledger) and `ValueError` (string `updated_ts`)
  - unknown session raises `KeyError` before delivery (initial check not swallowed)
  - prelog `ValueError` during staged send → raises, staged NOT cleared, busy NOT applied

## Required fixes
None to block acceptance. Two recommended (both trivial): NB1 widen prelog wrapper to include
`ValueError` (or document deliberate exclusion + amend deliverable #1); NB2 add a prelog
catch-branch test.
