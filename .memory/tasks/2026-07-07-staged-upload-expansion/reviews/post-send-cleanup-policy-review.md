# Clean-room review — post-confirmed-send staged attachment cleanup policy

**Commit:** `4963ba6 Report post-send attachment cleanup failures`
**Repo:** `/home/yiwen/codex-web-product-recovery` (branch `recovery/product-gaps`)
**Mode:** review only — no edits, no staging, no commits. Working tree left clean.

## Verdict: ACCEPTED (with one nonblocker + minor notes)

The commit correctly satisfies every stated deliverable for the **`ValueError` guard
path**. A post-success `clear_staged_attachments` `ValueError` no longer becomes a
send failure/500/502/504: it is caught, the confirmed-delivery response is returned
with an added `attachment_cleanup_error`, the route stays 200, and staged/pending
truth is preserved on both server and browser. Successful-cleanup behavior is
unchanged. No new backend commit path was introduced. Verified by reading the code,
by non-vacuous behavioral tests (mutation-tested against the pre-commit source), and
by a full-suite run.

The one substantive finding is a **nonblocker**: the catch is `ValueError`-only, so a
*non-`ValueError`* post-success cleanup failure (`OSError` from `unlink`, or from the
`save_staged_attachments`/`save_pending_attachments` persistence writes) still
escapes `send()` and reproduces the exact bug class this commit exists to prevent.
This is outside the deliverable's literal scope (which names `ValueError`), lower
probability (requires filesystem-level failure), and introduces no regression — hence
nonblocker, but worth closing.

---

## Note on repo state during review

While I reviewed, another session layered `a55922e "Record post-send cleanup policy
memory"` on top of `4963ba6`. HEAD is now `a55922e`. That commit touched only
`.memory/tasks/2026-07-07-staged-upload-expansion/{EPISTEMIC.md,OPS.md}`;
`git diff 4963ba6 HEAD -- codoxear/session_send.py codoxear/static/app.js tests/`
is empty. The reviewed code is unchanged, so this review of `4963ba6` stands.

---

## Deliverable-by-deliverable

**1. Confirmed delivery stays successful; ValueError cleanup ≠ 500/502/504/failure — MET.**
`codoxear/session_send.py:100-104`. The cleanup call is placed *after*
`apply_confirmed_send_success` (`session_send.py:91-97`) and wrapped:
```python
if staged_entries:
    try:
        self.clear_staged_attachments(session_id)
    except ValueError as exc:
        response = dict(response)
        response["attachment_cleanup_error"] = str(exc)
```
The `ValueError` is swallowed, so `send()` returns normally. `_handle_send`
(`codoxear/control_routes.py:208-232`) has no matching `except`, so a normal return
falls through to `json_response(handler, 200, res)`.

**2. JSON surfaces cleanup failure while preserving normal fields — MET.**
`response = dict(response)` shallow-copies the raw sock response
(`{"queued": ..., "queue_len": ...}`) and adds one top-level key
`attachment_cleanup_error`. The shallow copy is sufficient (only a top-level string
is added) and avoids mutating the shared response object. Route test asserts exactly
`(200, {"queued": False, "queue_len": 0, "attachment_cleanup_error": "..."})`.

**3. Staged-list / pending truth not falsely cleared — MET (server and browser).**
- Server: the deterministic guard `validate_staged_attachment_file_target`
  (`file_upload.py:106-137`) runs for *all* entries **before** any `unlink` and
  before `self.staged_attachments.pop(...)` (`session_store.py:442-451`). A guard
  `ValueError` therefore raises before the in-memory list is popped and before
  `_sync_pending_projection_locked`/`save_*` run (`session_pending_state.py:100-108`).
  In-memory staged list and pending projection are intact.
- Coordinator: on the caught `ValueError`, control skips the `else` branch, so
  `set_pending_attachment(session_id, False)` is **not** called
  (`session_send.py:105-106`). Pending flag preserved.
- Browser: on `cleanupError`, `setSelectedSessionPendingAttachment(false)` and
  `setAttachCount(0)` are **not** called (`app.js:7031-7034`). The badge is a MAX
  projection `Math.max(stagedAttachments.length, serverListCount, serverPending?1:0)`
  (`app.js:6650-6659`); since neither local nor server truth was cleared, the badge
  and chips stay visible. `refreshSessions()` re-syncs to the still-staged server
  truth, so local and server converge — **no stale badge/list divergence**.
- Behavioral test asserts `mgr._staged_attachments[sid] == [entry]`,
  `pending_attachment is True`, `sid in _pending_attachment_ids`.

**4. Successful cleanup unchanged — MET.**
Empty `attachment_cleanup_error` → `else` branch on the frontend runs
`setSelectedSessionPendingAttachment(false)` + `setAttachCount(0)` (badge/chips
cleared); backend clears staged/pending as before. Locked by the existing
`test_staged_attachments_compose_at_confirmed_send_boundary_and_clear_on_success`
(`tests/test_server_queue_persistence.py:481`), which returns the plain
`{"queued": False, "queue_len": 0}` with staged emptied. The new failure test is a
one-variable mirror of it (only the cleanup outcome changes), which isolates the
behavior cleanly.

**5. Pre-send / not-ready / injection / commit_unknown unchanged, staged preserved — MET.**
All of these raise *before* the cleanup block:
- preconditions/not-ready: `session_send.py:44-51` (before delivery);
- injection: `parse_confirmed_send_response` → `injection_error` → 502
  (`session_send.py:84-88`, `control_routes.py:226-227`);
- commit_unknown: `raise_commit_unknown` → 504 (`session_send.py:63-72`,
  `control_routes.py:228-230`).
Because the raise precedes the cleanup block, staged entries are untouched. Locked by
`test_staged_attachments_survive_commit_unknown_send`
(`tests/test_server_queue_persistence.py:558`),
`test_send_route_preserves_allow_pending_and_commit_unknown_status`, and
`test_pending_send_commit_error_preserves_pending_attachment`. None were modified by
this commit.

**6. No hidden PTY/key/write; send remains the sole commit boundary — MET.**
The backend diff adds only a `try/except` and a `dict(...)` copy. No new sock call,
no PTY write, no new key path. The frontend diff only changes toast text and the
guard around the existing UI-clear calls. `git show 4963ba6` confirms the entire
change surface.

---

## Audit questions

**Is catching only `ValueError` appropriate — hides too much/too little?**
Appropriate for the guard; leans toward hiding *too little*.
- *Too much:* negligible. The only realistic `ValueError` in the
  `clear_staged_attachments` chain is the guard
  (`validate_staged_attachment_file_target`, all branches raise `ValueError`).
  `json.dump` in the persistence writes would only raise `ValueError` for circular
  refs / non-finite floats, impossible for this flat string/float data. So the catch
  does not mask an unrelated logic bug in practice.
- *Too little:* real. After the guard passes, `session_store.clear_staged_attachments`
  runs `target.unlink()` (`file_upload.py:148-152`) and then
  `session_pending_state.clear_staged_attachments` runs
  `save_staged_attachments()`/`save_pending_attachments()`
  (`session_pending_state.py:106-107`). These can raise `OSError`
  (read-only mount / EROFS, ENOSPC, EACCES on the parent dir). `OSError` is **not**
  caught, so it propagates out of `send()`, hits no `except` in `_handle_send`, and
  surfaces as HTTP 500 after confirmed delivery — the exact resend-inviting harm the
  commit targets. The persistence-write case is slightly worse: the in-memory pop has
  already happened, so the server truth is *cleared* while the client sees a 500,
  a divergence in the opposite direction. See finding N1.

**Does the coordinator clear `commit_unknown_send` after cleanup failure because
delivery is confirmed?** Yes. `if queue_item_id is None:
self.set_commit_unknown_send(session_id, None)` (`session_send.py:107-108`) sits
*after* the cleanup block and runs unconditionally for non-queue sends. Behavioral
test asserts `clear_unknown_calls == [(sid, None)]`, `sid not in _commit_unknown_sends`,
`session.commit_unknown_send is None`.

**Does frontend success handling avoid restoring composer / implying resend while
surfacing the error and preserving staged UI?** Yes on all four. `sendText` returns
`true` on the cleanup-error path (`app.js:7042`), so the submit handler runs
`clearComposer()` (`app.js:7128`) — composer is cleared, no retry prompt, no resend
implied. The toast appends `; attachment cleanup failed: <err>` (`app.js:7029`).
Staged UI setters are skipped (`app.js:7031-7034`).

**Stale badge/list divergence from not calling
`setSelectedSessionPendingAttachment(false)`?** No. Badge/chips derive from
`Math.max(local, server, pending)`; on cleanup failure both local and server truth
remain populated, and `refreshSessions()` re-reads the still-staged server state, so
they converge (both show "still staged").

**Route status 200, not 500/504?** Yes for the `ValueError` path — verified by
`test_send_route_returns_200_when_confirmed_send_reports_attachment_cleanup_error`
and by code. (An `OSError` cleanup failure would be 500 — finding N1.)

**Tests non-vacuous and at the correct level?** Mixed but adequate:
- `test_staged_attachment_cleanup_failure_after_confirmed_send_is_success_with_visible_error`
  (`test_server_queue_persistence.py:511`): behavioral, correct level (drives real
  `SessionManager.send` through the coordinator). **Non-vacuous** — fails on
  pre-commit source (the raw `ValueError` escapes at `session_send.py:101`).
- The app.js source-string assertions
  (`test_attach_button_source.py:139-145`): **non-vacuous** — fail on pre-commit
  source. Brittle (exact-string/whitespace match) but consistent with this file's
  established source-test convention.
- `test_send_route_returns_200...` (`test_control_routes.py:172`): route-layer
  contract test that mocks `send()` to return the error dict. It **passes on the
  pre-commit source** because `control_routes.py` was not changed by this commit; it
  guards the route contract ("`attachment_cleanup_error` is a 200-carried field, not
  an error") rather than the commit's coordinator change. Legitimate regression guard,
  correct level for what it asserts, but note it does not exercise the code the commit
  actually changed — the coordinator test covers that.

---

## Findings

### N1 (nonblocker) — `ValueError`-only catch leaves the OSError/persistence path exposed to the same bug
- **Where:** `codoxear/session_send.py:100-104`; failure sources
  `codoxear/file_upload.py:148-152` (`unlink`) and
  `codoxear/session_pending_state.py:106-107` (`save_staged_attachments` /
  `save_pending_attachments`).
- **Mechanism:** post-`apply_confirmed_send_success`, any `OSError` from the file
  unlink or the JSON persistence writes propagates uncaught out of `send()`;
  `_handle_send` has no generic `except`, so it becomes HTTP 500 after confirmed
  delivery → duplicate-resend risk. The persistence-write variant additionally clears
  in-memory staged state before failing, so the browser (seeing a 500) diverges from a
  server that has already dropped the staged list.
- **Why nonblocker:** deliverable #1 and the problem statement are scoped to the
  deterministic *guard* failure, which is `ValueError`; that path is fully and
  correctly handled with no regression. The residual requires a filesystem-level
  failure (read-only mount, disk full, permission change on a server-owned dir), which
  is lower probability. The commit is a strict improvement.
- **Recommended fix (optional, closes the class):** broaden the catch to
  `(ValueError, OSError)`, or isolate the entire post-commit tail
  (`clear_staged_attachments` + `set_pending_attachment` + the trailing
  `set_commit_unknown_send(session_id, None)`) so nothing after the confirmed-delivery
  boundary can turn the send into an error; attach the message to
  `attachment_cleanup_error` and log it. Prefer a bounded widen over a blanket
  `except Exception` to avoid masking genuine logic errors. Note the trailing
  `set_commit_unknown_send(session_id, None)` → `save_commit_unknown_sends()`
  (`session_pending_state.py:153`) has the same pre-existing post-commit exposure and
  would be covered by tail-isolation.

### N2 (minor / informational) — dead plural alias in the frontend read
- **Where:** `codoxear/static/app.js:7026`
  (`res.attachment_cleanup_error || res.attachments_cleanup_error`).
- **Mechanism:** the backend only ever emits the singular
  `attachment_cleanup_error` (`session_send.py:103`). The plural
  `attachments_cleanup_error` branch is never produced. Harmless defensive
  redundancy; the test asserts the exact string, so it is intentional. No action
  required.

### N3 (informational, pre-existing — not introduced by this commit)
- If the session is pruned in the window between `apply_confirmed_send_success` and
  `clear_staged_attachments`, `session_pending_state.clear_staged_attachments` raises
  `KeyError("unknown session")` (`session_pending_state.py:102-103`), which
  `_handle_send` maps to **404** *after* delivery succeeded — a false failure. This
  behavior predates the commit (the pre-commit call was un-wrapped and hit the same
  route `except KeyError`) and is extremely unlikely mid-send. Flagged only so it is
  captured alongside N1; tail-isolation would also cover it.

---

## Required fixes

None blocking. Recommended (nonblocker): close N1 by widening the post-commit catch
to at least `(ValueError, OSError)` (or isolating the whole post-confirmation tail).

---

## Evidence checked

- Read: `session_send.py`, `control_routes.py:208-232`, `session_pending_state.py`
  (`clear_staged_attachments`, `set_pending_attachment`, `set_commit_unknown_send`),
  `session_store.py:442-451`, `file_upload.py:106-155`,
  `session_manager_factories.py:598-619` (per-call coordinator rebind confirms the
  behavioral test's `clear_staged_attachments` override is live),
  `app.js` `sendText`/badge setters/submit handler.
- `git show 4963ba6` (full diff); confirmed changed files match the stated set.
- `node --check codoxear/static/app.js` → OK.
- `pytest tests/test_control_routes.py tests/test_attach_button_source.py
  tests/test_server_queue_persistence.py` → **125 passed, 22 subtests**.
- Targeted 9 tests (route + coordinator + full `TestAttachButtonSource`) → **9 passed**.
- Full suite `pytest -q` → **1789 passed, 132 subtests** (matches parent).
- **Mutation / non-vacuity:** `git worktree` at `4963ba6~1`, copied the three
  post-commit test files onto the pre-commit source →
  `test_staged_attachment_cleanup_failure...` and the app.js source assertion
  **FAIL** (raw `ValueError` escapes `send()` at `session_send.py:101`);
  `test_send_route_returns_200...` **passes** (route unchanged → contract guard).
  Worktree removed; `git worktree prune` run.
- `git diff --check` → clean. `git status --short` → empty (working tree untouched by
  the review; the only new commit `a55922e` is memory-only and made by another
  session).

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Review-only task performed against commit 4963ba6 in /home/yiwen/codex-web-product-recovery with no edits/staging/commits; findings written to /tmp/post-send-cleanup-policy-review.md. Scope not widened; protected repo /home/yiwen/codex-web untouched."
    },
    {
      "id": "criterion-2",
      "status": "satisfied",
      "evidence": "Deliverable-by-deliverable verdict with file:line and mechanism, audit-question answers, one nonblocker (OSError/persistence residual) plus two informational notes, and reproduced evidence including a worktree mutation test proving the new tests are non-vacuous."
    }
  ],
  "changedFiles": [],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {"command": "git show 4963ba6", "result": "passed", "summary": "Confirmed diff = try/except+dict copy (session_send.py), toast/guard logic (app.js), 3 test files"},
    {"command": "node --check codoxear/static/app.js", "result": "passed", "summary": "JS parses clean"},
    {"command": "pytest tests/test_control_routes.py tests/test_attach_button_source.py tests/test_server_queue_persistence.py -q", "result": "passed", "summary": "125 passed, 22 subtests"},
    {"command": "pytest <9 targeted route+coordinator+source tests> -v", "result": "passed", "summary": "9 passed"},
    {"command": "pytest -q", "result": "passed", "summary": "1789 passed, 132 subtests"},
    {"command": "git worktree add --detach /tmp/precommit-wt 4963ba6~1 && cp new tests && pytest", "result": "passed", "summary": "Mutation test: coordinator+app.js tests FAIL on pre-commit source (non-vacuous); route test passes (route unchanged)"},
    {"command": "git diff --check && git status --short", "result": "passed", "summary": "Clean; working tree untouched"}
  ],
  "validationOutput": [
    "node --check: OK",
    "focused suites: 125 passed, 22 subtests",
    "targeted: 9 passed",
    "full suite: 1789 passed, 132 subtests",
    "mutation vs 4963ba6~1: test_staged_attachment_cleanup_failure_after_confirmed_send_is_success_with_visible_error FAILED (ValueError escapes send() at session_send.py:101); app.js source assertion FAILED; test_send_route_returns_200 PASSED",
    "git diff --check clean; git status --short empty"
  ],
  "residualRisks": [
    "N1 nonblocker: catch is ValueError-only; OSError from unlink (file_upload.py:148-152) or from save_staged_attachments/save_pending_attachments (session_pending_state.py:106-107) still escapes send() -> HTTP 500 after confirmed delivery -> resend risk; persistence-write variant also clears in-memory staged before failing (server/client divergence). Lower probability (needs FS-level failure); outside the ValueError-scoped deliverable; recommend widening to (ValueError, OSError) or isolating the post-commit tail.",
    "N2 minor: app.js reads a plural attachments_cleanup_error alias the backend never emits (harmless).",
    "N3 pre-existing: session pruned between confirm and cleanup -> KeyError -> 404 after delivery (unchanged by this commit).",
    "During review another session added memory-only commit a55922e on top of 4963ba6; reviewed code unchanged."
  ],
  "noStagedFiles": true,
  "diffSummary": "4963ba6 wraps clear_staged_attachments in try/except ValueError, copying the confirmed-delivery response and adding attachment_cleanup_error instead of raising; frontend surfaces the error in the toast and skips clearing staged/pending UI on cleanup failure while still clearing on success; three tests updated/added.",
  "reviewFindings": [
    "accepted: deliverables 1-6 met for the ValueError guard path (code + non-vacuous behavioral tests + full suite).",
    "nonblocker: session_send.py:100-104 catches only ValueError; OSError from unlink or from save_staged_attachments/save_pending_attachments still becomes a post-delivery 500 (same bug class) - recommend (ValueError, OSError) or post-commit tail isolation.",
    "minor: app.js:7026 dead plural alias attachments_cleanup_error (backend emits singular only).",
    "informational/pre-existing: clear_staged_attachments KeyError on mid-send prune maps to 404 after delivery."
  ],
  "manualNotes": "Review-only run; no files changed in the repo under review and the protected /home/yiwen/codex-web was not touched. HEAD advanced from 4963ba6 to a55922e during review via a memory-only commit by another session (codoxear code unchanged: git diff 4963ba6 HEAD -- codoxear/ tests/ is empty), so the review remains valid. Output written to /tmp/post-send-cleanup-policy-review.md per the runtime path override."
}
```
