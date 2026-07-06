# Clean-room adversarial review — staged attachment clear hardening (`75986c1`)

Repo: `/home/yiwen/codex-web-product-recovery` (branch `recovery/product-gaps`, HEAD `526ef9d`)
Scope reviewed: functional commit `75986c1 Harden staged attachment clear failures` + memory follow-up `526ef9d`.
Constraints honored: review-only. No edits, no staging, no commits. Working tree verified clean before and after (`git status --porcelain` empty; `git diff --cached` empty). Reproductions written to `/tmp` only; `/home/yiwen/codex-web` untouched.

---

## Verdict: **BLOCKER (narrow, one-line fix)**

The commit correctly hardens the `attachments/clear` route and, more importantly, moves the integrity guarantee into `SessionStore.clear_staged_attachments` (preflight-then-unlink), which protects **all** callers against partial cleanup / list-disk divergence. Requirements #2 (staged-list truth) and #3 (send boundary) are fully met.

It fails one part of its own stated charter. The deliverable is "convert deterministic staged-upload clear cleanup guard failures into explicit client-visible failure **rather than uncaught 500**." A second, reachable, UI-wired staged-clear route — `POST /api/sessions/<id>/pending_attachment/clear` — invokes the identical `clear_staged_attachments` cleanup but its handler catches only `KeyError`. On a tampered/symlinked upload dir it still produces an **uncaught HTTP 500** (with server-side traceback), which is exactly the failure mode the commit was written to eliminate.

Severity calibration: this is a completeness/consistency defect against the charter on a sibling route, **not** a data-integrity defect. No data loss, no false success, no list divergence occurs on that route (the store-level preflight protects it). The error message is still delivered (inside the 500 body). The gap is pre-existing (the guard already raised from this path before `75986c1`) and the fix is a one-line mirror of the `attachments/clear` handler. If the parent scopes acceptance strictly to the `attachments/clear` route named in the original memory finding, this downgrades to a nonblocker fast-follow — but as written, the broader charter ("staged-upload clear cleanup guard failures ... rather than uncaught 500") is not satisfied.

---

## Answers to the five review questions

### Q1 — Is `validate_staged_attachment_file_target` equivalent to the prior guard, and safe for symlink-at-file vs symlink-at-session-dir? — YES

The refactor is a **pure extraction**, verified byte-for-byte. The prior guard body inside `remove_staged_attachment_file` (pre-commit `file_upload.py` lines 114–135) is identical to the body of the new `validate_staged_attachment_file_target` (`file_upload.py:110–139`):

```
$ diff <(git show 75986c1^:codoxear/file_upload.py | sed -n '114,135p') <new validate body>
IDENTICAL: guard extraction is byte-for-byte equivalent
```

`remove_staged_attachment_file` (`file_upload.py:142–156`) now calls `validate_staged_attachment_file_target(...)` then performs the same `target.unlink()` / `FileNotFoundError→False`. Single-file delete/removal semantics (`SessionStore.remove_staged_attachment` → `file_upload.remove_staged_attachment_file`, `session_store.py:435`) are unchanged.

- **symlink-at-session-dir**: rejected. `subdir = root / sid`; `if subdir.is_symlink(): raise ValueError("session upload directory is a symlink")` (`file_upload.py:124–125`). The symlink is never resolved/followed. Confirmed by integration repro: the symlinked `uploads/s1` and its outside target `outside/doc.txt` are both preserved after the raise.
- **symlink-at-file**: allowed and unlinked as a link. The final check `if target.is_dir() and not target.is_symlink(): raise` (`file_upload.py:136–137`) rejects a **real** directory but permits a symlink (including symlink-to-dir, because `is_symlink()` is true). `target.parent.resolve()` resolves the *parent* (the real session dir), not the file target, so a file that is a symlink pointing outside still passes the parent==subdir check, and `unlink()` removes the link entry only — the target is never touched. This matches the documented intent and the prior behavior exactly.

### Q2 — Does `clear_staged_attachments` preflight all entries before unlinking? — YES

`session_store.py:442–451`:
```python
def clear_staged_attachments(self, session_id):
    removed = [dict(entry) for entry in self.staged_attachments.get(session_id, [])]
    uploads_root = self.paths.uploads_root
    if uploads_root is not None:
        for entry in removed:                                   # PASS 1: validate all
            validate_staged_attachment_file_target(uploads_root, session_id, entry["path"])
        for entry in removed:                                   # PASS 2: unlink all
            remove_staged_attachment_file(uploads_root, session_id, entry["path"])
    self.staged_attachments.pop(session_id, None)               # only after both passes
    return removed
```
If any entry fails validation, the exception is raised in PASS 1 **before any unlink** and **before** `self.staged_attachments.pop(...)`, so the in-memory list is left intact. Confirmed by the executor's new test `test_session_store_clear_staged_attachments_preflights_all_entries_before_unlinking` (valid `first.txt` is NOT deleted when a later entry is out-of-scope) and by integration repro (`staged list intact after failures: True`).

Deterministic PASS-2 partial failure after a clean PASS-1 was analyzed and ruled out: all staged files share one parent (the session dir), so `unlink` permission is all-or-nothing at entry 0 (a `PermissionError` would fire on the first unlink, deleting nothing); real directories are rejected in validate; symlink-to-dir `unlink()` removes the link (no `IsADirectoryError`); a missing file yields `FileNotFoundError→False` (not an error). The only residual is a concurrent-tamper TOCTOU race between the two passes, which is non-deterministic and requires an attacker writing into the upload dir while the manager lock is held — out of scope for the "deterministic" guarantee.

At the manager layer this is atomic on failure (`session_pending_state.py:100–108`): the store call is inside `with self.lock`; on `ValueError` it exits before `_sync_pending_projection_locked` (line 105) and before `save_staged_attachments()` (line 106), so neither the pending projection nor the persisted `staged_attachments.json` is mutated. No false success, no silent clear.

### Q3 — Does route-level `ValueError` mapping produce a visible 400 without hiding unknown-session 404? — YES (for the `attachments/clear` route it targets)

`_handle_attachments_clear` (`control_routes.py:179–191`): `except KeyError → 404` (line 185) precedes `except ValueError → 400 {"error": str(e)}` (line 188). `KeyError` and `ValueError` are independent subclasses of `Exception` (neither derives from the other), so ordering causes no shadowing. Unknown-session still yields `KeyError` from the manager **before** the store guard can run (`session_pending_state.py:102`), so 404 is never masked by the new 400 branch. Verified by `test_attachment_clear_maps_tamper_guard_failure_to_400` and by dispatch repro returning `(400, {'error': 'session upload directory is a symlink'})`.

Caveat feeding the blocker: this mapping was added to the `attachments/clear` handler only, not to the sibling `pending_attachment/clear` handler that reaches the same guard (see Finding 1).

### Q4 — Did the change weaken file mode, sibling preservation, symlink safety, or send-boundary invariants? — NO regression

- **File mode**: untouched. `stage_uploaded_file` still `os.chmod(out_path, 0o600)` (`file_upload.py:52`). The commit does not modify staging.
- **Sibling preservation**: `clear`/`remove` unlink only validated direct children of the session dir; no `rmtree` of the session dir, so non-staged siblings survive. `remove_session_uploads` (whole-dir delete on session deletion) is unchanged.
- **Symlink safety**: preserved — the guard extracted is byte-identical (Q1); symlink-at-session-dir rejected, symlink-at-file unlinked as a link and never followed (integration repro: outside target preserved).
- **Send boundary**: preserved. The hardening touches only filesystem validation/unlink in `file_upload.py` and `session_store.py`; no PTY/control-socket write is added to attach/remove/clear. In `session_send.py`, `call_confirmed_send` (commit boundary, line 75) precedes the post-commit `clear_staged_attachments` (line 101); ordering is unchanged. Crucially, the hardening does **not** widen the set of send-path inputs that raise: pre-commit `store.clear_staged_attachments` already called the guard-bearing `remove_staged_attachment_file` in a loop, so any entry that raises now also raised before. Send remains the only attachment commit boundary. (See Residual Risk 1 for a pre-existing post-commit issue this commit neither introduced nor fixed.)

### Q5 — Are tests adequate for the exact failure mechanism? — Adequate for what they cover; one gap let the blocker through

The three added tests correctly exercise: route `ValueError→400` mapping for `attachments/clear`; store rejection of a symlinked session dir with list + target preserved; and store preflight preventing partial cleanup. They are real (not vacuous) and all pass. Focused suite reproduced: `72 passed`.

Gaps:
- **No test drives `pending_attachment/clear`** guard mapping — a test there would have caught Finding 1.
- No test at the **manager** layer that a failed clear skips `save_staged_attachments`/projection sync (verified here only by code reading + integration repro).
- No explicit test that `clear` unlinks a symlink-at-file entry (covered transitively via the shared function), nor that a genuine non-staged sibling inside the session dir survives a clear.
- No test for the send-path post-commit clear failure (Residual Risk 1).

---

## Findings

### Finding 1 — BLOCKER: `pending_attachment/clear` still returns uncaught 500 on the staged-clear guard failure

**Files/lines:**
- `codoxear/control_routes.py:148–156` — `_handle_pending_attachment_clear` catches only `KeyError`; **no `except ValueError`**.
- Route wired at `codoxear/control_routes.py:74` — `("pending_attachment", "clear", _handle_pending_attachment_clear)`.
- `codoxear/session_pending_state.py:110–115` — `clear_pending_attachment` delegates unconditionally to `self.clear_staged_attachments(session_id)` (line 114).
- `codoxear/session_pending_state.py:100–108` — that path calls `self.store().clear_staged_attachments()` (line 104), which raises the new preflight `ValueError`.
- `codoxear/server_handler.py:286–287` → `codoxear/server_http.py:43–47` — uncaught `ValueError` is not `BadRequestError`, so it hits `traceback.print_exc()` + `json_response(handler, 500, ...)`.
- UI caller: `codoxear/static/app.js:7001` — `POST /api/sessions/${sessionId}/pending_attachment/clear` (the "Clear the browser pending-attachment state" confirm flow).

**Mechanism:** Both clear routes converge on `store.clear_staged_attachments`, whose preflight now raises `ValueError` for a tampered/symlinked upload dir. `attachments/clear` maps that to 400; `pending_attachment/clear` does not catch it, so it propagates to the generic handler and becomes an uncaught 500 + traceback — the exact "uncaught 500" the commit set out to convert. The list-truth invariant still holds on this route (the store preflight deletes nothing and the manager skips persistence), so the only harm is the wrong, noisy status code, but that is precisely what the charter forbids.

**Evidence (reproductions, `/tmp`, no repo edits):**
- Route dispatch repro: `attachments/clear -> ('returned', True, [(400, {'error': 'session upload directory is a symlink'})])` vs `pending_attachment/clear -> ('RAISED', 'ValueError', 'session upload directory is a symlink')`.
- Integration repro with the real `SessionPendingStateCoordinator` + real `SessionStore` (symlinked `uploads/s1`): `clear_pending_attachment -> ValueError session upload directory is a symlink`; `staged list intact after failures: True`; `symlink dir preserved: True`; `outside target file preserved: True`.

### Finding 2 — Note: memory follow-up scopes the fix to `attachments/clear` only

`526ef9d` EPISTEMIC.md records the fix for `attachments/clear` and removes the "Harden `attachments/clear`" nonblocking item, but neither the EPISTEMIC nor OPS note mentions the sibling `pending_attachment/clear` route or the send-path post-commit surface. The recorded scope ("no send/queue/frontend paths changed; staged-upload commit boundary remains unchanged") is accurate for what changed but leaves the sibling-route gap undocumented.

---

## Residual risks (pre-existing; not introduced by `75986c1`)

1. **Send path returns 500 after a successful commit on post-commit cleanup failure.** `session_send.py:101` runs `clear_staged_attachments` *after* `call_confirmed_send` (line 75) when sending with staged attachments. `_handle_send` (`control_routes.py:207–228`) catches `KeyError`/not-ready/injection/commit-unknown but **not** `ValueError`, so a guard failure there → 500 *after the message was delivered to the agent*, risking a user re-send / duplicate delivery. Pre-existing (the guard raised from this path before the commit); this commit slightly improves it by removing partial cleanup before the raise. Fixing it properly is a design choice (post-commit cleanup failure arguably should log and still return success), so it is out of scope for a "clear route" hardening but worth tracking.
2. **TOCTOU between preflight and unlink** in `clear_staged_attachments` — non-deterministic, requires concurrent tampering while the manager lock is held; acceptable for the stated "deterministic" guarantee.

---

## Required fix

Mirror the `attachments/clear` handler in `_handle_pending_attachment_clear` (`control_routes.py:148–156`): add `except ValueError as e: deps.json_response(handler, 400, {"error": str(e)}); return` after the existing `except KeyError`. Add a route-level test that drives `pending_attachment/clear` with a manager whose clear raises `ValueError` and asserts `(400, {...})` (parallel to `test_attachment_clear_maps_tamper_guard_failure_to_400`).

Optional (tracks Residual Risk 1): decide send-path post-commit cleanup-failure policy and either map `ValueError` in `_handle_send` or make the post-commit clear non-fatal to the send response.

---

## Evidence checked

- `git show 75986c1` (full diff), `git show 526ef9d`, `git show --stat` for both.
- Byte-for-byte guard equivalence: `diff` of pre-commit `remove_staged_attachment_file` guard vs new `validate_staged_attachment_file_target` → IDENTICAL.
- Full current source: `codoxear/file_upload.py`, `codoxear/session_store.py:412–451`, `codoxear/session_pending_state.py:14–115`, `codoxear/control_routes.py:143–228`, `codoxear/session_send.py`, `codoxear/server_handler.py:250–288`, `codoxear/server_http.py:29–47`.
- Caller/blast-radius map via grep for `clear_staged_attachments` / `clear_pending_attachment` (routes: `attachments/clear` hardened; `pending_attachment/clear` unhardened; `send` post-commit unhardened; `attachments/delete` already maps `ValueError`).
- Focused suite: `python3 -m pytest -q tests/test_control_routes.py tests/test_file_upload.py tests/test_session_store.py` → **72 passed in 1.73s**.
- `git diff --check 75986c1^ 75986c1` → clean.
- Two standalone reproductions in `/tmp` (route dispatch + real-coordinator integration) confirming the 400-vs-uncaught-500 asymmetry and the preserved list-truth/symlink safety.
- Post-review `git status --porcelain` empty; `git diff --cached` empty (no edits, nothing staged).

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Clean-room review confined to /home/yiwen/codex-web-product-recovery commits 75986c1 and 526ef9d; no source edits, no staging, no commits; /home/yiwen/codex-web untouched; reproductions written only to /tmp."
    },
    {
      "id": "criterion-2",
      "status": "satisfied",
      "evidence": "Findings cite file:line and mechanism; byte-for-byte guard-equivalence diff; two runnable reproductions (route dispatch + real-coordinator integration); focused pytest 72 passed and git diff --check clean reproduced; verdict and required fix stated."
    }
  ],
  "changedFiles": [],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "git show 75986c1 / git show 526ef9d / git show --stat",
      "result": "passed",
      "summary": "Full functional + memory diffs inspected"
    },
    {
      "command": "diff <(git show 75986c1^:codoxear/file_upload.py | sed -n '114,135p') <new validate body>",
      "result": "passed",
      "summary": "Guard extraction byte-for-byte identical"
    },
    {
      "command": "python3 -m pytest -q tests/test_control_routes.py tests/test_file_upload.py tests/test_session_store.py",
      "result": "passed",
      "summary": "72 passed in 1.73s (executor claim reproduced)"
    },
    {
      "command": "git diff --check 75986c1^ 75986c1",
      "result": "passed",
      "summary": "No whitespace/conflict errors"
    },
    {
      "command": "python3 /tmp/repro_pending_clear.py",
      "result": "passed",
      "summary": "attachments/clear -> 400; pending_attachment/clear -> RAISED ValueError (uncaught)"
    },
    {
      "command": "python3 /tmp/repro_integration_pending_clear.py",
      "result": "passed",
      "summary": "Real coordinator: clear_pending_attachment raises guard ValueError; staged list + symlink + outside target all preserved"
    },
    {
      "command": "git status --porcelain / git diff --cached --name-only",
      "result": "passed",
      "summary": "Working tree clean; nothing staged before and after review"
    }
  ],
  "validationOutput": [
    "72 passed in 1.73s",
    "git diff --check 75986c1^ 75986c1: clean",
    "guard extraction: IDENTICAL (byte-for-byte)",
    "route dispatch repro: attachments/clear=(400, error json); pending_attachment/clear=RAISED ValueError (propagates to 500)",
    "integration repro: clear_pending_attachment raises 'session upload directory is a symlink'; staged list intact=True; symlink dir preserved=True; outside target preserved=True",
    "git status --porcelain: empty (no repo mutation, nothing staged)"
  ],
  "residualRisks": [
    "BLOCKER: pending_attachment/clear (control_routes.py:148-156) catches only KeyError; delegates to the same clear_staged_attachments guard and returns uncaught HTTP 500 (server_http.py:43-47) on a tampered/symlinked upload dir - the exact failure mode the commit charter targets. List-truth is still preserved on that route; fix is a one-line except ValueError mirror.",
    "Pre-existing: send path (session_send.py:101) runs clear_staged_attachments after commit; _handle_send does not map ValueError, so a post-commit guard failure yields 500 after the message was delivered (possible re-send/duplicate). Not introduced by this commit.",
    "Non-deterministic TOCTOU between preflight and unlink passes in clear_staged_attachments; requires concurrent tampering under the held manager lock; acceptable for the deterministic guarantee.",
    "Test gap: no coverage for pending_attachment/clear guard mapping, manager-level save/projection skip on failure, or symlink-at-file within clear."
  ],
  "noStagedFiles": true,
  "diffSummary": "75986c1 extracts the staged-file guard into validate_staged_attachment_file_target (byte-identical to prior guard), makes clear_staged_attachments preflight all entries before unlinking (no deterministic partial cleanup / list divergence), and maps ValueError->400 on the attachments/clear route. Store-level integrity holds for all callers; send boundary and file mode unchanged. Gap: sibling pending_attachment/clear route reaches the same guard but lacks the ValueError->400 mapping, so it still returns uncaught 500.",
  "reviewFindings": [
    "blocker: control_routes.py:148-156 - _handle_pending_attachment_clear reaches the same clear_staged_attachments guard as attachments/clear but catches only KeyError, so a tampered/symlinked upload dir yields an uncaught HTTP 500 (server_http.py:43-47) instead of the charter-required visible 400. Confirmed by route-dispatch and real-coordinator integration repros.",
    "nonblocker: session_send.py:101 + control_routes.py:207-228 - post-commit clear_staged_attachments can raise ValueError after a successful send commit; _handle_send does not map it -> 500 after delivery. Pre-existing, not introduced by 75986c1.",
    "pass: validate_staged_attachment_file_target is byte-for-byte equivalent to the prior single-file guard; symlink-at-session-dir rejected, symlink-at-file unlinked as link.",
    "pass: clear_staged_attachments preflights all entries before unlinking; list-truth preserved in-memory and on disk on failure.",
    "pass: attachments/clear maps ValueError->400 without masking unknown-session 404; send boundary, file mode 0o600, and sibling preservation unchanged."
  ],
  "manualNotes": "Verdict is blocker against the deliverable's explicit 'rather than uncaught 500' wording, on the reachable UI-wired pending_attachment/clear route (app.js:7001). It is narrow: no data loss/false success/list divergence (store-level preflight protects it), pre-existing, and fixable in one line by mirroring the attachments/clear handler. If the parent scopes acceptance strictly to the attachments/clear route from the original memory finding, this is a nonblocker fast-follow; the store-level preflight and byte-identical guard extraction are correct and accepted. Reproductions are in /tmp/repro_pending_clear.py and /tmp/repro_integration_pending_clear.py (outside the repo)."
}
```
