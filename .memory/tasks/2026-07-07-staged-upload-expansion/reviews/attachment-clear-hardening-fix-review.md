# Clean-room verification — attachment-clear hardening blocker fix (`b7148bb`)

Repo: `/home/yiwen/codex-web-product-recovery` (branch `recovery/product-gaps`).
Scope reviewed: fix commit `b7148bb Map legacy attachment clear guard failures`, verified against prior hardening `75986c1` and the blocker recorded in `.memory/tasks/2026-07-07-staged-upload-expansion/reviews/attachment-clear-hardening-review.md`.
Constraints honored: **review-only — no edits, no staging, no commits**. `/home/yiwen/codex-web` untouched. The only artifact I created is `/tmp/repro_pending_clear_e2e.py` (outside the repo).

---

## Verdict: **ACCEPTED — blocker resolved**

The prior blocker was: `_handle_pending_attachment_clear` caught only `KeyError`, so the deterministic staged-cleanup guard `ValueError` (raised by `SessionStore.clear_staged_attachments`'s preflight) propagated to an uncaught HTTP 500 + traceback — the exact failure mode the hardening charter set out to eliminate. `b7148bb` applies precisely the one-line fix the prior review prescribed and adds a parallel route-level test. Both deliverable clauses are confirmed end-to-end against live code.

---

## What the fix does (`git show b7148bb`)

Two files, **+24 lines, 0 deletions** — minimal and correctly scoped:

- `codoxear/control_routes.py` — adds to `_handle_pending_attachment_clear`, after the existing `except KeyError → 404`:
  ```python
  except ValueError as e:
      deps.json_response(handler, 400, {"error": str(e)})
      return
  ```
  This makes the handler mirror the already-hardened `_handle_attachments_clear`.
- `tests/test_control_routes.py` — adds `test_pending_attachment_clear_maps_tamper_guard_failure_to_400`, a route-dispatch test using a `TamperedManager` whose `clear_pending_attachment` raises `ValueError`; asserts the route returns `True`, invokes the manager once, reads no body (`read_body_count == 0`, correct for this handler), and produces `(400, {"error": "session upload directory is a symlink"})`.

`git show 27ca144 --name-only` and `git show b7148bb --name-only` confirm no store/manager/send/PTY file was touched by the fix (or by the later unrelated frontend commit `27ca144`).

---

## Deliverable clause 1 — `pending_attachment/clear` maps guard `ValueError` → 400, preserves 404 and store-list truth: **CONFIRMED**

**Route mapping (current source at live HEAD):**
```python
def _handle_pending_attachment_clear(handler, *, session_id, manager, deps):
    if not _authorized(handler, deps):
        return
    try:
        res = manager.clear_pending_attachment(session_id)
    except KeyError:
        deps.json_response(handler, 404, {"error": "unknown session"})
        return
    except ValueError as e:
        deps.json_response(handler, 400, {"error": str(e)})
        return
    deps.json_response(handler, 200, res)
```

- **404 not shadowed by the new branch.** `KeyError` and `ValueError` are independent `Exception` subclasses (neither derives from the other), so clause ordering causes no shadowing. More decisively, unknown-session raises `KeyError` inside `SessionPendingStateCoordinator.clear_pending_attachment` **before** the store guard can run (`session_pending_state.py`: the `if session_id not in self.sessions(): raise KeyError` check precedes the delegated `clear_staged_attachments` call). So an unknown session can never reach the `ValueError` branch.
- **Only the deterministic tamper guard reaches the route as `ValueError`.** Tracing `clear_pending_attachment → clear_staged_attachments (manager) → store().clear_staged_attachments() → validate_staged_attachment_file_target`, the sole `ValueError` source on this path is the preflight guard (symlinked session dir / out-of-scope path / real-directory target). No try/except swallows it between the store and the route. Mapping it to 400 is exact, not over-broad.
- **Store-level staged-list truth invariant preserved by construction.** `b7148bb` does not touch `session_store.py`, `file_upload.py`, or `session_pending_state.py`. `git log --oneline -- codoxear/session_store.py codoxear/file_upload.py` shows those files last changed at `75986c1`. The preflight-then-unlink structure (validate **all** entries, only then unlink any, only then `pop` the in-memory list) and the byte-identical `validate_staged_attachment_file_target` guard are unchanged from the previously-accepted hardening. On a guard failure nothing is unlinked, the in-memory list is left intact, and the manager exits before `_sync_pending_projection_locked` / `save_staged_attachments`, so neither the projection nor `staged_attachments.json` is mutated.

**End-to-end proof through the REAL stack** (`/tmp/repro_pending_clear_e2e.py`: real `handle_control_post_route` + real `SessionPendingStateCoordinator` + real `SessionStore` + real guard; only mocks are auth/json-response capture and no-op persistence hooks):

```
CASE tamper:          handled=True raised=None responses=[(400, {'error': 'session upload directory is a symlink'})]
                      list_intact=True symlink=True target=True saves={'staged': 0, 'pending': 0} -> PASS
CASE unknown-session: handled=True raised=None responses=[(404, {'error': 'unknown session'})] -> PASS
CASE happy-path:      handled=True raised=None responses=[(200, {'ok': True, 'pending_attachment': False})]
                      cleared=True file_gone=True saves={'staged': 1, 'pending': 1} -> PASS
ALL PASS
```
- **tamper** (symlinked `uploads/s1 → outside/`): route returns, does **not** raise, emits 400; staged list intact, symlink and outside target both preserved, zero persistence writes — the previously-uncaught 500 is now a clean 400 with the list-truth invariant held.
- **unknown-session**: 404 preserved; the new `ValueError` branch does not intercept it.
- **happy-path** (real non-tampered file): 200, file unlinked, list cleared, persistence written — the new branch does **not** fire spuriously; no happy-path regression.

## Deliverable clause 2 — no send-boundary or PTY/key-write behavior changed: **CONFIRMED**

- `b7148bb --name-only` = `codoxear/control_routes.py`, `tests/test_control_routes.py` only. `_handle_send` and `codoxear/session_send.py` are byte-unchanged; the send commit boundary (`call_confirmed_send` before post-commit `clear_staged_attachments`) is untouched.
- `_handle_pending_attachment_clear` contains no PTY/socket/key-write. The only `inject_keys` in `control_routes.py` is in the unrelated interrupt handler (`control_routes.py:287`), not on this path.
- The fix does not widen the set of inputs that raise on the send path; it only changes how the *clear* route renders an already-possible `ValueError`.

---

## Evidence checked

- `git show b7148bb` / `--stat` / `--name-only` — 2 files, +24/-0; matches prescribed fix + parallel test.
- Current `codoxear/control_routes.py:148–159` (`_handle_pending_attachment_clear`) and `:184–194` (`_handle_attachments_clear`) — mirror confirmed.
- `codoxear/session_pending_state.py` — `clear_pending_attachment` / `clear_staged_attachments` raise `KeyError` before the guard; no `ValueError` swallow.
- `codoxear/session_store.py:442–451` + `codoxear/file_upload.py:106–156` — preflight-then-unlink and guard unchanged since `75986c1` (`git log --oneline -- ...`).
- `python3 -m pytest -q tests/test_control_routes.py::test_attachment_clear_maps_tamper_guard_failure_to_400 tests/test_control_routes.py::test_pending_attachment_clear_maps_tamper_guard_failure_to_400 tests/test_control_routes.py tests/test_file_upload.py tests/test_session_store.py` → **73 passed** (reproduces parent's claim).
- `python3 -m pytest -v <two named mapping tests>` → **2 passed** individually.
- `python3 /tmp/repro_pending_clear_e2e.py` → **ALL PASS** (tamper→400 no-raise / unknown→404 / happy→200), list-truth + symlink safety + no-spurious-fire confirmed through real code.
- `git diff --check` clean; caller map via `grep clear_staged_attachments|clear_pending_attachment` (routes: `attachments/clear` and `pending_attachment/clear` both map `ValueError`; `send` post-commit still unmapped — see residual).

**Concurrency note (not introduced by this review):** during review the repo advanced from `731da2c` to `27ca144 Add paste and drop attachment producers` (another session's work). `27ca144` touches only frontend files (`app.css`, `app.js`, `app_file_helpers.js`) and their source tests — no backend clear/send/store path — so it does not affect this verification; all checks above were re-confirmed against live HEAD `27ca144`. The single untracked path `.memory/.../browser-artifacts/upload-producers-19343/` is that session's artifact, not mine. I made no edits, stages, or commits.

---

## Residual risks (all pre-existing; correctly out of scope for this narrowly-scoped fix)

1. **Send-path post-commit cleanup still returns 500** — `session_send.py:101` runs `clear_staged_attachments` *after* `call_confirmed_send`, and `_handle_send` maps `KeyError`/not-ready/injection/commit-unknown but **not** `ValueError`. A guard failure there yields 500 *after* the message was delivered (possible user re-send). This is the prior review's Residual Risk 1, pre-existing, and explicitly excluded by the deliverable ("No send-boundary … behavior changed"). Not a blocker; worth tracking as a fast-follow with an explicit post-commit-cleanup-failure policy.
2. **TOCTOU between preflight and unlink** in `clear_staged_attachments` — non-deterministic, requires concurrent tampering under the held manager lock; acceptable for the stated "deterministic" guarantee.

## Verdict rationale

The blocker's exact mechanism — `pending_attachment/clear` → uncaught 500 on the staged-clear guard — is closed by an `except ValueError → 400` mirror, proven end-to-end through the real route/coordinator/store. Unknown-session 404 and the store-level staged-list truth invariant are preserved (verified live). The send boundary and PTY/key-write surface are untouched. The change is minimal, scoped, and covered by a real (non-vacuous) test. **Accepted.**

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "b7148bb is a scoped 2-file, +24/-0 change: adds `except ValueError -> 400` to _handle_pending_attachment_clear (control_routes.py) mirroring _handle_attachments_clear, plus a parallel route test. No store/manager/send/PTY file touched; store-list-truth guard unchanged since 75986c1. Clean-room honored: no repo edits/stages/commits; /home/yiwen/codex-web untouched."
    },
    {
      "id": "criterion-2",
      "status": "satisfied",
      "evidence": "Verdict backed by: git show diff inspection; live-HEAD source read of the handler + call chain (KeyError before guard => 404 never shadowed; only deterministic guard ValueError reaches route); focused pytest 73 passed and 2 named mapping tests passed individually; real end-to-end repro (/tmp/repro_pending_clear_e2e.py) proving tamper->400 no-raise with list/symlink/target preserved, unknown->404, happy->200 with no spurious fire; git diff --check clean; concurrency note explaining HEAD advance 731da2c->27ca144 is unrelated frontend work."
    }
  ],
  "changedFiles": [],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "git show b7148bb / --stat / --name-only",
      "result": "passed",
      "summary": "Fix = +24/-0 across control_routes.py + tests/test_control_routes.py; adds except ValueError->400 + parallel test"
    },
    {
      "command": "sed -n control_routes.py _handle_pending_attachment_clear (live HEAD 27ca144)",
      "result": "passed",
      "summary": "Handler mirrors _handle_attachments_clear: KeyError->404, ValueError->400, else 200"
    },
    {
      "command": "git log --oneline -- codoxear/session_store.py codoxear/file_upload.py codoxear/session_pending_state.py",
      "result": "passed",
      "summary": "Store/guard/coordinator last changed at 75986c1; b7148bb did not touch them => list-truth invariant preserved by construction"
    },
    {
      "command": "python3 -m pytest -q tests/test_control_routes.py::test_attachment_clear_maps_tamper_guard_failure_to_400 tests/test_control_routes.py::test_pending_attachment_clear_maps_tamper_guard_failure_to_400 tests/test_control_routes.py tests/test_file_upload.py tests/test_session_store.py",
      "result": "passed",
      "summary": "73 passed (reproduces parent claim)"
    },
    {
      "command": "python3 -m pytest -v <two named mapping tests>",
      "result": "passed",
      "summary": "2 passed individually"
    },
    {
      "command": "python3 /tmp/repro_pending_clear_e2e.py",
      "result": "passed",
      "summary": "Real route+coordinator+store: tamper->400 no-raise (list/symlink/target preserved, 0 saves); unknown->404; happy->200 (file unlinked, list cleared). ALL PASS"
    },
    {
      "command": "git diff --check",
      "result": "passed",
      "summary": "clean"
    },
    {
      "command": "git status --porcelain / git log --oneline",
      "result": "passed",
      "summary": "No changes by this review; HEAD advanced to unrelated frontend commit 27ca144; single untracked browser-artifacts dir belongs to that session"
    }
  ],
  "validationOutput": [
    "73 passed in 1.84s (full focused suite)",
    "2 passed in 1.65s (test_attachment_clear_maps_tamper_guard_failure_to_400, test_pending_attachment_clear_maps_tamper_guard_failure_to_400)",
    "e2e tamper: handled=True raised=None responses=[(400, {'error':'session upload directory is a symlink'})] list_intact=True symlink=True target=True saves={'staged':0,'pending':0} -> PASS",
    "e2e unknown-session: handled=True raised=None responses=[(404, {'error':'unknown session'})] -> PASS",
    "e2e happy-path: handled=True raised=None responses=[(200,{'ok':True,'pending_attachment':False})] cleared=True file_gone=True saves={'staged':1,'pending':1} -> PASS",
    "git diff --check: clean",
    "b7148bb --name-only: codoxear/control_routes.py, tests/test_control_routes.py (no send/PTY/store paths)",
    "27ca144 --name-only: frontend only (app.css, app.js, app_file_helpers.js, *source tests) - does not affect reviewed surface"
  ],
  "residualRisks": [
    "Pre-existing (out of scope, not a blocker): send path session_send.py:101 runs clear_staged_attachments after commit; _handle_send does not map ValueError, so a post-commit guard failure yields HTTP 500 after delivery (possible re-send). Explicitly excluded by the 'no send-boundary change' constraint; track as fast-follow with a defined post-commit-cleanup-failure policy.",
    "Non-deterministic TOCTOU between preflight and unlink passes in clear_staged_attachments; requires concurrent tampering under the held manager lock; acceptable for the deterministic guarantee.",
    "Repo advanced under review (731da2c -> 27ca144) via another session's frontend-only work; does not affect the reviewed backend path but means the branch is live and moving."
  ],
  "noStagedFiles": true,
  "diffSummary": "b7148bb adds `except ValueError as e: json_response(400, {'error': str(e)})` to _handle_pending_attachment_clear (mirroring _handle_attachments_clear) plus a parallel route-dispatch test. Closes the blocker where the deterministic staged-cleanup guard ValueError propagated to an uncaught HTTP 500. Store/manager/send/PTY paths untouched; unknown-session 404 and store-level staged-list-truth invariant preserved (verified end-to-end).",
  "reviewFindings": [
    "no blockers: pending_attachment/clear now maps deterministic staged-cleanup guard ValueError -> 400 (control_routes.py _handle_pending_attachment_clear), verified end-to-end through real route+coordinator+store",
    "no blockers: unknown-session 404 preserved - KeyError raised in coordinator before the store guard; new ValueError branch cannot shadow it (e2e unknown-session case -> 404)",
    "no blockers: store-level staged-list truth invariant preserved by construction - b7148bb does not touch session_store.py/file_upload.py; preflight-then-unlink unchanged since 75986c1; tamper case deletes nothing and leaves list intact",
    "no blockers: no send-boundary or PTY/key-write change - fix touches only the clear handler + test; _handle_send and session_send.py byte-unchanged; handler has no PTY/socket write",
    "note (pre-existing, out of scope): _handle_send still does not map ValueError from the post-commit clear (session_send.py:101) -> 500 after delivery; track as fast-follow, not a blocker for this deliverable"
  ],
  "manualNotes": "Clean-room honored: zero repo edits/stages/commits by this review; only artifact created is /tmp/repro_pending_clear_e2e.py (outside repo). During review the branch advanced 731da2c -> 27ca144 ('Add paste and drop attachment producers', frontend-only) and an untracked .memory/.../browser-artifacts/upload-producers-19343/ dir appeared - both belong to a concurrent session, not this review, and do not affect the reviewed backend surface; all checks re-confirmed against live HEAD 27ca144. Output written to /tmp/attachment-clear-hardening-fix-review.md per run-authoritative path."
}
```
