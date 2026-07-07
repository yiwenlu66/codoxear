# Clean-room review: `18cd64c Remove immediate attachment key injection`

## Verdict

**Accepted.** No blockers and no required fixes.

The commit removes the obsolete pre-send attachment key/PTY-write mechanism while preserving the staged upload invariant: upload producers stage bytes through `/inject_file`/`/inject_image`; explicit send is still the only backend commit boundary; send still prepends generated `Attachment N: <path>` lines from staged entries.

## Findings

### 1. Scope matches the old immediate-key path removal

No blocker.

Mechanism removed:
- `codoxear/session_attachment.py` is deleted; its former `SessionAttachmentCoordinator.inject_attachment_keys()` path performed input locking, `attachment_injection_ready()`, `inject_keys(..., track_request_sent=True)`, and pending-attachment mutation after broker key writes (deleted file, former lines 7-43 in `18cd64c^`).
- `codoxear/session_manager_factories.py` removes the `SessionAttachmentCoordinator` import and `attachment_coordinator_for_manager()` factory (diff former lines 8 and 288-298).
- `codoxear/session_manager_method_bindings.py` removes the `_attachment_coordinator_for_manager`, `attachment_injection_ready`, and `inject_attachment_keys` bindings (diff former lines 38, 101, and 136).

Staged route preserved:
- `/api/sessions/<sid>/inject_file` and `/inject_image` still route to `_handle_inject_attachment` (`codoxear/control_routes.py:86-90`).
- The handler checks `manager.attachment_staging_ready(session_id)` before decode/stage (`codoxear/control_routes.py:293-324`), writes staged bytes through `deps.stage_uploaded_file` (`codoxear/control_routes.py:328-329`), and records staged metadata with `manager.add_staged_attachment(...)` (`codoxear/control_routes.py:334-357`). There is no attachment backend send/key call in that handler.
- Browser upload producers still post only `{ filename, data_b64 }` to `/inject_file` (`codoxear/static/app.js:6853-6902`).

### 2. No tracked runtime references to the removed attachment API remain

No blocker.

`git grep` across tracked `codoxear` and `tests` for `SessionAttachmentCoordinator|session_attachment|attachment_injection_ready|inject_attachment_keys|_attachment_coordinator_for_manager` returns only negative/source assertions:
- `tests/test_file_upload_module_source.py:59,83,85`
- `tests/test_session_manager_method_bindings.py:141,145,146`

`codoxear/session_attachment.py` is absent from the tracked tree. `SessionManager` still exposes generic `inject_keys()` for interrupt/control use (`codoxear/server.py:986-992`; `/interrupt` route at `codoxear/control_routes.py:281-290`), but no attachment route calls it.

Ignored local generated artifacts under `build/`, `codoxear.egg-info/`, and `.pytest_cache/` contain stale old-symbol strings. They are git-ignored and not part of the reviewed commit or source runtime; a clean build should regenerate them before using `build/lib` as an import path.

### 3. Staging readiness keeps the needed blockers and correctly drops key-write-error support

No blocker.

`attachment_staging_ready()` now delegates to `_attachment_ready(..., allow_existing_pending=True)` (`codoxear/session_readiness.py:148-149`). The readiness mechanism still blocks:
- unknown session: `KeyError` before and after state refresh (`codoxear/session_readiness.py:117-119`, `133-135`);
- commit-unknown send: `SessionNotReadyError` before and after refresh (`codoxear/session_readiness.py:120-121`, `136-137`);
- missing confirmed-send capability: `sync_send_supported` check (`codoxear/session_readiness.py:122-123`);
- active queue send item: `queue_sending_item_id` checks (`codoxear/session_readiness.py:126-127`, `140-141`);
- local queue: `queue_len` checks before and after broker state (`codoxear/session_readiness.py:128-129`, `142-144`);
- broker/log runtime busy: refreshed state plus `runtime_status_from_state_and_log(...)`, then `session_runtime_readiness(...).direct_send` (`codoxear/session_readiness.py:131`, `145-146`).

The removed `key_write_errors_supported` precondition is correct for stage-only upload because staging no longer writes keys before send. This is directly covered by `tests/test_server_queue_persistence.py:892-899`, which sets `key_write_errors_supported = False` and expects `attachment_staging_ready()` to succeed.

### 4. Immediate-key commit/error tests were removed for a deleted API; staged pending behavior remains tested through real staged state

No blocker.

The removed tests were for `SessionManager.inject_attachment_keys()` response classification and readiness recheck, which no longer has an API surface. The replacement coverage exercises the live mechanism:
- stage readiness blockers: `tests/test_server_queue_persistence.py:269-297`, `299-308`, `310-334`, `353-401`, `1667-1672`, `1862-1874`;
- pending attachment blocks queue/direct send until explicit send: `tests/test_server_queue_persistence.py:403-428`;
- send-boundary composition and success cleanup from staged entries: `tests/test_server_queue_persistence.py:471-499`;
- commit-unknown send preserves staged entry and records the composed text: `tests/test_server_queue_persistence.py:703-720`;
- route-level stage-only behavior with no backend paste: `tests/test_control_routes.py:359-381`;
- legacy `attachment_index` remains ignored: `tests/test_control_routes.py:480-488`.

### 5. Send-boundary attachment numbering remains intact

No blocker.

`file_upload.attachment_inject_text()` still implements the numbering format and rejects non-positive indexes (`codoxear/file_upload.py:155-159`). The send coordinator imports it (`codoxear/session_send.py:7`), enumerates staged entries starting at 1 (`codoxear/session_send.py:77-81`), prepends the prefix to the user text (`codoxear/session_send.py:82`), and sends exactly one confirmed `cmd: send` payload (`codoxear/session_send.py:95-104`). Staged cleanup happens only after confirmed response parsing (`codoxear/session_send.py:105-130`).

Tests preserve the numbering semantics:
- direct helper tests: `tests/test_file_upload.py:264-277`;
- one staged attachment send payload: `tests/test_server_queue_persistence.py:403-428`;
- two staged attachments numbered 1/2: `tests/test_server_queue_persistence.py:471-499`.

### 6. Compatibility risk is deliberate and not material for the current architecture

No blocker.

A non-HTTP or in-process caller that directly used `SessionManager.inject_attachment_keys()` will now get `AttributeError`. That is an intentional deletion of a dead compatibility API, not a live product regression:
- `codoxear.__all__` exports only `__version__` (`codoxear/__init__.py:1-3`), so the removed method was not a declared package API.
- Tracked source grep finds no remaining in-repo caller.
- The current architecture’s attachment contract is HTTP upload staging plus explicit send-boundary composition, not in-process key injection.
- Keeping the old method would preserve the exact hidden PTY/key-write path this commit is supposed to make impossible.

## Required fixes

None.

## Evidence checked

- Reviewed commit diff for `18cd64c^..18cd64c` and changed-file list.
- Inspected current staged route, readiness coordinator, send coordinator, pending-state/store/listing state, frontend upload/send calls, and tests.
- `git grep` for removed symbols in tracked runtime/tests: only negative/source assertions remain.
- `python3 -m pytest -q tests/test_server_queue_persistence.py tests/test_file_upload_module_source.py tests/test_control_routes.py tests/test_session_manager_method_bindings.py tests/test_file_upload.py tests/test_send_ack.py` → `163 passed, 18 subtests passed in 3.92s`.
- `python3 -m pytest -q` → `1788 passed, 128 subtests passed in 24.42s`.
- `git diff --check` → clean.
- `git status --short --branch` → `## recovery/product-gaps` with no modified tracked files.
- `git diff --cached --quiet` → no staged files.

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Commit 18cd64c deletes the immediate-key attachment coordinator/API and its manager bindings, while /inject_file and /inject_image still stage bytes via attachment_staging_ready and add_staged_attachment; send still composes Attachment N lines from staged entries."
    },
    {
      "id": "criterion-2",
      "status": "satisfied",
      "evidence": "Review records changed files, source/grep evidence, targeted and full pytest output, diff-check output, and no-staged-files evidence."
    }
  ],
  "changedFiles": [
    "codoxear/server.py",
    "codoxear/session_attachment.py",
    "codoxear/session_manager_factories.py",
    "codoxear/session_manager_method_bindings.py",
    "codoxear/session_readiness.py",
    "tests/test_control_routes.py",
    "tests/test_file_upload_module_source.py",
    "tests/test_server_queue_persistence.py",
    "tests/test_session_manager_method_bindings.py"
  ],
  "testsAddedOrUpdated": [
    "tests/test_control_routes.py",
    "tests/test_file_upload_module_source.py",
    "tests/test_server_queue_persistence.py",
    "tests/test_session_manager_method_bindings.py"
  ],
  "commandsRun": [
    {
      "command": "git diff --name-status 18cd64c^ 18cd64c && git diff --stat 18cd64c^ 18cd64c",
      "result": "passed",
      "summary": "Confirmed 9 changed files: one deleted coordinator, source binding/readiness edits, and test updates."
    },
    {
      "command": "git diff --no-ext-diff --unified=0 18cd64c^ 18cd64c -- codoxear/session_attachment.py codoxear/session_manager_factories.py codoxear/session_manager_method_bindings.py codoxear/session_readiness.py",
      "result": "passed",
      "summary": "Confirmed old SessionAttachmentCoordinator/factory/bindings removed and readiness no longer uses require_key_write_errors."
    },
    {
      "command": "git grep -n -E 'SessionAttachmentCoordinator|session_attachment|attachment_injection_ready|inject_attachment_keys|_attachment_coordinator_for_manager' -- codoxear tests || true",
      "result": "passed",
      "summary": "Only negative/source assertions remain in tracked tests; no tracked runtime reference remains."
    },
    {
      "command": "git grep -n -E 'attachment_staging_ready|add_staged_attachment|clear_staged_attachments|attachment_inject_text|staged_attachments_for_session' -- codoxear tests",
      "result": "passed",
      "summary": "Confirmed staged route/state and send-boundary attachment composition are still present."
    },
    {
      "command": "python3 -m pytest -q tests/test_server_queue_persistence.py tests/test_file_upload_module_source.py tests/test_control_routes.py tests/test_session_manager_method_bindings.py tests/test_file_upload.py tests/test_send_ack.py",
      "result": "passed",
      "summary": "163 passed, 18 subtests passed in 3.92s."
    },
    {
      "command": "python3 -m pytest -q",
      "result": "passed",
      "summary": "1788 passed, 128 subtests passed in 24.42s."
    },
    {
      "command": "git diff --check",
      "result": "passed",
      "summary": "No whitespace errors."
    },
    {
      "command": "git status --short --branch && git diff --cached --quiet",
      "result": "passed",
      "summary": "Branch shown with no tracked modifications and no staged files."
    }
  ],
  "validationOutput": [
    "Targeted pytest: 163 passed, 18 subtests passed in 3.92s.",
    "Full pytest: 1788 passed, 128 subtests passed in 24.42s.",
    "git diff --check: clean.",
    "Removed-symbol grep over tracked codoxear/tests: only negative/source assertions remain.",
    "No staged files."
  ],
  "residualRisks": [
    "Non-HTTP/in-process callers of SessionManager.inject_attachment_keys will break; no tracked caller remains, the method was not a declared package API, and preserving it would keep the forbidden pre-send key-write mechanism.",
    "Ignored local build/cache artifacts contain stale old-symbol strings; they are outside the commit and should be regenerated before using build/lib as a runtime import path."
  ],
  "noStagedFiles": true,
  "diffSummary": "Deleted the immediate attachment key injection coordinator and manager binding, removed attachment_injection_ready, simplified staging readiness to sync-send-only capability, and updated tests/source assertions around staged state and send-boundary behavior.",
  "reviewFindings": [
    "no blockers",
    "no required fixes"
  ],
  "manualNotes": "Review performed from current branch recovery/product-gaps; source after 18cd64c is unchanged by later memory-only commits."
}
```
