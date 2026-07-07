# Clean-room audit: staged attachment path redaction

Verdict: **ACCEPT for the scoped staged-list/browser redaction invariant.** I found no blocker in the current tree. One nonblocking boundary note remains: commit-unknown recovery state can expose the already-composed send text with absolute attachment paths after an explicit send attempt whose receipt is unknown; that is outside the staged-list public projection proved by `f4f38dc`/`e7a02cb`, but should be treated as a separate policy decision if "not confirmed" must mean no path in recovery previews.

## Findings

### Blockers
- **No blockers.** Public staged attachment entries are projected without `path`; confirmed send still uses the internal path; immediate attachment key injection has not reappeared.

### Nonblockers / residual risks
- **Recovery-state path exposure after commit-unknown send attempt.** `codoxear/session_send.py:84-89` stores `commit_unknown_send.text` as `committed_text`, which includes generated `Attachment N: <absolute path>` lines. `codoxear/session_listing.py:218-220` exposes `commit_unknown_send_text` in `/api/sessions`, and `codoxear/static/app.js:2859-2862` / `3439-3443` can display that text in recovery UI. Existing coverage intentionally asserts this behavior in `tests/test_server_queue_persistence.py:747-752`. This is not pre-send staged attachment data, and it occurs only after the explicit send boundary, but it is browser/API-visible before confirmed receipt if the broker times out.

## Success-criteria audit

1. **Browser/API-visible staged attachment data before confirmed send:** **passes for staged attachment payloads.**
   - Internal entries keep `path`, but `public_staged_attachment()` returns only `id`, `display_name`, `filename`, `size`, and `created_ts` (`codoxear/session_store.py:22-52`).
   - Staged list/add/remove coordinator methods return public projections (`codoxear/session_pending_state.py:63-91`, `93-106`).
   - Control routes defensively strip `path` from `attachment`, `attachments`, `removed`, and top-level route payloads (`codoxear/control_routes.py:41-61`), including `/attachments` (`72-82`) and `/inject_file`/`/inject_image` (`318-382`).
   - `/api/sessions` rows redact staged entries both while building active rows and while producing public runtime rows (`codoxear/session_listing.py:216-217`, `496-500`).
   - The browser drops any incoming `item.path` during staged-list normalization and renders chip identity from name, size, and id only (`codoxear/static/app.js:6545-6561`, `6600-6604`).

2. **Confirmed send preserves backend-readable absolute paths:** **passes.**
   - Store cleaning persists internal `path` (`codoxear/session_store.py:385-411`), and staging records `path=str(path)` (`codoxear/session_pending_state.py:71-79`).
   - Send composition reads internal staged entries and prepends `attachment_inject_text(idx, Path(entry["path"]))` before the confirmed send call (`codoxear/session_send.py:73-82`, `95-104`).
   - Regression coverage asserts the backend send payload contains absolute internal upload paths and clears staged state after success (`tests/test_server_queue_persistence.py:503-531`).
   - Docker proof recorded exactly one send containing `/home/tester/.local/share/codoxear/uploads/...` paths and user text (`browser-artifacts/path-redaction-19371/docker-calls-summary.json:7-14`).

3. **Immediate key injection remains retired:** **passes.**
   - Upload route readiness calls `attachment_staging_ready`, stages bytes, and records staged metadata; it does not call `inject_keys` or any attachment-key API (`codoxear/control_routes.py:334-382`). The only `manager.inject_keys` route left in `control_routes.py` is interrupt (`codoxear/control_routes.py:311`).
   - Method bindings expose `attachment_staging_ready` and staged-list methods, with no attachment-key binding (`codoxear/session_manager_method_bindings.py:70-100`).
   - Removed-symbol grep over `codoxear`/`tests` found no implementation of `inject_attachment_keys`, `attachment_injection_ready`, `SessionAttachmentCoordinator`, or `session_attachment`; only negative source assertions remain.
   - Frontend has exactly one `/inject_file` producer path through `stageFiles()` and no `attachment_index` fallback (`tests/test_attach_button_source.py:42-67`).
   - Browser proof used a fake broker with `key_write_errors:false` and recorded `key_count: 0` (`VERIFICATION-REPORT.md:17-20`; `docker-calls-summary.json:10-14`).

4. **Tests/proof strength:** **sufficient for the scoped invariant.**
   - Unit/source coverage checks route redaction, store public projection, session listing public rows, frontend normalization, confirmed-send composition, key-write-error independence, and commit-unknown preservation (`tests/test_control_routes.py:342-420`, `478-495`; `tests/test_session_store.py:39-59`; `tests/test_session_listing.py:138-223`, `255-292`; `tests/test_attach_button_source.py:42-67`; `tests/test_server_queue_persistence.py:503-531`, `924-931`, `735-752`).
   - Browser/Docker proof exercises the real hidden `#imgInput` change listener, captures `/inject_file`, `/attachments`, and `/api/sessions` staged entries without `path`, verifies chip text/title contain no slash, then confirms a single backend send with absolute paths and zero key calls (`VERIFICATION-REPORT.md:7-21`; `browser-redaction-result.json:1`; `docker-calls-summary.json:1-14`).
   - The proof does not cover commit-unknown redaction; current tests assert commit-unknown keeps absolute path text. That is the only identified coverage boundary.

## Validation run in this audit

- `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider ...` focused redaction/send/key-retirement slice: **10 passed in 1.70s**.
- `node --check codoxear/static/app.js`: **passed**.
- Removed-symbol grep: only negative/source-test assertions for retired immediate attachment key symbols.
- `git status --short`: **clean/no staged files** before and after validation.

Prior GLM critic run `1e0e84e3` was ignored as evidence because it failed before review.

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Review-only audit stayed within staged attachment path redaction scope; no repository files were edited."
    },
    {
      "id": "criterion-2",
      "status": "satisfied",
      "evidence": "Report provides file/line evidence for public redaction, internal send composition, key-injection retirement, tests/proof credibility, and residual commit-unknown boundary."
    }
  ],
  "changedFiles": [],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "cd /home/yiwen/codex-web-product-recovery && git branch --show-current && git rev-parse --short HEAD && git status --short",
      "result": "passed",
      "summary": "On recovery/product-gaps at d38f587; status clean."
    },
    {
      "command": "Read required memory/proof artifacts and implementation/test files listed in the prompt",
      "result": "passed",
      "summary": "Inspected VERIFICATION-REPORT.md, browser-redaction-result.json, docker-calls-summary.json, session_store.py, session_pending_state.py, control_routes.py, session_listing.py, session_send.py, app.js, and required tests."
    },
    {
      "command": "PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider [10 focused tests]",
      "result": "passed",
      "summary": "10 passed in 1.70s."
    },
    {
      "command": "node --check codoxear/static/app.js",
      "result": "passed",
      "summary": "JavaScript syntax check passed with no output."
    },
    {
      "command": "rg -n \"inject_attachment_keys|attachment_injection_ready|SessionAttachmentCoordinator|session_attachment\" codoxear tests || true",
      "result": "passed",
      "summary": "No implementation occurrences; only negative test assertions remain."
    },
    {
      "command": "git status --short && git diff --stat",
      "result": "passed",
      "summary": "No repository changes or staged files."
    }
  ],
  "validationOutput": [
    "Focused pytest slice: 10 passed in 1.70s.",
    "Docker proof artifact: preSendContainsUploadRoot=false, preSendStagedEntriesHavePathKey=false, chipTitlesContainSlash=false, chipTextsContainSlash=false.",
    "Docker command summary: send_count=1, key_count=0, send payload contained absolute internal upload paths and user text.",
    "Docker gate artifact: 1790 passed, 1 skipped, 128 subtests passed; smoke 401 before login and 200 /api/sessions after login."
  ],
  "residualRisks": [
    "Commit-unknown recovery state exposes committed_text with absolute attachment paths after an explicit send attempt whose receipt is unknown; current tests assert this and the path-redaction proof does not cover it."
  ],
  "noStagedFiles": true,
  "diffSummary": "No repository diff; wrote this review report only to /tmp/staged-attachment-path-redaction-review.md.",
  "reviewFindings": [
    "no blockers",
    "nonblocker: codoxear/session_send.py:84-89 + codoxear/session_listing.py:218-220 + codoxear/static/app.js:2859-2862/3439-3443 expose absolute paths in commit-unknown recovery text after an explicit but unconfirmed send attempt"
  ],
  "manualNotes": "Final recommendation: accept scoped staged-list redaction; decide separately whether commit-unknown recovery prompt previews should redact generated Attachment N paths."
}
```
