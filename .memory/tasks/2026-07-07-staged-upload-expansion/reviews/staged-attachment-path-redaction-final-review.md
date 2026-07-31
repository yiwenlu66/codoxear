# Final clean-room audit: staged attachment path redaction

Verdict: **ACCEPT**. I found **no blockers** against the requested redaction invariant at current `recovery/product-gaps` HEAD `39d3efa400d2df075df438edd4e9278ba827d9d5`.

## Blockers

None.

## Nonblockers / boundaries

- Legacy commit-unknown redaction is intentionally narrow: it redacts leading server-generated `Attachment N: /...` prefix lines (`codoxear/session_listing.py:145-161`). It is not a general sanitizer for arbitrary user-authored absolute paths elsewhere in prompt text.
- Public attachment `display_name` remains the client-provided filename (`codoxear/control_routes.py:360-364`, `codoxear/session_pending_state.py:71-79`). Browser producers supply `File.name` basenames; the audited invariant is that the server-generated backend-readable staged `path` is not projected publicly.

## Success-criteria audit

1. **Pre-send browser/API staged attachment payloads do not expose internal upload paths: passes.**
   - `public_staged_attachment()` omits stored `path` and returns only `id`, `display_name`, `filename`, `size`, `created_ts` (`codoxear/session_store.py:22-56`).
   - The private cleaner still stores `path`, but the public list methods and pending-state coordinator return only public projections (`codoxear/session_store.py:385-411`, `449-453`; `codoxear/session_pending_state.py:63-91`, `93-116`).
   - Control routes defensively project attachment responses and remove any top-level `path` before returning `/attachments`, `/inject_file`, `/inject_image`, delete, or clear responses (`codoxear/control_routes.py:41-61`, `72-82`, `180-218`, `318-382`).
   - `/api/sessions` sanitizes active and public rows (`codoxear/session_listing.py:212-242`, `518-523`).
   - The browser normalization drops any incoming `item.path` and renders chips from name/size/id only (`codoxear/static/app.js:6545-6561`, `6600-6604`).
   - Docker/browser proof: `path-redaction-19371/browser-redaction-result.json` has `preSendContainsUploadRoot:false`, `preSendStagedEntriesHavePathKey:false`, `chipTitlesContainSlash:false`, and `chipTextsContainSlash:false`.

2. **Confirmed send and commit-unknown private state preserve backend-readable absolute paths: passes.**
   - Internal staged records retain `path` (`codoxear/session_store.py:385-411`; `codoxear/session_pending_state.py:71-79`).
   - Send composition reads internal `entry["path"]` and prepends generated attachment lines before the confirmed send call (`codoxear/session_send.py:73-104`).
   - Commit-unknown records store private committed text plus public display text (`codoxear/session_send.py:84-89`), and the cleaner preserves `display_text` (`codoxear/session_pending_state.py:125-145`).
   - Regression coverage asserts send payloads contain absolute attachment paths and commit-unknown private state keeps full `text` while setting `display_text` to the user prompt (`tests/test_server_queue_persistence.py:503-531`, `735-753`, `849-862`).
   - Docker proof: `path-redaction-19371/docker-calls-summary.json` recorded one send, zero keys, and absolute `/home/tester/.local/share/codoxear/uploads/...` attachment paths. `commit-unknown-redaction-19373/docker-private-state.txt` shows private `text` with the upload path and `display_text` without it.

3. **Commit-unknown public recovery preview no longer exposes generated attachment paths, including legacy records without `display_text`: passes.**
   - Public commit-unknown text prefers `display_text`; legacy records fall back through `_redact_generated_attachment_prefix_paths()` (`codoxear/session_listing.py:145-171`).
   - Active and orphan recovery rows both use `_commit_unknown_text()` (`codoxear/session_listing.py:240-242`, `479-481`).
   - Browser recovery UI consumes only `commit_unknown_send_text` (`codoxear/static/app.js:2856-2866`, `2986-2999`, `3439-3443`; `codoxear/static/app_recovery.js:153-170`). On immediate 504, the local optimistic preview is the raw user prompt, not the server-generated attachment prefix (`codoxear/static/app.js:7108-7116`).
   - Tests cover display-text use and legacy fallback redaction (`tests/test_session_listing.py:374-401`), and commit-unknown send persistence (`tests/test_server_queue_persistence.py:735-753`).
   - Docker/browser proof: `commit-unknown-redaction-19373/browser-commit-unknown-result.json` has `rowCommitUnknownTextEqualsPrompt:true`, `rowCommitUnknownTextContainsAttachmentLine:false`, `rowContainsUploadRoot:false`, `publicPayloadContainsUploadRoot:false`, and `bodyContainsUploadRoot:false`.

4. **Immediate key injection remains retired: passes.**
   - Upload route readiness is `attachment_staging_ready`; it stages bytes and records metadata, with no key/PTY write path (`codoxear/control_routes.py:334-382`). The only `manager.inject_keys` use in `control_routes.py` is the interrupt route (`codoxear/control_routes.py:306-315`).
   - SessionManager bindings expose staged-list methods and `attachment_staging_ready`, with no attachment-key binding (`codoxear/session_manager_method_bindings.py:71-100`).
   - Staging readiness no longer depends on key-write-error support (`codoxear/session_readiness.py:114-149`; `tests/test_server_queue_persistence.py:927-934`).
   - Removed-symbol grep found no implementation of `inject_attachment_keys`, `attachment_injection_ready`, `SessionAttachmentCoordinator`, or `session_attachment`; only negative/source assertions remain. Browser/Docker proofs recorded `key_count:0`.

5. **Tests/proofs are adequate: passes.**
   - Focused local validation over the requested test files passed: `160 passed, 18 subtests passed in 1.89s`.
   - `node --check codoxear/static/app.js` passed.
   - Path-redaction proof exercised the real hidden `#imgInput` change listener, public `/inject_file` + `/attachments` + `/api/sessions` payloads, chip rendering, explicit send, and fake-broker command summary.
   - Commit-unknown proof deterministically forced `commit_unknown:true`, then checked browser/API public preview, preserved staged entry without `path`, private JSON state, send payload, and zero key calls.
   - Docker gates recorded in artifacts passed: path redaction `1790 passed, 1 skipped, 128 subtests passed` plus smoke 401/200; commit-unknown redaction `1792 passed, 1 skipped, 128 subtests passed` plus smoke 401/200.

## Commands run in this audit

- `cd /home/yiwen/codex-web-product-recovery && git rev-parse --abbrev-ref HEAD && git rev-parse HEAD && git status --short --branch`
- Read task/project memory and the required implementation/test/proof artifacts.
- `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider tests/test_session_listing.py tests/test_server_queue_persistence.py tests/test_control_routes.py tests/test_session_store.py tests/test_attach_button_source.py` → `160 passed, 18 subtests passed in 1.89s`.
- `node --check codoxear/static/app.js` → passed.
- `rg -n "inject_attachment_keys|attachment_injection_ready|SessionAttachmentCoordinator|session_attachment" codoxear tests || true` → only negative/source-test assertions.
- `git diff --stat && git status --short --branch` → no diff; clean branch status.

Final recommendation: **accept**.

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Review-only audit stayed within the requested staged attachment path redaction scope; no files in the repository were edited and no protected checkout/runtime dirs were touched."
    },
    {
      "id": "criterion-2",
      "status": "satisfied",
      "evidence": "Report gives file/line evidence for public redaction, private send/commit-unknown preservation, legacy preview redaction, key-injection retirement, and proof/test credibility."
    }
  ],
  "changedFiles": [],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "cd /home/yiwen/codex-web-product-recovery && git rev-parse --abbrev-ref HEAD && git rev-parse HEAD && git status --short --branch",
      "result": "passed",
      "summary": "On recovery/product-gaps at 39d3efa400d2df075df438edd4e9278ba827d9d5; status clean."
    },
    {
      "command": "PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider tests/test_session_listing.py tests/test_server_queue_persistence.py tests/test_control_routes.py tests/test_session_store.py tests/test_attach_button_source.py",
      "result": "passed",
      "summary": "160 passed, 18 subtests passed in 1.89s."
    },
    {
      "command": "node --check codoxear/static/app.js",
      "result": "passed",
      "summary": "JavaScript syntax check passed."
    },
    {
      "command": "rg -n \"inject_attachment_keys|attachment_injection_ready|SessionAttachmentCoordinator|session_attachment\" codoxear tests || true",
      "result": "passed",
      "summary": "No implementation occurrences; only negative/source-test assertions remain."
    },
    {
      "command": "git diff --stat && git status --short --branch",
      "result": "passed",
      "summary": "No repository diff and no staged files."
    }
  ],
  "validationOutput": [
    "Focused audit test suite: 160 passed, 18 subtests passed in 1.89s.",
    "Path-redaction browser artifact: preSendContainsUploadRoot=false, preSendStagedEntriesHavePathKey=false, chipTitlesContainSlash=false, chipTextsContainSlash=false.",
    "Path-redaction broker summary: send_count=1, key_count=0, send payload contained absolute upload paths and user text.",
    "Commit-unknown browser artifact: rowCommitUnknownTextEqualsPrompt=true, rowCommitUnknownTextContainsAttachmentLine=false, rowContainsUploadRoot=false, publicPayloadContainsUploadRoot=false, bodyContainsUploadRoot=false.",
    "Commit-unknown private artifact: private text retained absolute upload path; display_text retained only the prompt.",
    "Docker gates in artifacts: 1790 passed/1 skipped/128 subtests and 1792 passed/1 skipped/128 subtests; smoke gates returned 401 before login and 200 /api/sessions after login."
  ],
  "residualRisks": [
    "Legacy commit-unknown fallback redacts only leading generated Attachment N absolute-path prefix lines, not arbitrary absolute paths elsewhere in user text.",
    "Public display_name reflects the client-provided filename; browser producers provide basename File.name values, and generated internal staged paths are omitted."
  ],
  "noStagedFiles": true,
  "diffSummary": "Review-only: no repository files changed by this audit. Audited implementation/proof commits touched the listed code, tests, and .memory proof artifacts for staged attachment redaction and commit-unknown preview redaction.",
  "reviewFindings": [
    "no blockers",
    "nonblocker: codoxear/session_listing.py:145-161 redacts the generated leading Attachment N prefix shape only",
    "nonblocker: display_name is client-provided filename, while generated internal path is omitted from public projections"
  ],
  "manualNotes": "Final recommendation: accept current HEAD for staged attachment path redaction."
}
```
