# Upload producer polish clean-room audit

Verdict: **ACCEPT**. I found no blockers against the stated upload-producer invariants.

## Blockers

- None.

## Nonblockers / residual proof boundaries

- The new Docker/browser proof directly exercises mixed text+file paste, jpeg/webp pasted fallback names, window-leave highlight clearing, off-composer drop non-staging, and send-boundary delivery. It does not separately browser-exercise text-only paste, file-only paste, or png/gif fallback names in this proof artifact; those paths are covered by source structure and `tests/test_attach_button_source.py` assertions.
- Task memory is slightly stale: `.memory/tasks/2026-07-07-staged-upload-expansion/EPISTEMIC.md` still lists the producer polish nits as remaining. This is not a product/code blocker for the audited commits.

## File/line evidence

- `codoxear/static/app.js:6688-6697` centralizes attachment producer blockers in `attachmentBlockerForSession()`.
- `codoxear/static/app.js:6832-6848` maps pasted/captured image MIME types to extensions, including jpeg/jpg, png, gif, and webp.
- `codoxear/static/app.js:6857-6906` is the single shared `stageFiles()` path; it checks the blocker before the only `/inject_file` call and sends only `filename` + `data_b64`.
- `codoxear/static/app.js:6949,6968,7002,7036` are the picker, capture, paste, and drop producer calls into `stageFiles()`.
- `codoxear/static/app.js:6996-7002` preserves mixed pasted text by reading `text/plain`, preventing default only for file-bearing paste, inserting text, then staging files. Text-only paste returns before `preventDefault()`.
- `codoxear/static/app.js:7005-7057` clears `.drop-active` on composer drop, window leave, dragend, and any window drop; the window drop handler prevents file navigation and does not call `stageFiles()`.
- `codoxear/control_routes.py:318-367` keeps `/inject_file` stage-only: readiness check, decode, `stage_uploaded_file()`, and `add_staged_attachment()`; no backend key/PTY write.
- `codoxear/session_send.py:77-99` composes `Attachment N: <path>` lines only inside confirmed send, immediately before `call_confirmed_send()`.
- `tests/test_attach_button_source.py:45-66` asserts the shared `stageFiles()` pipeline, one `/inject_file` occurrence, no `attachment_index`, and no public path normalization.
- `tests/test_attach_button_source.py:86-99` asserts the MIME extension helper and capture routing through `stageFiles()`.
- `tests/test_attach_button_source.py:111-131` asserts mixed paste text insertion ordering, drop/highlight clearing, window drop fail-safe, and no off-composer `stageFiles()` call.

## Proof/test credibility

- `.memory/tasks/2026-07-07-staged-upload-expansion/browser-artifacts/upload-producer-polish-19377/browser-producer-polish-result.json` has `success:true`; it records `pasteDefaultPrevented:true`, `textareaAfterMixedPaste:"prefix-MIXED-TEXTsuffix"`, two `/inject_file` filenames ending `.jpg` and `.webp`, `activeAfterWindowLeave:false`, `offComposerDropDefaultPrevented:true`, `offComposerDropDidNotStage:true`, and `attachmentsAfterSend.attachments:[]`.
- `.memory/tasks/2026-07-07-staged-upload-expansion/browser-artifacts/upload-producer-polish-19377/docker-calls-summary.json` records `send_count:1`, `key_count:0`, and the only send payload containing the two generated attachment path lines plus the user prompt.
- `.memory/tasks/2026-07-07-staged-upload-expansion/browser-artifacts/upload-producer-polish-19377/VERIFICATION-REPORT.md` describes an isolated Docker sandbox on port 19377 with fake broker capabilities `sync_send:true` and `key_write_errors:false`, which discriminates stage-only upload from the removed pre-send key-write path.
- The same artifact directory records Docker gate output: `1792 passed, 1 skipped, 128 subtests passed` and smoke `pre_login_api_me_status=401`, `post_login_sessions_status=200`, app dir `/home/tester/.local/share/codoxear`.

## Recommendation

Accept `b4b018b` and `54a66d1` for the upload producer polish. The code implements the requested behavior without widening the commit boundary or reintroducing pre-send backend writes.

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Audited commits are scoped to app.js producer polish, source tests, and proof artifacts; all file producers route through stageFiles(), and send remains the only backend commit boundary."
    },
    {
      "id": "criterion-2",
      "status": "satisfied",
      "evidence": "Report includes file/line evidence, artifact evidence, commands run, validation output, residual risks, and clean git status."
    }
  ],
  "changedFiles": [
    "codoxear/static/app.js",
    "tests/test_attach_button_source.py",
    ".memory/tasks/2026-07-07-staged-upload-expansion/browser-artifacts/upload-producer-polish-19377/VERIFICATION-REPORT.md",
    ".memory/tasks/2026-07-07-staged-upload-expansion/browser-artifacts/upload-producer-polish-19377/browser-producer-polish-result.json",
    ".memory/tasks/2026-07-07-staged-upload-expansion/browser-artifacts/upload-producer-polish-19377/docker-calls-summary.json",
    ".memory/tasks/2026-07-07-staged-upload-expansion/browser-artifacts/upload-producer-polish-19377/upload-producer-polish-driver.js",
    ".memory/tasks/2026-07-07-staged-upload-expansion/browser-artifacts/upload-producer-polish-19377/fake_upload_session.py",
    ".memory/tasks/2026-07-07-staged-upload-expansion/browser-artifacts/upload-producer-polish-19377/docker-test-19378.txt",
    ".memory/tasks/2026-07-07-staged-upload-expansion/browser-artifacts/upload-producer-polish-19377/docker-smoke-19378.txt"
  ],
  "testsAddedOrUpdated": [
    "tests/test_attach_button_source.py"
  ],
  "commandsRun": [
    {
      "command": "cd /home/yiwen/codex-web-product-recovery && git status --short && git branch --show-current && git log --oneline -5",
      "result": "passed",
      "summary": "Confirmed target branch recovery/product-gaps at 54a66d1 with clean status."
    },
    {
      "command": "cd /home/yiwen/codex-web-product-recovery && git show --stat --oneline b4b018b && git show --stat --oneline 54a66d1 && git diff --name-only b4b018b^ 54a66d1",
      "result": "passed",
      "summary": "Confirmed implementation scope: app.js, tests/test_attach_button_source.py, and upload-producer-polish proof artifacts."
    },
    {
      "command": "cd /home/yiwen/codex-web-product-recovery && node --check codoxear/static/app.js && python3 -m pytest -q tests/test_attach_button_source.py && git status --short",
      "result": "passed",
      "summary": "JavaScript syntax check succeeded; source test file reported 8 passed; git status remained clean."
    },
    {
      "command": "cd /home/yiwen/codex-web-product-recovery && git grep -n \"inject_attachment_keys\\|attachment_injection_ready\" -- codoxear tests || true; count /inject_file and stageFiles callers in app.js",
      "result": "passed",
      "summary": "Removed-symbol grep found only negative/source assertions; app.js contains one /inject_file occurrence and producer callers only for picker/capture/paste/drop into stageFiles()."
    },
    {
      "command": "cd /home/yiwen/codex-web-product-recovery && git diff --check && git status --short",
      "result": "passed",
      "summary": "Whitespace check clean; no staged or unstaged changes in target repo."
    }
  ],
  "validationOutput": [
    "node --check codoxear/static/app.js: passed with no output",
    "python3 -m pytest -q tests/test_attach_button_source.py: 8 passed in 0.43s",
    "artifact docker-test-19378.txt: 1792 passed, 1 skipped, 128 subtests passed in 46.15s",
    "artifact docker-smoke-19378.txt: pre-login /api/me 401; post-login /api/sessions 200; app dir /home/tester/.local/share/codoxear",
    "browser-producer-polish-result.json: success true; mixed paste text preserved; jpg/webp names staged; off-composer drop did not stage; attachments cleared after send",
    "docker-calls-summary.json: send_count 1; key_count 0; send payload contained generated jpg/webp attachment paths and prompt"
  ],
  "residualRisks": [
    "Current browser proof samples jpeg/webp fallback names but not png/gif; source helper/tests cover png/gif.",
    "Current browser proof does not separately exercise text-only or file-only paste after this polish; source structure and prior producer proof cover those paths.",
    "Task EPISTEMIC memory still lists the now-addressed producer polish nits as remaining."
  ],
  "noStagedFiles": true,
  "diffSummary": "b4b018b adds MIME-derived pasted/captured names, mixed paste text insertion, and stronger drag/drop highlight clearing while preserving the shared stageFiles upload path; 54a66d1 records Docker/browser proof artifacts.",
  "reviewFindings": [
    "no blockers",
    "nonblocker: proof matrix could add dedicated browser cases for text-only paste, file-only paste, and png/gif pasted fallback names",
    "nonblocker: task memory EPISTEMIC has not been updated to mark producer polish nits closed"
  ],
  "manualNotes": "Review-only audit. I did not edit files in /home/yiwen/codex-web-product-recovery and did not touch protected /home/yiwen/codex-web or live runtime dirs; the only written artifact is /tmp/upload-producer-polish-review.md."
}
```
