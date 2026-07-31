# Photo attachment affordance clean-room review

Recommendation: **accept**. I found **no blockers**.

## Findings

### Blockers

- None.

### Nonblockers / proof boundaries

- The updated unit coverage in `tests/test_attach_button_source.py` is source-string coverage, not executable UI behavior. The Docker/browser artifact covers the real `#captureInput` `change` listener with a synthetic `File`, which is sufficient for this product decision because the accepted invariant is “add/stage a photo truthfully,” not guaranteed hardware camera invocation.
- The working tree was already dirty at review start with an unstaged `.memory/tasks/2026-07-07-staged-upload-expansion/OPS.md` change. I did not edit repo files; `git diff --cached --stat --exit-code` confirmed no staged files.

## Evidence

- Visible affordance is truthful: `codoxear/static/app.js:1242-1248` defines `#captureBtn` with `title: "Add photo"` and `aria-label: "Add photo"`; `codoxear/static/app.js:6703-6719` keeps the dynamic enabled label at `Add photo (max ...)`.
- Mobile capture support remains: `codoxear/static/app.js:1255-1256` keeps the hidden `#captureInput` as `type="file"`, `accept="image/*"`, `capture="environment"`.
- No-name capture fallback now describes a photo: `codoxear/static/app.js:6855-6858` returns `photo-<seed>[...].<ext>`.
- Capture still uses the shared staged path: `codoxear/static/app.js:6958-6974` gates the button with `attachmentBlockerForSession`, opens the hidden input, and the `change` listener calls `stageFiles(files, { sid, source: "capture" })`.
- Shared blocker and upload boundary are preserved: `codoxear/static/app.js:6861-6910` re-confirms selection, rechecks `latestAttachmentBlockerForSession()` before progress/compression/`arrayBuffer()`, and posts to the sole `/inject_file` call with `{ filename, data_b64 }`.
- Public staged entries still omit backend path in frontend normalization: `codoxear/static/app.js:6545-6555` maps only `id`, `display_name`, `filename`, `size`, and `created_ts`.
- Source tests pin the relevant invariants: `tests/test_attach_button_source.py:99-121` asserts Add-photo copy, retained hidden input attributes, `photoFileName`, `staging photo...`, and `stageFiles(...source: "capture")`; `tests/test_attach_button_source.py:70` and `:122` assert only one `/inject_file` occurrence; `tests/test_attach_button_source.py:58` asserts no `attachment_index`; `tests/test_attach_button_source.py:61-65` assert no frontend path fallback.
- The reviewed functional diff did not touch backend route/state/key/PTY/send code: `537fd18` changes only `codoxear/static/app.js` and `tests/test_attach_button_source.py`; `107af43` adds only proof artifacts under `.memory/tasks/.../photo-affordance-19381/`.
- Browser proof exercises the real listener and visible behavior: `VERIFICATION-REPORT.md:7-15` records Add-photo title/aria, retained `accept/capture`, real `#captureInput.files` + `change` dispatch, `staging photo...`, and `photo-*.jpg` display name.
- Public/private and send-boundary proof holds: `VERIFICATION-REPORT.md:18-27` records no public `path`, no upload-root leak, zero pre-send broker `send`/`keys`, then exactly one explicit-send broker `send` with generated `Attachment 1: <path>`.
- Raw compact call logs match the report: `docker-calls-after-stage-compact.json:1-8` has only `state` calls with `send_count=0`, `key_count=0`; `docker-calls-after-send-compact.json:1-11` has one `send`, zero `keys`, and the generated attachment payload.

## Proof/test credibility

The test layer is brittle but directly guards the intended source invariants. The browser proof is stronger: its stage driver wraps `fetch` only to delay `/inject_file`, installs a no-name JPEG on `#captureInput.files`, dispatches the real `change` event, reads `/attachments` and `/api/sessions`, and checks broker call summaries. That distinguishes the mechanism under audit: photo producer → shared staged upload → no backend write before explicit send.

## Verdict

Accept `537fd18` + `107af43` for this slice. The visible product promise now says “Add photo,” the mobile capture hint remains, and no evidence shows a widened backend or send-boundary scope.

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Only frontend copy/fallback/progress and source-test guards changed in the functional commit; capture still uses #captureInput and shared stageFiles(source: \"capture\") with the same /inject_file send boundary."
    },
    {
      "id": "criterion-2",
      "status": "satisfied",
      "evidence": "Report cites source lines, source tests, browser/Docker artifacts, compact broker call logs, and locally rerun node/pytest/diff checks."
    }
  ],
  "changedFiles": [
    "codoxear/static/app.js",
    "tests/test_attach_button_source.py",
    ".memory/tasks/2026-07-07-staged-upload-expansion/browser-artifacts/photo-affordance-19381/VERIFICATION-REPORT.md",
    ".memory/tasks/2026-07-07-staged-upload-expansion/browser-artifacts/photo-affordance-19381/api-login.json",
    ".memory/tasks/2026-07-07-staged-upload-expansion/browser-artifacts/photo-affordance-19381/api-me-before-login.json",
    ".memory/tasks/2026-07-07-staged-upload-expansion/browser-artifacts/photo-affordance-19381/api-me-before-login.status",
    ".memory/tasks/2026-07-07-staged-upload-expansion/browser-artifacts/photo-affordance-19381/api-sessions-initial.json",
    ".memory/tasks/2026-07-07-staged-upload-expansion/browser-artifacts/photo-affordance-19381/browser-after-photo-stage.json",
    ".memory/tasks/2026-07-07-staged-upload-expansion/browser-artifacts/photo-affordance-19381/browser-after-send.json",
    ".memory/tasks/2026-07-07-staged-upload-expansion/browser-artifacts/photo-affordance-19381/browser-ready.json",
    ".memory/tasks/2026-07-07-staged-upload-expansion/browser-artifacts/photo-affordance-19381/docker-calls-after-send-compact.json",
    ".memory/tasks/2026-07-07-staged-upload-expansion/browser-artifacts/photo-affordance-19381/docker-calls-after-stage-compact.json",
    ".memory/tasks/2026-07-07-staged-upload-expansion/browser-artifacts/photo-affordance-19381/docker-final-state.txt",
    ".memory/tasks/2026-07-07-staged-upload-expansion/browser-artifacts/photo-affordance-19381/docker-smoke-19382.txt",
    ".memory/tasks/2026-07-07-staged-upload-expansion/browser-artifacts/photo-affordance-19381/docker-start-19381.txt",
    ".memory/tasks/2026-07-07-staged-upload-expansion/browser-artifacts/photo-affordance-19381/docker-test-19382.txt",
    ".memory/tasks/2026-07-07-staged-upload-expansion/browser-artifacts/photo-affordance-19381/fake-start.txt",
    ".memory/tasks/2026-07-07-staged-upload-expansion/browser-artifacts/photo-affordance-19381/fake_photo_session.py",
    ".memory/tasks/2026-07-07-staged-upload-expansion/browser-artifacts/photo-affordance-19381/photo-affordance-send-driver.js",
    ".memory/tasks/2026-07-07-staged-upload-expansion/browser-artifacts/photo-affordance-19381/photo-affordance-stage-driver.js"
  ],
  "testsAddedOrUpdated": [
    "tests/test_attach_button_source.py"
  ],
  "commandsRun": [
    {
      "command": "git status --short --branch",
      "result": "passed",
      "summary": "On recovery/product-gaps; one pre-existing unstaged OPS.md modification visible."
    },
    {
      "command": "git show --stat --oneline --decorate 537fd18 107af43 --",
      "result": "passed",
      "summary": "Confirmed functional commit changes app.js/test only; proof commit adds photo-affordance artifacts."
    },
    {
      "command": "node --check codoxear/static/app.js",
      "result": "passed",
      "summary": "No JavaScript syntax errors."
    },
    {
      "command": "python3 -m pytest -q tests/test_attach_button_source.py",
      "result": "passed",
      "summary": "8 passed in 0.44s."
    },
    {
      "command": "git diff --check 537fd18^..107af43 -- codoxear/static/app.js tests/test_attach_button_source.py .memory/tasks/2026-07-07-staged-upload-expansion/browser-artifacts/photo-affordance-19381",
      "result": "passed",
      "summary": "No whitespace errors reported."
    },
    {
      "command": "git diff --cached --stat --exit-code",
      "result": "passed",
      "summary": "No staged files."
    }
  ],
  "validationOutput": [
    "node --check codoxear/static/app.js: no output, exit 0",
    "python3 -m pytest -q tests/test_attach_button_source.py: 8 passed in 0.44s",
    "git diff --check reviewed range: no output, exit 0",
    "docker artifact report: 1792 passed, 1 skipped, 128 subtests passed; smoke 401 pre-login and 200 post-login"
  ],
  "residualRisks": [
    "Browser proof uses a synthetic File/DataTransfer rather than a physical camera picker; nonblocking because the product invariant is truthful add-photo staging while preserving capture=environment for mobile browsers.",
    "Source tests are string-based; browser/Docker proof covers the real listener and network/broker boundary."
  ],
  "noStagedFiles": true,
  "diffSummary": "Functional change renames visible capture affordance to Add photo, changes no-name capture fallback from captured-* to photo-*, changes capture progress copy to staging photo, and updates source guards. Proof commit records Docker/browser evidence only. No backend route/state/key/PTY/send-boundary files changed.",
  "reviewFindings": [
    "no blockers",
    "nonblocker: proof boundary is synthetic file selection rather than hardware camera invocation"
  ],
  "manualNotes": "Review-only task honored. I wrote only /tmp/photo-affordance-review.md. Current git status still shows a pre-existing unstaged .memory/tasks/2026-07-07-staged-upload-expansion/OPS.md modification; nothing is staged."
}
```
