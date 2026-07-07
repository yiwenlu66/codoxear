# Upload batch blocker recheck clean-room review

Verdict: **ACCEPT**. I found no blockers in commits `ec37237` and `1a986a9` for the stated upload-batch blocker invariant.

## Blockers

- None.

## Nonblockers / residual risks

- The browser proof stops the batch through a `commit_unknown` marker; the public-row `busy` blocker is verified by source/code inspection (`info.busy`) and source tests, not by an isolated busy-only browser scenario.
- If readiness flips after the per-file client check but before the `/inject_file` request, the server route remains authoritative and will reject staging; this client loop would count that as an upload failure until polling refreshes the row. That is a residual race, but it does not widen the commit boundary or contradict the requested before-work recheck.

## File/line evidence

- `codoxear/static/app.js:6688-6701` centralizes attachment blockers, now including `if (info && info.busy)`, and `latestAttachmentBlockerForSession()` re-reads the current `sessionIndex` row for the target session.
- `codoxear/static/app.js:6872-6880` performs the per-file loop check in the required order: confirm `selected === sessionId`, then call `latestAttachmentBlockerForSession(sessionId)`, then break before doing upload work.
- `codoxear/static/app.js:6881-6907` shows progress toast, compression, `arrayBuffer()`, base64 conversion, and the sole `/inject_file` request all occur after the blocker recheck.
- `codoxear/static/app.js:6911-6914` applies successful staged-list responses immediately, so a staged partial batch remains visible before a later blocker stops remaining files.
- `codoxear/static/app.js:6923-6928` reports partial stop distinctly: `attached ${successes}; stopped: ${stoppedByBlocker}`.
- `codoxear/static/app.js:6955,6974,7008,7042` shows picker, capture, paste, and drop all route through `stageFiles()`; `grep` found exactly one `/inject_file` occurrence, at `app.js:6907`.
- `tests/test_attach_button_source.py:48-52` asserts the public row busy guard, latest blocker helper, and stop-on-blocker branch exist.
- `tests/test_attach_button_source.py:72-84` asserts the ordering: selected-session check < latest-blocker check < progress toast < compression < arrayBuffer < `/inject_file`.

## Proof/test credibility

- The source test is narrow but aimed at the invariant most likely to regress: the upload pipeline ordering and single shared `/inject_file` route.
- The browser proof uses the real `#imgInput` change listener and real `/inject_file` route. Its fetch wrapper delays the first upload response, creates an external `commit_unknown` blocker through the real `/send` route, then releases the first response so `stageFiles()` must re-read the updated session state before file 2.
- `browser-batch-blocker-result.json` shows `injectRequestCount: 1`, `injectFilenames: ["first.txt"]`, `stoppedToast: true`, final toast `attached 1; stopped: Resolve the unknown send before attaching a file`, and `stagedCountAfterBatch: 1` with no public `path` key.
- `docker-calls-summary.json` shows `key_count: 0`, `max_staged_file_count: 1`, and `send_count: 1` equal to `marker_send_count: 1`; the one send is the deliberate proof marker, not an upload producer write.
- `VERIFICATION-REPORT.md:27-32` records local syntax/source/full-suite validation plus Docker test/smoke gates.

## Recommendation

Accept. The implementation preserves the staged-upload model: client producers still funnel through `stageFiles()`, blockers are rechecked before each file’s expensive/API work, partial success remains visible, and no backend route/state/key/PTY path is added.

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Diff is limited to frontend blocker/recheck logic, source tests, and proof artifacts; no backend route/state changes and app.js still has exactly one /inject_file call inside stageFiles()."
    },
    {
      "id": "criterion-2",
      "status": "satisfied",
      "evidence": "Code line evidence, source ordering tests, browser proof JSON, docker broker summary, and re-run validation are sufficient for independent acceptance review."
    }
  ],
  "changedFiles": [
    "codoxear/static/app.js",
    "tests/test_attach_button_source.py",
    ".memory/tasks/2026-07-07-staged-upload-expansion/browser-artifacts/upload-batch-blocker-19379/VERIFICATION-REPORT.md",
    ".memory/tasks/2026-07-07-staged-upload-expansion/browser-artifacts/upload-batch-blocker-19379/browser-batch-blocker-result.json",
    ".memory/tasks/2026-07-07-staged-upload-expansion/browser-artifacts/upload-batch-blocker-19379/docker-calls-summary.json",
    ".memory/tasks/2026-07-07-staged-upload-expansion/browser-artifacts/upload-batch-blocker-19379/upload-batch-blocker-driver.js",
    ".memory/tasks/2026-07-07-staged-upload-expansion/browser-artifacts/upload-batch-blocker-19379/fake_batch_blocker_session.py"
  ],
  "testsAddedOrUpdated": [
    "tests/test_attach_button_source.py"
  ],
  "commandsRun": [
    {
      "command": "cd /home/yiwen/codex-web-product-recovery && git status --short --branch && git log --oneline -5",
      "result": "passed",
      "summary": "Confirmed branch recovery/product-gaps at 1a986a9 with no working-tree output."
    },
    {
      "command": "cd /home/yiwen/codex-web-product-recovery && git show --stat --oneline ec37237 1a986a9 && git diff --name-status ec37237^..1a986a9",
      "result": "passed",
      "summary": "Scoped diff: app.js and source test in ec37237; proof artifacts in 1a986a9."
    },
    {
      "command": "cd /home/yiwen/codex-web-product-recovery && grep -n \"stageFiles(files\\|/inject_file\" codoxear/static/app.js && grep -o \"/inject_file\" codoxear/static/app.js | wc -l",
      "result": "passed",
      "summary": "All producers call stageFiles(); exactly one /inject_file occurrence."
    },
    {
      "command": "cd /home/yiwen/codex-web-product-recovery && node --check codoxear/static/app.js",
      "result": "passed",
      "summary": "No JavaScript syntax errors."
    },
    {
      "command": "cd /home/yiwen/codex-web-product-recovery && PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider tests/test_attach_button_source.py",
      "result": "passed",
      "summary": "8 passed in 0.44s."
    },
    {
      "command": "cd /home/yiwen/codex-web-product-recovery && git diff --check ec37237^..1a986a9 && git status --short",
      "result": "passed",
      "summary": "No whitespace errors and no working-tree/staged changes."
    }
  ],
  "validationOutput": [
    "app.js: selected-session check precedes latestAttachmentBlockerForSession; blocker precedes progress/compression/arrayBuffer/API work.",
    "browser-batch-blocker-result.json: injectRequestCount=1, injectFilenames=[first.txt], stoppedToast=true, stagedCountAfterBatch=1, stagedEntriesHavePathKey=false.",
    "docker-calls-summary.json: key_count=0, send_count=1, marker_send_count=1, max_staged_file_count=1.",
    "Artifact Docker gate: 1792 passed, 1 skipped, 128 subtests passed; smoke: /api/me 401 pre-login and /api/sessions 200 post-login."
  ],
  "residualRisks": [
    "Busy-only row blocker lacks an isolated browser artifact; it is covered by direct app.js source and source test evidence.",
    "A server-side readiness flip between client check and /inject_file is handled as route rejection/failure until polling catches up; server authority prevents staging/backend writes."
  ],
  "noStagedFiles": true,
  "diffSummary": "ec37237 adds public-row busy blocking and per-file latest blocker checks inside stageFiles before upload work; tests assert the order. 1a986a9 records Docker/browser proof artifacts.",
  "reviewFindings": [
    "no blockers",
    "nonblocker: proof uses commit_unknown marker rather than pure busy-only browser scenario",
    "nonblocker: route-rejected race after client check would be reported as failures, not stoppedByBlocker"
  ],
  "manualNotes": "Review-only audit; no project files edited. Findings written to /tmp/upload-batch-blocker-review.md."
}
```
