# Post-fix Docker/browser verification — HEAD 1421d20

Repository: `/home/yiwen/codex-web-product-recovery`, branch `recovery/product-gaps`.


## Durable committed copies

The original run artifacts were captured under `/tmp/codoxear-postfix-browser-19132/artifacts/`. The decisive screenshots were copied into this task's `browser-artifacts/` directory for durable review:

- `browser-artifacts/postfix-d1-editor-unavailable.png`
- `browser-artifacts/postfix-binary-download-only.png`
- `browser-artifacts/postfix-git-file-diff-browser.png`
- `browser-artifacts/postfix-mobile-file-viewer-390x844.png`
- `browser-artifacts/postfix-cc-fallback-bound-browser.png`

## Sandbox boundary

- Docker unit sandbox: port `19131`, name `codoxear-postfix-test-19131`, root `/tmp/codoxear-postfix-test-19131`.
- Browser/smoke sandbox: port `19132`, name `codoxear-postfix-browser-19132`, root `/tmp/codoxear-postfix-browser-19132`.
- Server/container cleanup used exact Docker sandbox teardown only: `scripts/codoxear-docker-sandbox stop` for `codoxear-postfix-browser-19132`.
- No host brokers, host servers, host sessiond/tmux, host throwaway-HOME repros, host runtime dirs, or host process-pattern cleanup were used.

## Commands and observations

### Docker sandbox test/smoke

- `CODOXEAR_DOCKER_PORT=19131 CODOXEAR_DOCKER_NAME=codoxear-postfix-test-19131 CODOXEAR_DOCKER_ROOT=/tmp/codoxear-postfix-test-19131 scripts/codoxear-docker-sandbox test`
  - Result: pass — `1626 passed, 1 skipped, 132 subtests passed in 43.83s`.
- `CODOXEAR_DOCKER_PORT=19132 CODOXEAR_DOCKER_NAME=codoxear-postfix-browser-19132 CODOXEAR_DOCKER_ROOT=/tmp/codoxear-postfix-browser-19132 scripts/codoxear-docker-sandbox smoke`
  - Result: pass — `pre_login_api_me_status=401`, `post_login_sessions_status=200`, `container_app_dir=/home/tester/.local/share/codoxear`.
- `curl http://127.0.0.1:19132/monaco/vs/loader.js`
  - Result: `404`, so the browser editor path exercised the Monaco-unavailable path.

### Workbench D1 — editor unavailable state

Fixture: synthetic control socket + sidecar + throwaway git repo inside the Docker app/container home. Session `fake-001` had cwd `/home/tester/work/repo` and file `notes.txt`.

Browser evidence:

- Opened `http://127.0.0.1:19132/`, logged in with the sandbox password, selected/opened `notes.txt` in the file viewer.
- The viewer rendered plain fallback: `Plain text fallback — monaco loader timed out. Showing a read-only plain-text view.`
- `#fileEditBtn` state from browser DOM:
  - `disabled: false`
  - `aria-disabled: "true"`
  - `aria-label/title: "Editing is unavailable because the code editor failed to load. Read-only preview remains available."`
- Dispatching a click event to `#fileEditBtn` reached the handler and produced both a toast/status line with the exact required message: `Editing is unavailable because the code editor failed to load. Read-only preview remains available.`

Artifact: `/tmp/codoxear-postfix-browser-19132/artifacts/d1-editor-unavailable.png`.

Status: **pass**.

### File/git/mobile surfaces

API evidence against session `fake-001`:

- `GET /api/sessions/fake-001/file/list`
  - Result: `README.md`, `binary.bin`, `notes.txt`, `untracked.txt`.
- `GET /api/sessions/fake-001/file/search?q=notes&limit=10`
  - Result: `matches[0].path = notes.txt`.
- `GET /api/sessions/fake-001/file/read?path=notes.txt`
  - Result: `kind=text`, `editable=true`, text returned.
- `GET /api/sessions/fake-001/file/read?path=binary.bin`
  - Result: `kind=download_only`, `reason=binary`, `size=11`.
- `GET /api/sessions/fake-001/git/changed_files`
  - Result: `entries[0].path=notes.txt`, `additions=2`, `deletions=1`, `unstaged=["notes.txt"]`.
- `GET /api/sessions/fake-001/git/diff?path=notes.txt`
  - Result: unified diff contained `-beta line`, `+beta line changed`, `+new worktree line`.

Browser evidence:

- Binary file viewer showed: `binary.bin - download only - 11 B` and `Preview unavailable — binary.bin is not renderable as text, markdown, image, or PDF. Use Download instead.` Download remained enabled; edit was disabled.
  - Artifact: `/tmp/codoxear-postfix-browser-19132/artifacts/binary-download-only.png`.
- Changed git-file option `notes.txt git root · changed +2 -1` opened diff mode; diff toggle was enabled/active and the viewer showed `notes.txt - diff` with plain fallback because Monaco was unavailable.
  - Artifact: `/tmp/codoxear-postfix-browser-19132/artifacts/git-file-diff-browser.png`.
- Mobile viewport `390x844` rendered the file viewer at width `390`, with readable `notes.txt` plain fallback content and no layout collapse.
  - Artifact: `/tmp/codoxear-postfix-browser-19132/artifacts/mobile-file-viewer-390x844.png`.

Status: **pass**.

### Claude Code residual binding proof

Docker-only proof was completed. A fake Claude Code binary was installed inside the container only at `/home/tester/bin/fake-claude` and launched through `python3 -m codoxear.broker` with:

`CODEX_WEB_AGENT_BACKEND=cc CODEX_WEB_OWNER=web CLAUDE_CONFIG_DIR=/home/tester/.claude CLAUDE_BIN=/home/tester/bin/fake-claude python3 -m codoxear.broker --cwd /home/tester/work/cc_requested -- --model sonnet`

The fake Claude process wrote a fresh JSONL transcript under the Claude projects directory with divergent cwd:

- Broker cwd: `/home/tester/work/cc_requested`
- Log path: `/home/tester/.claude/projects/-home-tester-work-cc-actual/aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee.jsonl`
- Log row cwd: `/home/tester/work/cc_actual`
- Log fd was closed after writing; the fake process stayed alive only to keep the broker socket discoverable until container teardown.

API evidence:

- `GET /api/sessions` listed `broker-2543` with `agent_backend=cc`, `cwd=/home/tester/work/cc_requested`, and `log_path=/home/tester/.claude/projects/-home-tester-work-cc-actual/aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee.jsonl`.
- `GET /api/sessions/broker-2543/messages/tail` returned `transcript_state=bound`, the divergent `log_path`, user text `cc fallback bind request`, and assistant text `CC-FALLBACK-BOUND`.

Browser evidence:

- Opening `/#session=broker-2543` displayed both transcript messages, including `CC-FALLBACK-BOUND`.
- Artifact: `/tmp/codoxear-postfix-browser-19132/artifacts/cc-fallback-bound-browser.png`.

Status: **pass** for log binding/transcript projection. The fake process intentionally remained alive, so this run does not make a CC idle-state claim.

## Artifacts

Primary artifacts under `/tmp/codoxear-postfix-browser-19132/artifacts/`:

- `d1-editor-unavailable.png`
- `binary-download-only.png`
- `git-file-diff-browser.png`
- `mobile-file-viewer-390x844.png`
- `cc-fallback-bound-browser.png`
- API captures: `file-list.json`, `file-search-notes.json`, `file-read-notes.json`, `file-read-binary.json`, `git-changed-files.json`, `git-diff-notes.json`, `sessions-cc-fallback.json`, `cc-fallback-messages-tail.raw`, `cc-fallback-messages-export.raw`

Temporary helper files created outside the repo:

- `/tmp/codoxear_fake_session.py`
- `/tmp/fake_claude.py`
- Docker roots `/tmp/codoxear-postfix-test-19131` and `/tmp/codoxear-postfix-browser-19132`

Repo status after verification had no staged files and no tracked diff. The only repo untracked entries shown by `git status` were pre-existing `.memory/tasks/2026-07-03-usable-product-ui-architecture/browser-artifacts/*.png` files already present before this run; this run created no new untracked repo files.

## Residual risks

- The Workbench D1 click evidence used browser DOM event dispatch because the high-level `agent-browser click` primitive refuses to click an `aria-disabled` element. The DOM state proves the button is not `disabled`, is `aria-disabled=true`, and its click handler receives/prevents the event and surfaces the exact message.
- The CC fake-broker proof validates the cwd-mismatch fallback bind and transcript projection, not CC process idle semantics.

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Verification scope only: no source edits, staging, commits, host brokers/servers/sessiond/tmux, or host runtime access; all broker/server/session proof ran inside Docker sandboxes on ports 19131/19132."
    },
    {
      "id": "criterion-2",
      "status": "satisfied",
      "evidence": "Docker test/smoke passed; browser/API evidence covers Monaco-unavailable editor message, file text/binary, git changed_files/diff, mobile file viewer, and Docker-only fake Claude cwd-mismatch log binding."
    }
  ],
  "changedFiles": [],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "CODOXEAR_DOCKER_PORT=19131 CODOXEAR_DOCKER_NAME=codoxear-postfix-test-19131 CODOXEAR_DOCKER_ROOT=/tmp/codoxear-postfix-test-19131 scripts/codoxear-docker-sandbox test",
      "result": "passed",
      "summary": "1626 passed, 1 skipped, 132 subtests passed in 43.83s"
    },
    {
      "command": "CODOXEAR_DOCKER_PORT=19132 CODOXEAR_DOCKER_NAME=codoxear-postfix-browser-19132 CODOXEAR_DOCKER_ROOT=/tmp/codoxear-postfix-browser-19132 scripts/codoxear-docker-sandbox smoke",
      "result": "passed",
      "summary": "pre-login /api/me 401, post-login /api/sessions 200, APP_DIR /home/tester/.local/share/codoxear"
    },
    {
      "command": "curl http://127.0.0.1:19132/monaco/vs/loader.js",
      "result": "passed",
      "summary": "Returned 404, exercising Monaco-unavailable browser path"
    },
    {
      "command": "agent-browser login/open file viewer for fake-001 notes.txt and inspect #fileEditBtn",
      "result": "passed",
      "summary": "Plain fallback rendered; Edit had aria-disabled=true, disabled=false, exact unavailable message in aria/title; click dispatch produced exact toast/status"
    },
    {
      "command": "GET /api/sessions/fake-001/file/list, /file/search, /file/read notes.txt, /file/read binary.bin",
      "result": "passed",
      "summary": "Text file readable/editable; binary returned download_only with reason=binary"
    },
    {
      "command": "GET /api/sessions/fake-001/git/changed_files and /git/diff?path=notes.txt",
      "result": "passed",
      "summary": "Changed file notes.txt reported with +2/-1 and diff body returned"
    },
    {
      "command": "agent-browser mobile viewport 390x844 file viewer",
      "result": "passed",
      "summary": "notes.txt plain fallback remained readable at 390x844; screenshot captured"
    },
    {
      "command": "Docker-only fake Claude Code broker with CLAUDE_BIN=/home/tester/bin/fake-claude and divergent log cwd",
      "result": "passed",
      "summary": "/api/sessions bound log_path despite cwd mismatch; /messages/tail and browser transcript showed CC-FALLBACK-BOUND"
    },
    {
      "command": "CODOXEAR_DOCKER_PORT=19132 CODOXEAR_DOCKER_NAME=codoxear-postfix-browser-19132 CODOXEAR_DOCKER_ROOT=/tmp/codoxear-postfix-browser-19132 scripts/codoxear-docker-sandbox stop",
      "result": "passed",
      "summary": "Exact Docker sandbox teardown; no matching container remained"
    },
    {
      "command": "git status --porcelain=v1 && git diff --cached --name-only",
      "result": "passed",
      "summary": "No staged files and no tracked diff; only pre-existing untracked memory screenshots were listed"
    }
  ],
  "validationOutput": [
    "Docker test: 1626 passed, 1 skipped, 132 subtests passed in 43.83s",
    "Docker smoke: pre_login_api_me_status=401; post_login_sessions_status=200; container_app_dir=/home/tester/.local/share/codoxear",
    "Workbench D1: exact message surfaced: Editing is unavailable because the code editor failed to load. Read-only preview remains available.",
    "File API: notes.txt kind=text editable=true; binary.bin kind=download_only reason=binary",
    "Git API: changed_files notes.txt +2/-1; diff returned expected line changes",
    "CC API/browser: divergent-cwd log_path bound and transcript contained CC-FALLBACK-BOUND"
  ],
  "residualRisks": [
    "agent-browser high-level click refuses aria-disabled controls, so event-receiving was verified by DOM click dispatch plus disabled=false/aria-disabled=true state rather than a trusted pointer click",
    "Fake Claude proof covers log binding/transcript projection, not CC idle-state behavior"
  ],
  "noStagedFiles": true,
  "diffSummary": "No source changes; verification artifacts only under /tmp and pre-existing untracked repo screenshots unchanged.",
  "reviewFindings": [
    "no blockers"
  ],
  "manualNotes": "Docker/browser verification completed at HEAD 1421d20. The browser/smoke container was stopped by exact sandbox name after artifacts were captured."
}
```
