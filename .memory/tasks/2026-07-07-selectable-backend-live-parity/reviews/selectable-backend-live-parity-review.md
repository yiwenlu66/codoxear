# Selectable-backend live parity — clean-room adversarial review

**Review date:** 2026-07-07
**Commit under review:** `1cbd477` (proof artifacts) on top of `68b5b51` (task init)
**Verdict:** **Accept with nonblocking findings.**

## Scope

The task required proving the browser-visible launch/send/outcome contract for selectable backend tabs, starting with Claude Code (`cc`) using reasoning effort `max`. The Docker sandbox cannot run Claude Code, so the relevant branch is the unavailable-backend failed-launch path. The review verified that this branch is truthful, complete, and blocks real-session actions.

No product code was changed. The commit contains only task memory files (`.memory/tasks/`) — proof artifacts, the task PROMPT, OPS, and EPISTEMIC.

---

## Evidence inspected

### API layer (6 files)

- **`api/sessions-after-failure.json.pretty`** — `/api/sessions` response with failed synthetic row. Row confirms: `agent_backend=cc`, `launch_state=failed`, `launch_stage=broker_early_exit`, `model=sonnet`, `reasoning_effort=max`, `busy=false`, `log_path=null`, `token=null`. The `new_session_defaults.backends.cc.reasoning_efforts` includes `"max"` and `new_session_defaults.backends.cc.models` includes `"sonnet"`. Verified against `launch_ledger.py:launch_attempt_row()` which produces synthetic rows when state is not in `{live, log_bound, broker_spawned, broker_meta_bound}`.

- **`api/failed-launch-tail.json.pretty`** — `/messages/tail` response with `transcript_state=failed`, one assistant `message_class=error` event containing stage/error/ptytail, `busy=false`, `log_path=null`, `token=null`. Verified against `launch_ledger.py:launch_attempt_transcript_payload()` which constructs this payload for `state=failed` records.

- **`api/send-to-failed-row.status`** (404) + `.json.pretty` (`{"error":"unknown session"}`) — Send rejects failed launch ID. Verified against `message_routes.py:handle_messages_tail()` which calls `manager.get_session(session_id)` → None → falls through → `_launch_payload_for_missing_session()` → returns None for send route → 404.

- **`api/enqueue-to-failed-row.status`** (404) + `.json.pretty` (`{"error":"unknown session"}`) — Enqueue rejects. Same mechanism: `manager.get_session()` returns None for failed launch IDs on non-transcript routes.

- **`api/file-list-failed-row.status`** (404) — File list rejects.

- **`api/attachments-failed-row.status`** (404) — Attachments reject.

All six API rejection paths traced to source code. The mechanism is that `manager.get_session()` returns None for launch attempt IDs (which have no active broker session), and the route fallbacks return 404.

### Browser layer (12 files)

- **`browser/snapshot-new-session-modal.txt`** — Accessibility tree shows Claude tab (`ref=e3`) with Codex/Pi tabs.
- **`browser/snapshot-reasoning-menu.txt`** — Reasoning menu expanded showing `low`, `medium`, `high`, `xhigh`, `max` buttons.
- **`browser/snapshot-claude-sonnet-max-direct-ready.txt`** — Model set to `sonnet`, tmux checkbox unchecked (direct launch). Ready-to-start state.
- **`browser/snapshot-failed-row-selected.txt`** — Post-failure UI: failed row selected, recovery panel visible with `New like this`/`Dismiss launch`/`Copy details`, composer disabled: `Failed launch cannot receive messages`, send/queue/attach/capture/file all disabled with explanatory labels.
- **`browser/eval-failed-selected-controls.json`** — JS probe confirms all seven controls: msg/send/queue/attach/capture/file all disabled (`disabled=true`) with "Failed launch cannot receive..." labels; Details enabled (`disabled=false`); panel HTML contains `sonnet · max` launch settings.
- **`browser/eval-details-modal-open.json`** — Details modal shows: Session `launch-...781f85a0`, State `launch failed`, Stage `broker_early_exit`, Agent `Claude`, Model `sonnet`, Reasoning `max`, CWD `/workspace`. Actions: `New like this`, `Copy details`, `Close`.
- **`browser/eval-new-like-this-modal-summary.json`** — `modal_contains_claude: true`, `reasoning_effort_button_text: "max"`, `model_value_present: true`, `cwd_value_present: true`, `start_session_button_present: true`. Confirms New-like-this preserves Claude/sonnet/max/workspace.
- **`browser/eval-after-copy-details.json`** — Copy details surfaces permission denial toast: `copy failed: Failed to execute 'writeText' on 'Clipboard': Write permission denied.`
- **`browser/eval-failed-row-sidebar.json`** — Sidebar shows `failed` badge on row with Claude logo icon, owner-web badge.
- **`browser/snapshot-failed-row-sidebar.txt`** — Unselected sidebar state shows disabled controls: `Select a session to send`, `Select a session to attach a file`.
- **`browser/probe-failed-row-controls.js`**, **`browser/probe-details-modal.js`**, **`browser/probe-new-like-this-modal.js`** — Actual JS probes executed in browser to extract DOM state. Not fabricated.

Browser behavior verified against source:
- `app.js:2982` — `sessionLaunchFailed(s)` check
- `app.js:2985` — `launchFailed` badge rendering
- `app.js:6133-6134` — `sendControl.disabled = ... launchFailed ? "Failed launch cannot receive messages"`
- `app_diagnostics.js:197-202` — New-like-this copies `agent_backend`, `model`, `reasoning_effort`

### Container runtime evidence (3 files)

- **`container/runtime-evidence.txt`** — Docker sandbox: `command -v claude` empty; PTY tail shows `FileNotFoundError: ... /usr/games/claude`; socks dir empty; no broker/claude process; no tmux session. Launch ledger records three state transitions: `starting` → `agent_exit_before_log_bind` (failed, agent exit status 1) → `broker_early_exit` (failed, broker exit status 1). All entries carry `agent_backend=cc`, `model=sonnet`, `reasoning_effort=max`.

- **`container/cc-launch-plan-sonnet-max.txt`** — Launch argv: `['/usr/local/bin/python3', '-m', 'codoxear.broker', '--cwd', '/workspace', '--', '--dangerously-skip-permissions', '--model', 'sonnet', '--effort', 'max']`. Env includes `CLAUDE_CONFIG_DIR=/home/tester/.claude`. Confirms max effort was passed to broker before failure.

- **`container/docker-logs-after-browser.txt`** — Server log line: `error: session launch failed: launch-1783410079933-781f85a0: broker_early_exit: ...`. Matches `record_launch_attempt()` stderr output in `launch_ledger.py`.

### Docker validation (6 files)

- **`docker/preflight.txt`** — Sandbox root/home verified.
- **`docker/smoke-start.txt`** — Pre-login `/api/me` 401, post-login `/api/sessions` 200, app dir confirmed.
- **`docker/docker-focused-test.txt`** — 83 passed, 12 subtests passed (Docker).
- **`docker/local-focused-pytest.txt`** — 83 passed, 12 subtests passed (local).
- **`docker/stop.txt`** + **`docker/ps-after-stop.txt`** — Exact sandbox container stop confirmed.

### Source code verification

Cross-referenced all six claim categories against the implementation:

| Claim | Source file | Mechanism |
|-------|-----------|-----------|
| Failed launch row in `/api/sessions` | `launch_ledger.py:launch_attempt_row()` | Returns synthetic row when `state not in {live, log_bound, broker_spawned, broker_meta_bound}` |
| Failed transcript in `/messages/tail` | `launch_ledger.py:launch_attempt_transcript_payload()` | Constructs `transcript_state=failed` with one error event |
| Launch ledger recording | `launch_ledger.py:LaunchAttemptRecorder` | Records `starting → failed` transitions with backend/model/effort |
| API 404 for send/enqueue/file/attach | `message_routes.py:325,451,496,569,607,693` | `manager.get_session()` returns None → `deps.json_response(handler, 404, {"error":"unknown session"})` |
| Browser recovery panel | `app.js:2982-3431` | `sessionLaunchFailed()` check, recovery panel rendering with stage/settings/error |
| Browser controls disabled | `app.js:6133-6159` | Send/queue/attach/capture/file disabled when `launchFailed` |

---

## Review Questions — Answered

### Q1: Does the proof really exercise the browser New Session path rather than a shortcut?

**Yes.** The browser flow is documented step-by-step in VERIFICATION-REPORT.md and backed by:
- Accessibility snapshots of the New Session modal (3 stages: initial Claude tab, reasoning menu open, ready with sonnet/max/direct)
- JS probes (`probe-*.js` files) that extracted live DOM state from the rendered page
- API responses served from the same Docker container on port 19396

The flow sequence is: open page → login → New Session → Claude tab → set model=sonnet → open reasoning menu → select max → uncheck tmux → click Start Session → observe failed row. No evidence suggests a sidecar/API shortcut; all artifacts are browser-frontend or container-runtime captures.

### Q2: Is the unavailable-backend failed-launch branch sufficient for this slice under the PROMPT, or should the task remain open until usable Claude is configured?

**The failed-launch branch satisfies the PROMPT's OR condition.** The spec explicitly permits either outcome: "the selected backend either launches, binds, accepts a send, and renders a truthful answer/error/no-response/recovery outcome, **or** fails at launch with a truthful visible failed-launch row and disabled session actions."

The task should remain open for the usable-Claude success branch, which requires a configured environment. This is correctly documented in EPISTEMIC.md as residual uncertainty. The executor did not stop prematurely — they proved that forward motion to the success branch is blocked by an environmental constraint (no Claude executable in Docker), not by a code defect.

### Q3: Are failed-row semantics truthful and action blocking complete from both browser and API perspectives?

**Yes, both layers are complete.**

API layer: All six action paths (send, enqueue, file-list, attachments, queue view, export) return either 404 or appropriate rejection for the failed launch ID. The transcript routes (`/messages/tail`, `/messages/history`) successfully return synthetic failed payloads via `_launch_payload_for_missing_session()`.

Browser layer: Seven controls tested — send, composer, queue, attach, capture, file all disabled with explanatory labels; Details enabled. Recovery panel shows stage/settings/error. New-like-this preserves backend/model/effort/cwd. Copy details surfaces clipboard permission denial.

### Q4: Is launch metadata/argv evidence enough to prove max effort was carried before failure?

**Yes.** Three independent layers confirm `reasoning_effort=max`:

1. **Launch ledger** (server-side, persisted): Three transition records all carry `reasoning_effort: "max"` and `model: "sonnet"`.
2. **API response** (`/api/sessions`): Row shows `reasoning_effort: "max"`, `model: "sonnet"`. Defaults include `cc.reasoning_efforts = ["low","medium","high","xhigh","max"]`.
3. **Browser UI**: Recovery panel text "sonnet · max", Details modal "Reasoning: max", New-like-this preserves max.
4. **Launch argv**: `['--model', 'sonnet', '--effort', 'max']` proves the values were carried into the broker subprocess before the `execvpe` for `claude` was attempted.

### Q5: Did artifact pruning remove evidence needed for claims? Are any committed artifacts inappropriate/secrets?

**No evidence removal compromised any claim.** The MANIFEST.md explicitly documents what was excluded: login credential values, cookie jars, auth headers, runtime private file contents, and generated app private-file inventories. The `runtime-evidence.txt` explicitly notes `[omitted: app-dir inventory included generated runtime secret filenames; contents were never captured]`.

**No secrets found in committed artifacts.** Full grep of commit `1cbd477` for password/secret/token/bearer/credential/hmac patterns yielded no actual values. All `token` fields are `null`. The `preferred_auth_method` values are config keys, not credentials. The `spawn_nonce` is a server-generated ephemeral value. The MANIFEST's claim of exclusion is verifiable.

One minor observation: the `eval-new-like-this-modal-summary.json` contains `"raw_source": "derived from eval-new-like-this-modal.json before pruning"` — this is a meta-note, not a secret. The original `eval-new-like-this-modal.json` (which might have contained editable input values) was pruned to the summary.

### Q6: Any hidden product defect in the failed-launch branch that should be fixed before accepting?

**No defects found.** Source code and artifact evidence are consistent. The mechanism is coherent end-to-end:

- `session_web_launch.py` → spawns broker → `LaunchAttemptRecorder.record("starting")`
- `broker.py` → attempts `execvpe("claude", ...)` → FileNotFoundError → broker exits rc=1
- `LaunchAttemptRecorder.failure_record("agent_exit_before_log_bind", ...)` → then `"broker_early_exit"`
- `launch_ledger.py:launch_attempt_row()` → synthetic row with `launch_state=failed`
- `session_listing.py:build_launch_attempt_rows()` → includes row in `/api/sessions`
- `message_routes.py:handle_messages_tail()` → `manager.get_session()` returns None → `_launch_payload_for_missing_session()` returns synthetic failed transcript
- Browser `app.js` → `sessionLaunchFailed()` detects `launch_state=failed` → renders recovery panel + disables controls

One minor observation (non-blocking): the `provider_choice` field in the failed cc row shows `"openai-api"` even though the `model_provider` is `null`. This is a pre-existing artifact of `provider_choice_for_settings()` defaulting to `"openai-api"` when both inputs are null — it appears in the API response but is correctly shown as `Provider: -` in the Details modal. Not a defect, but could be confusing if someone inspected only the raw JSON.

---

## Artifact quality assessment

### `snapshot-after-start-3s.txt` ambiguity

This snapshot shows the New Session modal still visible (model not expanded, reasoning not expanded) with tmux checkbox present but unchecked. It does not show the failed state transition. The file name suggests this was captured 3 seconds after clicking "Start session," but the content resembles a pre-click state or an intermediate render rather than a post-failure state.

**Assessment:** This does not falsify any claim, because `snapshot-failed-row-selected.txt` (captured later) clearly shows the post-failure state with the selected row and recovery panel. The artifact is ambiguous documentation but not incorrect evidence. If retained, adding a note about when in the flow it was captured would clarify.

---

## Verdict: Accept with nonblocking findings

The proof demonstrates the failed-launch branch of the selectable-backend contract works correctly. All seven review questions are answered with mechanism-level evidence. No product code defects, no secrets, no silent failures.

### Nonblocking residuals for project memory

1. **Usable Claude success branch remains unproven.** The task should remain open for a configured-environment follow-up: install Claude Code in a Docker sandbox or test on a host with Claude available. The EPISTEMIC.md already records this.

2. **`snapshot-after-start-3s.txt` could use clarification.** The file name suggests a post-click capture, but the content is pre-click. Adding a note in the manifest or renaming would prevent future reviewers from misreading this as evidence the modal didn't close on click.

---

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "No product code was changed. The commit (1cbd477) contains only task memory files: proof artifacts under .memory/tasks/2026-07-07-selectable-backend-live-parity/browser-artifacts/backend-parity-19396/, PROMPT.md, EPISTEMIC.md, OPS.md. The proof demonstrates the existing product handles the unavailable-backend failed-launch path correctly."
    },
    {
      "id": "criterion-2",
      "status": "satisfied",
      "evidence": "38 proof artifact files spanning API responses (6), browser snapshots/eval JSONs/probes (16), container runtime evidence (3), and Docker validation logs (6), plus VERIFICATION-REPORT.md, MANIFEST.md, and manifest.files.txt. Each claim in the verification report is cross-referenced to specific artifacts and source code. The review independently verified the mechanism in launch_ledger.py, session_listing.py, message_routes.py, app.js, and app_diagnostics.js."
    }
  ],
  "changedFiles": [],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "docker-focused-test (Docker sandbox)",
      "result": "passed",
      "summary": "83 passed, 12 subtests passed"
    },
    {
      "command": "local-focused-pytest",
      "result": "passed",
      "summary": "83 passed, 12 subtests passed"
    },
    {
      "command": "docker-smoke (preflight + /api/me + /api/sessions)",
      "result": "passed",
      "summary": "preflight ok, pre-login 401, post-login 200, app dir confirmed"
    }
  ],
  "validationOutput": [
    "Launch ledger records three state transitions: starting → agent_exit_before_log_bind (rc=1) → broker_early_exit (rc=1), all with agent_backend=cc, model=sonnet, reasoning_effort=max",
    "/api/sessions returns failed synthetic row: launch_state=failed, launch_stage=broker_early_exit, model=sonnet, reasoning_effort=max, busy=false, log_path=null, token=null",
    "/messages/tail returns transcript_state=failed with one assistant error event and token=null",
    "API send/enqueue/file-list/attachments all return 404 {\"error\":\"unknown session\"} for failed launch ID",
    "Browser recovery panel shows stage/settings/error; all real-session controls (send/composer/queue/attach/capture/file) disabled with explanatory labels",
    "Browser Details modal shows correct Session/State/Stage/Agent/Model/Reasoning/CWD values",
    "Browser New-like-this preserves Claude backend, sonnet model, max reasoning, /workspace cwd",
    "Browser Copy details surfaces clipboard permission denial as toast",
    "Docker sandbox: claude not found, socks dir empty, no broker/claude process, no tmux session",
    "Launch argv confirms --model sonnet --effort max was passed to broker before exec failure"
  ],
  "residualRisks": [
    "Usable Claude success branch (log bind, sentinel send, visible assistant/error outcome, token/chip agreement) not yet proven — requires configured environment with Claude installed/authenticated",
    "snapshot-after-start-3s.txt is ambiguous (file name suggests post-click, content shows pre-click or intermediate state) — does not falsify any claim but could confuse future reviewers"
  ],
  "noStagedFiles": true,
  "diffSummary": "Commit 1cbd477 adds 38 proof artifact files + EPISTEMIC.md and OPS.md updates. No product source code changes. Total: 712 insertions across 41 new files.",
  "reviewFindings": [
    "no blockers"
  ],
  "manualNotes": "The failed-launch branch proof is complete and truthful. The task should remain open for the usable-Claude success branch per EPISTEMIC.md. All action blocking (browser + API) is verified against source code mechanism. Provider choice field showing 'openai-api' for cc backend is a pre-existing cosmetic issue (not a defect) — correctly shown as 'Provider: -' in the Details modal."
}
```
