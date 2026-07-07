# Selectable-backend live parity proof — port 19396

## Outcome branch

Visible failed launch. The Docker sandbox does not have a usable Claude Code executable, so the browser-created Claude session failed before a transcript log was bound. The product did not silently idle: it projected a failed launch row, a failed transcript payload, Details/New-like-this/Copy details actions, and blocked real-session actions.

## Browser flow exercised

Actual browser UI path, not sidecar/API shortcut:

1. Opened `http://127.0.0.1:19396/` and logged in.
2. Opened **New session**.
3. Selected the **Claude** backend tab.
4. Entered cwd `/workspace`.
5. Opened reasoning menu and selected **max**.
6. Entered model `sonnet`.
7. Unchecked **Create in tmux** to exercise direct web launch.
8. Clicked **Start session**.

Browser artifacts:
- `browser/snapshot-new-session-modal.txt`
- `browser/snapshot-reasoning-menu.txt` shows options `low`, `medium`, `high`, `xhigh`, `max`.
- `browser/snapshot-claude-sonnet-max-direct-ready.txt` shows Claude tab ready with model `sonnet`.
- `browser/snapshot-after-start-3s.txt` shows the launch failure surfaced after the start attempt.
- `browser/snapshot-failed-row-selected.txt` and `browser/eval-failed-selected-controls.json` show selected failed row and blocked controls.
- `browser/eval-details-modal-open.json` shows Details modal values.
- `browser/eval-new-like-this-modal.json` shows New-like-this reopens the New Session dialog with Claude, `sonnet`, and `max`.

## Claims proven

### 1. Claude max is exposed in launch defaults and UI

`api/sessions-after-failure.json.pretty` includes:

- `new_session_defaults.backends.cc.reasoning_efforts = ["low", "medium", "high", "xhigh", "max"]`
- `new_session_defaults.backends.cc.models = ["sonnet", "opus", "fable"]`

`browser/snapshot-reasoning-menu.txt` shows a **max** button in the actual New Session reasoning menu.

### 2. The browser-selected launch carried backend/model/effort into the launch ledger

`container/runtime-evidence.txt` launch ledger records:

- `agent_backend: "cc"`
- `cwd: "/workspace"`
- `model: "sonnet"`
- `reasoning_effort: "max"`
- `transport: "direct"`
- `launch_id: "launch-1783410079933-781f85a0"`

`container/cc-launch-plan-sonnet-max.txt` proves the same request shape builds argv:

```text
['/usr/local/bin/python3', '-m', 'codoxear.broker', '--cwd', '/workspace', '--', '--dangerously-skip-permissions', '--model', 'sonnet', '--effort', 'max']
```

### 3. Docker lacks usable Claude, so failed-launch semantics are the relevant branch

`container/runtime-evidence.txt` shows:

- `command -v claude` produced no path.
- The PTY tail includes `FileNotFoundError: [Errno 2] No such file or directory: b'/usr/games/claude'`.
- `socks/` is empty after failure.
- Process tree contains only `python3 -m codoxear.server`; no broker or Claude process remains.
- No tmux session exists.

### 4. The browser/API row is a truthful failed launch, not silent idle

`api/sessions-after-failure.json.pretty` row values:

- `session_id: "launch-1783410079933-781f85a0"`
- `agent_backend: "cc"`
- `launch_state: "failed"`
- `launch_stage: "broker_early_exit"`
- `launch_error: "broker exited early ... claude exited with status 1 before a session log was bound"`
- `model: "sonnet"`
- `reasoning_effort: "max"`
- `busy: false`
- `log_path: null`
- `queue_len: 0`
- `token: null`

`api/failed-launch-tail.json.pretty` projects:

- `transcript_state: "failed"`
- one assistant `message_class: "error"` event: `Session launch failed before a transcript log was created...`
- `busy: false`
- `log_path: null`

### 5. Failed row blocks real-session actions

`browser/eval-failed-selected-controls.json` shows:

- composer disabled with `Failed launch cannot receive messages`
- send disabled with `Failed launch cannot receive messages`
- queue disabled with `Failed launch cannot receive queued messages`
- attach and capture disabled with `Failed launch cannot receive file attachments`
- file viewer disabled with `Failed launch has no file browser`
- Details remains enabled for local launch details

API route probes also reject the failed launch id as a real session:

- `api/send-to-failed-row.status`: `404`, body `{"error":"unknown session"}`
- `api/enqueue-to-failed-row.status`: `404`, body `{"error":"unknown session"}`
- `api/file-list-failed-row.status`: `404`
- `api/attachments-failed-row.status`: `404`

### 6. Details / Copy / New-like-this behavior is visible

`browser/eval-details-modal-open.json` shows Details modal:

- Session `launch-1783410079933-781f85a0`
- State `launch failed`
- Stage `broker_early_exit`
- Agent `Claude`
- Model `sonnet`
- Reasoning `max`
- actions `New like this`, `Copy details`, `Close`

`browser/eval-new-like-this-modal-summary.json` shows New-like-this opens New Session with Claude, model `sonnet`, cwd `/workspace`, and reasoning `max`.

`browser/eval-after-copy-details.json` shows clipboard permission denial is surfaced as a toast: `copy failed: Failed to execute 'writeText' on 'Clipboard': Write permission denied.`

## Tests and commands

- Local focused tests: `83 passed, 12 subtests passed` (`docker/local-focused-pytest.txt`).
- Docker focused tests: `83 passed, 12 subtests passed` (`docker/docker-focused-test.txt`).
- Docker preflight: `preflight ok: root=/tmp/codoxear-docker-sandbox-19396 home=/tmp/codoxear-docker-sandbox-19396/home`.
- Docker smoke: pre-login `/api/me` `401`, post-login `/api/sessions` `200`, app dir `/home/tester/.local/share/codoxear` (`docker/smoke-start.txt`).
- Docker stopped with exact sandbox stop; `docker/ps-after-stop.txt` is empty.

## Defects fixed

None. No product-code defect was observed in the failed-launch branch. No code was edited.

## Residual risk

This proof did not exercise the usable Claude branch because the sandbox lacks Claude Code. A configured environment with Claude installed/authenticated still needs the success-branch proof: log bind, sentinel send, visible assistant/error/no-response/recovery outcome, and token/chip agreement if usage appears.
