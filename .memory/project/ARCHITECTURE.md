# Codoxear architecture (distilled)

Active development checkout: `/home/yiwen/codex-web-product-recovery`, branch `recovery/product-gaps`.
Protected checkout: `/home/yiwen/codex-web` on `main` — never edit/restart/merge/promote without explicit user approval.

## Ownership map (post-refactor, 2026-07)

- **Backend adapters** (`agent_backend.py`): CodexBackend / PiBackend / ClaudeCodeBackend own launch argv, env, log path recognition, session-id extraction, run-settings, row-busy predicates, chat-event parsing hooks, launch defaults/validation. `backend_launch.py`, `session_log_paths.py`, `process_log_paths.py`, `launch_config.py`, `session_log_metadata.py`, `rollout_chat_events.py` are compatibility facades over adapters, not branch owners.
- **Runtime status authority** (`session_runtime.py`): `RuntimeStatus` is the single busy/idle/interrupted-idle synthesis; `SessionRuntimeReadiness` projects send/queue/attachment/unattended eligibility. Routes and coordinators consume `manager._runtime_status_from_state_and_log(...)`; they must not recompute busy/idle.
- **Turn-state reducer** (`broker_turn_state.py`): the only log→busy reducer. Broker and sessiond both consume it (`_apply_log_objects_to_state`, `_should_clear_busy_state`, `_mark_busy_state_idle`, `_mark_explicit_interrupt_request`, `_update_busy_from_pty_text`). Do not add a second reducer.
- **Persistent state** (`session_store.py`): SessionStore owns per-session lifecycle (reset/load/delete/save-ordering/prune/recent-cwd). New per-session maps must extend SessionStore lifecycle methods, not add ad-hoc deletion/load code in coordinators.
- **SessionManager** is a thin coordinator host: methods are bound via `session_manager_method_bindings.py` forwarders to coordinator objects (`session_queue.py`, `session_readiness.py`, `session_recent_cwd.py`, etc.).
- **sessiond** (`sessiond.py`): supported headless runner. Same control-state schema as broker: `busy`, `queue_len`, `token`, `interrupted_idle`. Intentionally no foreground terminal UX.
- **Frontend**: `app.js` is the app shell/wiring. Stateful subsystems live in `app_file_viewer.js`, `app_file_editor.js`, `app_file_picker.js`, `app_transcript.js`, `app_message_rows.js`, `app_launch.js`, `app_new_session.js`, `app_queue.js`, `app_diagnostics.js`, `app_recovery.js`, `app_unattended.js`, `app_chat_navigation.js`, `app_chat_search.js`, `app_voice.js`, plus helper modules. Ownership pattern: controller modules own their state/actions/rendering; app.js owns DOM construction when still coupled to the shell and delegates through thin wrappers with fail-loud module checks. Chat search owns loaded/all-history search orchestration while app.js retains transcript rendering and shared row-helper authority; Voice owns settings/notification/announcement/audio orchestration while app.js retains dialog/button/audio DOM construction and thin wrappers. Remaining app.js work is residual shell projection/code-size cleanup, not a single known stateful subsystem concentration.

## Product model invariants


- Selectable backend tabs are product promises. If a turn records user input and then terminates without assistant output or explicit backend error, normalization must emit a truthful visible no-response/failure event; ordinary idle silence violates the projection contract. Claude Code terminal `system/api_error` rows are visible error outcomes only when retries are exhausted; transient retry notices stay out of the transcript by themselves.
- Product surface must be justified by user workflow, not verification convenience. Sandbox-only flags, credentials, broken local packages, and certification workarounds are environment/ops boundaries unless a real user scenario independently requires UI. Do not add visible controls merely to make an isolated test path pass.
- Minimal UI: GTD-style flat sidebar, sparse chat rendering, mobile-first companion (phone is a view/controller of local sessions).
- Fail loud: no silent fallbacks; contract violations return explicit errors.
- JSON API responses must never carry raw surrogateescape path strings. Filesystem paths discovered from `os.walk` can contain lone surrogates for non-UTF-8 bytes; user-facing display fields must use `git_ops.path_json_text`, while reversible operations require an explicit token field such as `api_path`/`path_token`.
- File display path is not identity. Same-rendering paths (for example raw-byte `bad<ff>name.txt` and literal `bad\\xffname.txt`) must preserve token identity through picker/open/download/write/recent flows and must be visibly disambiguated before selection (`non-UTF bytes` vs `literal name` style hints are the current convention).
- Repository diff truth must not depend on Monaco. When the rich editor is unavailable, the read-only fallback must render unified repository diff text (`/git/diff?head=1`) rather than showing working-tree file content under a diff heading.
- File editing must not depend on Monaco. Monaco is an optional rich editor path; the certified baseline editor is `plain-edit`, a textarea surface for editable text files that saves through `/file/write`. `plain-fallback`, Git diff, preview, binary/download-only, oversize, and unavailable-session paths stay read-only.
- Mobile file/editor controls must preserve the companion-device contract. The file viewer header actions and touch dpad have separate 44px mobile target-size rules; the dpad grid tracks/spacers/buttons must move together so touch targets do not overflow smaller cells.
- Attachment upload is a staged-reference workflow, not cwd import. The browser supports a single-file paperclip flow: server stages bytes under app-dir `uploads/<session>/` with mode 0600 and injects `Attachment N: <absolute_path>` into the backend. Docker/Pi certification proved Pi can read that app-dir absolute path and return sentinel content; do not move staging into cwd or add drag/drop/multi-file/paste/capture surfaces without a product decision.
- Attachment badge authority is selected-session pending truth plus immediate local attach feedback. Server session-list refresh overwrites cached `pending_attachment`; direct successful attach/send/clear responses may update the selected cached value for immediate UI projection. Do not add independent badge state that can diverge from this contract.
- Deleting a session sends shutdown to the broker (terminal-owned sessions too) and removes only that session's staged-upload entry under `uploads/<session_id>`. Cleanup must treat symlink entries as links to unlink, never directories to follow, and must preserve sibling session uploads.
- Failed synthetic launch rows (`launch-*` ids) are not real sessions: no send/queue/attach/file-viewer; Details/Copy/New-like-this render from the session-list row locally.

## Validation norms (learned the hard way)

- `pytest` green is NOT acceptance. A live-route 500 (`/api/sessions` recent_cwds limit) shipped past 1344 passing tests because route tests used fake managers.
- Acceptance = full local pytest + `scripts/codoxear-docker-sandbox test` + `scripts/codoxear-docker-sandbox smoke` (real server, real login, real route) + browser evidence via agent-browser for UX claims.
- Docker sandbox: never port 8743; use `CODOXEAR_DOCKER_PORT=18790..19999`. See `.codex/skills/codoxear-docker-test/SKILL.md`.
- Browser automation cannot accept native `confirm()` dialogs — dismiss flows need API-level verification too.

## Known failure modes


- Runtime readiness treats a bound transcript log as the authority for real turn busy/idle. Broker PTY busy hints observed before log bind (for example Codex startup text containing `esc to interrupt`) are not sufficient to block the first browser input because no log-watcher idle path can clear them. After a confirmed send, the send-boundary mechanism blocks follow-up input until the log appears/advances or recovery is surfaced.
- `interrupted_idle` is an override for an interrupted non-final log tail, not a durable busy-state category. Once post-interrupt log activity proves the override stale, Codoxear suppresses repeated stale broker `interrupted_idle=true` reports until the broker reports false or the log/session resets; otherwise `/api/sessions` can project idle while the transcript log is non-idle. Every state-refresh path (broker, prune, discovery) must use `set_session_interrupted_idle()` rather than assigning `interrupted_idle` or `interrupted_idle_log_off` directly.
- Source-text tests and internal monkeypatch seams (e.g. patching `server.MANAGER`) hide live-contract breaks; prefer executable behavior tests with injected deps.
- Coordinator method signatures with required keyword-only args can break legacy manager call sites silently until a live route hits them.
- Stale docs propagate: sessiond schema docs omitted `token` and masked a parity gap for one review round.
