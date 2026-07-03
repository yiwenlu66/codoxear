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
- **Frontend**: `app.js` is the app shell/wiring. Stateful subsystems live in `app_file_viewer.js`, `app_file_editor.js`, `app_file_picker.js`, `app_transcript.js`, `app_message_rows.js`, `app_launch.js`, `app_new_session.js`, `app_queue.js`, `app_diagnostics.js`, `app_recovery.js`, plus helper modules. Ownership pattern: controller modules own their state/actions/rendering; app.js owns DOM construction when still coupled to the shell and delegates through thin wrappers with fail-loud module checks. Remaining app.js concentrations: chat search/navigation orchestration, unattended menu, voice/notifications.

## Product model invariants

- Minimal UI: GTD-style flat sidebar, sparse chat rendering, mobile-first companion (phone is a view/controller of local sessions).
- Fail loud: no silent fallbacks; contract violations return explicit errors.
- Deleting a session sends shutdown to the broker (terminal-owned sessions too).
- Failed synthetic launch rows (`launch-*` ids) are not real sessions: no send/queue/attach/file-viewer; Details/Copy/New-like-this render from the session-list row locally.

## Validation norms (learned the hard way)

- `pytest` green is NOT acceptance. A live-route 500 (`/api/sessions` recent_cwds limit) shipped past 1344 passing tests because route tests used fake managers.
- Acceptance = full local pytest + `scripts/codoxear-docker-sandbox test` + `scripts/codoxear-docker-sandbox smoke` (real server, real login, real route) + browser evidence via agent-browser for UX claims.
- Docker sandbox: never port 8743; use `CODOXEAR_DOCKER_PORT=18790..19999`. See `.codex/skills/codoxear-docker-test/SKILL.md`.
- Browser automation cannot accept native `confirm()` dialogs — dismiss flows need API-level verification too.

## Known failure modes

- Source-text tests and internal monkeypatch seams (e.g. patching `server.MANAGER`) hide live-contract breaks; prefer executable behavior tests with injected deps.
- Coordinator method signatures with required keyword-only args can break legacy manager call sites silently until a live route hits them.
- Stale docs propagate: sessiond schema docs omitted `token` and masked a parity gap for one review round.
