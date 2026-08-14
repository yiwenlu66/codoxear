# Operations evidence

- 2026-08-03T15:57:47+08:00 — Initial git state: `main` at `dd966f2e`; unrelated untracked `.memory/tasks/2026-07-31-issue-triage/OPS.md`, `.memory/tasks/2026-08-02-codex-tokens-subagents/`, and `.pi-subagents/` present and excluded from task scope.
- 2026-08-03T15:57:47+08:00 — Prediction: extraction must centralize SSE/poll state with live-delta handling; leaving cursor/event-source state split across app.js and controller would create competing authorities and likely break drop/resume.
- 2026-08-03T16:35:00+08:00 — Implemented `app_message_flow.js`; moved confirmed send, request cancellation, SSE retry/resume, polling fallback/backoff, shared live delta application, and typing reconciliation. `app.js` retains selected id + `pollGen` and open-session orchestration; composer injects send.
- 2026-08-03T16:42:00+08:00 — Initial targeted pytest attempt failed because host Python lacked pytest; isolated `/tmp/codoxear-test-venv` created and project installed editable for validation only.
- 2026-08-03T16:47:00+08:00 — Targeted behavior validation: 63 passed (`test_frontend_message_flow.py`, `test_chat_transcript_runtime.py`, `test_sse_live.py`, `test_message_routes.py`, `test_message_transcript_state.py`, `test_message_cursor.py`, `test_static_assets.py`).
- 2026-08-03T16:50:00+08:00 — Docker browser path unavailable (`docker: command not found`). Supervisor selected standalone real-browser controller/SSE harness as boundary; explicitly prohibited host/fake-broker server.
- 2026-08-03T16:53:00+08:00 — Agent-browser standalone harness observed confirmed send, user echo, assistant SSE row in 81ms, meta-delta tool count increment; after forced SSE error, poll resumed from `c2`, advanced to `c3`, and EventSource reconnected using `c3`.
