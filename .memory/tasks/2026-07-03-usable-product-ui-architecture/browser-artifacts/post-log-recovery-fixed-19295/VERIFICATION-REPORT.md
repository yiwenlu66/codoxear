# Post-log-bound backend death recovery — fixed proof

Verdict: **PASS** — a web-owned session whose backend dies after a transcript log is bound remains visible and usable as a recovered failed row. The original user transcript is preserved, a lifecycle assistant error is appended in memory, controls that would send to the dead backend are blocked, export/copy stays available, and server restart rediscovery sources the row from the durable launch ledger.

## Mechanism corrected

- Post-log death is recorded as a durable `launch_attempt` failure with stage `*_after_log_bind`, `session_id` equal to the browser route id, `thread_id` equal to the backend log/thread id, and `log_path` pointing at the bound backend log.
- Backend logs are read-only. Codoxear does not mutate the backend log; message routes project the log events plus one lifecycle error: `The backend process stopped before completing this turn.`
- Missing-session `tail`, `history`, `live`, `search`, and `export` use the same recovered transcript payload.
- Large recovered logs read a recent tail window, expose a signed history cursor at the truncation boundary, and attach a row cursor to the lifecycle event when no real chat row exists in the tail window, so the browser `Load older messages` path can page older context.

## API proof in Docker sandbox

Sandbox: `codoxear-sandbox-19295`, port `127.0.0.1:19295`, app dir `/home/tester/.local/share/codoxear`.

Fixture sessions:

- `post-log-recovery-fixed`: incomplete Pi log containing `POST_LOG_BOUND_DEATH_SENTINEL`.
- `post-log-completed-control`: completed idle Pi log; negative control.
- `post-log-large-cursor`: >2 MiB Codex-style log with `FIRST_EVENT_SENTINEL` near the head and no chat rows in the retained tail window.

Final API observations (`api-proof-after-final-restart.json`):

- `/api/sessions` returns recovered rows for `post-log-recovery-fixed` and `post-log-large-cursor`, both `busy:false`, `queue_len:0`, `launch_state:"failed"`, stage `broker_exit_after_log_bind`.
- `post-log-completed-control` has `control_row:null`; its tail is `404`, proving the completed/idle negative control does not get a false stopped recovery row.
- `tail`, `history`, `live`, `search`, and `export` for `post-log-recovery-fixed` return the preserved user prompt plus exactly one lifecycle assistant error.
- Search finds both `POST_LOG_BOUND_DEATH_SENTINEL` and `The backend process stopped before completing this turn.`
- `send`, `enqueue`, `inject_file`, and `unattended` for the recovered route all return `404 unknown session`, blocking backend-control actions through failed-row semantics.
- `post-log-large-cursor` tail returns only the lifecycle error in the recent tail window, `has_older:true`, a top-level `history_cursor`, and a row `history_cursor`; `/messages/history` with that cursor returns `FIRST_EVENT_SENTINEL` without duplicating the lifecycle error.
- After container restart, `container-after-final-restart.txt` shows an empty `socks/` directory and persisted failed records in `session_launches.jsonl`; the API still returns both recovered rows and transcripts. The row is therefore ledger-sourced, not socket-memory-sourced.

## Browser proof

Real Chrome against the Docker server:

- Desktop and mobile recovered session render `POST_LOG_BOUND_DEATH_SENTINEL`, the assistant error row, and a recovery panel whose copy says the session stopped after binding a transcript log.
- Send, queue, attach, and unattended controls are disabled; export/copy remains enabled.
- Large recovered log initially renders the lifecycle error with a row history cursor and visible `Load older messages` control.
- Activating `Load older messages` loads `FIRST_EVENT_SENTINEL` into the transcript with no visible older-load error.

Artifacts:

- `browser-dom-desktop-after-final-restart.json`
- `browser-dom-mobile-after-final-restart.json`
- `browser-large-before-load-older-final.json`
- `browser-large-after-load-older-final.json`
- `browser-desktop-after-final-restart.png`
- `browser-mobile-after-final-restart.png`
- `browser-large-after-load-older-final.png`

## Validation

- Full local pytest after functional commits: `1774 passed, 132 subtests passed`.
- Docker unit: `1773 passed, 1 skipped, 132 subtests passed` (`docker-test-19296.log`).
- Docker smoke: pre-login `/api/me` `401`; post-login `/api/sessions` `200`; app dir `/home/tester/.local/share/codoxear` (`docker-smoke-19297.log`).

## Boundary

The proof uses deterministic stale sidecars and synthetic backend logs in Docker. That targets the defect mechanism directly: lifecycle recovery after a bound log disappears from active socket discovery. It does not claim real provider/backend inference health.
