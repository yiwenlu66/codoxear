# Post-log-bind backend death proof scout

Verdict: **DEFECT**

HEAD: `ce401c313c103491738d2f433c9ad0adbe59407d`

## What was proven

A web-owned session with an already-bound backend log can disappear without any durable transcript outcome or recovery row if the backend/broker dies after the log contains a user message but before a terminal assistant/error/no-response/interruption row.

The deterministic proof creates a fake Pi session log with:

- session header
- one user message: `POST_LOG_BOUND_DEATH_SENTINEL`
- no assistant/error/no-response/interruption/terminal row

It then registers the session through the real discovery/registry path, reads messages through the real message route handlers, removes the fake socket/sidecar to simulate production broker cleanup, and invokes the real prune coordinator with dead pids.

## Key observations

Before simulated death:

- Session listed: `true`
- `/messages/tail` equivalent status: `200`
- Tail transcript state: `bound`
- Tail events: one user event containing `POST_LOG_BOUND_DEATH_SENTINEL`
- `/messages/search` equivalent status: `200`
- Search match count: `1`

After simulated post-bind death and prune:

- Session listed: `false`
- `/messages/tail` equivalent status: `404`, `unknown session`
- `/messages/search` equivalent status: `404`, `unknown session`
- Failure records written by prune: `0`
- Visible launch/recovery rows after prune: `0`
- Launch ledger state remains only `log_bound`
- The backend log still exists on disk, but no API-visible session or recovery row points to it

Raw proof files:

- `proof-output.json`
- `proof-summary.txt`
- `raw-api/before-sessions.json`
- `raw-api/before-tail.json`
- `raw-api/before-search.json`
- `raw-api/after-sessions.json`
- `raw-api/after-tail.json`
- `raw-api/after-search.json`
- `raw-api/launch-ledger-after.json`

## Mechanism

The durable failure paths all treat `log_path is None` as the condition for recovery-row creation:

- `codoxear/broker.py`: the web-owned exit guard records `agent_exit_before_log_bind` only when `st2.log_path is None or not st2.log_path.exists()`.
- `codoxear/session_discovery.py`: dead owned sidecars create `broker_exit_before_log_bind` only when `log_path is None`.
- `codoxear/session_prune.py`: `_record_pruned_launch_failure()` returns unless `session.owned and session.log_path is None and not existing_launch_failed`.
- `codoxear/message_routes.py`: once the session is gone, tail can only recover through `launch_attempt_transcript_for_session_id(session_id)`. The proof ledger contains only `log_bound`, which intentionally produces no failed launch row.

Therefore a bound-but-incomplete log disables every existing launch/recovery persistence path. When broker exit unlinks socket/sidecar and prune drops the in-memory session, the browser has neither a selectable session nor a launch/recovery transcript row. The user prompt survives only in an orphaned backend log path.

## Commands run

```bash
git status --short --untracked-files=all && git diff --stat && git diff --cached --stat
python3 .memory/tasks/2026-07-03-usable-product-ui-architecture/browser-artifacts/post-log-backend-death-scout/prove_post_log_backend_death.py
```

The first proof run exposed a JSON serialization bug in the artifact driver for `Path` objects; the driver was corrected and rerun. The final run produced `Verdict: DEFECT`.

## Isolation boundary

No Codoxear server, broker, sessiond, tmux, or backend process was started. No host live runtime, protected checkout, broad process cleanup, or Docker process was used. The proof uses in-process production components and proof-owned files under this artifact directory.

Because this was an in-process/API-route proof, it does not include browser DOM screenshots. Browser proof would need only to confirm the already-proven API projection: the session row disappears or becomes unavailable and no recovery panel/transcript outcome appears.

## Scope confirmation

No source files or tests were edited. Nothing was staged or committed. Files written are proof artifacts under:

`.memory/tasks/2026-07-03-usable-product-ui-architecture/browser-artifacts/post-log-backend-death-scout/`
