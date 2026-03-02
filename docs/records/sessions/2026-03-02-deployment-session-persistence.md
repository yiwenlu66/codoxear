# Session: 2026-03-02 Deployment Session Persistence

## Focus
Document deployment practices that keep active sessions visible across service updates/restarts.

## Requests
- Add this guidance as part of deployment tutorial documentation.

## Actions Taken
- Updated `docs/flows/DEPLOYMENT.md` with a dedicated section explaining why sessions disappear after restart and how to prevent it in production.
- Added a concrete deploy checklist for supervisord environments.
- Added a systemd note (`KillMode=process`) for deployments where service restarts previously killed child session processes.
- Corrected stale host notes in deployment docs after moving runtime from `/root/code/codoxear-gemini` back to `/root/code/codoxear`.
- Updated `docs/records/WORK_RECORDS.md` and added this session record.

## Outcomes
- Deployment docs now include explicit anti-session-loss practices.
- Operators have a repeatable update sequence that preserves active broker sessions when possible.

## Tests
- Not run (documentation-only changes).
