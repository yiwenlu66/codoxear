# Session: 2026-03-02 GitHub Update Notice

## Focus
Add automatic update detection so the web UI prompts when GitHub has newer commits.

## Requests
- Auto prompt when GitHub has a new version.

## Actions Taken
- Added `GET /api/update` in the server with short git command timeouts and cached results.
- Added server-side commit comparison logic using local `HEAD` vs remote branch head.
- Added UI periodic polling for update status.
- Added a sidebar `Update` button (shown only when an update is available) and one-time toast prompt per remote commit.
- Added unit tests for update helper parsing, status computation, and cache behavior.
- Updated feature/route/testing docs.

## Outcomes
- The UI now auto-checks for upstream updates and prompts users when a newer GitHub commit exists.
- Polling is lightweight due to server caching and short command timeouts.

## Tests
- `python3 -m unittest tests.test_update_check`
- `python3 -m unittest discover -s tests`
- Manual UI verification: keep server running, temporarily make `/api/update` return `update_available=true`, then confirm the sidebar shows `Update` and the toast appears once for that commit.
