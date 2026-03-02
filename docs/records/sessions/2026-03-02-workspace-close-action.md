# Session: 2026-03-02 Workspace Close Action

## Focus
Add a workspace-level control to close a whole workspace from the sidebar.

## Requests
- Add a button to close the entire workspace.

## Actions Taken
- Added a `Close` button in each workspace header in the sidebar.
- Implemented `closeWorkspace(...)` in UI logic to batch-delete web-owned sessions for that workspace.
- Added ownership-aware handling so non-web sessions are skipped (not deleted) with clear toast feedback.
- Reused existing local cleanup paths (cache, queue, pending echoes, drafts, unread/user-summary state) for each deleted session.
- Added workspace header action styles for the new close button.
- Updated UI feature documentation and work-record index.

## Outcomes
- Users can close a workspace in one action when sessions are web-owned.
- Mixed workspaces no longer require manual session-by-session cleanup for owned sessions; non-owned sessions remain safe.

## Tests
- `node --check codoxear/static/app.js`
- `python3 -m unittest discover -s tests`
- Manual UI verification: click workspace `Close`, confirm deletion prompt, and verify owned sessions are removed with toast feedback.
