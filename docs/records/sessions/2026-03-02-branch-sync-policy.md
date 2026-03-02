# Session: 2026-03-02 Branch Sync Policy

## Focus
Correct branch state after an accidental merge to `main`, and document the stable branch workflow.

## Requests
- Fix branch state so `main` aligns with upstream and development continues on `dev`.
- Write the operation steps into `AGENTS.md` and the corresponding flow docs.

## Actions Taken
- Moved development commits to `dev` and pushed `origin/dev`.
- Realigned `main` to `upstream/main` and force-updated `origin/main` with `--force-with-lease`.
- Added explicit branch policy and recovery flow to `AGENTS.md`.
- Added branch model, normal workflow, and recovery checklist to `docs/flows/DEVELOPMENT.md`.
- Added a README entry pointing contributors to the development flow doc.

## Outcomes
- `main`, `origin/main`, and `upstream/main` are aligned.
- `dev` now carries the latest development commits and remains the integration branch.
- The branch recovery procedure is documented for future incidents.

## Verification
- `git rev-list --left-right --count upstream/main...main` returned `0 0`.
- `git branch -vv` confirms `main` tracks `origin/main` at upstream-aligned commit.
- `git branch -vv` confirms `dev` tracks `origin/dev` at the latest development commit.
