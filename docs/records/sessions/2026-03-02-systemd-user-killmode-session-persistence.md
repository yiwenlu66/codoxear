# Session: 2026-03-02 Systemd User KillMode Session Persistence

## Focus
Document the verified `systemd --user` deployment fix that keeps sidebar sessions from disappearing after service restart.

## Requests
- Add the confirmed "correct deployment method" into the deployment tutorial.

## Actions Taken
- Expanded `docs/flows/DEPLOYMENT.md` with a dedicated `systemd --user` checklist for the restart/session-loss issue.
- Documented the verified root cause:
  - `KillMode=control-group` causes `restart` to kill broker/CLI child processes in the same cgroup.
- Added verification commands for:
  - `KillMode` / `SendSIGKILL`
  - unit definition (`systemctl --user cat`)
  - cgroup process tree in `systemctl --user status`
- Added remediation steps using `systemctl --user edit` with `KillMode=process` (and optional `SendSIGKILL=no`), plus reload/restart/re-verify commands.

## Outcomes
- Deployment docs now include a concrete, repeatable `systemd --user` flow to prevent restart-time session loss.
- Operators can validate both configuration and runtime cgroup behavior before and after changes.

## Tests
- Not run (documentation-only changes).
