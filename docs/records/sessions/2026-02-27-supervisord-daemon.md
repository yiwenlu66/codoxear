# Session: 2026-02-27 Supervisord Daemon

## Focus
Switch Codoxear startup to daemon management and document the real host runtime path.

## Requests
- Run Codoxear as a daemon.
- Record daemon operations in project docs.

## Actions Taken
- Verified `systemctl` is unavailable as the active service manager because PID 1 is not `systemd` on this host.
- Confirmed `codoxear` is configured in `/mlplatform/supervisord/supervisord.conf` with `autostart=true` and `autorestart=true`.
- Restarted the managed program via `supervisorctl restart codoxear`.
- Verified service health at `http://127.0.0.1:13780/` and `http://localhost:13780/`.
- Updated development/deployment flow docs to include `supervisorctl` daemon operations and host port notes.

## Outcomes
- Codoxear is running under daemon supervision (`supervisord`) on port `13780`.
- Documentation now reflects the actual host service manager and operator commands.

## Tests
- `supervisorctl status codoxear`
- `python3` URL probe for `http://127.0.0.1:13780/` and `http://localhost:13780/` (HTTP 200)
