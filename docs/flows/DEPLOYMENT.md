# Deployment

Codoxear is a single-process server intended to run on the same machine as Codex/Claude/Gemini CLI. In this deployment, the server is exposed to the public internet (direct port forward or reverse proxy). It does not provide TLS.

## Production environment (this host)
In this project, "production" refers to the service running on this host (the same machine that runs Codex/Claude/Gemini CLI), not a separate environment.

## Minimal deployment (public access)
1. Set `CODEX_WEB_PASSWORD` in `.env` (loaded from the server working directory) or the environment.
2. On this host, manage the server as a daemon with `supervisorctl` (`codoxear` program).
3. Start sessions with `codoxear-broker` and select CLI via `CODEX_WEB_CLI`:
   - `CODEX_WEB_CLI=codex codoxear-broker -- <codex args>`
   - `CODEX_WEB_CLI=claude codoxear-broker -- <claude args>`
   - `CODEX_WEB_CLI=gemini codoxear-broker -- <gemini args>`
4. Expose the configured port (this host currently uses `13780`) to the internet (or proxy it) and visit `http://<public-host>:<port>/` (or your HTTPS proxy URL).

## Network security
- The server does not provide TLS or strong authentication beyond a password.
- For public access, prefer a reverse proxy with TLS termination and IP allowlists if possible.
- If you terminate TLS in a proxy, set `CODEX_WEB_COOKIE_SECURE=1` or forward `X-Forwarded-Proto: https`.
- Treat the password as the only gate; assume anyone who can reach the port can observe or modify traffic.

## Public access options
- Direct port forward: expose your configured port (`13780` on this host) on your router or cloud firewall, and restrict source IPs if possible.
- Reverse proxy: terminate TLS and optionally mount under a subpath via `CODEX_WEB_URL_PREFIX`.

## URL prefix
Use `CODEX_WEB_URL_PREFIX=/codoxear` to serve the UI and API under a subpath. Cookie scope follows the prefix.

## Runtime paths
Runtime state is stored under `~/.local/share/codoxear`:
- `socks/` broker sockets and metadata
- `hmac_secret` cookie signing key
- `uploads/` temporary uploaded images

## Daemon operations (this host)
This host runs Codoxear under `supervisord` instead of `systemd`.

- Program name: `codoxear`
- Supervisor config: `/mlplatform/supervisord/supervisord.conf`
- Current runtime env in supervisor config: `CODEX_WEB_PORT=13780`, `CODEX_WEB_HOST=0.0.0.0`

Common commands:
- `supervisorctl status codoxear`
- `supervisorctl restart codoxear`
- `supervisorctl start codoxear`
- `supervisorctl stop codoxear`

## Gemini all-approve mode (this host)
To run web-owned Gemini sessions in "all approve" mode without changing app code:

1. Create wrapper script:
   - `/usr/local/bin/gemini-web`:
   - `#!/usr/bin/env bash`
   - `exec gemini --approval-mode yolo "$@"`
2. `chmod +x /usr/local/bin/gemini-web`
3. Add `GEMINI_BIN="/usr/local/bin/gemini-web"` to the `codoxear` `environment=` line in `/mlplatform/supervisord/supervisord.conf`.
4. Apply and restart:
   - `supervisorctl reread`
   - `supervisorctl update`
   - `supervisorctl restart codoxear`

As of 2026-03-02, this host uses exactly this wrapper + env pattern.

As of 2026-02-27, `systemctl` on this host reports the system is not booted with `systemd` as PID 1, so `systemd` units are not the active service manager here.

## Fixed startup (manual fallback)
Use `scripts/codoxear-server-dev` when you need a foreground/manual run from the repo root.

## This host (observed)
As of 2026-03-02, the running server process is managed by `supervisord` and started from `/root/code/codoxear-gemini`, so `.env` is read from `/root/code/codoxear-gemini/.env`.

## Env file location (this host)
- `/root/code/codoxear-gemini/.env` (because the server is running from `/root/code/codoxear-gemini`).

## Operational notes
- Terminal-owned sessions are attach-only and cannot be killed via the UI.
- Web-owned sessions can be deleted from the UI.
