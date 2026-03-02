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

## Keep sessions across updates/restarts
Session visibility is based on live broker sockets (`~/.local/share/codoxear/socks/*.sock`), not a durable "session database". If broker/CLI processes are killed during deployment, those sessions disappear from the sidebar after restart.

To avoid losing active sessions on updates:

1. Run web-owned sessions under tmux:
   - Set `CODEX_WEB_TMUX=1` in the service environment.
   - Keep `tmux` installed on the host.
2. Restart only the Codoxear server service during deploy:
   - Avoid blanket process-kill commands such as `pkill -f codoxear` that can kill broker children too.
3. Keep service user and `HOME` stable:
   - Runtime discovery depends on the current user's `~/.local/share/codoxear`.
4. Keep runtime state on persistent storage:
   - Do not clean `~/.local/share/codoxear/socks` during deploy/boot scripts.
5. For systemd deployments:
   - Prefer `KillMode=process` for the Codoxear unit so service restarts do not kill the whole cgroup session tree.

Systemd `--user` checklist for "restart clears all sessions":

If your service is managed by `systemd --user`, a common failure mode is `KillMode=control-group` (default). In this mode, `systemctl --user restart codoxear.service` kills every process in the unit cgroup, including broker/CLI children, so sidebar sessions disappear after restart.

1. Verify current kill behavior:
   - `systemctl --user show codoxear.service --property=KillMode,SendSIGKILL`
   - `systemctl --user cat codoxear.service`
2. Confirm broker/CLI are in the same cgroup as the web service:
   - `systemctl --user status codoxear.service`
   - Check the `CGroup:` process tree for `codoxear-broker` / `codex` / `claude` / `gemini`.
3. Apply an override:
   - `systemctl --user edit codoxear.service`
   - Add:
     - `[Service]`
     - `KillMode=process`
     - `SendSIGKILL=no` (optional; use only if you explicitly want to avoid hard-killing remaining children at timeout)
4. Reload and restart:
   - `systemctl --user daemon-reload`
   - `systemctl --user restart codoxear.service`
5. Re-verify:
   - `systemctl --user show codoxear.service --property=KillMode,SendSIGKILL`
   - `scripts/codoxear-status --web --list`
   - `ls ~/.local/share/codoxear/socks/*.sock`

Recommended deploy sequence (supervisord):

1. Update code in place (`git pull` on the running repo path).
2. Reinstall deps if needed (for example `python3 -m pip install -e .`).
3. Restart service:
   - `supervisorctl restart codoxear`
4. Verify active brokers still exist:
   - `supervisorctl status codoxear`
   - `scripts/codoxear-status --web --list`
   - `ls ~/.local/share/codoxear/socks/*.sock`

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
As of 2026-03-02, the running server process is managed by `supervisord` and started from `/root/code/codoxear`, so `.env` is read from `/root/code/codoxear/.env`.

## Env file location (this host)
- `/root/code/codoxear/.env` (because the server is running from `/root/code/codoxear`).

## Operational notes
- Terminal-owned sessions are attach-only and cannot be killed via the UI.
- Web-owned sessions can be deleted from the UI.
