---
name: codoxear-docker-test
description: Build and use an isolated Docker environment for codoxear web-session testing. Use when codoxear changes need browser/API verification involving session creation, tmux launch, Codex/Pi backends, launch ledgers, broker sockets, or cleanup behavior, and host live-server contamination must be avoided.
---

# Codoxear Docker Test

Use Docker as the default verification boundary for codoxear behavior that creates sessions. Do not create test sessions on the host. Do not use or mutate the host live server runtime state.

## Hard Boundaries

- Keep host `~/.local/share/codoxear`, host `~/.codex/sessions`, host `~/.pi/agent/sessions`, and the live codoxear service out of the test path.
- Do not run `codoxear-server` against the host app dir. Set `HOME` inside the container to a throwaway directory so `~/.local/share/codoxear` resolves inside the container.
- Do not point tests at the live host port, commonly `8743`. Pick a non-standard host port for each run, such as `18790` to `19999`, and verify the listener is the container.
- Do not copy host session logs, broker sockets, launch ledgers, or codoxear runtime JSON files into the container.
- Copy only configuration needed for backend CLIs to authenticate and select providers/models. Do not print config values in logs.
- Run browser verification from the host with `agent-browser` against the container-exposed port.

## Bootstrap

1. Choose an isolated work directory and port:

```bash
ROOT=/tmp/codoxear-docker-test-$(date +%Y%m%d-%H%M%S)
PORT=18790
mkdir -p "$ROOT"/{home,artifacts}
```

2. Prepare a throwaway home:

- Create `$ROOT/home/.codex` and `$ROOT/home/.pi/agent`.
- Copy needed Codex config files from host `~/.codex` into `$ROOT/home/.codex`, preserving names and permissions.
- Copy needed Pi config/model files from host `~/.pi/agent` into `$ROOT/home/.pi/agent`.
- Do not copy `~/.local/share/codoxear`, Codex rollout logs, Pi session logs, sockets, or prior Docker test homes.
- If a copied config points to absolute host paths, rewrite it inside `$ROOT/home` or fail before starting the container.

3. Build a container image from the current repo checkout:

- Include Python, pip, git, curl, node/npm if the Codex/Pi CLIs need them, and `tmux`.
- Install the repo with `python3 -m pip install -e /workspace`.
- Install or bind the same Codex and Pi CLI entrypoints the test needs. Verify inside the container with `command -v codex`, `command -v pi`, and `tmux -V`.

4. Start the service inside the container:

```bash
docker run --rm \
  --name codoxear-session-test \
  -p 127.0.0.1:${PORT}:${PORT} \
  -e HOME=/home/tester \
  -e CODEX_WEB_PASSWORD=test-password \
  -e CODEX_WEB_HOST=0.0.0.0 \
  -e CODEX_WEB_PORT=${PORT} \
  -v "$PWD":/workspace:ro \
  -v "$ROOT/home":/home/tester \
  -w /workspace \
  <image> \
  python3 -m codoxear.server
```

Use a stateful terminal or background process manager for this server. Capture container logs to `$ROOT/artifacts/server.log`.

## Verification Workflow

1. Confirm isolation before creating sessions:

```bash
curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:${PORT}/api/me
docker exec codoxear-session-test sh -lc 'echo "$HOME"; ls -la ~/.local/share/codoxear 2>/dev/null || true'
```

The first probe should show the container service, usually `401` before login. The app dir should be under the container home.

2. Run API checks against `http://127.0.0.1:${PORT}` only. Record:

- `/api/sessions` rows before, during, and after each launch.
- Container `~/.local/share/codoxear/session_launches.jsonl`.
- Container `~/.local/share/codoxear/socks/*.json`.
- `tmux list-sessions` and `tmux list-windows -a`.
- Container process tree for broker and backend CLI pids.

3. Run a real host-browser check with `agent-browser`:

```bash
agent-browser --cdp <port> open http://127.0.0.1:${PORT}/
agent-browser --cdp <port> snapshot -i -c --depth 12
```

Log in with the container password. Create, inspect, dismiss, or delete only container sessions. Screenshots or DOM snapshots should prove the UI state that API logs claim.

## Required Session Tests

For session-launch changes, cover at least:

- Codex direct start binds one broker row and one log.
- Codex tmux start creates one tmux pane and converges to one real session row.
- Codex tmux start-fresh does not inherit stale resume/provider/auth/model variables from tmux or server environment.
- Codex tmux resume carries the intended resume id only when explicitly requested.
- Metadata-delay path returns pending without a failed row when the pane is alive.
- Tmux command failure creates exactly one quiet failed launch row.
- Broker early exit creates exactly one quiet failed launch row.
- Failed launch row dismissal removes the row and does not try to kill a nonexistent broker.
- Real tmux session deletion stops the container broker/pane and does not create a synthetic failed row.
- Server restart inside the container does not resurrect dismissed failed rows.
- Pi direct and Pi tmux paths are checked when the touched code is backend-shared.

## Cleanup

Stop the container and remove only Docker test artifacts:

```bash
docker rm -f codoxear-session-test 2>/dev/null || true
```

Keep `$ROOT/artifacts` while reporting evidence. Remove `$ROOT` only after the investigation no longer needs raw logs.

Before touching the host live service, state what Docker evidence constrains and what it does not. A live smoke test, if needed, must use one fresh test session and must not kill existing brokers.
