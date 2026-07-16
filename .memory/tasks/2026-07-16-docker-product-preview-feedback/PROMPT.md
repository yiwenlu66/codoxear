## Objective
Run the refactored Codoxear product as a persistent, host-isolated Docker preview that the user can reach from another device and use against a real mounted project workspace. Keep the preview alive for continuous hands-on feedback, and iterate on every observed product defect in this task until the user ends the preview campaign.

Done means the exact recovery release is running in Docker; server, app dir, brokers, backend CLIs, sockets, logs, and tmux state remain inside the container/persistent preview home; `/home/yiwen/codex_ws` is available inside the container for real work; Pi, Codex, Claude Code, zsh, and environment configuration are available without copying host runtime/session state; the service is reachable on a non-8743 host port from the user's other devices; a real browser login/session round trip is verified; and operating/access instructions are recorded.

## Workbench
1. Build the persistent preview image/runtime without touching host Codoxear runtime.
2. Verify container identity, zsh/CLI/config behavior, project mount, network reachability, and browser product flows.
3. Keep an append-only feedback/evidence trail in OPS.md and maintain the live causal model in EPISTEMIC.md.
4. Convert each user observation into a reproducible mechanism, fix, isolated validation, and refreshed preview deployment.

## Context
- Active product checkout: `/home/yiwen/codex-web-product-recovery`, branch `recovery/product-gaps`, initial HEAD `15cccc8`.
- Protected live checkout/service: `/home/yiwen/codex-web`, `codoxear-server.service`, port `8743`; never modify or restart it in this task.
- Host live runtime: `/home/yiwen/.local/share/codoxear`; never mount or copy it.
- Preview project source: `/home/yiwen/codex_ws`, mounted read-write at `/home/tester/codex_ws` by explicit user request.
- Preview state root: `/home/yiwen/.local/share/codoxear-preview-19580`; container HOME is its `home/` subdirectory.
- Preview container/image/port: `codoxear-preview-19580`, `codoxear-preview:recovery-15cccc8`, host port `19580` to container port `19580`.
- Host reachability: Tailscale `100.76.186.22`, LAN `192.168.1.8`.
- Canonical Docker isolation rules: `.codex/skills/codoxear-docker-test/SKILL.md`.
- Project architecture/validation: `.memory/project/ARCHITECTURE.md`, `.memory/project/VALIDATION.md`.

## Task specifications
- Use the recovery checkout as read-only `/workspace` and run `python3 -m codoxear.server` from that source.
- Run as non-root user `tester` with `/usr/bin/zsh` as its login shell.
- Make `pi`, `codex`, and `claude` executable inside the container and verify their versions.
- Mount only configuration/auth material needed for backend operation. Host Pi/Codex/Claude session logs, histories, runtime sockets, app state, and live Codoxear files must not enter the preview.
- Mount host `.zshrc` and Codoxear `.env` read-only. The preview uses the password from that env; never print it into task memory, logs, commands shown to the user, or committed artifacts.
- Because host `.zshrc` contains credential environment variables, keep remote access password-gated and prefer the encrypted Tailscale address in user instructions. Do not expose an unauthenticated endpoint.
- Mount `/home/yiwen/codex_ws` read-write at `/home/tester/codex_ws`. This intentionally permits preview agents to modify host project files; no other host project root is writable.
- Do not mount the Docker socket, host process namespace, host tmux socket, or host `/tmp`.
- Container-generated `~/.local/share/codoxear`, backend sessions, sockets, logs, and tmux state must resolve below the preview HOME.
- Use a restartable named container with a persistent preview HOME. Rebuild/restart must preserve preview state unless a defect experiment explicitly requires a clean home.
- Bind a non-8743 port so another device can access the host. Verify unauthenticated `/api/me` returns 401 from localhost and the machine address, then verify authenticated `/api/sessions` returns 200 without retaining cookies in git artifacts.
- Verify through a real browser: login, New Session choices, project cwd, one backend launch, one confirmed prompt/response, transcript, and file access inside the mounted workspace. All resulting brokers/sockets/logs must be inside Docker.
- Provide user instructions for URL, password source, selecting `/home/tester/codex_ws`, creating a backend session, testing files/git/attachments/mobile controls, reporting feedback, checking logs, restarting, and stopping the preview.
- Every feedback item must record observation, mechanism, prediction, intervention, and user-facing verification. Do not call an issue fixed from tests alone.

## Constraints
- Do not edit, restart, stop, or reconfigure host `codoxear-server.service`.
- Do not use host port `8743`.
- Do not mount or copy `/home/yiwen/.local/share/codoxear`.
- Do not copy host Pi/Codex/Claude session histories or logs into the preview.
- Do not expose credential values in git, task memory, terminal output, Docker inspect summaries, screenshots, or final responses.
- Do not commit generated preview HOME, backend sessions, cookies, logs, sockets, or credentials.
- Do not use `pkill -f`, `killall`, broad `pgrep | xargs kill`, or `tmux kill-server`.
- Cleanup is exact named-container or exact preview tmux-session scoped.
- Keep functional code, deployment tooling, and task-memory commits atomic.
- Preserve the preview for user feedback after verification; do not stop it at task completion.
