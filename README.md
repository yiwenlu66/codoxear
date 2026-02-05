# Codoxear

<p align="center">
  <img src="codoxear/static/codoxear-icon.png" alt="Codoxear icon" width="140" />
</p>

Unofficial mobile handoff for Codex TUI sessions.

Codoxear runs a small web server on your computer and exposes a phone-friendly UI for continuing the same live Codex TUI session from mobile. Your environment stays local (filesystem, tools, credentials). The phone is a view/controller.

Name: "codoxear" = "codex dogear" (dog-ear a page so you can pick up where you left off), meaning you can seamlessly continue the same work from different devices.

Not affiliated with OpenAI. "Codex" is referenced only for compatibility with the Codex CLI TUI.

## Platform support

Supported:

- Linux (uses `/proc`, PTYs, and `strace` integration)

Not supported:

- macOS (no `/proc`, `strace` is Linux-specific)
- Windows (no POSIX PTY/termios model; use WSL2 if you want a Linux environment)

## Quick start

Requires Python 3.10+.

Install Codoxear (installs `codoxear-server` and `codoxear-broker`):

- `python3 -m pip install .`

1. Create `.env`:

   - Copy `.env.example` to `.env`
   - Set `CODEX_WEB_PASSWORD`
   - Codoxear reads `.env` from your current working directory

2. Start the server:

   - `codoxear-server`
    - Default bind: `::` (IPv6, usually reachable on LAN)
    - Default port: `8743`

3. Wrap your local `codex` with the broker (zsh/bash function, not an alias):

   Add to `~/.zshrc` or `~/.bashrc`:

   ```sh
   codex() {
     codoxear-broker -- "$@"
   }
   ```

   Restart your shell or `source` your rc file.

4. Start Codex in your terminal as usual (via the wrapper). Codoxear will discover the session.

5. On your phone, open `http://<your-computer>:8743`, enter the password, and select the session.

## User stories

- Desktop Linux: start Codex in your GUI terminal emulator, then continue the same live TUI session on your phone or a laptop browser.
- Headless Linux: start Codex inside `tmux`, then attach from your phone or a laptop browser. This avoids using a mobile terminal emulator for TUI interaction (for example Termius).
- Web-owned sessions: start a new Codex session from the Codoxear UI, use it from mobile, and kill it from the UI when finished.

## Session ownership

Codoxear shows two kinds of sessions:

- Terminal-owned: sessions started from your local terminal (via the `codex` wrapper). Codoxear can attach, but it does not offer a kill button.
- Web-owned: sessions started from the Codoxear UI ("New session"). These are owned by the web server and show a delete button in the session list.

If you start a web-owned session and later want to continue it in your terminal, use `codex resume`.

## Known limitations

### No elevated operations inside brokered Codex

Codex instances started via `codoxear-broker` cannot reliably run elevated operations (for example `sudo`, `pkexec`, or other setuid/file-capability programs).

Reason: the broker runs Codex under `strace -f` (ptrace) to detect which `rollout-*.jsonl` file is active (the rollout log can be opened by child processes, not just the top-level launcher).

Workaround: run privileged commands outside the ptrace-traced Codex process tree. If you want a persistent systemd-based shell for this, use PiloTY (`https://github.com/yiwenlu66/PiloTY`) and run your long-lived shell sessions as transient user units:

- `XDG_RUNTIME_DIR="/run/user/$(id -u)" DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$(id -u)/bus" systemd-run --user --pty bash --noprofile --norc`

Run `sudo ...` inside that shell.

## Security model

This project intentionally keeps security out of scope. It provides password gating only and does not provide TLS.

Assume anyone who can reach the port can:

- observe traffic (including the password)
- modify traffic

Use your own secure channel (VPN, SSH port-forward, reverse proxy with TLS) if you need network security.

## Configuration

Set these in `.env` (or in the process environment):

- `CODEX_WEB_PASSWORD` (required)
- `CODEX_WEB_HOST` (default `::`)
- `CODEX_WEB_PORT` (default `8743`)
- `CODEX_HOME` (default `~/.codex`)
- `CODEX_BIN` (default `codex`)

Runtime state is stored under `~/.local/share/codoxear` (or legacy `~/.local/share/codex-web`).

## License

MIT, see `LICENSE`.
