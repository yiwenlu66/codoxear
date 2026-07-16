# OPS — Docker product preview and continuous feedback

## 2026-07-16T17:40:45+08:00 — Task initialized
- User requested a persistent Docker deployment of the refactored product, remotely reachable on a non-8743 port, with container-local broker/socket/runtime state, real project access, backend configs, zsh environment, and a durable feedback/iteration task.
- Initial product source: `/home/yiwen/codex-web-product-recovery` on clean `recovery/product-gaps` at `15cccc8`.
- Host live service remains active on port 8743 and is out of scope for all preview operations.
- Candidate preview port `19580` was free. Host addresses observed: Tailscale `100.76.186.22`, LAN `192.168.1.8`.
- Candidate project `/home/yiwen/codex_ws` exists and is approximately 3.0 GiB. Observation: it contains sensitive personal and financial files as well as code/artifacts. Decision: remote preview access must remain password-gated and user guidance will prefer the encrypted Tailscale path. Project mount is read-write only because the user explicitly requested real-project testing.
- Host backend/tool observations: Pi 0.80.6, Codex CLI 0.142.2, Claude Code 2.1.173, zsh 5.9. Host `.zshrc` contains API credential exports and sources `$HOME/.env`; values were not printed.
- Isolation design commitment: host live Codoxear app dir, backend session logs, sockets, and tmux state will not be mounted or copied. Only required config/auth inputs are mounted read-only; generated backend/runtime state belongs under preview HOME.
