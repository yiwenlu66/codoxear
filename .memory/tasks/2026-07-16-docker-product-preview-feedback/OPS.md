# OPS — Docker product preview and continuous feedback

## 2026-07-16T17:40:45+08:00 — Task initialized
- User requested a persistent Docker deployment of the refactored product, remotely reachable on a non-8743 port, with container-local broker/socket/runtime state, real project access, backend configs, zsh environment, and a durable feedback/iteration task.
- Initial product source: `/home/yiwen/codex-web-product-recovery` on clean `recovery/product-gaps` at `15cccc8`.
- Host live service remains active on port 8743 and is out of scope for all preview operations.
- Candidate preview port `19580` was free. Host addresses observed: Tailscale `100.76.186.22`, LAN `192.168.1.8`.
- Candidate project `/home/yiwen/codex_ws` exists and is approximately 3.0 GiB. Observation: it contains sensitive personal and financial files as well as code/artifacts. Decision: remote preview access must remain password-gated and user guidance will prefer the encrypted Tailscale path. Project mount is read-write only because the user explicitly requested real-project testing.
- Host backend/tool observations: Pi 0.80.6, Codex CLI 0.142.2, Claude Code 2.1.173, zsh 5.9. Host `.zshrc` contains API credential exports and sources `$HOME/.env`; values were not printed.
- Isolation design commitment: host live Codoxear app dir, backend session logs, sockets, and tmux state will not be mounted or copied. Only required config/auth inputs are mounted read-only; generated backend/runtime state belongs under preview HOME.

## 2026-07-16T17:49:00+08:00 — Mount and network design corrected after adversarial review
- Critic `ce23fb8d-946f-468f-94d3-840d77657098` found the host `.zshrc` cannot be used verbatim: it aliases tmux through host systemd, assumes Oh My Zsh/plugins/zoxide, and contains host-only paths. Mechanism: an active verbatim mount would make interactive `tmux` fail inside the container even though the binary exists.
- Decision: mount `.zshrc` only as a read-only source and generate a container-tailored active shell configuration with container PATH/credential exports and no tmux alias. Required shell is still zsh.
- Critic found host Codex config contains many absolute `/home/yiwen` paths and its ChatGPT OAuth token is stale. Decision: use a rewritten preview-home copy and API-key provider variables from the shell source; do not rely on host-browser OAuth. Pi verification will use a configured inline-key provider rather than the five-day OAuth path. Claude hook paths are treated as preview config to sanitize.
- Network prediction corrected: raw `0.0.0.0:19580` would expose sensitive mounted workspace/config data over plaintext LAN HTTP. Docker will bind only `127.0.0.1:19580`; Tailscale Serve HTTPS `19581` will provide encrypted tailnet-only remote access and secure-context browser behavior.

## 2026-07-16T17:58:00+08:00 — Tailnet HTTPS endpoint established
- Command: `tailscale serve --bg --yes --https=19581 http://127.0.0.1:19580`.
- Observation: Tailscale registered `https://yiwen-workstation.tail0de6f7.ts.net:19581/` as a tailnet-only proxy to preview loopback port 19580.
- Direct HTTPS probe to `/api/me` returned 401 with certificate verification result 0 and remote address `100.76.186.22`. This proves encrypted remote routing reaches the preview password gate without exposing raw container HTTP to LAN.

## 2026-07-16T18:02:00+08:00 — First real browser launch exposed Pi extension dependency drift
- Browser `codoxear-preview` logged in through the Tailscale HTTPS endpoint, opened New Session, selected Pi provider `occ`, cwd `/home/tester/codex_ws`, tmux transport, and created session alias `Docker preview acceptance`.
- The browser accepted the prompt `Reply with exactly: DOCKER-PREVIEW-READY. Do not use tools or modify files.` and initially projected the session busy. The broker then exited before binding a Pi log; Codoxear truthfully replaced it with failed launch `launch-1784196154442-f5a897a7`, stage `agent_exit_before_log_bind`, error `pi exited with status 0 before a session log was bound`.
- Decisive `pty_tail` evidence: Pi lazily installed extensions, then `lsp-pi` failed because package subpath `./node.js` is not exported by preview `vscode-languageserver-protocol` 3.18.2. Host working Pi npm tree uses the same `lsp-pi` 1.0.5 with protocol 3.18.1 and `vscode-jsonrpc` 9.0.0; preview resolved protocol 3.18.2/jsonrpc 9.0.1.
- Interpretation: config/auth and Codoxear launch wiring worked through broker creation and prompt acceptance; the blocker is nondeterministic extension dependency resolution in the new preview HOME, matching the historical clean-container Pi extension failure. It is not a product UI defect and must be repaired in the preview environment without adding a user-facing no-extension control.
- Intervention dispatched: seed a writable isolated copy of the host working `~/.pi/agent/npm` tree into preview HOME, refresh when its package lock changes, restart the container, then rerun the exact browser discriminator.

## 2026-07-16T18:10:00+08:00 — Pi dependency repair and real browser acceptance passed
- The first two delegated repair attempts failed before edits because their provider account reached a seven-day usage limit. Main applied the bounded environment fix directly after confirming no partial changes.
- `scripts/codoxear-docker-preview` now requires host Pi npm lock/node_modules and seeds a writable copy into preview HOME only when missing or lock-different. `start` removes the named container before replacing the tree, so no running preview process observes the replacement. The host npm tree is not mounted.
- Post-restart discriminator: preview versions are `lsp-pi 1.0.5`, `vscode-languageserver-protocol 3.18.1`, and `vscode-jsonrpc 9.0.0`; preview and host package locks match; Docker mounts contain no `/home/tester/.pi/agent/npm` bind. A bounded TTY probe reached the Pi 0.80.6 prompt with the full extension list and no extension/subpath error.
- Browser reran the original path through Tailscale HTTPS: dismissed the prior failed launch, New Session -> Pi -> provider `occ`, cwd `/home/tester/codex_ws`, tmux transport, then sent the same prompt. Session `broker-514` rendered exact assistant text `DOCKER-PREVIEW-READY` and returned idle.
- Container evidence: `broker-514.json` advertises `control_protocol_version:2` with `sync_send:true` and `key_write_errors:true`; its socket exists inside preview `~/.local/share/codoxear/socks`; its Pi log is under `/home/tester/.pi/agent/sessions/...`; container tmux session `codoxear` owns one window. Host live socket directory still contains exactly the four pre-existing broker sidecars and no preview broker.
- File/product evidence: browser opened `agent-browser-download-payload.txt` from the mounted workspace; status reported 29 B, Monaco was loaded, and desktop had no horizontal overflow. At 390x844 mobile viewport the file viewer remained open, Monaco remained available, and document horizontal overflow was false. The temporary screenshot and automation browser session were removed; the preview session/container remain running for the user.
- Functional tooling committed as `0ef5114` (`Add isolated Docker product preview`). Initial task memory commit is `f66161f`.

## 2026-07-16T18:15:00+08:00 — Final clean-room review accepted preview
- Fresh critic `60b18e77-2b5e-4498-958c-ac08c5fc9955` audited the Dockerfile, script, live named container, mounts, endpoint, CLI versions, workspace/source writability, host-sidecar separation, and broker/log evidence without edits or runtime restart. Verdict: no blockers; preview is user-ready.
- Reviewer independently observed healthy container, loopback and HTTPS 401, HTTPS login page 200, Pi/Codex/Claude CLI availability, workspace read-write and source read-only behavior, unchanged host live sidecars, and `DOCKER-PREVIEW-READY` in the preview Pi log.
- One reviewer residual claimed Pi LSP could still fail under protocol 3.18.1. Direct evidence falsifies that prediction for the running environment: preview version is 3.18.1, bounded Pi startup loaded both `lsp-pi` extensions with `LSP pyright` visible and no extension error, and the subsequent real browser session completed. The failed first launch used 3.18.2. No product/environment change is justified against a contradicted residual.
- Reviewer-created workspace writability probe `.test` was removed; no project residue remains. Empty normal-operation `docker logs` is not a failure signal because the service is healthy and route/browser evidence is direct; errors remain available when emitted.
