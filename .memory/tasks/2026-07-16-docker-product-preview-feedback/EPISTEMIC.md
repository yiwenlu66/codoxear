# Epistemic model — Docker product preview and feedback campaign

## Phenomenon
The refactored product cannot yet replace the live service because current host brokers/runtime state are legacy, but it now runs as a persistent Docker preview against a real host workspace. This task owns continuous user feedback and iteration without risking the live service.

## Accepted mechanism
A named container with dedicated persistent HOME owns the complete preview control plane: Codoxear server/app state, protocol-v2 brokers, Unix sockets, backend logs/sessions, and tmux. Product source is read-only `/workspace`. Selective host config/auth sources are mounted read-only or copied/sanitized into preview HOME; host backend histories, Codoxear runtime, sockets, process namespace, tmux socket, Docker socket, and `/tmp` are absent.

The only intentional write boundary is `/home/yiwen/codex_ws` -> `/home/tester/codex_ws`. Docker HTTP binds to `127.0.0.1:19580`; Tailscale Serve terminates tailnet-only HTTPS at `https://yiwen-workstation.tail0de6f7.ts.net:19581/`. Password authentication uses the existing Codoxear env without recording its value. The preview must use a cookie name distinct from the main service because browser cookie identity ignores port; separate HMAC secrets alone do not isolate browser auth state.

The active container zsh configuration is generated from selected credential exports and container paths; the host systemd-backed tmux alias is excluded. Pi/Codex/Claude binaries are available, and tester's login shell is `/usr/bin/zsh`.

## Supported claims
- Container `codoxear-preview-19580` is healthy with restart policy `unless-stopped`; image is `codoxear-preview:recovery-15cccc8`; persistent state root is `/home/yiwen/.local/share/codoxear-preview-19580`.
- Tailscale HTTPS certificate/routing and unauthenticated 401 are proven. Real browser login and authenticated product bootstrap passed.
- Browser New Session -> Pi `occ/gpt-5.6-sol` -> `/home/tester/codex_ws` -> tmux -> confirmed prompt produced exact visible response `DOCKER-PREVIEW-READY`.
- Resulting `broker-514` advertises control protocol 2 and confirmed-send/key-error capabilities. Socket, Pi JSONL log, and tmux window exist only inside preview HOME/container. Host live broker sidecars remain the same four pre-existing files.
- Browser file picker opened a real mounted-workspace file; Monaco loaded. Desktop and 390x844 mobile had no horizontal overflow.
- The known clean-home Pi extension failure is resolved by copying the host working locked npm tree into preview HOME. This is an isolated writable copy, refreshed only when package lock differs, never a host bind.

## Anomaly absorbed
The first Pi launch exited before log bind because clean npm resolution paired `lsp-pi` 1.0.5 with `vscode-languageserver-protocol` 3.18.2/jsonrpc 9.0.1. The working host lock pins protocol 3.18.1/jsonrpc 9.0.0. Seeding the exact locked tree changed the same browser discriminator from durable failed-launch recovery to a bound protocol-v2 response. This localizes the problem to preview dependency resolution, not Codoxear launch/projection semantics.

## Resolved cross-service auth isolation
User feedback exposed that browser cookie identity ignores port: main and preview both set `codoxear_auth` at `Path=/` on one hostname, so each login overwrote the other's differently signed token. The preview now sets validated name `codoxear_preview_auth`; production retains default `codoxear_auth`. A same-profile two-tab proof shows both `/api/me` endpoints remain 200 after logging into both services. Runtime and browser-auth isolation are now independently supported.

## Boundaries
- Pi with configured `occ` provider is accepted. Codex and Claude binaries/config sources are present, but end-to-end inference is not yet accepted; host Codex ChatGPT OAuth is stale and host-only config paths are deliberately not active.
- The writable workspace contains sensitive documents. Anyone authenticated to the preview can direct agents or file views at that workspace; tailnet membership plus the password are the access boundary.
- `restart` restarts the container and therefore terminates active backend processes/tmux. Backend logs and Codoxear state persist, but active turns should be allowed to finish before restart.
- The source checkout is a read-only bind, not an immutable image copy. Product fixes become visible after a preview restart; the image tag names the initial product checkpoint rather than cryptographically pinning later task iterations.

## Current claim
The preview is ready for ongoing feedback with both runtime and browser-auth isolation supported. Remote access, simultaneous main/preview login, real mounted-project Pi work, protocol-v2 send/response, files/Monaco, and mobile layout are verified directly.

## Feedback protocol
Each reported issue becomes: user observation -> direct reproduction in this preview -> causal mechanism -> predicted fix -> isolated implementation/validation -> preview restart when active turns permit -> browser/user confirmation. OPS.md remains the append-only evidence trail; this file is rewritten as the model changes.

## Highest-value next question
What is the first mismatch the user observes between intended phone/laptop workflow and the running preview? That observation, not further speculative surface mining, determines the next intervention.
