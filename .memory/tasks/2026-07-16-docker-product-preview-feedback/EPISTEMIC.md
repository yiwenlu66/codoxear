# Epistemic model — Docker product preview and feedback campaign

## Phenomenon
The refactored product is strongly validated in isolated certification runs but cannot yet replace the live service because current brokers and runtime state are legacy. A persistent Docker preview is the shortest path to direct user experience without risking live sessions.

## Current mechanism
A named container with a dedicated persistent HOME can reproduce the complete Codoxear control plane: server, app state, backend brokers, Unix control sockets, logs, tmux sessions, and backend session files. A read-only source mount fixes the product revision; selective read-only config mounts provide real backend authentication; a single read-write project mount provides realistic work. Publishing only the HTTP port crosses the container boundary while leaving host process/runtime state isolated.

## Supported claims
- The recovery branch is a product-scale refactor with prior full-suite, Docker, desktop, and mobile browser evidence.
- Host port 19580 is currently free and distinct from live port 8743.
- `/home/yiwen/codex_ws` is a real 3.0 GiB workspace suitable for file/git/agent testing, but it also contains sensitive documents; access control and network path matter.
- Host Pi/Codex/Claude credentials and settings exist. Their runtime/session directories must remain outside the preview to preserve isolation.

## Live uncertainties
- Whether host CLI installations run correctly in the Debian preview image when mounted rather than freshly installed.
- Which minimal Pi/Codex/Claude config files are sufficient for authenticated real sessions without importing histories.
- Whether the host `.zshrc` initializes cleanly when optional host-only tooling is absent.
- Whether binding host port 19580 is reachable from the user's actual device through Tailscale/LAN and whether browser secure-context-dependent features need a later HTTPS proxy.
- Whether a real backend round trip succeeds with the selectively mounted config and container network.

## Ruled out
- Running the recovery server against host `~/.local/share/codoxear`: it would violate isolation and race the live service.
- Reusing live broker sockets or logs: recovery would reject legacy send capability and the test would contaminate production evidence.
- Mounting entire backend homes: this would expose/couple host session history and generated runtime state rather than emulate configuration.

## Current claim
The preview architecture is feasible and isolates the stateful Codoxear/backend control plane. It is not accepted until a real container browser session proves login, mounted-project launch, confirmed send/response, and container-local socket/log ownership.

## Highest-value next question
Can one selectively configured container launch a protocol-v2 backend session in `/home/tester/codex_ws`, receive a visible response, and leave every resulting socket/log/session artifact below the preview HOME?
