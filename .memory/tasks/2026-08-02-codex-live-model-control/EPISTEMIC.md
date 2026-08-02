# EPISTEMIC

## Phenomenon
A Codex TUI process owns the active session model and reasoning effort. Codoxear can inject text/keys into its shared PTY and can read backend JSONL, but needs an external, deterministic mutation surface for browser-selected model/effort values.

## Accepted mechanism
Codex CLI 0.133.0 introduced the experimental app-server method `thread/settings/update`. With `initialize.params.capabilities.experimentalApi=true`, an external client can send `{threadId, model, effort}`. The update starts no turn and adds no transcript item; it changes the loaded thread's next-turn settings. The app-server emits `thread/settings/updated` with the effective settings to subscribed clients, including the TUI. Upstream release tests prove subsequent provider requests use the updated model; a local 0.146.0 two-client probe independently reproduced model `mock-model-2` and effort `high` in the next Responses request. A second probe launched an actual Codex TUI against a loopback app-server via `--remote`, updated it from another client, then sent a prompt through the TUI PTY; the provider request used the external model and effort.

The normal TUI embeds its app-server client/server in-process and exposes no socket. The deployable Codoxear mechanism is therefore to start an app-server transport owned by the broker, launch the same TUI with `--remote unix://<broker-private-socket>`, and forward typed setting updates through the broker control socket to `thread/settings/update`. This preserves the shared PTY while making the TUI's existing authoritative thread settings externally reachable.

## Config and signals
The native TUI picker persists `model` and `model_reasoning_effort` to `config.toml`, but persistence and live mutation are separate operations: picker handlers first update TUI/app-server thread state and separately write the defaults. App-server `config/batchWrite(reloadUserConfig=true)` explicitly excludes session-static model, reasoning-effort, Plan effort, service tier, and personality from loaded-thread refresh. Upstream core tests confirm runtime refresh retains the original model. No TUI SIGHUP reload handler exists. App-server SIGHUP is a graceful shutdown/restart signal, not config reload.

## CLI and environment
`--model` and `-c model_reasoning_effort=...` are invocation-time config overrides. They can select settings when starting or resuming another Codex process, but cannot mutate an already-running process. Source search found no model environment variable or runtime flag channel.

## Extensions and protocols
The extension API exposes lifecycle/tool integration but no model/effort setter for a running TUI. App-server is the only typed live mutation surface found. It is experimental and requires explicit capability negotiation; Codoxear must probe support and advertise browser commands only when the broker-private app-server supports the method.

## PTY picker assessment
PTY driving is unjustified. `/model` takes no inline arguments and opens a non-searchable `ListSelectionView`. Quick auto models have a hard-coded order, but the complete model list retains server catalog order; selection can require an `All models` hop, then a nested effort picker, and Max/Ultra add another nested picker. Plan mode can add a scope dialog. The broker retains an ANSI byte tail, not an authoritative terminal screen model. Arrow counts, first letters, and row text are therefore neither stable selection identity nor deterministic readback.

## Residual uncertainty
`thread/settings/update` remains experimental, so protocol or CLI launch syntax may change. Runtime capability probing prevents unsupported Codex versions from advertising the web picker. The update acknowledgement means accepted by app-server; immediate effective-setting notification is consumed by the TUI, while Codoxear's sidebar remains log-authoritative and reflects the new model/effort on the next `turn_context`. Until then, the effort picker is scoped to the last logged model. TUI-only provider layers (`--profile`, `--oss`, `--local-provider`, or caller-owned `--remote`) intentionally disable the broker-private transport because a sibling app-server cannot reproduce them honestly.

## Current claim
An honest external mechanism exists for released Codex 0.133+ when the TUI shares a reachable app-server. Broker-owned Unix-socket app-server transport is the smallest mechanism that preserves Codoxear's terminal/browser shared-session invariant.
