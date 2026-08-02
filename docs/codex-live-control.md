# Codex live model and reasoning control

## Recommendation

Run Codex TUI sessions against a broker-owned app-server Unix socket and change the loaded thread with the experimental `thread/settings/update` method. This is the only mechanism found that is typed, targets the same running conversation, and gives the TUI an authoritative settings notification.

Codoxear now:

1. starts `codex app-server --listen unix://<broker-private-socket>`;
2. probes `thread/settings/update` after negotiating `experimentalApi: true`;
3. launches the ordinary shared TUI PTY with `codex --remote unix://<socket> ...` only when the probe succeeds;
4. advertises `/model` and `/effort` to the browser only for capable sessions;
5. forwards the browser selection through the broker control socket to `thread/settings/update`.

The app-server update applies to subsequent turns and creates no user or assistant transcript item. Codoxear keeps the sidebar log-authoritative: the next Codex `turn_context` records the effective model and effort, then the existing log projection updates the row.

This requires Codex CLI 0.133.0 or newer. The method is still experimental, so the broker probes it rather than assuming version compatibility. Unsupported versions continue with the embedded TUI and do not advertise the browser picker. Terminal invocations that supply TUI-only provider layers (`--profile`, `--oss`, `--local-provider`, or their own `--remote`) also retain embedded/external behavior without Codoxear advertising the private-socket control.

## Evidence by question

### 1. Config persistence and reload

The native picker persists `model` and `model_reasoning_effort` to the user config, but persistence is separate from active-thread mutation. In current source, model selection sends TUI `UpdateModel` / `UpdateReasoningEffort` events, synchronizes the active thread with `thread/settings/update`, and separately calls `config/batchWrite` to save defaults:

- [`codex-rs/tui/src/chatwidget/model_popups.rs`](https://github.com/openai/codex/blob/2b5bdcf67547860f2e5c5a605009a70026796b2b/codex-rs/tui/src/chatwidget/model_popups.rs)
- [`codex-rs/tui/src/app/thread_settings.rs`](https://github.com/openai/codex/blob/2b5bdcf67547860f2e5c5a605009a70026796b2b/codex-rs/tui/src/app/thread_settings.rs)
- [`codex-rs/tui/src/app/event_dispatch.rs`](https://github.com/openai/codex/blob/2b5bdcf67547860f2e5c5a605009a70026796b2b/codex-rs/tui/src/app/event_dispatch.rs)

Codex explicitly excludes model and reasoning defaults from hot reload into loaded threads. The app-server documentation states that `config/batchWrite` with `reloadUserConfig: true` reloads runtime config, while model, reasoning effort, Plan effort, service tier, and personality remain session-static. The core test `refresh_runtime_config_updates_runtime_refreshable_fields_and_keeps_session_static_settings` writes a different model into the next config snapshot and asserts that the running session retains its original model:

- [`codex-rs/app-server/README.md`](https://github.com/openai/codex/blob/2b5bdcf67547860f2e5c5a605009a70026796b2b/codex-rs/app-server/README.md)
- [`codex-rs/core/src/session/tests.rs`](https://github.com/openai/codex/blob/2b5bdcf67547860f2e5c5a605009a70026796b2b/codex-rs/core/src/session/tests.rs)

No TUI SIGHUP reload handler exists. The app-server installs SIGHUP as a graceful shutdown signal; repeated SIGHUP waits for active turns and then exits. It is a lifecycle signal, not config reload:

- [`codex-rs/app-server/src/lib.rs`](https://github.com/openai/codex/blob/2b5bdcf67547860f2e5c5a605009a70026796b2b/codex-rs/app-server/src/lib.rs)
- [`connection_handling_websocket_unix.rs`](https://github.com/openai/codex/blob/2b5bdcf67547860f2e5c5a605009a70026796b2b/codex-rs/app-server/tests/suite/v2/connection_handling_websocket_unix.rs)

**Answer:** writing `config.toml` plus SIGHUP is not a live model or effort mechanism.

### 2. Environment and runtime flags

`--model` and `-c model_reasoning_effort=...` are configuration inputs consumed when a Codex process starts. They can be supplied to a new `codex resume` invocation, which creates another TUI process over the stored conversation, but cannot mutate the process that is already running. Source and CLI help expose no model environment variable or attach-to-running-process flag.

Official references:

- [Codex CLI reference](https://developers.openai.com/codex/cli/reference/)
- [Codex configuration reference](https://developers.openai.com/codex/config-reference/)

**Answer:** launch and resume overrides exist; no environment or CLI flag changes an already-running TUI.

### 3. Extension and protocol surfaces

Codex CLI 0.133.0 added experimental `thread/settings/update` in [PR #23502](https://github.com/openai/codex/pull/23502) / [commit `771a4e7`](https://github.com/openai/codex/commit/771a4e74ac319c3d8379c62c7caa3aec1ad53382). Its typed params include `threadId`, `model`, and `effort`. Upstream tests prove that it:

- returns an empty acknowledgement;
- emits `thread/settings/updated` with effective settings;
- starts no model request by itself;
- applies the selected model to future turns;
- accepts updates while a turn is active for the next-turn state.

Sources:

- [`ThreadSettingsUpdateParams`](https://github.com/openai/codex/blob/2b5bdcf67547860f2e5c5a605009a70026796b2b/codex-rs/app-server-protocol/src/protocol/v2/thread.rs)
- [`thread_settings_update.rs`](https://github.com/openai/codex/blob/2b5bdcf67547860f2e5c5a605009a70026796b2b/codex-rs/app-server/tests/suite/v2/thread_settings_update.rs)
- [App-server documentation](https://developers.openai.com/codex/app-server/)

A normal TUI embeds the app server in-process, so it has no external socket. Current Codex also supports a TUI connected to an app-server WebSocket or Unix socket through `--remote`; that makes the same thread reachable by another client. The TUI consumes `thread/settings/updated` and updates its local model/reasoning state.

The extension API contains lifecycle and tool integration but no model or effort setter. No separate TUI control socket was found.

**Answer:** app-server is the honest protocol mechanism, provided the TUI is launched against a reachable app-server transport.

### 4. PTY picker drive

PTY driving cannot provide stable identity selection:

- `/model` does not accept inline arguments.
- The model popup uses a non-searchable `ListSelectionView`.
- Only quick auto models have fixed ordering; the full model list retains catalog order.
- A choice can traverse `All models` → model → effort.
- Max/Ultra efforts add another nested picker.
- Plan mode can add a scope dialog.
- The broker stores an ANSI output tail, not a terminal screen state with row identity and selected-cell semantics.

Sources:

- [`slash_command.rs`](https://github.com/openai/codex/blob/2b5bdcf67547860f2e5c5a605009a70026796b2b/codex-rs/tui/src/slash_command.rs)
- [`model_popups.rs`](https://github.com/openai/codex/blob/2b5bdcf67547860f2e5c5a605009a70026796b2b/codex-rs/tui/src/chatwidget/model_popups.rs)
- [`list_selection_view.rs`](https://github.com/openai/codex/blob/2b5bdcf67547860f2e5c5a605009a70026796b2b/codex-rs/tui/src/bottom_pane/list_selection_view.rs)

**Answer:** arrow counts, catalog order, first-letter keys, and raw PTY text do not form a reliable control protocol.

### 5. Reasoning effort

The conclusions are the same as for model selection:

- `model_reasoning_effort` is persisted as a default but remains session-static under config reload.
- `-c model_reasoning_effort=...` is an invocation-time override.
- the native `/model` flow controls model and effort together, with additional nested branches for advanced levels;
- `thread/settings/update` has a typed `effort` field and applies it to subsequent turns.

## Independent probes

All probes used an unpacked Codex 0.146.0 npm artifact under `/tmp`; nothing was installed globally.

1. **Two-client app-server probe:** client A started a thread; client B sent model `mock-model-2` and effort `high`; client A received the effective settings notification; the next mock Responses request contained `model=mock-model-2` and `reasoning.effort=high`.
2. **Actual remote-TUI probe:** a Codex TUI ran under an isolated PTY against a loopback app-server. A separate client changed the loaded thread to model `mock-model-remote`, effort `high`. A prompt then injected through the TUI PTY produced a provider request with exactly those settings.

The probes cleaned up only their own process groups. Existing Codoxear brokers and backend CLIs were not signaled.

## Residual risk

`thread/settings/update`, remote TUI, and app-server socket transports are experimental. The runtime probe protects capability advertisement, but a future Codex release can still change method semantics after accepting the probe. The broker therefore treats the app-server acknowledgement as delivery acceptance and retains Codex JSONL `turn_context` as the displayed model/effort authority. Until that next turn records the new model, the effort picker remains scoped to the last logged model; app-server validation remains the final guard against an effort unsupported by the newly selected model.
