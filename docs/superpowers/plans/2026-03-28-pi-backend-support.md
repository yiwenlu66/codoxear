# Pi Backend Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add initial `pi-coding-agent` support to Codoxear by introducing a backend adapter path that preserves existing Codex behavior and enables web-owned Pi sessions through `pi --mode rpc`.

**Architecture:** Keep the current Codex path intact and add a parallel Pi path behind the same server and browser API. Session discovery stays socket-sidecar based, while Pi session control and message streaming are handled by a new RPC-driven broker that normalizes Pi events into the existing `user`/`assistant` chat feed expected by the UI.

**Tech Stack:** Python 3.10+, stdlib JSONL/socket/subprocess/threading, existing Codoxear HTTP server + browser UI, `pi --mode rpc`, `unittest`/`pytest`

---

## Scope

This plan covers the MVP only:
- Add **web-owned Pi sessions** from the browser
- Preserve all existing Codex flows
- Keep the current HTTP API shape for the browser
- Add backend-aware diagnostics and session listing

Out of scope for this implementation:
- Attaching to an already-running interactive Pi TUI session
- A dedicated terminal wrapper like `codoxear-pi-broker`
- Full parity for every Codex-specific model/provider control in the Pi path

## File Map

**Create:**
- `codoxear/pi_broker.py` - Pi RPC subprocess manager + Unix socket control server
- `codoxear/pi_rpc.py` - JSONL RPC client for `pi --mode rpc`
- `codoxear/pi_messages.py` - Pi event/session-file normalization into browser chat events
- `tests/test_pi_fixtures.py` - stable Pi RPC/session fixtures used by transport and message tests
- `tests/test_pi_rpc.py` - RPC framing/state unit tests
- `tests/test_pi_broker.py` - broker command contract tests for Pi
- `tests/test_pi_server_backend.py` - server/session-list/messages/interrupt/delete integration tests for Pi backend

**Modify:**
- `codoxear/server.py` - backend-aware spawn/list/messages/diagnostics plumbing
- `codoxear/broker.py` - write `backend="codex"` into broker sidecar metadata
- `codoxear/sessiond.py` - write `backend="codex"` into sessiond sidecar metadata
- `codoxear/static/app.js` - backend selector for new sessions + backend-aware defaults/diagnostics
- `README.md` - document Pi MVP support and limitations

**Verify with:**
- `python3 -m pytest tests/test_pi_fixtures.py tests/test_pi_rpc.py -q`
- `python3 -m pytest tests/test_pi_broker.py -q`
- `python3 -m pytest tests/test_pi_server_backend.py -q`
- `python3 -m pytest tests/test_launch_defaults.py tests/test_send_ack.py tests/test_server_chat_flags.py tests/test_sessiond_fail_closed.py -q`
- `python3 -m pytest -q`

---

### Task 0: Lock the Pi contract with fixtures before implementation

**Files:**
- Create: `tests/test_pi_fixtures.py`
- Test: `tests/test_pi_rpc.py`
- Test: `tests/test_pi_server_backend.py`

- [ ] **Step 1: Capture the Pi MVP contract in fixtures instead of guessing it ad hoc**

Fixture coverage:
- RPC response lines for `prompt`, `abort`, and `get_state`
- streamed turn/tool/message events used by the broker
- persisted Pi session entries needed for restart/replay behavior

- [ ] **Step 2: Write failing fixture-based tests that assert the expected Pi command/event shapes**

```python
class TestPiFixtures(unittest.TestCase):
    def test_rpc_fixture_covers_prompt_abort_and_get_state(self) -> None:
        ...

    def test_session_fixture_can_replay_user_and_assistant_history(self) -> None:
        ...
```

- [ ] **Step 3: Run the fixture tests and verify they fail until the fixtures/helpers exist**

Run: `python3 -m pytest tests/test_pi_fixtures.py -q`
Expected: failure because the Pi fixture helpers do not yet exist

- [ ] **Step 4: Add the shared fixtures and make the fixture tests pass**

Run: `python3 -m pytest tests/test_pi_fixtures.py -q`
Expected: PASS

---

### Task 1: Add backend metadata and server routing

**Files:**
- Modify: `codoxear/server.py`
- Modify: `codoxear/broker.py`
- Modify: `codoxear/sessiond.py`
- Test: `tests/test_pi_server_backend.py`

- [ ] **Step 1: Write the failing server tests for backend metadata and spawn routing**

```python
class TestPiBackendRouting(unittest.TestCase):
    def test_discover_existing_reads_backend_from_sidecar(self) -> None:
        ...

    def test_spawn_web_session_dispatches_to_pi_backend(self) -> None:
        ...

    def test_create_session_ignores_codex_only_fields_for_pi_backend(self) -> None:
        ...
```

- [ ] **Step 2: Run the targeted test file and verify it fails for the missing Pi backend path**

Run: `python3 -m pytest tests/test_pi_server_backend.py -q`
Expected: failure mentioning missing backend support or missing Pi routing helpers

- [ ] **Step 3: Add a backend field to sidecar metadata for existing Codex flows**

Implementation notes:
- In `codoxear/broker.py`, include `"backend": "codex"` in `_write_meta()`.
- In `codoxear/sessiond.py`, include `"backend": "codex"` in `_write_meta()`.
- Keep missing `backend` values backward-compatible by treating them as Codex in server discovery.

- [ ] **Step 4: Add backend-aware session discovery and spawn dispatch in `codoxear/server.py`**

Implementation notes:
- Extend the in-memory `Session` model with `backend: str`.
- Parse `backend` from socket sidecars in `_discover_existing()` and `refresh_session_meta()`.
- Add a request-level backend parameter for `POST /api/sessions`; default to `codex`.
- Split `spawn_web_session()` into backend-aware dispatch:
  - Codex -> existing broker path unchanged
  - Pi -> new Pi broker module (Task 2)
- On the Pi path, server-side validation must ignore or reject Codex-only create-session fields instead of blindly forwarding them.

- [ ] **Step 5: Preserve the existing process contract while making diagnostics backend-aware**

Implementation notes:
- Include `backend` in session rows and diagnostics payload.
- Keep `codex_pid` populated for Pi as a compatibility shim in the sidecar and `Session` object so discovery, pruning, delete, and existing diagnostics paths continue to work.
- Optionally add a future-facing `agent_pid`, but only if all server/UI call sites are updated in the same task.
- Keep existing fields (`busy`, `queue_len`, `token`, `provider_choice`, etc.) present where practical.
- For Pi, return `None`/fallback values instead of inventing Codex-only semantics.

- [ ] **Step 6: Re-run the targeted tests until green**

Run: `python3 -m pytest tests/test_pi_server_backend.py -q`
Expected: PASS

---

### Task 2: Build the Pi RPC transport and broker contract

**Files:**
- Create: `codoxear/pi_rpc.py`
- Create: `codoxear/pi_broker.py`
- Test: `tests/test_pi_fixtures.py`
- Test: `tests/test_pi_rpc.py`
- Test: `tests/test_pi_broker.py`

- [ ] **Step 1: Write failing RPC framing tests**

```python
class TestPiRpc(unittest.TestCase):
    def test_send_command_writes_jsonl_and_correlates_response(self) -> None:
        ...

    def test_event_reader_collects_async_events_without_blocking_responses(self) -> None:
        ...
```

- [ ] **Step 2: Write failing broker contract tests**

```python
class TestPiBroker(unittest.TestCase):
    def test_state_returns_busy_queue_and_token(self) -> None:
        ...

    def test_send_maps_to_prompt_command(self) -> None:
        ...

    def test_shutdown_stops_subprocess(self) -> None:
        ...
```

- [ ] **Step 3: Run the targeted Pi transport tests and verify failure**

Run: `python3 -m pytest tests/test_pi_rpc.py tests/test_pi_broker.py -q`
Expected: failure because Pi modules do not yet exist

- [ ] **Step 4: Implement `codoxear/pi_rpc.py` as a small JSONL RPC client**

Implementation notes:
- Start `pi --mode rpc` with stdin/stdout pipes and a session file on disk.
- Write newline-delimited JSON commands.
- Correlate `{"type": "response", "id": ...}` messages to pending requests.
- Stream non-response events into a thread-safe event buffer for the broker.
- Keep the transport minimal; no UI logic in this module.

- [ ] **Step 5: Implement `codoxear/pi_broker.py` with the same Unix socket command surface used by `server.py`**

Implementation notes:
- Support `state`, `send`, `keys`, `tail`, and `shutdown`.
- `send` -> Pi RPC `prompt`
- `keys` on the Pi path should translate the server interrupt flow into Pi RPC `abort` instead of writing ESC bytes.
- Maintain sidecar metadata under `~/.local/share/codoxear/socks/*.json`
- Store `backend="pi"`, `session_id`, `cwd`, `sock_path`, `broker_pid`, compatibility `codex_pid`, and session file path if known.

- [ ] **Step 6: Track busy state and context-ish state inside the Pi broker**

Implementation notes:
- Use `get_state` plus streamed events to maintain `busy`.
- Prefer a conservative token payload; if Pi state does not expose Codex-like context numbers, return `None`.
- Keep `queue_len` compatible with the server’s own queue model.

- [ ] **Step 7: Add explicit interrupt/delete broker tests before greening the task**

```python
class TestPiBroker(unittest.TestCase):
    def test_keys_command_maps_interrupt_to_abort(self) -> None:
        ...

    def test_shutdown_allows_web_owned_session_cleanup(self) -> None:
        ...
```

- [ ] **Step 8: Re-run the targeted Pi transport tests until green**

Run: `python3 -m pytest tests/test_pi_fixtures.py tests/test_pi_rpc.py tests/test_pi_broker.py -q`
Expected: PASS

---

### Task 3: Normalize Pi messages into the existing browser feed

**Files:**
- Create: `codoxear/pi_messages.py`
- Modify: `codoxear/server.py`
- Test: `tests/test_pi_server_backend.py`

- [ ] **Step 1: Write failing normalization tests for Pi events and restart replay**

```python
class TestPiMessageNormalization(unittest.TestCase):
    def test_prompt_and_turn_events_emit_user_and_assistant_rows(self) -> None:
        ...

    def test_tool_events_do_not_break_chat_rendering(self) -> None:
        ...

    def test_session_file_replay_restores_history_after_restart(self) -> None:
        ...
```

- [ ] **Step 2: Run the server/backend test file and verify failure**

Run: `python3 -m pytest tests/test_pi_server_backend.py -q`
Expected: failure because Pi `/messages` output is missing or malformed

- [ ] **Step 3: Implement `codoxear/pi_messages.py` normalization helpers**

Implementation notes:
- Convert Pi session entries and/or streamed RPC events into the browser event shape:

```python
{"role": "user", "text": "...", "ts": 1710000000.0}
{"role": "assistant", "text": "...", "ts": 1710000001.0}
```

- Map Pi `toolResult`, `bashExecution`, and intermediate tool events into metadata only; do not emit new browser roles.
- Preserve enough state to set `turn_start`, `turn_end`, and `turn_aborted`.

- [ ] **Step 4: Add a backend switch in `GET /api/sessions/<id>/messages`**

Implementation notes:
- Codex backend keeps the current rollout parser.
- Pi backend reads normalized events from the Pi broker/session file and returns the same outer JSON shape already consumed by the browser.
- Prefer persisted session-file replay as the source of truth for history; use live RPC events only to fill the gap between polls.
- Keep `has_older`, `next_before`, and `offset` meaningful for Pi, even if the implementation is simpler than Codex’s rolling log index.

- [ ] **Step 5: Verify that Pi diagnostics and polling remain stable**

Implementation notes:
- `busy` should be derived from broker state.
- `token` can be `None` if Pi does not expose an equivalent metric.
- `thread_id` and `log_path` should become backend-neutral values where needed.

- [ ] **Step 6: Re-run the backend tests until green**

Run: `python3 -m pytest tests/test_pi_server_backend.py -q`
Expected: PASS

---

### Task 4: Add a backend selector to the browser UI

**Files:**
- Modify: `codoxear/static/app.js`
- Modify: `codoxear/server.py`
- Test: `tests/test_pi_server_backend.py`

- [ ] **Step 1: Write or extend server tests for new-session backend selection payloads**

```python
def test_new_session_post_accepts_backend_pi() -> None:
    ...

def test_interrupt_route_uses_pi_abort_path() -> None:
    ...

def test_delete_route_cleans_up_web_owned_pi_session() -> None:
    ...
```

- [ ] **Step 2: Run the targeted test and verify failure**

Run: `python3 -m pytest tests/test_pi_server_backend.py -q`
Expected: failure because `backend="pi"` is rejected or ignored

- [ ] **Step 3: Add backend-aware new-session defaults in `codoxear/static/app.js`**

Implementation notes:
- Add a backend picker in the new-session dialog: `Codex` and `Pi`.
- Default to `Codex` for backward compatibility.
- When backend is `Pi`, do not force Codex-specific provider/auth choices into the request body.
- Keep the rest of the session list and message rendering unchanged.

- [ ] **Step 4: Keep diagnostics readable for both backends**

Implementation notes:
- Show backend label in diagnostics.
- For Pi sessions, avoid misleading `Codex PID` wording if the payload now represents a generic agent process.
- Preserve old behavior for Codex sessions.

- [ ] **Step 5: Re-run targeted backend tests**

Run: `python3 -m pytest tests/test_pi_server_backend.py -q`
Expected: PASS

---

### Task 5: Document, regressions, and acceptance verification

**Files:**
- Modify: `README.md`
- Test: `tests/test_launch_defaults.py`
- Test: `tests/test_send_ack.py`
- Test: `tests/test_server_chat_flags.py`
- Test: `tests/test_sessiond_fail_closed.py`
- Test: `tests/test_pi_rpc.py`
- Test: `tests/test_pi_broker.py`
- Test: `tests/test_pi_server_backend.py`

- [ ] **Step 1: Document the Pi MVP in `README.md`**

Implementation notes:
- Explain that Pi support currently covers browser-created sessions.
- Call out that attaching to an already-running interactive Pi TUI session is not part of this change.
- Document the runtime requirement that `pi` is available in `PATH`.

- [ ] **Step 2: Run focused regression tests for existing Codex behavior**

Run: `python3 -m pytest tests/test_launch_defaults.py tests/test_send_ack.py tests/test_server_chat_flags.py tests/test_sessiond_fail_closed.py -q`
Expected: PASS

- [ ] **Step 3: Run focused Pi tests**

Run: `python3 -m pytest tests/test_pi_rpc.py tests/test_pi_broker.py tests/test_pi_server_backend.py -q`
Expected: PASS

- [ ] **Step 4: Run the full test suite**

Run: `python3 -m pytest -q`
Expected: PASS

- [ ] **Step 5: Perform acceptance review against the goal**

Acceptance checklist:
- Browser can create a `Pi` session via `POST /api/sessions`
- Session appears in `/api/sessions` with `backend="pi"`
- Browser polling on `/api/sessions/<id>/messages` returns normalized `user`/`assistant` events
- Pi history survives broker/server restart by replaying the persisted session file
- Interrupt uses Pi abort semantics instead of terminal ESC injection
- Web-owned Pi sessions can be deleted cleanly from the UI/server path
- Existing Codex tests still pass unchanged

- [ ] **Step 6: Record any deferred work explicitly**

Deferred items to note in the final report if not implemented:
- terminal-owned Pi wrapper / session attachment
- richer Pi token/context metrics
- model/provider parity between Codex and Pi create-session flows

---

## Suggested Execution Order

1. Task 0 - lock the Pi contract with fixtures
2. Task 1 - backend metadata and routing
3. Task 2 - Pi RPC transport and broker
4. Task 3 - Pi message normalization
5. Task 4 - browser backend selection
6. Task 5 - docs, regressions, acceptance

## Review Protocol

For each task:
1. Implement with a fresh worker subagent
2. Review spec compliance with a fresh reviewer subagent
3. Review code quality with a fresh reviewer subagent
4. Fix issues before moving to the next task

## Acceptance Bar

Do not declare this project complete until there is fresh evidence for all of the following:
- targeted Pi tests pass
- targeted Codex regression tests pass
- full test suite passes
- README documents the MVP scope and limitations
- session listing, diagnostics, create-session, send, interrupt, and message polling all work for the new Pi backend path
