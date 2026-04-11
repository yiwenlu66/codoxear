# Pi Terminal RPC Broker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make newly launched terminal-owned Pi sessions run on the same RPC broker control plane as web-owned Pi sessions so browser `ask_user` replies close the real terminal prompt through `ui_response` instead of `/send` fallback.

**Architecture:** Promote `codoxear/pi_broker.py` into the single live Pi runtime owner, add a foreground terminal attachment mode there, route new terminal Pi launches through that broker, and make `server.py` capability-aware so real Pi RPC sessions never silently degrade `ui_response` into `send`. Keep legacy Pi PTY sessions readable but non-live.

**Tech Stack:** Python 3, Pi RPC JSONL transport, Unix domain sockets, existing Codoxear server/session metadata, pytest, Vitest for existing ask-user UI guards.

---

## File Structure

### Existing files to modify

- `codoxear/pi_broker.py` - add foreground terminal mode, capability metadata, and terminal input/output bridging while preserving the current RPC-backed broker commands.
- `codoxear/broker.py` - stop owning Pi PTY semantics for new Pi sessions; delegate Pi launches to foreground `pi_broker`.
- `codoxear/server.py` - gate Pi `ui_response` fallback on explicit legacy capability, not generic `unknown cmd`.
- `tests/test_pi_broker.py` - add failing tests for foreground mode, metadata, and live UI behavior.
- `tests/test_pi_server_backend.py` - add failing tests for capability-aware `ui_response` handling.
- `web/src/components/conversation/AskUserCard.test.tsx` - keep the existing historical-only guard green after backend capability changes.

### New files planned

- None. Keep this implementation inside the existing Pi broker, generic broker, server, and test modules.

---

### Task 1: Lock in capability-aware backend behavior with failing tests

**Files:**
- Modify: `tests/test_pi_broker.py`
- Modify: `tests/test_pi_server_backend.py`
- Test: `tests/test_pi_rpc.py`

- [ ] **Step 1: Add a failing broker metadata test for live Pi capability**

```python
class TestPiBroker(unittest.TestCase):
    def test_write_meta_marks_rpc_transport_and_live_ui_support(self) -> None:
        rpc = _FakeRpc()
        with tempfile.TemporaryDirectory() as td:
            sock_path = Path(td) / "pi.sock"
            broker = PiBroker(cwd="/tmp")
            broker.state = PiBrokerState(
                session_id="pi-session-001",
                codex_pid=123,
                sock_path=sock_path,
                session_path=Path(td) / "pi-session.jsonl",
                start_ts=0.0,
                rpc=rpc,
            )

            broker._write_meta()

            meta = json.loads(sock_path.with_suffix(".json").read_text(encoding="utf-8"))

        self.assertEqual(meta["transport"], "pi-rpc")
        self.assertTrue(meta["supports_live_ui"])
        self.assertEqual(meta["ui_protocol_version"], 1)
```

- [ ] **Step 2: Add a failing broker test for foreground terminal mode bookkeeping**

```python
class TestPiBroker(unittest.TestCase):
    def test_run_foreground_mode_writes_meta_without_pty_wrapper(self) -> None:
        rpc = _FakeRpc()
        broker = PiBroker(cwd="/tmp", rpc=rpc)

        with tempfile.TemporaryDirectory() as td, \
             patch("codoxear.pi_broker.SOCK_DIR", Path(td)), \
             patch("codoxear.pi_broker.PI_SESSION_DIR", Path(td)), \
             patch.object(PiBroker, "_sock_server", side_effect=lambda: None):
            exit_code = broker.run(foreground=False)

        self.assertEqual(exit_code, 0)
        self.assertIsNotNone(broker.state)
        self.assertEqual(broker.state.backend, "pi")
```

- [ ] **Step 3: Add a failing server test proving live-capable Pi sessions do not fall back to `/send`**

```python
def test_ui_response_route_does_not_fallback_to_send_for_live_pi_rpc_session(self) -> None:
    mgr = _make_manager()
    body = json.dumps({"id": "ui-req-1", "value": "Details"}).encode("utf-8")
    forwarded: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory() as td:
        sock = Path(td) / "pi.sock"
        sock.touch()
        mgr._sessions["pi-session"] = Session(
            session_id="pi-session",
            thread_id="pi-thread-001",
            agent_backend="pi",
            backend="pi",
            broker_pid=3333,
            codex_pid=4444,
            owned=True,
            start_ts=123.0,
            cwd="/tmp",
            log_path=None,
            sock_path=sock,
            session_path=Path("/tmp/pi-session.jsonl"),
            transport="pi-rpc",
        )

        def _sock_call(_sock: Path, req: dict[str, object], timeout_s: float = 0.0) -> dict[str, object]:
            forwarded.append(dict(req))
            if req["cmd"] == "ui_response":
                return {"ok": True}
            if req["cmd"] == "send":
                raise AssertionError("live pi-rpc session must not use send fallback")
            return {}

        mgr._sock_call = _sock_call  # type: ignore[method-assign]
        handler = _HandlerHarness("/api/sessions/pi-session/ui_response", body=body)
        with patch("codoxear.server._require_auth", return_value=True), patch("codoxear.server.MANAGER", mgr):
            Handler.do_POST(handler)  # type: ignore[arg-type]

    payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert handler.status == 200
    assert payload == {"ok": True}
    assert forwarded == [{"cmd": "ui_response", "id": "ui-req-1", "value": "Details"}]
```

- [ ] **Step 4: Add a failing legacy-session test that keeps fallback only for non-live Pi sessions**

```python
def test_ui_response_route_legacy_pi_session_still_uses_send_fallback(self) -> None:
    mgr = _make_manager()
    body = json.dumps({"id": "ui-req-1", "value": "Details"}).encode("utf-8")
    forwarded: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory() as td:
        sock = Path(td) / "pi.sock"
        sock.touch()
        mgr._sessions["pi-session"] = Session(
            session_id="pi-session",
            thread_id="pi-thread-001",
            agent_backend="pi",
            backend="pi",
            broker_pid=3333,
            codex_pid=4444,
            owned=True,
            start_ts=123.0,
            cwd="/tmp",
            log_path=None,
            sock_path=sock,
            session_path=Path("/tmp/pi-session.jsonl"),
            transport="pty",
        )

        def _sock_call(_sock: Path, req: dict[str, object], timeout_s: float = 0.0) -> dict[str, object]:
            forwarded.append(dict(req))
            if req["cmd"] == "ui_response":
                return {"error": "unknown cmd"}
            if req["cmd"] == "send":
                return {"queued": False, "queue_len": 0}
            return {}

        mgr._sock_call = _sock_call  # type: ignore[method-assign]
        handler = _HandlerHarness("/api/sessions/pi-session/ui_response", body=body)
        with patch("codoxear.server._require_auth", return_value=True), patch("codoxear.server.MANAGER", mgr):
            Handler.do_POST(handler)  # type: ignore[arg-type]

    payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert handler.status == 200
    assert payload == {"ok": True}
    assert forwarded == [
        {"cmd": "ui_response", "id": "ui-req-1", "value": "Details"},
        {"cmd": "send", "text": "Details"},
    ]
```

- [ ] **Step 5: Run the targeted backend tests and verify failure**

Run: `python3 -m pytest tests/test_pi_broker.py tests/test_pi_server_backend.py -q`
Expected: FAIL with missing `transport/supports_live_ui` metadata and current server fallback behavior still too permissive.

- [ ] **Step 6: Commit the red tests**

```bash
git add tests/test_pi_broker.py tests/test_pi_server_backend.py
git commit -m "test: lock pi rpc broker capability behavior"
```

---

### Task 2: Add foreground terminal mode to `pi_broker`

**Files:**
- Modify: `codoxear/pi_broker.py`
- Test: `tests/test_pi_broker.py`

- [ ] **Step 1: Add a failing test for line-oriented terminal prompt submission**

```python
class TestPiBroker(unittest.TestCase):
    def test_handle_terminal_line_submits_prompt_via_rpc(self) -> None:
        rpc = _FakeRpc()
        broker = PiBroker(cwd="/tmp", rpc=rpc)
        broker.state = PiBrokerState(
            session_id="pi-session-001",
            codex_pid=123,
            sock_path=Path("/tmp/pi.sock"),
            session_path=Path("/tmp/pi-session.jsonl"),
            start_ts=0.0,
            rpc=rpc,
        )

        broker._submit_terminal_prompt("hello from tty")

        self.assertIn(("prompt", "hello from tty"), rpc.calls)
        self.assertTrue(broker.state.busy)
```

- [ ] **Step 2: Add a failing test for terminal interrupt mapping**

```python
class TestPiBroker(unittest.TestCase):
    def test_handle_terminal_interrupt_maps_to_rpc_abort(self) -> None:
        rpc = _FakeRpc()
        broker = PiBroker(cwd="/tmp", rpc=rpc)
        broker.state = PiBrokerState(
            session_id="pi-session-001",
            codex_pid=123,
            sock_path=Path("/tmp/pi.sock"),
            session_path=Path("/tmp/pi-session.jsonl"),
            start_ts=0.0,
            rpc=rpc,
            last_turn_id="turn-001",
            busy=True,
        )

        broker._interrupt_terminal_turn()

        self.assertIn(("abort", "turn-001"), rpc.calls)
        self.assertFalse(broker.state.busy)
```

- [ ] **Step 3: Run the focused broker tests and verify failure**

Run: `python3 -m pytest tests/test_pi_broker.py -k "transport or terminal" -q`
Expected: FAIL because `PiBroker` has no foreground terminal helpers yet.

- [ ] **Step 4: Implement the foreground terminal helpers in `codoxear/pi_broker.py`**

```python
class PiBroker:
    def _submit_terminal_prompt(self, text: str) -> dict[str, Any]:
        st = self._get_state_snapshot()
        if not st:
            raise RuntimeError("no state")
        result = st.rpc.prompt(text)
        with self._lock:
            if self.state is st:
                st.last_turn_id = _extract_turn_id(result) or st.last_turn_id
                st.busy = True
                st.prompt_sent_at = time.monotonic()
        return result

    def _interrupt_terminal_turn(self) -> dict[str, Any]:
        st = self._get_state_snapshot()
        if not st:
            raise RuntimeError("no state")
        result = st.rpc.abort(st.last_turn_id)
        with self._lock:
            if self.state is st:
                st.busy = False
        return result
```

- [ ] **Step 5: Implement a minimal foreground run path without PTY ownership**

```python
def run(self, *, foreground: bool = True) -> int:
    SOCK_DIR.mkdir(parents=True, exist_ok=True)
    PI_SESSION_DIR.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    session_path = self.session_path or (PI_SESSION_DIR / f"{token}.jsonl")
    sock_path = SOCK_DIR / f"{token}.sock"
    rpc = self.rpc or PiRpcClient(cwd=self.cwd, session_path=session_path)
    self.state = State(
        session_id=None,
        codex_pid=rpc.pid or os.getpid(),
        sock_path=sock_path,
        session_path=session_path,
        start_ts=time.time(),
        rpc=rpc,
    )
    self._write_meta()
    threading.Thread(target=self._bg_sync_loop, name="pi-bg-sync", daemon=True).start()
    if foreground and sys.stdin.isatty() and sys.stdout.isatty():
        threading.Thread(target=self._stdin_loop, name="pi-stdin", daemon=True).start()
        threading.Thread(target=self._stdout_loop, name="pi-stdout", daemon=True).start()
    self._sock_server()
    return 0
```

- [ ] **Step 6: Re-run the broker tests until green**

Run: `python3 -m pytest tests/test_pi_broker.py -q`
Expected: PASS

- [ ] **Step 7: Commit the broker runtime changes**

```bash
git add codoxear/pi_broker.py tests/test_pi_broker.py
git commit -m "feat: add foreground pi rpc broker mode"
```

---

### Task 3: Route new terminal Pi launches through `pi_broker`

**Files:**
- Modify: `codoxear/broker.py`
- Test: `tests/test_pi_server_backend.py`
- Test: `tests/test_pi_broker.py`

- [ ] **Step 1: Add a failing launch-path test for Pi delegation**

```python
class TestPiLaunchPath(unittest.TestCase):
    def test_pi_backend_launches_pi_broker_instead_of_pty_agent(self) -> None:
        broker = Broker(cwd="/tmp", codex_args=[])
        with patch("codoxear.broker.AGENT_BACKEND", "pi"), \
             patch.object(broker, "_run_pi_foreground_broker", return_value=0) as run_pi:
            exit_code = broker.run()

        self.assertEqual(exit_code, 0)
        run_pi.assert_called_once()
```

- [ ] **Step 2: Run the focused launch-path test and verify failure**

Run: `python3 -m pytest tests/test_pi_broker.py tests/test_pi_server_backend.py -k "launch" -q`
Expected: FAIL because `broker.py` still forks a PTY for Pi sessions.

- [ ] **Step 3: Implement a Pi delegation branch in `codoxear/broker.py`**

```python
def run(self) -> int:
    if AGENT_BACKEND == "pi":
        return self._run_pi_foreground_broker()
    # existing Codex PTY path remains unchanged
```

- [ ] **Step 4: Add the delegation helper with explicit session file handling**

```python
def _run_pi_foreground_broker(self) -> int:
    session_path = _session_log_path_from_args(
        args=self.codex_args,
        agent_backend="pi",
        sessions_dir=self.sessions_dir,
    )
    return PiBroker(cwd=self.cwd, session_path=session_path).run(foreground=True)
```

- [ ] **Step 5: Re-run the focused tests until green**

Run: `python3 -m pytest tests/test_pi_broker.py tests/test_pi_server_backend.py -k "launch or foreground" -q`
Expected: PASS

- [ ] **Step 6: Commit the Pi launch-path change**

```bash
git add codoxear/broker.py tests/test_pi_broker.py tests/test_pi_server_backend.py
git commit -m "refactor: route pi terminal sessions through pi broker"
```

---

### Task 4: Make `server.py` capability-aware for live Pi UI

**Files:**
- Modify: `codoxear/server.py`
- Modify: `tests/test_pi_server_backend.py`

- [ ] **Step 1: Add a failing unit test for capability detection**

```python
def test_get_ui_state_respects_live_pi_transport_metadata(self) -> None:
    mgr = _make_manager()
    session = Session(
        session_id="pi-session",
        thread_id="pi-thread-001",
        agent_backend="pi",
        backend="pi",
        broker_pid=3333,
        codex_pid=4444,
        owned=True,
        start_ts=123.0,
        cwd="/tmp",
        log_path=None,
        sock_path=Path("/tmp/pi.sock"),
        session_path=Path("/tmp/pi-session.jsonl"),
        transport="pi-rpc",
    )
```

- [ ] **Step 2: Run the server backend tests and verify failure**

Run: `python3 -m pytest tests/test_pi_server_backend.py -k "ui_response or ui_state" -q`
Expected: FAIL because server logic still keys fallback only on `unknown cmd`.

- [ ] **Step 3: Add explicit helper methods in `codoxear/server.py` for Pi live capability**

```python
def _session_supports_live_pi_ui(session: Session) -> bool:
    if session.backend != "pi":
        return False
    transport = (session.transport or "").strip().lower()
    return transport == "pi-rpc"
```

- [ ] **Step 4: Use capability checks inside `get_ui_state()` and `submit_ui_response()`**

```python
if resp.get("error") == "unknown cmd":
    if _session_supports_live_pi_ui(s):
        raise ValueError("live ui interactions are unavailable for this pi session")
    return {"requests": []}

if resp.get("error") == "unknown cmd":
    if _session_supports_live_pi_ui(s):
        raise ValueError("live ui responses are unavailable for this pi session")
    # legacy fallback remains here only
```

- [ ] **Step 5: Re-run the server tests until green**

Run: `python3 -m pytest tests/test_pi_server_backend.py -q`
Expected: PASS

- [ ] **Step 6: Commit the server capability gate**

```bash
git add codoxear/server.py tests/test_pi_server_backend.py
git commit -m "fix: gate pi ui fallback on legacy transport"
```

---

### Task 5: Reconfirm web ask-user behavior and ship safely

**Files:**
- Test: `web/src/components/conversation/AskUserCard.test.tsx`
- Test: `tests/test_pi_broker.py`
- Test: `tests/test_pi_server_backend.py`

- [ ] **Step 1: Run the existing frontend ask-user guard tests**

Run: `cd web && npm test -- --run src/components/conversation/AskUserCard.test.tsx`
Expected: PASS

- [ ] **Step 2: Run the full targeted backend suite for Pi broker behavior**

Run: `python3 -m pytest tests/test_pi_rpc.py tests/test_pi_broker.py tests/test_pi_server_backend.py -q`
Expected: PASS

- [ ] **Step 3: Do a manual smoke check with a new terminal Pi session**

Run:

```bash
CODEX_WEB_AGENT_BACKEND=pi codoxear-broker --
```

Expected:
- terminal Pi session starts through the RPC broker path
- sidecar metadata shows `transport: "pi-rpc"`
- a browser-originated `ask_user` reply resolves the prompt without sending plain chat text

- [ ] **Step 4: Commit the verification checkpoint**

```bash
git add codoxear/pi_broker.py codoxear/broker.py codoxear/server.py \
  tests/test_pi_broker.py tests/test_pi_server_backend.py
 git commit -m "test: verify pi terminal rpc broker integration"
```

---

## Plan Self-Review

### Spec coverage

- Single Pi runtime owner: covered in Tasks 2 and 3
- Capability-aware server behavior: covered in Task 4
- No silent live-session fallback: covered in Task 4
- Terminal-owned Pi broker path: covered in Tasks 2 and 3
- Regression tests: covered in Tasks 1, 4, and 5

### Placeholder scan

- No `TODO`/`TBD` placeholders remain.
- Each task includes exact files, commands, and representative code changes.

### Type and flow consistency

- `transport="pi-rpc"` is the canonical live-session marker across tests and server logic.
- `PiBroker.run(foreground=True|False)` is the proposed entrypoint used consistently across tasks.
- `prompt`, `abort`, `ui_state`, and `ui_response` remain the broker command vocabulary throughout the plan.
