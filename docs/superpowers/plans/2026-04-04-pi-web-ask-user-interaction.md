# Pi Web Ask-User Interaction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a general Pi RPC interaction bridge so the web UI can render full `ask_user` cards and answer pending interactions directly from the browser.

**Architecture:** Split the work into two coordinated paths: durable `ask_user` history comes from Pi session-log normalization in `codoxear/pi_messages.py`, while live pending interaction state flows through a new Pi RPC UI bridge in `codoxear/pi_rpc.py`, `codoxear/pi_broker.py`, and `codoxear/server.py`. The browser merges those sources in `codoxear/static/app.js` so one card can move from pending-interactive to log-backed resolved state without faking answers as normal chat sends.

**Tech Stack:** Python 3.10+, stdlib `json`/`socket`/`unittest`, existing Pi RPC JSONL protocol, vanilla JS/CSS, `python3 -m pytest`

---

## File Map

**Create:**
- `tests/test_pi_ask_user_normalization.py` - focused log-normalization tests for `ask_user` history pairing and fallback behavior
- `tests/test_pi_ask_user_ui_source.py` - focused source-level UI tests for `ask_user` rendering and live interaction polling

**Modify:**
- `tests/pi_fixtures.py` - reusable Pi RPC UI request/response fixtures and `ask_user` log fixtures
- `tests/test_pi_rpc.py` - Pi RPC client tests for retaining UI requests and sending UI responses
- `tests/test_pi_broker.py` - broker tests for pending interaction state, response forwarding, and single-consumption semantics
- `tests/test_pi_server_backend.py` - server/API tests for `/ui_state`, `/ui_response`, and the integration contract with message loading
- `codoxear/pi_rpc.py` - retain `extension_ui_request` events and add `send_ui_response()`
- `codoxear/pi_broker.py` - track pending UI requests and expose broker socket commands
- `codoxear/server.py` - add Pi-only interaction routes and session-manager helpers
- `codoxear/pi_messages.py` - normalize historical `ask_user` tool call/result pairs into first-class events
- `codoxear/static/app.js` - fetch/merge live interaction state and render full interactive `ask_user` cards
- `codoxear/static/app.css` - style `ask_user` cards, option buttons, input states, and status badges

**Verify with:**
- `python3 -m pytest tests/test_pi_rpc.py -q`
- `python3 -m pytest tests/test_pi_broker.py -q`
- `python3 -m pytest tests/test_pi_ask_user_normalization.py -q`
- `python3 -m pytest tests/test_pi_server_backend.py -k "ui_state or ui_response or ask_user" -q`
- `python3 -m pytest tests/test_pi_ask_user_ui_source.py -q`
- `python3 -m pytest tests/test_pi_rpc.py tests/test_pi_broker.py tests/test_pi_ask_user_normalization.py tests/test_pi_server_backend.py tests/test_pi_ask_user_ui_source.py -q`

## Task 1: Extend Pi RPC fixtures and client support for UI requests

**Files:**
- Modify: `tests/pi_fixtures.py`
- Modify: `tests/test_pi_rpc.py`
- Modify: `codoxear/pi_rpc.py`

- [ ] **Step 1: Write the failing RPC fixture and client tests**

```python
# tests/pi_fixtures.py

def pi_ui_request_event() -> dict[str, object]:
    return {
        "type": "extension_ui_request",
        "id": "ui-req-1",
        "method": "select",
        "title": "Pick a location",
        "options": ["Details", "Sidebar"],
        "timeout": 10000,
    }


def pi_ui_response_payload() -> dict[str, object]:
    return {
        "type": "extension_ui_response",
        "id": "ui-req-1",
        "value": "Details",
    }
```

```python
# tests/test_pi_rpc.py
from tests.pi_fixtures import pi_ui_request_event, pi_ui_response_payload


def test_event_reader_preserves_extension_ui_requests(self) -> None:
    proc = _FakeProc()
    client = PiRpcClient(proc=proc)
    try:
        proc.stdout.put_line(json.dumps(pi_ui_request_event()) + "\n")
        deadline = time.time() + 1.0
        seen = []
        while not seen:
            if time.time() >= deadline:
                self.fail("extension_ui_request was not captured")
            seen = client.drain_events()
            time.sleep(0.01)
        self.assertEqual(seen, [pi_ui_request_event()])
    finally:
        client.close()


def test_send_ui_response_writes_raw_jsonl_without_waiting_for_response(self) -> None:
    proc = _FakeProc()
    client = PiRpcClient(proc=proc)
    try:
        client.send_ui_response("ui-req-1", value="Details")
        self.assertEqual(json.loads(proc.stdin.getvalue()), pi_ui_response_payload())
    finally:
        client.close()
```

- [ ] **Step 2: Run the RPC tests to verify they fail**

Run: `python3 -m pytest tests/test_pi_rpc.py -k "extension_ui_request or send_ui_response" -q`
Expected: FAIL because `PiRpcClient` does not yet provide `send_ui_response()` and the fixtures are missing.

- [ ] **Step 3: Implement raw UI-response writing in `codoxear/pi_rpc.py`**

```python
def _write_jsonl(self, payload: dict[str, Any]) -> None:
    stdin = getattr(self._proc, "stdin", None)
    if stdin is None:
        raise RuntimeError("pi rpc stdin is unavailable")
    stdin.write(json.dumps(payload) + "\n")
    stdin.flush()


def send_ui_response(
    self,
    request_id: str,
    *,
    value: str | None = None,
    confirmed: bool | None = None,
    cancelled: bool = False,
) -> None:
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("request_id required")
    body: dict[str, Any] = {"type": "extension_ui_response", "id": request_id}
    if cancelled:
        body["cancelled"] = True
    elif confirmed is not None:
        body["confirmed"] = bool(confirmed)
    else:
        body["value"] = value
    self._write_jsonl(body)
```

- [ ] **Step 4: Re-run the RPC tests to verify they pass**

Run: `python3 -m pytest tests/test_pi_rpc.py -k "extension_ui_request or send_ui_response" -q`
Expected: PASS with the new fixtures and UI-response writer in place.

- [ ] **Step 5: Commit the RPC client changes**

```bash
git add tests/pi_fixtures.py tests/test_pi_rpc.py codoxear/pi_rpc.py
git commit -m "feat(pi): bridge rpc ui responses"
```

## Task 2: Track pending Pi UI requests in the broker

**Files:**
- Modify: `tests/test_pi_broker.py`
- Modify: `codoxear/pi_broker.py`

- [ ] **Step 1: Write the failing broker tests for `ui_state` and `ui_response`**

```python
# tests/test_pi_broker.py
class _FakeRpc:
    def __init__(self) -> None:
        self.ui_responses: list[dict[str, object]] = []
        self.events: list[dict[str, object]] = []

    def send_ui_response(self, request_id: str, **payload: object) -> None:
        self.ui_responses.append({"id": request_id, **payload})


def test_ui_state_returns_pending_requests_after_event_drain(self) -> None:
    rpc = _FakeRpc()
    rpc.events = [
        {
            "type": "extension_ui_request",
            "id": "ui-req-1",
            "method": "select",
            "title": "Pick a location",
            "options": ["Details", "Sidebar"],
        }
    ]
    broker = PiBroker(cwd="/tmp")
    broker.state = PiBrokerState(
        session_id="pi-session-001",
        codex_pid=123,
        sock_path=Path("/tmp/pi.sock"),
        session_path=Path("/tmp/pi-session.jsonl"),
        start_ts=0.0,
        rpc=rpc,
    )
    broker._sync_output_from_rpc()

    server_sock, client_sock = socket.socketpair()
    try:
        thread = threading.Thread(target=broker._handle_conn, args=(server_sock,), daemon=True)
        thread.start()
        client_sock.sendall((json.dumps({"cmd": "ui_state"}) + "\n").encode("utf-8"))
        resp = json.loads(_recv_line(client_sock).decode("utf-8"))
        self.assertEqual(resp["requests"][0]["id"], "ui-req-1")
    finally:
        client_sock.close()


def test_ui_response_forwards_value_once_and_rejects_replay(self) -> None:
    rpc = _FakeRpc()
    broker = PiBroker(cwd="/tmp")
    broker.state = PiBrokerState(
        session_id="pi-session-001",
        codex_pid=123,
        sock_path=Path("/tmp/pi.sock"),
        session_path=Path("/tmp/pi-session.jsonl"),
        start_ts=0.0,
        rpc=rpc,
        pending_ui_requests={
            "ui-req-1": {
                "id": "ui-req-1",
                "method": "select",
                "status": "pending",
                "options": ["Details", "Sidebar"],
            }
        },
    )

    first = _roundtrip_json(broker, {"cmd": "ui_response", "id": "ui-req-1", "value": "Details"})
    second = _roundtrip_json(broker, {"cmd": "ui_response", "id": "ui-req-1", "value": "Sidebar"})

    self.assertEqual(first, {"ok": True})
    self.assertEqual(rpc.ui_responses, [{"id": "ui-req-1", "value": "Details"}])
    self.assertEqual(second["error"], "request already resolved")
```

- [ ] **Step 2: Run the broker tests to verify they fail**

Run: `python3 -m pytest tests/test_pi_broker.py -k "ui_state or ui_response" -q`
Expected: FAIL because `PiBroker` has no pending UI state or broker socket commands for UI requests.

- [ ] **Step 3: Implement pending-request tracking and broker commands in `codoxear/pi_broker.py`**

```python
from dataclasses import dataclass, field


@dataclass
class State:
    ...
    pending_ui_requests: dict[str, dict[str, Any]] = field(default_factory=dict)


def _record_ui_request(st: State, event: dict[str, Any]) -> None:
    request_id = event.get("id")
    if not isinstance(request_id, str) or not request_id:
        return
    st.pending_ui_requests[request_id] = {
        "id": request_id,
        "method": event.get("method"),
        "title": event.get("title"),
        "message": event.get("message"),
        "options": event.get("options") if isinstance(event.get("options"), list) else [],
        "timeout_ms": event.get("timeout"),
        "status": "pending",
    }


if event.get("type") == "extension_ui_request":
    _record_ui_request(st, event)
    continue

if cmd == "ui_state":
    _send_socket_json_line(conn, {"requests": list(st.pending_ui_requests.values())})
    return

if cmd == "ui_response":
    request_id = req.get("id")
    pending = st.pending_ui_requests.get(request_id)
    if pending is None:
        _send_socket_json_line(conn, {"error": "unknown or expired request"})
        return
    if pending.get("status") != "pending":
        _send_socket_json_line(conn, {"error": "request already resolved"})
        return
    pending["status"] = "resolved"
    st.rpc.send_ui_response(
        request_id,
        value=req.get("value"),
        confirmed=req.get("confirmed"),
        cancelled=bool(req.get("cancelled")),
    )
    _send_socket_json_line(conn, {"ok": True})
    return
```

- [ ] **Step 4: Re-run the broker tests to verify they pass**

Run: `python3 -m pytest tests/test_pi_broker.py -k "ui_state or ui_response" -q`
Expected: PASS with pending interaction state visible and single-consumption behavior enforced.

- [ ] **Step 5: Commit the broker changes**

```bash
git add tests/test_pi_broker.py codoxear/pi_broker.py
git commit -m "feat(pi-web): track pending ui requests"
```

## Task 3: Add server routes for live Pi interaction state and responses

**Files:**
- Modify: `tests/test_pi_server_backend.py`
- Modify: `codoxear/server.py`

- [ ] **Step 1: Write the failing server-route tests**

```python
# tests/test_pi_server_backend.py

def test_ui_state_route_returns_pending_requests_for_pi_session(self) -> None:
    mgr = _make_manager()
    mgr._sessions["pi-session"] = Session(
        session_id="pi-session",
        thread_id="pi-thread-001",
        backend="pi",
        broker_pid=3333,
        codex_pid=4444,
        owned=True,
        start_ts=123.0,
        cwd="/tmp",
        log_path=None,
        sock_path=Path("/tmp/pi.sock"),
        session_path=Path("/tmp/pi-session.jsonl"),
    )
    with patch("codoxear.server.MANAGER", mgr), patch.object(
        mgr,
        "_sock_call",
        return_value={"requests": [{"id": "ui-req-1", "method": "select"}]},
    ), patch("codoxear.server._require_auth", return_value=True):
        harness = _HandlerHarness("/api/sessions/pi-session/ui_state")
        Handler.do_GET(harness)

    self.assertEqual(harness.status, 200)
    payload = json.loads(harness.wfile.getvalue().decode("utf-8"))
    self.assertEqual(payload["requests"][0]["id"], "ui-req-1")


def test_ui_response_route_forwards_browser_answer(self) -> None:
    mgr = _make_manager()
    mgr._sessions["pi-session"] = Session(
        session_id="pi-session",
        thread_id="pi-thread-001",
        backend="pi",
        broker_pid=3333,
        codex_pid=4444,
        owned=True,
        start_ts=123.0,
        cwd="/tmp",
        log_path=None,
        sock_path=Path("/tmp/pi.sock"),
        session_path=Path("/tmp/pi-session.jsonl"),
    )
    body = json.dumps({"id": "ui-req-1", "value": "Details"}).encode("utf-8")
    with patch("codoxear.server.MANAGER", mgr), patch.object(
        mgr,
        "_sock_call",
        return_value={"ok": True},
    ), patch("codoxear.server._require_auth", return_value=True):
        harness = _HandlerHarness("/api/sessions/pi-session/ui_response", body=body)
        Handler.do_POST(harness)

    self.assertEqual(harness.status, 200)
    payload = json.loads(harness.wfile.getvalue().decode("utf-8"))
    self.assertTrue(payload["ok"])
```

- [ ] **Step 2: Run the server-route tests to verify they fail**

Run: `python3 -m pytest tests/test_pi_server_backend.py -k "ui_state_route or ui_response_route" -q`
Expected: FAIL because `server.py` does not expose `/ui_state` or `/ui_response`.

- [ ] **Step 3: Implement the SessionManager helpers and HTTP routes in `codoxear/server.py`**

```python
def get_ui_state(self, session_id: str) -> dict[str, Any]:
    with self._lock:
        s = self._sessions.get(session_id)
        if not s:
            raise KeyError("unknown session")
        if s.backend != "pi":
            raise ValueError("ui interactions are only supported for pi sessions")
        sock = s.sock_path
    return self._sock_call(sock, {"cmd": "ui_state"}, timeout_s=1.5)


def submit_ui_response(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    with self._lock:
        s = self._sessions.get(session_id)
        if not s:
            raise KeyError("unknown session")
        if s.backend != "pi":
            raise ValueError("ui interactions are only supported for pi sessions")
        sock = s.sock_path
    return self._sock_call(sock, {"cmd": "ui_response", **payload}, timeout_s=3.0)
```

```python
if path.startswith("/api/sessions/") and path.endswith("/ui_state"):
    if not _require_auth(self):
        self._unauthorized()
        return
    ...
    payload = MANAGER.get_ui_state(session_id)
    _json_response(self, 200, payload)
    return

if path.startswith("/api/sessions/") and path.endswith("/ui_response"):
    if not _require_auth(self):
        self._unauthorized()
        return
    ...
    payload = MANAGER.submit_ui_response(session_id, obj)
    _json_response(self, 200, payload)
    return
```

- [ ] **Step 4: Re-run the server-route tests to verify they pass**

Run: `python3 -m pytest tests/test_pi_server_backend.py -k "ui_state_route or ui_response_route" -q`
Expected: PASS with Pi-only live interaction routes available.

- [ ] **Step 5: Commit the server-route changes**

```bash
git add tests/test_pi_server_backend.py codoxear/server.py
git commit -m "feat(pi-web): expose live ui interaction routes"
```

## Task 4: Normalize historical `ask_user` interactions from Pi logs

**Files:**
- Create: `tests/test_pi_ask_user_normalization.py`
- Modify: `tests/pi_fixtures.py`
- Modify: `codoxear/pi_messages.py`

- [ ] **Step 1: Write the failing `ask_user` normalization tests**

```python
import unittest

from codoxear import pi_messages


class TestPiAskUserNormalization(unittest.TestCase):
    def test_ask_user_tool_call_and_result_emit_single_event(self) -> None:
        entries = [
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "toolCall",
                            "id": "ask-1",
                            "name": "ask_user",
                            "arguments": {
                                "context": "Pick where todo should appear.",
                                "question": "Where should the todo live?",
                                "options": ["Details", "Sidebar"],
                                "allowFreeform": True,
                            },
                        }
                    ],
                },
            },
            {
                "type": "message",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "ask-1",
                    "toolName": "ask_user",
                    "details": {
                        "answer": "Details",
                        "cancelled": False,
                        "wasCustom": False,
                    },
                    "content": [{"type": "text", "text": "User answered: Details"}],
                },
            },
        ]

        events, _meta, _flags, _diag = pi_messages.normalize_pi_entries(entries, include_system=True)

        self.assertEqual(
            events,
            [
                {
                    "type": "ask_user",
                    "tool_call_id": "ask-1",
                    "question": "Where should the todo live?",
                    "context": "Pick where todo should appear.",
                    "options": ["Details", "Sidebar"],
                    "allow_freeform": True,
                    "answer": "Details",
                    "cancelled": False,
                    "was_custom": False,
                    "resolved": True,
                    "ts": 0.0,
                }
            ],
        )

    def test_ask_user_without_result_stays_unresolved(self) -> None:
        entries = [
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "toolCall",
                            "id": "ask-2",
                            "name": "ask_user",
                            "arguments": {"question": "Pick one", "options": ["A", "B"]},
                        }
                    ],
                },
            }
        ]

        events, _meta, _flags, _diag = pi_messages.normalize_pi_entries(entries, include_system=True)
        self.assertEqual(events[0]["type"], "ask_user")
        self.assertFalse(events[0]["resolved"])
        self.assertIsNone(events[0].get("answer"))
```

- [ ] **Step 2: Run the normalization tests to verify they fail**

Run: `python3 -m pytest tests/test_pi_ask_user_normalization.py -q`
Expected: FAIL because `normalize_pi_entries()` currently emits generic `tool` / `tool_result` rows for `ask_user`.

- [ ] **Step 3: Implement `ask_user` pairing in `codoxear/pi_messages.py`**

```python
def _normalize_ask_user_args(args: Any) -> dict[str, Any]:
    if isinstance(args, str):
        try:
            args = _json.loads(args)
        except (TypeError, ValueError):
            args = {}
    if not isinstance(args, dict):
        args = {}
    return {
        "question": args.get("question") if isinstance(args.get("question"), str) else "",
        "context": args.get("context") if isinstance(args.get("context"), str) else "",
        "options": list(args.get("options")) if isinstance(args.get("options"), list) else [],
        "allow_freeform": bool(args.get("allowFreeform")),
        "allow_multiple": bool(args.get("allowMultiple")),
        "timeout_ms": args.get("timeout") if isinstance(args.get("timeout"), int) else None,
    }
```

```python
pending_interactions: dict[str, dict[str, Any]] = {}

if tc_name == "ask_user" and isinstance(call_id, str) and call_id:
    pending_interactions[call_id] = {
        "type": "ask_user",
        "tool_call_id": call_id,
        **_normalize_ask_user_args(item.get("arguments")),
        "resolved": False,
        "ts": float(_event_ts_value(preferred_ts=assistant_ts, fallback_ts=fallback_ts)),
    }
    fallback_ts += 0.1
    continue

if payload.get("role") == "toolResult" and payload.get("toolName") == "ask_user":
    call_id = payload.get("toolCallId")
    event = pending_interactions.pop(call_id, None) if isinstance(call_id, str) else None
    details = _tool_result_details(payload) or {}
    if event is None:
        event = {"type": "ask_user", "tool_call_id": call_id, "resolved": False, "ts": ts}
    event["answer"] = details.get("answer") if isinstance(details.get("answer"), str) else None
    event["cancelled"] = bool(details.get("cancelled"))
    event["was_custom"] = bool(details.get("wasCustom"))
    event["resolved"] = True
    events.append(event)
    fallback_ts = ts + 0.1
    continue
```

- [ ] **Step 4: Re-run the normalization tests to verify they pass**

Run: `python3 -m pytest tests/test_pi_ask_user_normalization.py -q`
Expected: PASS with `ask_user` log history emitted as a dedicated event.

- [ ] **Step 5: Commit the normalization changes**

```bash
git add tests/test_pi_ask_user_normalization.py tests/pi_fixtures.py codoxear/pi_messages.py
git commit -m "feat(pi-web): normalize ask_user history"
```

## Task 5: Render merged historical/live `ask_user` cards in the web UI

**Files:**
- Create: `tests/test_pi_ask_user_ui_source.py`
- Modify: `codoxear/static/app.js`
- Modify: `codoxear/static/app.css`

- [ ] **Step 1: Write the failing source-level UI tests**

```python
import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"
APP_CSS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.css"


class TestPiAskUserUiSource(unittest.TestCase):
    def test_app_js_declares_ui_state_fetch_and_ask_user_renderer(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("async function fetchSessionUiState(sid)", source)
        self.assertIn("function makeAskUserRow(ev)", source)
        self.assertIn('if (ev.type === "ask_user") return makeAskUserRow(ev).row;', source)

    def test_app_js_submits_ui_responses_without_using_send(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("/ui_response", source)
        self.assertIn("submitSessionUiResponse(", source)
        self.assertIn("resolved-awaiting-log", source)

    def test_app_css_contains_ask_user_card_styles(self) -> None:
        source = APP_CSS.read_text(encoding="utf-8")
        self.assertIn(".askUserCard", source)
        self.assertIn(".askUserOption", source)
        self.assertIn(".askUserAnswer", source)
```

- [ ] **Step 2: Run the UI source tests to verify they fail**

Run: `python3 -m pytest tests/test_pi_ask_user_ui_source.py -q`
Expected: FAIL because the browser code does not yet fetch `ui_state` or render dedicated `ask_user` cards.

- [ ] **Step 3: Implement the minimal UI fetch/merge/render flow in `codoxear/static/app.js` and `codoxear/static/app.css`**

```javascript
async function fetchSessionUiState(sid) {
  return api(`/api/sessions/${sid}/ui_state`);
}

async function submitSessionUiResponse(sid, payload) {
  return api(`/api/sessions/${sid}/ui_response`, { method: "POST", body: payload });
}

function makeAskUserRow(ev) {
  const row = el("div", { class: `msg-row ask-user-row ${ev.status || "resolved"}` });
  const card = el("div", { class: "askUserCard" });
  if (ev.context) card.appendChild(el("div", { class: "askUserContext", text: ev.context }));
  card.appendChild(el("div", { class: "askUserQuestion", text: ev.question || "Choose an answer" }));
  const options = el("div", { class: "askUserOptions" });
  for (const option of Array.isArray(ev.options) ? ev.options : []) {
    const label = typeof option === "string" ? option : option && typeof option.title === "string" ? option.title : "";
    const btn = el("button", { class: "askUserOption", type: "button", text: label });
    btn.disabled = ev.status === "submitting" || ev.status === "resolved-awaiting-log";
    btn.onclick = () => handleAskUserOption(ev, label);
    options.appendChild(btn);
  }
  card.appendChild(options);
  row.appendChild(card);
  return { row };
}
```

```css
.askUserCard {
  border: 1px solid var(--border);
  border-radius: 16px;
  background: linear-gradient(180deg, #fffdf6 0%, #ffffff 100%);
  padding: 12px;
}

.askUserOptions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.askUserOption {
  border: 1px solid rgba(15, 23, 42, 0.16);
  border-radius: 999px;
  background: #fff;
  padding: 8px 12px;
}

.askUserAnswer {
  margin-top: 10px;
  font-weight: 600;
}
```

- [ ] **Step 4: Re-run the UI source tests to verify they pass**

Run: `python3 -m pytest tests/test_pi_ask_user_ui_source.py -q`
Expected: PASS with dedicated `ask_user` card rendering and live-response submission hooks present.

- [ ] **Step 5: Commit the UI changes**

```bash
git add tests/test_pi_ask_user_ui_source.py codoxear/static/app.js codoxear/static/app.css
git commit -m "feat(pi-web): render interactive ask_user cards"
```

## Task 6: Run the bridge end-to-end regression suite

**Files:**
- Modify: `tests/test_pi_server_backend.py`
- Test: `tests/test_pi_rpc.py`
- Test: `tests/test_pi_broker.py`
- Test: `tests/test_pi_ask_user_normalization.py`
- Test: `tests/test_pi_ask_user_ui_source.py`

- [ ] **Step 1: Add one end-to-end regression in `tests/test_pi_server_backend.py` that exercises the full contract**

```python
def test_pi_session_messages_and_ui_state_can_coexist_for_ask_user(self) -> None:
    mgr = _make_manager()
    with tempfile.TemporaryDirectory() as td:
        session_path = Path(td) / "pi-session.jsonl"
        _write_jsonl(
            session_path,
            [
                {"type": "session", "id": "pi-session-001"},
                {
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "toolCall",
                                "id": "ask-1",
                                "name": "ask_user",
                                "arguments": {"question": "Where?", "options": ["Details", "Sidebar"]},
                            }
                        ],
                    },
                },
            ],
        )
        mgr._sessions["pi-session"] = Session(
            session_id="pi-session",
            thread_id="pi-thread-001",
            backend="pi",
            broker_pid=3333,
            codex_pid=4444,
            owned=True,
            start_ts=123.0,
            cwd=td,
            log_path=None,
            sock_path=Path(td) / "pi.sock",
            session_path=session_path,
        )
        with patch.object(
            mgr,
            "_sock_call",
            return_value={
                "busy": False,
                "queue_len": 0,
                "token": None,
                "requests": [{"id": "ui-req-1", "method": "select", "tool_call_id": "ask-1"}],
            },
        ):
            payload = mgr.get_messages_page("pi-session", offset=0, init=True, limit=20)
        self.assertEqual(payload["events"][0]["type"], "ask_user")
```

- [ ] **Step 2: Run the focused regression suite**

Run: `python3 -m pytest tests/test_pi_rpc.py tests/test_pi_broker.py tests/test_pi_ask_user_normalization.py tests/test_pi_server_backend.py tests/test_pi_ask_user_ui_source.py -q`
Expected: PASS with no regressions across the RPC bridge, broker, server routes, log normalization, or browser render pipeline.

- [ ] **Step 3: Run a broader Pi regression command before shipping**

Run: `python3 -m pytest tests/test_pi_rpc.py tests/test_pi_broker.py tests/test_pi_server_backend.py tests/test_pi_chat_ui_source.py tests/test_pi_continue_ui.py tests/test_pi_details_todo_ui.py -q`
Expected: PASS so the new interaction bridge does not break existing Pi web behavior.

- [ ] **Step 4: Commit the regression pass updates**

```bash
git add tests/test_pi_server_backend.py
git commit -m "test(pi-web): cover ask_user interaction bridge"
```

## Self-Review Notes

- Spec coverage is complete: live bridge, historical normalization, merged UI rendering, stale/error handling, and verification all map to explicit tasks.
- Placeholder scan fixed the only risky area by using a concrete `Session(...)` setup in the server tests instead of an ellipsis.
- Type consistency is maintained across the plan: `tool_call_id`, `send_ui_response()`, `ui_state`, and `ui_response` use the same names in every task.
