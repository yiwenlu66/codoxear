# Pi Session Todo Details Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show the latest Pi todo list snapshot inside the existing web `Details` modal for a session.

**Architecture:** Keep todo extraction out of the chat event pipeline. Add a focused Pi session-file reader in `codoxear/pi_messages.py`, expose its output through `/api/sessions/<id>/diagnostics` in `codoxear/server.py`, then render the returned snapshot in the existing diagnostics modal in `codoxear/static/app.js` with a small CSS extension in `codoxear/static/app.css`.

**Tech Stack:** Python 3.10+, stdlib JSON/Path/unittest, existing Codoxear Pi session JSONL format, browser-side vanilla JS/CSS, `python3 -m pytest`, Node-backed UI snippet tests

---

## File Map

**Create:**
- `tests/test_pi_todo_snapshot.py` - focused backend tests for Pi todo snapshot parsing and diagnostics payloads
- `tests/test_pi_details_todo_ui.py` - focused UI tests for rendering the todo section inside the diagnostics modal

**Modify:**
- `codoxear/pi_messages.py` - add bounded tail-scan todo snapshot reader + normalization helper
- `codoxear/server.py` - attach `todo_snapshot` to diagnostics responses without breaking existing fields
- `codoxear/static/app.js` - render a `Todo list` section inside the existing `Details` modal
- `codoxear/static/app.css` - add lightweight diagnostics modal styles for todo summary, items, and status chips

**Verify with:**
- `python3 -m pytest tests/test_pi_todo_snapshot.py -q`
- `python3 -m pytest tests/test_pi_details_todo_ui.py -q`
- `python3 -m pytest tests/test_pi_todo_snapshot.py tests/test_pi_details_todo_ui.py tests/test_pi_server_backend.py tests/test_pi_continue_ui.py -q`

## Task 1: Extract the latest Pi todo snapshot from the session file

**Files:**
- Create: `tests/test_pi_todo_snapshot.py`
- Modify: `codoxear/pi_messages.py`

- [ ] **Step 1: Write the failing parser tests**

```python
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from codoxear import pi_messages


def _write_jsonl(path: Path, entries: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8")


def _todo_result_entry(
    todos: object,
    *,
    call_id: str = "call_todo_1",
    is_error: bool = False,
) -> dict[str, object]:
    return {
        "type": "message",
        "message": {
            "role": "toolResult",
            "toolCallId": call_id,
            "toolName": "manage_todo_list",
            "isError": is_error,
            "details": {
                "operation": "write",
                "todos": todos,
            },
            "content": [
                {
                    "type": "text",
                    "text": "Todos have been modified successfully.",
                }
            ],
        },
    }


class TestPiTodoSnapshot(unittest.TestCase):
    def test_read_latest_pi_todo_snapshot_returns_newest_valid_snapshot(self) -> None:
        with TemporaryDirectory() as td:
            session_path = Path(td) / "pi-session.jsonl"
            _write_jsonl(
                session_path,
                [
                    {"type": "session", "id": "pi-session-001"},
                    _todo_result_entry(
                        [
                            {
                                "id": 1,
                                "title": "Older task",
                                "description": "old snapshot",
                                "status": "completed",
                            }
                        ],
                        call_id="call_old",
                    ),
                    _todo_result_entry(
                        [
                            {
                                "id": 1,
                                "title": "Explore project context",
                                "description": "inspect files first",
                                "status": "completed",
                            },
                            {
                                "id": 2,
                                "title": "Ask clarifying questions",
                                "description": "confirm desired surface",
                                "status": "in-progress",
                            },
                            {
                                "id": 3,
                                "title": "Present design",
                                "description": "share recommended approach",
                                "status": "not-started",
                            },
                        ],
                        call_id="call_new",
                    ),
                ],
            )

            snapshot = pi_messages.read_latest_pi_todo_snapshot(session_path)

        assert snapshot is not None
        self.assertEqual(snapshot["progress_text"], "1/3 completed")
        self.assertEqual(
            snapshot["counts"],
            {
                "total": 3,
                "completed": 1,
                "in_progress": 1,
                "not_started": 1,
            },
        )
        self.assertEqual(snapshot["items"][0]["title"], "Explore project context")
        self.assertEqual(snapshot["items"][1]["status"], "in-progress")

    def test_read_latest_pi_todo_snapshot_skips_newer_malformed_result(self) -> None:
        with TemporaryDirectory() as td:
            session_path = Path(td) / "pi-session.jsonl"
            _write_jsonl(
                session_path,
                [
                    {"type": "session", "id": "pi-session-001"},
                    _todo_result_entry(
                        [
                            {
                                "id": 1,
                                "title": "Stable todo",
                                "description": "valid older entry",
                                "status": "completed",
                            }
                        ],
                        call_id="call_valid",
                    ),
                    _todo_result_entry({"broken": True}, call_id="call_bad"),
                ],
            )

            snapshot = pi_messages.read_latest_pi_todo_snapshot(session_path)

        assert snapshot is not None
        self.assertEqual(snapshot["progress_text"], "1/1 completed")
        self.assertEqual(snapshot["items"][0]["title"], "Stable todo")
```

- [ ] **Step 2: Run the parser tests and verify they fail**

Run: `python3 -m pytest tests/test_pi_todo_snapshot.py -k "read_latest_pi_todo_snapshot" -q`
Expected: FAIL with `AttributeError: module 'codoxear.pi_messages' has no attribute 'read_latest_pi_todo_snapshot'`

- [ ] **Step 3: Implement the bounded tail-scan snapshot reader in `codoxear/pi_messages.py`**

```python
from .rollout_log import _read_jsonl_tail as _read_jsonl_tail

_PI_TODO_SCAN_START_BYTES = 64 * 1024
_PI_TODO_SCAN_MAX_BYTES = 2 * 1024 * 1024
_PI_TODO_STATUS_KEYS = {
    "completed": "completed",
    "in-progress": "in_progress",
    "not-started": "not_started",
}


def _normalize_pi_todo_snapshot(todos: list[dict[str, Any]]) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    counts = {
        "total": 0,
        "completed": 0,
        "in_progress": 0,
        "not_started": 0,
    }
    for todo in todos:
        if not isinstance(todo, dict):
            continue
        title = todo.get("title")
        status = todo.get("status")
        if not isinstance(title, str) or not title.strip():
            continue
        item = {
            "id": todo.get("id"),
            "title": title.strip(),
            "description": todo.get("description") if isinstance(todo.get("description"), str) else "",
            "status": status if isinstance(status, str) and status else "not-started",
        }
        items.append(item)
        counts["total"] += 1
        key = _PI_TODO_STATUS_KEYS.get(item["status"])
        if key is not None:
            counts[key] += 1
    return {
        "items": items,
        "counts": counts,
        "progress_text": str(counts["completed"]) + "/" + str(counts["total"]) + " completed",
    }


def read_latest_pi_todo_snapshot(session_path: Path, *, max_scan_bytes: int = _PI_TODO_SCAN_MAX_BYTES) -> dict[str, Any] | None:
    if not session_path.exists():
        return None
    scan = min(_PI_TODO_SCAN_START_BYTES, int(max_scan_bytes))
    while scan > 0 and scan <= int(max_scan_bytes):
        entries = _read_jsonl_tail(session_path, scan)
        for entry in reversed(entries):
            payload = _payload_for_entry(entry)
            if not isinstance(payload, dict):
                continue
            if payload.get("role") != "toolResult":
                continue
            if payload.get("toolName") != "manage_todo_list":
                continue
            if payload.get("isError") is True:
                continue
            details = payload.get("details")
            if not isinstance(details, dict):
                continue
            todos = details.get("todos")
            if isinstance(todos, list):
                return _normalize_pi_todo_snapshot(todos)
        if scan == int(max_scan_bytes):
            break
        scan = min(scan * 2, int(max_scan_bytes))
    return None
```

- [ ] **Step 4: Run the parser tests and verify they pass**

Run: `python3 -m pytest tests/test_pi_todo_snapshot.py -k "read_latest_pi_todo_snapshot" -q`
Expected: PASS

- [ ] **Step 5: Commit the parser work**

```bash
git add tests/test_pi_todo_snapshot.py codoxear/pi_messages.py
git commit -m "test(pi-todo): lock snapshot parser"
```

## Task 2: Expose the snapshot through diagnostics

**Files:**
- Modify: `tests/test_pi_todo_snapshot.py`
- Modify: `codoxear/server.py`

- [ ] **Step 1: Add failing diagnostics tests**

```python
import io
from unittest.mock import patch

from codoxear.server import Handler
from codoxear.server import Session


class _HandlerHarness:
    def __init__(self, path: str) -> None:
        self.path = path
        self.headers = {"Content-Length": "0"}
        self.rfile = io.BytesIO(b"")
        self.wfile = io.BytesIO()
        self.status: int | None = None
        self.sent_headers: list[tuple[str, str]] = []

    def send_response(self, status: int) -> None:
        self.status = status

    def send_header(self, key: str, value: str) -> None:
        self.sent_headers.append((key, value))

    def end_headers(self) -> None:
        return


class TestPiTodoDiagnostics(unittest.TestCase):
    def test_diagnostics_includes_todo_snapshot_for_pi_session(self) -> None:
        handler = _HandlerHarness("/api/sessions/pi-session/diagnostics")
        session = Session(
            session_id="pi-session",
            thread_id="pi-thread-001",
            backend="pi",
            broker_pid=3333,
            codex_pid=4444,
            owned=True,
            start_ts=123.0,
            cwd="/tmp/pi-cwd",
            log_path=None,
            sock_path=Path("/tmp/pi.sock"),
            session_path=Path("/tmp/pi-session.jsonl"),
        )
        snapshot = {
            "items": [{"id": 1, "title": "Explore project context", "description": "inspect files first", "status": "completed"}],
            "counts": {"total": 1, "completed": 1, "in_progress": 0, "not_started": 0},
            "progress_text": "1/1 completed",
        }

        with patch("codoxear.server._require_auth", return_value=True), \
             patch("codoxear.server.MANAGER") as manager, \
             patch("codoxear.server._current_git_branch", return_value=None), \
             patch("codoxear.server._pi_messages.read_latest_pi_todo_snapshot", return_value=snapshot):
            manager.refresh_session_meta.return_value = None
            manager.get_session.return_value = session
            manager.get_state.return_value = {"busy": False, "queue_len": 0, "token": None}
            manager.sidebar_meta_get.return_value = {
                "priority_offset": 0.0,
                "snooze_until": None,
                "dependency_session_id": None,
            }

            Handler.do_GET(handler)  # type: ignore[arg-type]

        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(handler.status, 200)
        self.assertEqual(payload["todo_snapshot"]["available"], True)
        self.assertEqual(payload["todo_snapshot"]["progress_text"], "1/1 completed")
        self.assertEqual(payload["todo_snapshot"]["items"][0]["title"], "Explore project context")

    def test_diagnostics_marks_todo_unavailable_on_reader_error(self) -> None:
        handler = _HandlerHarness("/api/sessions/pi-session/diagnostics")
        session = Session(
            session_id="pi-session",
            thread_id="pi-thread-001",
            backend="pi",
            broker_pid=3333,
            codex_pid=4444,
            owned=True,
            start_ts=123.0,
            cwd="/tmp/pi-cwd",
            log_path=None,
            sock_path=Path("/tmp/pi.sock"),
            session_path=Path("/tmp/pi-session.jsonl"),
        )

        with patch("codoxear.server._require_auth", return_value=True), \
             patch("codoxear.server.MANAGER") as manager, \
             patch("codoxear.server._current_git_branch", return_value=None), \
             patch("codoxear.server._pi_messages.read_latest_pi_todo_snapshot", side_effect=OSError("boom")):
            manager.refresh_session_meta.return_value = None
            manager.get_session.return_value = session
            manager.get_state.return_value = {"busy": False, "queue_len": 0, "token": None}
            manager.sidebar_meta_get.return_value = {
                "priority_offset": 0.0,
                "snooze_until": None,
                "dependency_session_id": None,
            }

            Handler.do_GET(handler)  # type: ignore[arg-type]

        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(payload["todo_snapshot"], {"available": False, "error": True, "items": []})
```

- [ ] **Step 2: Run the diagnostics tests and verify they fail**

Run: `python3 -m pytest tests/test_pi_todo_snapshot.py -k "diagnostics" -q`
Expected: FAIL because `/diagnostics` does not yet include `todo_snapshot`

- [ ] **Step 3: Implement diagnostics enrichment in `codoxear/server.py`**

```python
def _todo_snapshot_payload_for_session(s: Session) -> dict[str, Any]:
    empty = {"available": False, "error": False, "items": []}
    if s.backend != "pi":
        return empty
    if s.session_path is None or (not s.session_path.exists()):
        return empty
    try:
        snapshot = _pi_messages.read_latest_pi_todo_snapshot(s.session_path)
    except OSError:
        return {"available": False, "error": True, "items": []}
    if snapshot is None:
        return empty
    return {
        "available": True,
        "error": False,
        "items": snapshot["items"],
        "counts": snapshot["counts"],
        "progress_text": snapshot["progress_text"],
    }
```

Add the helper result to the existing diagnostics response body:

```python
"dependency_session_id": sidebar_meta["dependency_session_id"],
"todo_snapshot": _todo_snapshot_payload_for_session(s),
```

- [ ] **Step 4: Run the diagnostics tests and verify they pass**

Run: `python3 -m pytest tests/test_pi_todo_snapshot.py -k "diagnostics" -q`
Expected: PASS

- [ ] **Step 5: Commit the diagnostics work**

```bash
git add tests/test_pi_todo_snapshot.py codoxear/server.py
git commit -m "feat(pi-todo): expose diagnostics snapshot"
```

## Task 3: Render the todo snapshot inside the Details modal

**Files:**
- Create: `tests/test_pi_details_todo_ui.py`
- Modify: `codoxear/static/app.js`

- [ ] **Step 1: Write the failing UI tests**

```python
import json
import subprocess
import textwrap
import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


def eval_render_diag_todo_snapshot(snapshot: dict[str, object]) -> str:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("function renderDiagTodoSnapshot(snapshot) {")
    end = source.index("async function showDiagViewer() {", start)
    snippet = source[start:end]
    injected = json.dumps(snippet + "\nglobalThis.__test_renderDiagTodoSnapshot = renderDiagTodoSnapshot;\n")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        function makeNode(tag, attrs = {{}}, children = []) {{
          const node = {{
            tagName: String(tag || "div").toUpperCase(),
            className: attrs.class || "",
            textContent: attrs.text || "",
            children: [],
            appendChild(child) {{ this.children.push(child); return child; }},
          }};
          for (const child of children) node.appendChild(child);
          return node;
        }}
        function flatten(node) {{
          if (!node) return "";
          const own = node.textContent ? String(node.textContent) : "";
          const kids = Array.isArray(node.children) ? node.children.map(flatten).join(" ") : "";
          return (own + " " + kids).trim();
        }}
        const ctx = {{ el: makeNode }};
        vm.createContext(ctx);
        vm.runInContext({injected}, ctx);
        const root = ctx.__test_renderDiagTodoSnapshot({json.dumps(snapshot)});
        process.stdout.write(JSON.stringify(flatten(root)));
        """
    )
    proc = subprocess.run(
        ["node", "-e", js],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(proc.stdout)


class TestPiDetailsTodoUi(unittest.TestCase):
    def test_render_diag_todo_snapshot_shows_summary_and_items(self) -> None:
        text = eval_render_diag_todo_snapshot(
            {
                "available": True,
                "error": False,
                "progress_text": "2/3 completed",
                "items": [
                    {"id": 1, "title": "Explore project context", "description": "inspect files first", "status": "completed"},
                    {"id": 2, "title": "Ask clarifying questions", "description": "confirm scope", "status": "in-progress"},
                ],
            }
        )

        self.assertIn("Todo list", text)
        self.assertIn("2/3 completed", text)
        self.assertIn("Explore project context", text)
        self.assertIn("Ask clarifying questions", text)
        self.assertIn("completed", text)
        self.assertIn("in-progress", text)

    def test_render_diag_todo_snapshot_distinguishes_empty_and_error_states(self) -> None:
        self.assertIn(
            "No todo list yet",
            eval_render_diag_todo_snapshot({"available": False, "error": False, "items": []}),
        )
        self.assertIn(
            "Todo list unavailable",
            eval_render_diag_todo_snapshot({"available": False, "error": True, "items": []}),
        )

    def test_show_diag_viewer_calls_render_diag_todo_snapshot(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        block_start = source.index("async function showDiagViewer() {")
        block_end = source.index("function hideDiagViewer() {", block_start)
        block = source[block_start:block_end]

        self.assertIn("const todoSection = renderDiagTodoSnapshot", block)
        self.assertIn("diagContent.appendChild(todoSection)", block)
```

- [ ] **Step 2: Run the UI tests and verify they fail**

Run: `python3 -m pytest tests/test_pi_details_todo_ui.py -q`
Expected: FAIL because `renderDiagTodoSnapshot` does not yet exist in `codoxear/static/app.js`

- [ ] **Step 3: Implement the render helper and wire it into `showDiagViewer()`**

```javascript
function renderDiagTodoSnapshot(snapshot) {
  const section = el("div", { class: "detailsSection todoSnapshotSection" });
  section.appendChild(el("div", { class: "detailsSectionTitle", text: "Todo list" }));
  const snap = snapshot && typeof snapshot === "object" ? snapshot : null;
  if (!snap || snap.available !== true) {
    section.appendChild(
      el("div", {
        class: "detailsSectionEmpty",
        text: snap && snap.error ? "Todo list unavailable" : "No todo list yet",
      })
    );
    return section;
  }
  if (snap.progress_text) {
    section.appendChild(el("div", { class: "todoSnapshotSummary", text: String(snap.progress_text) }));
  }
  const list = el("div", { class: "todoSnapshotList" });
  const items = Array.isArray(snap.items) ? snap.items : [];
  for (const item of items) {
    const row = el("div", { class: "todoSnapshotItem" });
    const head = el("div", { class: "todoSnapshotHead" });
    head.appendChild(el("div", { class: "todoSnapshotTitle", text: String(item && item.title ? item.title : "Untitled todo") }));
    head.appendChild(el("span", { class: "todoStatusChip", text: String(item && item.status ? item.status : "not-started") }));
    row.appendChild(head);
    if (item && item.description) {
      row.appendChild(el("div", { class: "todoSnapshotDescription", text: String(item.description) }));
    }
    list.appendChild(row);
  }
  section.appendChild(list);
  return section;
}
```

Add the helper call near the end of `showDiagViewer()` after the existing diagnostics rows:

```javascript
const todoSection = renderDiagTodoSnapshot(d && d.todo_snapshot ? d.todo_snapshot : null);
if (todoSection) diagContent.appendChild(todoSection);
```

- [ ] **Step 4: Run the UI tests and verify they pass**

Run: `python3 -m pytest tests/test_pi_details_todo_ui.py -q`
Expected: PASS

- [ ] **Step 5: Commit the UI rendering work**

```bash
git add tests/test_pi_details_todo_ui.py codoxear/static/app.js
git commit -m "feat(pi-todo): render details snapshot"
```

## Task 4: Style the Details todo section and run final verification

**Files:**
- Modify: `tests/test_pi_details_todo_ui.py`
- Modify: `codoxear/static/app.css`

- [ ] **Step 1: Add the failing CSS coverage test**

Add `APP_CSS` near the existing `APP_JS` constant, then add one more method to the existing `TestPiDetailsTodoUi` class:

```python
APP_CSS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.css"


class TestPiDetailsTodoUi(unittest.TestCase):
    def test_details_todo_css_selectors_exist(self) -> None:
        source = APP_CSS.read_text(encoding="utf-8")
        self.assertIn(".detailsSection", source)
        self.assertIn(".todoSnapshotSection", source)
        self.assertIn(".todoSnapshotItem", source)
        self.assertIn(".todoStatusChip", source)
```

- [ ] **Step 2: Run the CSS test and verify it fails**

Run: `python3 -m pytest tests/test_pi_details_todo_ui.py -k "css" -q`
Expected: FAIL because the new diagnostics todo selectors are not yet defined in `codoxear/static/app.css`

- [ ] **Step 3: Add the diagnostics todo styles in `codoxear/static/app.css`**

```css
.detailsSection {
  border-top: 1px solid rgba(15, 23, 42, 0.06);
  padding: 12px 10px;
}

.detailsSectionTitle {
  font-size: 12px;
  font-weight: 700;
  color: rgba(17, 24, 39, 0.62);
  text-transform: uppercase;
  letter-spacing: 0.04em;
  margin-bottom: 8px;
}

.detailsSectionEmpty,
.todoSnapshotSummary,
.todoSnapshotDescription {
  color: var(--muted);
  font-size: 13px;
  line-height: 1.4;
}

.todoSnapshotList {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.todoSnapshotItem {
  border: 1px solid rgba(15, 23, 42, 0.08);
  border-radius: 10px;
  background: rgba(248, 250, 252, 0.92);
  padding: 10px;
}

.todoSnapshotHead {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 10px;
}

.todoSnapshotTitle {
  font-size: 13px;
  font-weight: 600;
  color: var(--text);
}

.todoStatusChip {
  flex: 0 0 auto;
  border-radius: 999px;
  padding: 2px 8px;
  font-size: 11px;
  line-height: 1.5;
  background: rgba(15, 23, 42, 0.08);
  color: rgba(15, 23, 42, 0.72);
}

.todoSnapshotDescription {
  margin-top: 6px;
}

@media (max-width: 520px) {
  .todoSnapshotHead {
    flex-direction: column;
    align-items: flex-start;
  }
}
```

- [ ] **Step 4: Run the CSS test, then run the full targeted verification set**

Run: `python3 -m pytest tests/test_pi_details_todo_ui.py -k "css" -q`
Expected: PASS

Run: `python3 -m pytest tests/test_pi_todo_snapshot.py tests/test_pi_details_todo_ui.py tests/test_pi_server_backend.py tests/test_pi_continue_ui.py -q`
Expected: PASS

- [ ] **Step 5: Commit the styling and verification work**

```bash
git add tests/test_pi_details_todo_ui.py codoxear/static/app.css
git commit -m "style(pi-todo): polish details section"
```

## Self-Review Checklist

- Spec coverage: parser helper, diagnostics payload, modal rendering, empty state, unavailable state, and bounded tail scanning are each covered by a dedicated task.
- Placeholder scan: no `TBD`, `TODO`, or vague "handle edge cases" instructions remain.
- Type consistency: the plan uses the same names everywhere: `read_latest_pi_todo_snapshot`, `_todo_snapshot_payload_for_session`, `renderDiagTodoSnapshot`, and `todo_snapshot`.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-03-pi-session-todo-details.md`.

Two execution options:

1. Subagent-Driven (recommended) - I dispatch a fresh subagent per task, review between tasks, fast iteration
2. Inline Execution - Execute tasks in this session using `executing-plans`, batch execution with checkpoints
