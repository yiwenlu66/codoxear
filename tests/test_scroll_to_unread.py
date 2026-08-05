"""Behavioral coverage for the persisted unread sidebar and transcript-scroll pipeline."""

import json
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from codoxear.session_routes import SessionRouteDeps
from codoxear.session_routes import _handle_read_post
from codoxear.session_routes import _handle_unread_get
from codoxear.unread_store import UnreadStore


ROOT = Path(__file__).resolve().parents[1]
APP_SESSION_HELPERS = ROOT / "codoxear" / "static" / "app_session_helpers.js"
APP_SESSIONS = ROOT / "codoxear" / "static" / "app_sessions.js"
APP_TRANSCRIPT = ROOT / "codoxear" / "static" / "app_transcript.js"
APP_UNREAD = ROOT / "codoxear" / "static" / "app_unread.js"


class _Handler:
    pass


def _deps(payload, responses):
    return SessionRouteDeps(
        require_auth=lambda _handler: True,
        json_response=lambda _handler, status, body: responses.append((status, body)),
        json_response_with_etag=lambda *_args: None,
        read_json_body=lambda _handler: payload,
        read_new_session_defaults=lambda: {},
        tmux_available=lambda: False,
        tmux_session_name="",
        metrics_snapshot=lambda: {},
        record_metric=lambda *_args: None,
        perf_counter=lambda: 0.0,
        normalize_agent_backend=lambda value, default: default,
        default_agent_backend="pi",
        resolve_dir_target=lambda *_args, **_kwargs: Path("."),
        describe_session_cwd=lambda _path: {},
        list_resume_candidates_for_cwd=lambda *_args, **_kwargs: [],
        first_user_message_preview_from_log=lambda _path: "",
        parse_new_session_launch_request=lambda _obj: None,
        launch_request_validation_error=ValueError,
        session_launch_error=RuntimeError,
    )


def _unread_response(manager, responses):
    _handle_unread_get(_Handler(), session_id="sid", manager=manager, deps=_deps({}, responses))
    assert responses and responses[0][0] == 200
    return responses[0][1]


def _run_frontend(unread):
    sources = [
        APP_SESSION_HELPERS.read_text(encoding="utf-8"),
        APP_SESSIONS.read_text(encoding="utf-8"),
        APP_TRANSCRIPT.read_text(encoding="utf-8"),
        APP_UNREAD.read_text(encoding="utf-8"),
    ]
    script = f"""
const vm = require("vm");
const ctx = {{ window: {{}}, console }};
vm.createContext(ctx);
(async () => {{
{''.join(f'vm.runInContext({json.dumps(source)}, ctx);' for source in sources)}

class Node {{
  constructor(tag, attrs = {{}}, children = []) {{
    this.tag = tag; this.attributes = attrs; this.children = []; this.dataset = {{}};
    this.style = {{}}; this.parentNode = null; this.offsetTop = 0; this.offsetHeight = 100;
    this.textContent = attrs.text || ""; this.isConnected = true;
    this.classList = {{ add: () => {{}}, remove: () => {{}}, contains: () => false }};
    children.filter(Boolean).forEach((child) => this.appendChild(child));
  }}
  get childElementCount() {{ return this.children.length; }}
  set innerHTML(value) {{ if (value !== "") throw new Error("harness only clears HTML"); this.children = []; }}
  appendChild(child) {{ this.children.push(child); child.parentNode = this; return child; }}
  addEventListener() {{}}
  setPointerCapture() {{}}
  releasePointerCapture() {{}}
}}
function el(tag, attrs = {{}}, children = []) {{ return new Node(tag, attrs, children); }}
function walk(node, predicate) {{
  if (predicate(node)) return node;
  for (const child of node.children || []) {{ const found = walk(child, predicate); if (found) return found; }}
  return null;
}}

const unread = {json.dumps(unread)};
const calls = [];
const session = {{ session_id: "sid", unread_count: 0, cwd: "/work", agent_backend: "pi" }};
const controller = ctx.window.CodoxearUnread.createUnreadController({{
  api: async (path, options = {{}}) => {{
    calls.push({{ path, body: options.body || null }});
    return path.endsWith("/unread") ? unread : {{ ok: true }};
  }},
  patchSessionInfo: (_sid, patch) => Object.assign(session, patch),
}});
await controller.refreshSidebarCounts([session]);

const sessionsWrap = new Node("div");
const sidebar = ctx.window.CodoxearSessions.createSessionsController({{
  sessionsWrap, sidebarEmptyHint: new Node("div"), el, iconSvg: () => "",
  sidebarRenderSignature: ctx.window.CodoxearSessionHelpers.sidebarRenderSignature,
  sidebarSessionEntries: ctx.window.CodoxearSessionHelpers.sidebarSessionEntries,
  sessionDisplayName: () => "Unread session", sessionLaunchFailed: () => false, sessionLaunchPending: () => false,
  redactedLaunchErrorText: () => "", fmtRelativeAge: () => "now", sidebarEffortCode: () => "", sidebarModelText: () => "",
  baseName: () => "work", sessionIsFast: () => false, agentBackendLogoPath: () => "", agentBackendDisplayName: () => "",
  sessionAgentBackend: () => "pi", sessionLaunchIcon: () => "", sessionLaunchLabel: () => "",
  confirmAction: async () => false, api: async () => ({{}}), clearDeletedSessionClientState: () => {{}}, refreshSessions: async () => {{}},
  setToast: () => {{}}, openEditSession: () => {{}}, duplicateSession: async () => {{}}, selectSession: async () => {{}}, setSidebarOpen: () => {{}}, now: () => 0,
}});
sidebar.renderSessions([session]);
const unreadBadge = walk(sessionsWrap, (node) => String(node.attributes && node.attributes.class || "").split(/\\s+/).includes("unread"));

await controller.loadForOpen("sid");
const rows = [1, 2, 3, 4, 5, 6, 7].map((index) => {{
  const row = new Node("div"); row.dataset.messageId = `m${{index}}`; row.offsetTop = index * 100; row.offsetHeight = 100;
  row.scrollIntoView = () => {{ row.scrolled = true; }};
  return row;
}});
const root = {{
  querySelectorAll: () => rows,
  insertBefore: () => {{}},
}};
let autoScrollDisabled = 0;
const runtime = ctx.window.CodoxearTranscript.createTranscriptRenderRuntime({{
  root, bottomSentinel: new Node("div"), document: {{ createDocumentFragment: () => ({{ appendChild: () => {{}} }}) }},
  safeMakeRow: () => ({{ row: new Node("div") }}), normalizeEvents: (events) => events,
  consumePendingUserIfMatches: () => false, isDuplicateEvent: () => false, isAdjacentAssistantDuplicateEvent: () => false,
  markEventSeen: () => {{}}, markFirstPaint: () => {{}}, restorePendingRows: () => {{}}, resetRecentEvents: () => {{}}, setOlderState: () => {{}},
  firstVisibleMessageRow: () => null, getScrollTop: () => 0, getSelectedSessionId: () => "sid",
  domRuntime: {{ clear: () => {{}}, rebuildDecorations: () => {{}}, trimRenderedRows: () => {{}} }},
  scrollRuntime: {{ shouldStickToBottom: () => false, snapshot: () => ({{}}), syncJumpButton: () => {{}}, scheduleScrollToBottom: () => {{}}, markLiveTail: () => {{}},
    disableAutoScroll: () => {{ autoScrollDisabled += 1; }}, setRenderedAtLiveTail: () => {{}}, setScrollTop: () => {{}} }},
  typingRowRuntime: {{ anchor: () => new Node("div") }},
}});
const firstUnreadEventId = controller.firstUnreadForInitialRender("sid");
runtime.renderTranscript([{{ role: "assistant", message_id: "m1" }}], {{ firstUnreadEventId }});
const beforePast = await controller.markReadIfScrolledPast("sid", rows, 799);
const afterPast = await controller.markReadIfScrolledPast("sid", rows, 800);
process.stdout.write(JSON.stringify({{
  badge: unreadBadge && unreadBadge.textContent,
  firstUnreadEventId,
  scrolledTo: rows.filter((row) => row.scrolled).map((row) => row.dataset.messageId),
  autoScrollDisabled,
  beforePast, afterPast, readCalls: calls.filter((call) => call.path.endsWith("/read")), unreadCountAfterScroll: session.unread_count,
}}));
}})();
"""
    completed = subprocess.run(
        ["node"],
        input=script,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_sidebar_unread_badge_scroll_target_and_read_watermark_advance() -> None:
    with tempfile.TemporaryDirectory() as td:
        events = [
            {"role": "assistant", "text": f"answer {index}", "message_id": f"m{index}"}
            for index in range(1, 4)
        ]
        store = UnreadStore(Path(td) / "session_unread.json")
        session = SimpleNamespace(log_path=Path(td) / "transcript.jsonl")
        manager = SimpleNamespace(_unread_store=store, get_session=lambda session_id: session if session_id == "sid" else None)
        with patch("codoxear.session_routes._transcript_events_for_unread", side_effect=lambda _session: list(events)):
            responses = []
            _handle_read_post(_Handler(), session_id="sid", manager=manager, deps=_deps({"event_id": "m3"}, responses))
            assert responses == [(200, {"ok": True, "event_id": "m3"})]

            events.extend(
                {"role": "assistant", "text": f"answer {index}", "message_id": f"m{index}"}
                for index in range(4, 8)
            )
            unread = _unread_response(manager, [])
            frontend = _run_frontend(unread)

            assert frontend == {
                "badge": "unread 4",
                "firstUnreadEventId": "m4",
                "scrolledTo": ["m4"],
                "autoScrollDisabled": 1,
                "beforePast": False,
                "afterPast": True,
                "readCalls": [{"path": "/api/sessions/sid/read", "body": {"event_id": "m7"}}],
                "unreadCountAfterScroll": 0,
            }

            responses = []
            _handle_read_post(_Handler(), session_id="sid", manager=manager, deps=_deps(frontend["readCalls"][0]["body"], responses))
            assert responses == [(200, {"ok": True, "event_id": "m7"})]
            assert _unread_response(manager, []) == {"count": 0, "first_unread_event_id": None, "last_unread_event_id": None}
            assert store.watermark("sid") == "m7"
