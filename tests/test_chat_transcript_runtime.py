from frontend_module_loader import module_path
import json
import subprocess
import textwrap
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = module_path("app_message_history.js")
APP_POLLING_JS = module_path("app_polling.js")
APP_COMPOSER_JS = module_path("app_composer.js")
APP_TRANSCRIPT_JS = module_path("app_transcript.js")
APP_TRANSCRIPT_RENDER_JS = module_path("app_transcript_render.js")
APP_MESSAGE_FLOW_JS = module_path("app_message_flow.js")
APP_SESSION_REFRESH_JS = module_path("app_session_refresh.js")
APP_SESSION_LIFECYCLE_JS = module_path("app_session_lifecycle.js")
APP_SESSION_CATALOG_JS = module_path("app_session_catalog.js")
APP_SESSION_STATE_JS = module_path("app_session_state.js")
APP_MESSAGE_IDENTITY_JS = module_path("app_transcript.js")


MESSAGE_FLOW_HARNESS_JS = """
function createMessageFlow(state, overrides = {}) {
  const noop = () => {};
  const active = state.active || { state: "bound", liveCursor: "c1", logPath: "/tmp/log.jsonl" };
  const typingRowRuntime = {
    snapshot: () => ({ stats: { thinking: 0, thinkingTokens: 0, thinkingMode: "blocks", tools: 0 } }),
    updateTypingStats: noop,
    updateSubagentGauge: noop,
    resetTypingStats: () => { state.resets = (state.resets || 0) + 1; },
  };
  const sessionState = state.sessionState || ctx.window.CodoxearSessionState.createSessionState({ consoleError: noop });
  state.sessionState = sessionState;
  const sessionCatalog = state.sessionCatalog || ctx.window.CodoxearSessionCatalog.createSessionCatalog({ consoleError: noop });
  if (!state.sessionCatalog) sessionCatalog.set("latestSessions", [state.session || { session_id: state.selected || "sid", agent_backend: "pi" }]);
  state.sessionCatalog = sessionCatalog;
  sessionState.applyRuntime({ selected: state.selected || "sid", running: Boolean(state.running) });
  Object.defineProperty(state, "running", {
    get: () => sessionState.get("running"),
    set: (value) => sessionState.applyRuntime({ running: Boolean(value) }),
    configurable: true,
  });
  return ctx.window.CodoxearMessageFlow.createMessageFlowController({
    sessionState, sessionCatalog,
    currentGeneration: () => 1, isAppDisposed: () => false,

    sessionLaunchFailed: () => false,
    api: async () => ({ queued: false, queue_len: 0 }), resolveAppUrl: (path) => `http://example.test${path}`,
    handleAppAuthLoss: noop, refreshSessions: async () => [], openSession: async () => null,
    clearSelectedSessionAfterRemoval: noop, activeTranscriptSnapshot: () => active,
    updateSessionTranscriptSlot: () => ({ ignoredStaleBound: false, current: { state: "bound" } }),
    renderPendingTranscriptSlot: noop, renderSessionTail: noop, applySessionRuntimeFromTail: noop,
    resetChatRenderState: noop, setAttachCount: noop, setLiveCursor: (cursor) => { active.liveCursor = cursor; },
    appendEvent: noop, appendTailSnapshotEvents: noop, setStatus: noop, setContext: noop, setTyping: noop,
    setSubagentsRunning: noop, updateSessionTitle: noop, initPageLimit: () => 60, typingRowRuntime,
    getSending: () => Boolean(state.sending), setSending: (value) => { state.sending = Boolean(value); },
    getCurrentRunning: () => Boolean(state.running), setCurrentRunning: (value) => { state.running = Boolean(value); },
    getStagedAttachments: () => [], normalizedStagedAttachments: () => [], setSelectedSessionPendingAttachment: noop,
    syncSendButtonState: noop, syncAttachButtonState: noop, syncQueueSubmitState: noop, syncRecoveryUiForSession: noop,
    confirmAction: async () => false, setToast: (text) => { state.toast = text; }, isTranscriptRenewalCommand: () => false,
    nextLocalEchoId: () => 1, renderedAtLiveTail: () => true, clearTranscriptDom: noop,
    clearRenderedTranscriptRange: noop, setOlderState: noop, getSessionTranscriptSlot: () => ({ epoch: 0 }),
    addPendingUser: noop, deleteTailCache: noop, beginTranscriptRenewal: noop, clearLiveCursor: noop,
    invalidateOlderLoad: noop, dropPendingUser: noop, removePendingUserRow: noop, hasPendingForSession: () => false,
    visibilityState: () => "visible", navigatorValue: () => ({ onLine: true }), EventSource: null,
    AbortController: null, setTimeout: () => 0, clearTimeout: noop, now: () => 1000,
    consoleWarn: noop, consoleError: noop,
    ...overrides,
  });
}
"""


def _run_node(js: str) -> dict:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".js", encoding="utf-8") as script:
        script.write(js)
        script.flush()
        proc = subprocess.run(
            ["node", script.name],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    return json.loads(proc.stdout)


def _source_between(start: str, end: str) -> str:
    source = APP_JS.read_text(encoding="utf-8")
    i = source.index(start)
    j = source.index(end, i)
    return source[i:j]


class TestChatTranscriptRuntime(unittest.TestCase):
    def test_session_snapshot_busy_to_idle_clears_store_projected_typing_row(self) -> None:
        polling_source = APP_POLLING_JS.read_text(encoding="utf-8")
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        transcript_render_source = APP_TRANSCRIPT_RENDER_JS.read_text(encoding="utf-8")
        message_flow_source = APP_MESSAGE_FLOW_JS.read_text(encoding="utf-8")
        session_refresh_source = APP_SESSION_REFRESH_JS.read_text(encoding="utf-8")
        session_state_source = APP_SESSION_STATE_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const vm = require("vm");
            const ctx = {{ window: {{}}, console, Date, URL, encodeURIComponent }};
            vm.createContext(ctx);
            vm.runInContext({json.dumps(polling_source)}, ctx);
            vm.runInContext({json.dumps(transcript_source)}, ctx);
            vm.runInContext({json.dumps(session_state_source)}, ctx);
            vm.runInContext({json.dumps(APP_SESSION_CATALOG_JS.read_text(encoding="utf-8"))}, ctx);
            vm.runInContext({json.dumps(transcript_render_source)}, ctx);
            vm.runInContext({json.dumps(message_flow_source)}, ctx);
            vm.runInContext({json.dumps(session_refresh_source)}, ctx);

            function node(attrs = {{}}, children = []) {{
              const out = {{ ...attrs, children: [], dataset: {{}}, isConnected: false, parentNode: null }};
              out.appendChild = (child) => {{ out.children.push(child); child.parentNode = out; return child; }};
              out.insertBefore = (child, before) => {{
                if (child.parentNode) child.parentNode.children = child.parentNode.children.filter((item) => item !== child);
                const index = out.children.indexOf(before);
                out.children.splice(index < 0 ? out.children.length : index, 0, child);
                child.parentNode = out;
                child.isConnected = true;
                return child;
              }};
              out.remove = () => {{
                if (out.parentNode) out.parentNode.children = out.parentNode.children.filter((item) => item !== out);
                out.parentNode = null;
                out.isConnected = false;
              }};
              Object.defineProperty(out, "nextSibling", {{ get: () => out.parentNode ? out.parentNode.children[out.parentNode.children.indexOf(out) + 1] || null : null }});
              for (const child of children) out.appendChild(child);
              return out;
            }}

            const root = node();
            const bottom = node();
            root.appendChild(bottom);
            bottom.isConnected = true;
            const typingRowRuntime = ctx.window.CodoxearTranscript.createTypingRowRuntime({{
              root, bottomSentinel: bottom, el: (_tag, attrs, children) => node(attrs, children),
              shouldAutoScroll: () => false, scheduleScrollToBottom: () => {{}},
            }});
            const sessionState = ctx.window.CodoxearSessionState.createSessionState({{ consoleError: () => {{}} }});
            sessionState.applyRuntime({{ selected: "sid", running: true, turnOpen: true }});
            const projection = ctx.window.CodoxearTranscriptRender.createTypingRowStoreProjection({{ sessionState, typingRowRuntime }});
            typingRowRuntime.updateTypingStats({{ tools: 5, thinkingTokens: 1200, thinkingMode: "tokens" }});
            const noop = () => {{}};
            const sessionCatalog = ctx.window.CodoxearSessionCatalog.createSessionCatalog({{ consoleError: noop }});
            sessionCatalog.set("latestSessions", [{{ session_id: "sid", agent_backend: "pi" }}]);
            const flow = ctx.window.CodoxearMessageFlow.createMessageFlowController({{
              sessionState, sessionCatalog, currentGeneration: () => 1, isAppDisposed: () => false,
              sessionLaunchFailed: () => false,
              api: async () => ({{}}), resolveAppUrl: (path) => path, handleAppAuthLoss: noop,
              refreshSessions: async () => [], openSession: async () => null, clearSelectedSessionAfterRemoval: noop,
              activeTranscriptSnapshot: () => ({{ state: "bound", liveCursor: "cursor", logPath: "/tmp/log" }}),
              updateSessionTranscriptSlot: () => ({{ ignoredStaleBound: false, current: {{ state: "bound" }} }}),
              renderPendingTranscriptSlot: noop, renderSessionTail: noop, applySessionRuntimeFromTail: noop,
              resetChatRenderState: noop, setAttachCount: noop, setLiveCursor: noop, appendEvent: noop,
              appendTailSnapshotEvents: noop, updateSessionTitle: noop, initPageLimit: () => 60, typingRowRuntime,
              getStagedAttachments: () => [], normalizedStagedAttachments: () => [], setSelectedSessionPendingAttachment: noop,
              syncSendButtonState: noop, syncAttachButtonState: noop, syncQueueSubmitState: noop, syncRecoveryUiForSession: noop,
              confirmAction: async () => false, setToast: noop, isTranscriptRenewalCommand: () => false,
              nextLocalEchoId: () => 1, renderedAtLiveTail: () => true, getSessionTranscriptSlot: () => ({{ epoch: 0 }}),
              addPendingUser: noop, deleteTailCache: noop, beginTranscriptRenewal: noop, clearLiveCursor: noop,
              invalidateOlderLoad: noop, dropPendingUser: noop, removePendingUserRow: noop, hasPendingForSession: () => false,
              visibilityState: () => "hidden", navigatorValue: () => ({{ onLine: true }}), EventSource: null,
              AbortController: null, setTimeout: () => 0, clearTimeout: noop, now: () => 0, consoleWarn: noop, consoleError: noop,
            }});
            const listedSession = {{ session_id: "sid", busy: false, queue_len: 0, token: null, subagents_running: 0, tools: 0, thinking_tokens: 0 }};
            const refresh = ctx.window.CodoxearSessionRefresh.createSessionRefreshController({{
              sessionState, sessionCatalog,
              api: async () => ({{ sessions: [listedSession] }}),
              isDisposed: () => false, apiResponseNotModified: () => false,
              emptyDefaults: () => ({{}}), clearFileDiscoveryCaches: noop, useDesktopSessionActions: () => true,
 clearSelectedSessionAfterRemoval: noop, applySessionListTranscriptIdentity: noop,
              syncRecoveryUiForSession: noop, syncAttachments: noop, clearAttachments: noop,
              renderSessions: () => true, hasDeferredRefresh: () => false, setTitle: noop, sessionTitle: () => "sid",
              updateTypingStats: flow.updateTypingStatsFromSession, updateUnattendedButton: noop,
              syncComposerSendButton: noop, syncQueueSubmitState: noop, maybeSelectPendingHashSession: noop,
            }});
            const before = {{ running: sessionState.get("running"), turnOpen: sessionState.get("turnOpen"), children: root.children.length, connected: typingRowRuntime.snapshot().connected }};
            (async () => {{
              await refresh.refreshSessions();
              const after = {{ running: sessionState.get("running"), turnOpen: sessionState.get("turnOpen"), children: root.children.length, connected: typingRowRuntime.snapshot().connected, stats: typingRowRuntime.snapshot().stats }};
              projection.dispose();
              process.stdout.write(JSON.stringify({{ before, after }}));
            }})().catch((error) => {{ console.error(error); process.exit(1); }});
            """
        )
        out = _run_node(js)
        self.assertEqual(out["before"], {"running": True, "turnOpen": True, "children": 2, "connected": True})
        self.assertEqual(out["after"], {
            "running": False,
            "turnOpen": False,
            "children": 1,
            "connected": False,
            "stats": {"thinking": 0, "thinkingTokens": 0, "thinkingMode": "blocks", "tools": 0},
        })

    def test_recovery_refresh_updates_catalog_subscribers_without_imperative_projection_calls(self) -> None:
        session_refresh_source = APP_SESSION_REFRESH_JS.read_text(encoding="utf-8")
        session_state_source = APP_SESSION_STATE_JS.read_text(encoding="utf-8")
        catalog_source = APP_SESSION_CATALOG_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const vm = require("vm");
            const ctx = {{ window: {{}}, console }};
            vm.createContext(ctx);
            vm.runInContext({json.dumps(session_state_source)}, ctx);
            vm.runInContext({json.dumps(catalog_source)}, ctx);
            vm.runInContext({json.dumps(session_refresh_source)}, ctx);
            const noop = () => {{}};
            const sessionState = ctx.window.CodoxearSessionState.createSessionState({{ consoleError: noop }});
            sessionState.set("selected", "sid");
            const sessionCatalog = ctx.window.CodoxearSessionCatalog.createSessionCatalog({{ consoleError: noop }});
            sessionCatalog.set("latestSessions", [{{ session_id: "sid", commit_unknown_send: false, queue_len: 0 }}]);
            const projections = [];
            for (const name of ["attachment", "queue", "composer", "unattended"]) {{
              sessionCatalog.subscribe("sessionIndex", () => {{
                const row = sessionCatalog.get("sessionIndex").get("sid");
                projections.push([name, row.commit_unknown_send, sessionCatalog.get("newSessionDefaults").generation]);
              }});
            }}
            const refresh = ctx.window.CodoxearSessionRefresh.createSessionRefreshController({{
              sessionState, sessionCatalog,
              api: async () => ({{
                sessions: [{{ session_id: "sid", commit_unknown_send: true, queue_len: 3 }}],
                new_session_defaults: {{ generation: 2 }}, tmux_available: true, recent_cwds: ["/next"],
              }}),
              isDisposed: () => false, apiResponseNotModified: () => false, emptyDefaults: () => ({{}}),
              clearFileDiscoveryCaches: noop, useDesktopSessionActions: () => true, clearSelectedSessionAfterRemoval: noop,
              applySessionListTranscriptIdentity: noop, syncAttachments: noop, clearAttachments: noop,
              renderSessions: () => true, hasDeferredRefresh: () => false, updateTypingStats: (session) => {{
                sessionState.applyRuntime({{ queueLen: Number(session.queue_len) || 0 }});
              }}, maybeSelectPendingHashSession: noop,
            }});
            refresh.refreshSessions().then(() => process.stdout.write(JSON.stringify({{
              projections, queueLen: sessionState.get("queueLen"),
              defaults: sessionCatalog.get("newSessionDefaults").generation,
            }})));
            """
        )
        out = _run_node(js)
        self.assertEqual(out, {
            "projections": [
                ["attachment", True, 2], ["queue", True, 2],
                ["composer", True, 2], ["unattended", True, 2],
            ],
            "queueLen": 3,
            "defaults": 2,
        })

    def test_first_unread_message_row_scrolls_to_matching_transcript_row(self) -> None:
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const vm = require("vm");
            const ctx = {{ window: {{}} }};
            vm.createContext(ctx);
            vm.runInContext({json.dumps(transcript_source)}, ctx);
            const scrolled = [];
            const rows = ["m-old", "m-first-unread", "m-newer"].map((messageId, index) => ({{
              dataset: {{ messageId }}, offsetTop: (index + 1) * 100,
              scrollIntoView: () => scrolled.push(messageId),
            }}));
            const root = {{ querySelectorAll: () => rows }};
            let autoScrollDisabled = 0;
            const runtime = ctx.window.CodoxearTranscript.createTranscriptRenderRuntime({{
              root, bottomSentinel: {{}}, document: {{ createDocumentFragment: () => ({{ appendChild: () => {{}} }}) }},
              safeMakeRow: () => ({{ row: {{}} }}), normalizeEvents: (events) => events,
              consumePendingUserIfMatches: () => false, isDuplicateEvent: () => false,
              isAdjacentAssistantDuplicateEvent: () => false, markEventSeen: () => {{}}, markFirstPaint: () => {{}},
              restorePendingRows: () => {{}}, resetRecentEvents: () => {{}}, setOlderState: () => {{}},
              firstVisibleMessageRow: () => null, getScrollTop: () => 0, getSelectedSessionId: () => "sid",
              domRuntime: {{ clear: () => {{}}, rebuildDecorations: () => {{}}, trimRenderedRows: () => {{}} }},
              scrollRuntime: {{ shouldStickToBottom: () => false, snapshot: () => ({{}}), syncJumpButton: () => {{}},
                scheduleScrollToBottom: () => {{}}, markLiveTail: () => {{}}, disableAutoScroll: () => {{ autoScrollDisabled += 1; }},
                setRenderedAtLiveTail: () => {{}}, setScrollTop: () => {{}} }},
              typingRowRuntime: {{ anchor: () => ({{}}) }},
            }});
            const found = runtime.scrollToFirstUnread("m-first-unread");
            const missing = runtime.scrollToFirstUnread("missing");
            process.stdout.write(JSON.stringify({{ found, missing, scrolled, autoScrollDisabled }}));
            """
        )
        self.assertEqual(_run_node(js), {"found": True, "missing": False, "scrolled": ["m-first-unread"], "autoScrollDisabled": 1})

    def test_typing_row_runtime_projects_activity_stats_without_replacing_dots(self) -> None:
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const ctx = {{ window: {{}} }};
            const vm = require("vm");
            vm.createContext(ctx);
            vm.runInContext({json.dumps(transcript_source)}, ctx);
            function node(attrs = {{}}, children = []) {{
              const out = {{ ...attrs, children: [], dataset: {{}}, isConnected: false }};
              out.appendChild = (child) => {{ out.children.push(child); return child; }};
              out.querySelector = (selector) => {{
                if (selector === ".typingStats" && out.class === "typingStats") return out;
                for (const child of out.children) {{
                  const found = child && typeof child.querySelector === "function" ? child.querySelector(selector) : null;
                  if (found) return found;
                }}
                return null;
              }};
              out.remove = () => {{ out.isConnected = false; }};
              for (const child of children) out.appendChild(child);
              return out;
            }}
            const bottom = node();
            const root = {{
              insertBefore: (row) => {{ row.isConnected = true; }},
            }};
            const runtime = ctx.window.CodoxearTranscript.createTypingRowRuntime({{
              root,
              bottomSentinel: bottom,
              el: (tag, attrs, children) => node(attrs, children),
              shouldAutoScroll: () => false,
              scheduleScrollToBottom: () => {{}},
            }});
            runtime.setVisible(true);
            const statsNode = runtime.anchor().querySelector(".typingStats");
            const initial = {{ text: statsNode.textContent, stats: runtime.snapshot().stats }};
            runtime.updateTypingStats({{ tools: 2, thinking: 1 }}, {{ delta: true }});
            const incremented = {{ text: statsNode.textContent, stats: runtime.snapshot().stats }};
            runtime.updateTypingStats({{ tools: 7, thinking: 3 }});
            const replaced = {{ text: statsNode.textContent, stats: runtime.snapshot().stats }};
            runtime.updateTypingStats({{ tools: 7, thinking: 3, thinkingTokens: 999, thinkingMode: "tokens" }});
            const token999 = {{ text: statsNode.textContent, stats: runtime.snapshot().stats }};
            runtime.updateTypingStats({{ thinkingTokens: 201 }}, {{ delta: true }});
            const token1200 = {{ text: statsNode.textContent, stats: runtime.snapshot().stats }};
            runtime.updateTypingStats({{ thinkingTokens: 1500000, thinkingMode: "tokens", tools: 7 }});
            const tokenMillion = {{ text: statsNode.textContent, stats: runtime.snapshot().stats }};
            runtime.updateTypingStats({{ tools: 7, thinking: 3, thinkingMode: "blocks" }});
            const nonPiBlocks = {{ text: statsNode.textContent, stats: runtime.snapshot().stats }};
            runtime.updateSubagentGauge(2);
            const withGauge = {{ text: statsNode.textContent, stats: runtime.snapshot().stats }};
            runtime.updateSubagentGauge(0);
            const gaugeCleared = {{ text: statsNode.textContent, stats: runtime.snapshot().stats }};
            runtime.setVisible(false);
            const hidden = {{ text: statsNode.textContent, stats: runtime.snapshot().stats }};
            process.stdout.write(JSON.stringify({{ initial, incremented, replaced, token999, token1200, tokenMillion, nonPiBlocks, withGauge, gaugeCleared, hidden }}));
            """
        )
        out = _run_node(js)
        # With no counts yet the bubble communicates "working" instead of three
        # cryptic dots; the counts replace the hint as they arrive.
        self.assertEqual(out["initial"], {"text": "working", "stats": {"thinking": 0, "thinkingTokens": 0, "thinkingMode": "blocks", "tools": 0}})
        # Block counts remain state for reconciliation, but the bubble renders
        # only authoritative reasoning-token counts; it must not imply that
        # opaque blocks and tokens are equivalent.
        self.assertEqual(out["incremented"], {"text": "tools: 2", "stats": {"thinking": 1, "thinkingTokens": 0, "thinkingMode": "blocks", "tools": 2}})
        self.assertEqual(out["replaced"], {"text": "tools: 7", "stats": {"thinking": 3, "thinkingTokens": 0, "thinkingMode": "blocks", "tools": 7}})
        self.assertEqual(out["token999"], {"text": "tools: 7 · thinking: 999", "stats": {"thinking": 3, "thinkingTokens": 999, "thinkingMode": "tokens", "tools": 7}})
        self.assertEqual(out["token1200"], {"text": "tools: 7 · thinking: 1.2k", "stats": {"thinking": 3, "thinkingTokens": 1200, "thinkingMode": "tokens", "tools": 7}})
        self.assertEqual(out["tokenMillion"], {"text": "tools: 7 · thinking: 1.5M", "stats": {"thinking": 0, "thinkingTokens": 1500000, "thinkingMode": "tokens", "tools": 7}})
        self.assertEqual(out["nonPiBlocks"], {"text": "tools: 7", "stats": {"thinking": 3, "thinkingTokens": 0, "thinkingMode": "blocks", "tools": 7}})
        self.assertEqual(out["withGauge"], {"text": "tools: 7 · subagents: 2", "stats": {"thinking": 3, "thinkingTokens": 0, "thinkingMode": "blocks", "tools": 7}})
        self.assertEqual(out["gaugeCleared"], {"text": "tools: 7", "stats": {"thinking": 3, "thinkingTokens": 0, "thinkingMode": "blocks", "tools": 7}})
        self.assertEqual(out["hidden"], {"text": "", "stats": {"thinking": 0, "thinkingTokens": 0, "thinkingMode": "blocks", "tools": 0}})

    def test_idle_subagent_store_projection_materializes_and_removes_activity_row(self) -> None:
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        transcript_render_source = APP_TRANSCRIPT_RENDER_JS.read_text(encoding="utf-8")
        session_state_source = APP_SESSION_STATE_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const vm = require("vm");
            const ctx = {{ window: {{}}, console }};
            vm.createContext(ctx);
            vm.runInContext({json.dumps(transcript_source)}, ctx);
            vm.runInContext({json.dumps(session_state_source)}, ctx);
            vm.runInContext({json.dumps(APP_SESSION_CATALOG_JS.read_text(encoding="utf-8"))}, ctx);
            vm.runInContext({json.dumps(transcript_render_source)}, ctx);

            function node(attrs = {{}}, children = []) {{
              const out = {{ ...attrs, children: [], dataset: {{}}, isConnected: false, parentNode: null }};
              out.appendChild = (child) => {{ out.children.push(child); child.parentNode = out; return child; }};
              out.insertBefore = (child, before) => {{
                if (child.parentNode) child.parentNode.children = child.parentNode.children.filter((item) => item !== child);
                const index = out.children.indexOf(before);
                out.children.splice(index < 0 ? out.children.length : index, 0, child);
                child.parentNode = out;
                child.isConnected = true;
                return child;
              }};
              out.remove = () => {{
                if (out.parentNode) out.parentNode.children = out.parentNode.children.filter((item) => item !== out);
                out.parentNode = null;
                out.isConnected = false;
              }};
              Object.defineProperty(out, "nextSibling", {{ get: () => out.parentNode ? out.parentNode.children[out.parentNode.children.indexOf(out) + 1] || null : null }});
              for (const child of children) out.appendChild(child);
              return out;
            }}

            const root = node();
            const bottom = node();
            root.appendChild(bottom);
            bottom.isConnected = true;
            const typingRowRuntime = ctx.window.CodoxearTranscript.createTypingRowRuntime({{
              root, bottomSentinel: bottom, el: (_tag, attrs, children) => node(attrs, children),
              shouldAutoScroll: () => false, scheduleScrollToBottom: () => {{}},
            }});
            const sessionState = ctx.window.CodoxearSessionState.createSessionState({{ consoleError: () => {{}} }});
            const projection = ctx.window.CodoxearTranscriptRender.createTypingRowStoreProjection({{ sessionState, typingRowRuntime }});
            const countActivityRows = () => root.children.filter((child) => child.class === "msg-row assistant subagent-activity-row").length;
            const initial = {{ children: root.children.length, activityRows: countActivityRows() }};
            sessionState.applyRuntime({{ subagentsRunning: 1 }});
            const afterOne = {{ children: root.children.length, activityRows: countActivityRows(), text: typingRowRuntime.anchor().children[0].children[1].textContent }};
            sessionState.applyRuntime({{ subagentsRunning: 0 }});
            const afterZero = {{ children: root.children.length, activityRows: countActivityRows() }};
            projection.dispose();
            process.stdout.write(JSON.stringify({{ initial, afterOne, afterZero }}));
            """
        )
        self.assertEqual(_run_node(js), {
            "initial": {"children": 1, "activityRows": 0},
            "afterOne": {"children": 2, "activityRows": 1, "text": "▸1 subagent working"},
            "afterZero": {"children": 1, "activityRows": 0},
        })

    def test_typing_token_mode_requires_authoritative_positive_tokens(self) -> None:
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const vm = require("vm");
            const ctx = {{ window: {{}} }};
            vm.createContext(ctx);
            vm.runInContext({json.dumps(transcript_source)}, ctx);
            const mode = ctx.window.CodoxearTranscript.thinkingModeForTokens;
            process.stdout.write(JSON.stringify({{
              pi: mode(1200),
              codex: mode(48),
              claude: mode(7),
              absent: mode(undefined),
              zero: mode(0),
              malformed: mode("unknown"),
            }}));
            """
        )
        self.assertEqual(_run_node(js), {
            "pi": "tokens",
            "codex": "tokens",
            "claude": "tokens",
            "absent": "blocks",
            "zero": "blocks",
            "malformed": "blocks",
        })

    def test_idle_subagent_activity_row_is_static_and_replaced(self) -> None:
        source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const vm = require("vm");
            const ctx = {{ window: {{}} }};
            vm.createContext(ctx);
            vm.runInContext({json.dumps(source)}, ctx);
            function node(attrs = {{}}, children = []) {{
              const out = {{ ...attrs, children: [], dataset: {{}}, isConnected: false, appendChild(child) {{ out.children.push(child); return child; }}, remove() {{ out.isConnected = false; }} }};
              out.nextSibling = null;
              for (const child of children) out.appendChild(child);
              return out;
            }}
            const bottom = node();
            const root = {{ insertBefore: (row) => {{ row.isConnected = true; }} }};
            const runtime = ctx.window.CodoxearTranscript.createTypingRowRuntime({{
              root, bottomSentinel: bottom, el: (tag, attrs, children) => node(attrs, children),
              shouldAutoScroll: () => false, scheduleScrollToBottom: () => {{}},
            }});
            runtime.updateSubagentGauge(2);
            runtime.setSubagentVisible(true);
            const first = runtime.anchor();
            const firstBubble = first.children[0];
            const firstText = firstBubble.children[1].textContent;
            runtime.updateSubagentGauge(3);
            const replacedText = firstBubble.children[1].textContent;
            runtime.setSubagentVisible(false);
            process.stdout.write(JSON.stringify({{ className: first.class, squares: firstBubble.children[0].children.length, firstText, replacedText, connected: first.isConnected }}));
            """
        )
        proc = subprocess.run(["node", "-e", js], check=True, capture_output=True, text=True)
        self.assertEqual(json.loads(proc.stdout), {
            "className": "msg-row assistant subagent-activity-row",
            "squares": 2,
            "firstText": "▸2 subagents working",
            "replacedText": "▸3 subagents working",
            "connected": False,
        })

    def test_typing_count_window_starts_only_from_idle(self) -> None:
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const ctx = {{ window: {{}} }};
            const vm = require("vm");
            vm.createContext(ctx);
            vm.runInContext({json.dumps(transcript_source)}, ctx);
            const starts = ctx.window.CodoxearTranscript.startsTypingCountWindow;
            const hasHuman = ctx.window.CodoxearTranscript.hasHumanOriginatedUserEvent;
            const shouldReset = (wasTurnOpen, turnStart, nowBusy, events) =>
              starts({{ wasTurnOpen, turnStart, nowBusy }}) && hasHuman(events);
            process.stdout.write(JSON.stringify({{
              idleUser: starts({{ wasTurnOpen: false, turnStart: true, nowBusy: true }}),
              idleBusyFallback: starts({{ wasTurnOpen: false, turnStart: false, nowBusy: true }}),
              steer: starts({{ wasTurnOpen: true, turnStart: true, nowBusy: true }}),
              openSnapshot: starts({{ wasTurnOpen: true, turnStart: false, nowBusy: true }}),
              resetForHuman: shouldReset(false, true, true, [{{ role: "user", text: "human request" }}]),
              resetForSubagentResult: shouldReset(false, true, true, [{{ role: "user", text: "**📨 From subagent-result** (/workspace)" }}]),
              resetForTaggedSubagentControl: shouldReset(false, true, true, [{{ role: "user", text: "ignored", agent_internal_delivery: true }}]),
              resetForBusyWithoutUser: shouldReset(false, false, true, []),
            }}));
            """
        )
        out = _run_node(js)
        self.assertEqual(
            out,
            {
                "idleUser": True,
                "idleBusyFallback": True,
                "steer": False,
                "openSnapshot": False,
                "resetForHuman": True,
                "resetForSubagentResult": False,
                "resetForTaggedSubagentControl": False,
                "resetForBusyWithoutUser": False,
            },
        )

    def test_transcript_scroll_runtime_owns_bottom_lock_and_input_policy(self) -> None:
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const ctx = {{ window: {{}} }};
            const vm = require("vm");
            vm.createContext(ctx);
            vm.runInContext({json.dumps(transcript_source)}, ctx);
            const rafs = [];
            const chat = {{ scrollTop: 0, scrollHeight: 500, clientHeight: 100 }};
            const jumpButton = {{ style: {{ display: "" }} }};
            const timeChip = {{ style: {{ display: "" }}, textContent: "" }};
            let selected = true;
            let searchOpen = false;
            let cancelOlder = 0;
            let autoLoadOlder = 0;
            let cancelEligible = false;
            const firstRow = {{ dataset: {{ ts: "1000" }} }};
            const runtime = ctx.window.CodoxearTranscript.createTranscriptScrollRuntime({{
              chat,
              jumpButton,
              timeChip,
              requestAnimationFrame: (fn) => rafs.push(fn),
              hasSelection: () => selected,
              isSearchOpen: () => searchOpen,
              firstVisibleMessageRow: () => firstRow,
              dayLabel: () => "Day",
              time24: () => "12:34",
              shouldCancelOlderLoad: () => cancelEligible,
              cancelOlderLoad: () => {{ cancelOlder += 1; }},
              autoLoadOlder: () => {{ autoLoadOlder += 1; }},
              bottomThresholdPx: 80,
              olderTopTriggerPx: 1,
              olderCancelPx: 48,
            }});
            runtime.syncJumpButton();
            const initialProjection = {{ jump: jumpButton.style.display, time: timeChip.style.display, text: timeChip.textContent }};
            runtime.scrollToBottom();
            const afterBottom = {{ top: chat.scrollTop, snapshot: runtime.snapshot() }};
            chat.scrollTop = 100;
            const upScroll = runtime.handleScroll();
            runtime.markDetachedWindow();
            const detachedTime = runtime.syncVisibleTimeIndicator();
            searchOpen = true;
            const searchHidden = runtime.syncVisibleTimeIndicator();
            searchOpen = false;
            cancelEligible = true;
            chat.scrollTop = 50;
            runtime.handleScroll();
            chat.scrollTop = 0;
            runtime.handleScroll();
            const afterThresholds = {{ cancelOlder, autoLoadOlder, snapshot: runtime.snapshot(), jump: jumpButton.style.display }};
            chat.scrollTop = 400;
            runtime.handleScroll();
            const afterNearBottom = runtime.snapshot();
            chat.scrollTop = 10;
            runtime.handleWheel({{ deltaY: -1 }});
            const afterWheelAwayFromTop = {{ autoLoadOlder, snapshot: runtime.snapshot() }};
            chat.scrollTop = 0;
            runtime.handleWheel({{ deltaY: -1 }});
            runtime.handleTouchStart({{ touches: [{{ clientY: 10 }}] }});
            runtime.handleTouchMove({{ touches: [{{ clientY: 20 }}] }});
            const afterWheelTouchAtTop = {{ autoLoadOlder, snapshot: runtime.snapshot() }};
            chat.scrollTop = 5;
            chat.scrollHeight = 800;
            runtime.markLiveTail();
            runtime.enableAutoScroll();
            runtime.scheduleScrollToBottom({{ double: true, syncJump: true }});
            const scheduledBeforeRun = rafs.length;
            rafs.shift()();
            const afterFirstRaf = {{ top: chat.scrollTop, queued: rafs.length }};
            rafs.shift()();
            const afterSecondRaf = {{ top: chat.scrollTop, jump: jumpButton.style.display }};
            runtime.reset({{ scrollTop: 0 }});
            const reset = {{ snapshot: runtime.snapshot(), top: chat.scrollTop, jump: jumpButton.style.display, time: timeChip.style.display }};
            let missingError = "";
            try {{ ctx.window.CodoxearTranscript.createTranscriptScrollRuntime({{ chat, jumpButton, timeChip }}); }} catch (err) {{ missingError = err && err.message ? err.message : String(err); }}
            process.stdout.write(JSON.stringify({{
              initialProjection,
              afterBottom,
              upScroll,
              detachedTime,
              searchHidden,
              afterThresholds,
              afterNearBottom,
              afterWheelAwayFromTop,
              afterWheelTouchAtTop,
              scheduledBeforeRun,
              afterFirstRaf,
              afterSecondRaf,
              reset,
              missingError,
              frozen: Object.isFrozen(runtime),
            }}));
            """
        )
        out = _run_node(js)
        self.assertEqual(out["initialProjection"], {"jump": "none", "time": "none", "text": ""})
        self.assertEqual(out["afterBottom"]["top"], 500)
        self.assertTrue(out["afterBottom"]["snapshot"]["autoScroll"])
        self.assertLess(out["upScroll"]["delta"], 0)
        self.assertFalse(out["upScroll"]["autoScroll"])
        self.assertEqual(out["detachedTime"], {"visible": True, "text": "Day · 12:34"})
        self.assertEqual(out["searchHidden"], {"visible": False, "text": ""})
        self.assertEqual(out["afterThresholds"]["cancelOlder"], 1)
        self.assertEqual(out["afterThresholds"]["autoLoadOlder"], 1)
        self.assertEqual(out["afterThresholds"]["jump"], "inline-flex")
        self.assertTrue(out["afterNearBottom"]["autoScroll"])
        self.assertEqual(out["afterWheelAwayFromTop"]["autoLoadOlder"], 1)
        self.assertFalse(out["afterWheelAwayFromTop"]["snapshot"]["autoScroll"])
        self.assertEqual(out["afterWheelTouchAtTop"]["autoLoadOlder"], 3)
        self.assertFalse(out["afterWheelTouchAtTop"]["snapshot"]["autoScroll"])
        self.assertEqual(out["scheduledBeforeRun"], 1)
        self.assertEqual(out["afterFirstRaf"], {"top": 800, "queued": 1})
        self.assertEqual(out["afterSecondRaf"], {"top": 800, "jump": "none"})
        self.assertEqual(out["reset"]["snapshot"], {"autoScroll": True, "renderedAtLiveTail": True, "lastScrollTop": 0})
        self.assertEqual(out["reset"]["top"], 0)
        self.assertEqual(out["reset"]["jump"], "none")
        self.assertContains("transcript dependency missing: requestAnimationFrame", out["missingError"])
        self.assertTrue(out["frozen"])

    def test_session_scroll_memory_restores_exact_positions_and_discards_stale_heights(self) -> None:
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const vm = require("vm");
            const ctx = {{ window: {{}} }};
            vm.createContext(ctx);
            vm.runInContext({json.dumps(transcript_source)}, ctx);
            const rafs = [];
            const chat = {{ scrollTop: 120, scrollHeight: 500, clientHeight: 100 }};
            const jumpButton = {{ style: {{ display: "" }} }};
            const timeChip = {{ style: {{ display: "" }}, textContent: "" }};
            const runtime = ctx.window.CodoxearTranscript.createTranscriptScrollRuntime({{
              chat, jumpButton, timeChip,
              requestAnimationFrame: (fn) => rafs.push(fn),
              hasSelection: () => true, isSearchOpen: () => false,
              firstVisibleMessageRow: () => ({{ dataset: {{ ts: "1000" }} }}),
              dayLabel: () => "Day", time24: () => "12:34",
              shouldCancelOlderLoad: () => false, cancelOlderLoad: () => {{}}, autoLoadOlder: () => {{}},
            }});
            const saved = runtime.saveSessionScrollPosition("sid");
            const remembered = runtime.sessionScrollPosition("sid");
            chat.scrollTop = 0;
            runtime.scheduleScrollToBottom({{ double: true }});
            const restoreExact = runtime.restoreSessionScrollPosition("sid");
            while (rafs.length) rafs.shift()();
            const exact = {{ top: chat.scrollTop, autoScroll: runtime.snapshot().autoScroll, jump: jumpButton.style.display }};
            chat.scrollTop = 0;
            chat.scrollHeight = 700;
            const restoreChanged = runtime.restoreSessionScrollPosition("sid");
            while (rafs.length) rafs.shift()();
            const changed = {{ top: chat.scrollTop, autoScroll: runtime.snapshot().autoScroll, jump: jumpButton.style.display }};
            const cleared = runtime.clearSessionScrollPosition("sid");
            const restoreMissing = runtime.restoreSessionScrollPosition("sid");
            process.stdout.write(JSON.stringify({{ saved, remembered, restoreExact, exact, restoreChanged, changed, cleared, restoreMissing }}));
            """
        )
        self.assertEqual(_run_node(js), {
            "saved": True,
            "remembered": {"scrollTop": 120, "scrollHeight": 500},
            "restoreExact": True,
            "exact": {"top": 120, "autoScroll": False, "jump": "inline-flex"},
            "restoreChanged": True,
            "changed": {"top": 700, "autoScroll": True, "jump": "none"},
            "cleared": True,
            "restoreMissing": False,
        })

    def test_transcript_event_runtime_owns_recent_events_and_pending_echoes(self) -> None:
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        identity_source = APP_MESSAGE_IDENTITY_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const ctx = {{ window: {{}} }};
            const vm = require("vm");
            vm.createContext(ctx);
            vm.runInContext({json.dumps(identity_source)}, ctx);
            vm.runInContext({json.dumps(transcript_source)}, ctx);
            const identity = ctx.window.CodoxearMessageIdentity;
            const runtime = ctx.window.CodoxearTranscript.createTranscriptEventRuntime({{
              eventKey: identity.eventKey,
              pendingMatchKey: identity.pendingMatchKey,
              normalizePendingText: identity.normalizeTextForPendingMatch,
              assistantDedupeKey: identity.chatAssistantDedupeKey,
              maxRecentEventKeys: 2,
            }});
            const firstSeen = runtime.markEventSeen({{ role: "assistant", text: "one", ts: 1 }});
            const firstDuplicate = runtime.isDuplicateEvent({{ role: "assistant", text: "one", ts: 1 }});
            runtime.markEventSeen({{ role: "assistant", text: "two", ts: 2 }});
            runtime.markEventSeen({{ role: "assistant", text: "three", ts: 3 }});
            const afterEvict = {{ keys: runtime.snapshot().recentEventKeys, evictedDuplicate: runtime.isDuplicateEvent({{ role: "assistant", text: "one", ts: 1 }}) }};
            const adjacentTrue = runtime.isAdjacentAssistantDuplicateEvent(
              {{ role: "assistant", text: "same final text", message_class: "final_response", ts: 4 }},
              {{ renderedAtLiveTail: true, rows: [{{ dataset: {{ role: "assistant", assistantDedupeKey: "final_response|same final text" }} }}] }}
            );
            const adjacentFalseOffTail = runtime.isAdjacentAssistantDuplicateEvent(
              {{ role: "assistant", text: "same final text", message_class: "final_response", ts: 5 }},
              {{ renderedAtLiveTail: false, rows: [{{ dataset: {{ role: "assistant", assistantDedupeKey: "final_response|same final text" }} }}] }}
            );
            const id1 = runtime.nextLocalEchoId();
            runtime.addPendingUser({{ id: id1, sessionId: "sid", epoch: 1, text: "hello  ", t0: 10 }});
            runtime.addPendingUser({{ sessionId: "sid", epoch: 1, text: "later", t0: 12 }});
            runtime.addPendingUser({{ sessionId: "sid", epoch: 2, text: "other epoch", t0: 8 }});
            const pendingEpoch1 = runtime.pendingUsersForSession("sid", 1).map((item) => [item.id, item.text, item.epoch]);
            const exactMatch = runtime.takePendingUserMatch({{ role: "user", text: "hello", ts: 10.2 }}, "sid", 1);
            const hasAfterExact = runtime.hasPendingForSession("sid");
            const noUntimed = runtime.takePendingUserMatch({{ role: "user", text: "unrelated" }}, "sid", 1, {{ allowUntimedCommit: false }});
            const fallbackTimed = runtime.takePendingUserMatch({{ role: "user", text: "unrelated", ts: 20 }}, "sid", 1);
            const dropped = runtime.dropPendingUsers("sid", (item) => item.epoch === 2);
            const finalSnapshot = runtime.snapshot();
            let missingError = "";
            try {{ ctx.window.CodoxearTranscript.createTranscriptEventRuntime({{ eventKey: () => "" }}); }} catch (err) {{ missingError = err && err.message ? err.message : String(err); }}
            process.stdout.write(JSON.stringify({{
              firstSeen,
              firstDuplicate,
              afterEvict,
              adjacentTrue,
              adjacentFalseOffTail,
              id1,
              pendingEpoch1,
              exactMatch: exactMatch && {{ id: exactMatch.id, text: exactMatch.text, epoch: exactMatch.epoch }},
              hasAfterExact,
              noUntimed,
              fallbackTimed: fallbackTimed && {{ text: fallbackTimed.text, epoch: fallbackTimed.epoch }},
              dropped: dropped.map((item) => item.text),
              finalSnapshot,
              missingError,
              frozen: Object.isFrozen(runtime),
            }}));
            """
        )
        out = _run_node(js)
        self.assertTrue(out["firstSeen"])
        self.assertTrue(out["firstDuplicate"])
        self.assertEqual(out["afterEvict"], {"keys": ["assistant|2000|two", "assistant|3000|three"], "evictedDuplicate": False})
        self.assertTrue(out["adjacentTrue"])
        self.assertFalse(out["adjacentFalseOffTail"])
        self.assertEqual(out["id1"], 1)
        self.assertEqual(out["pendingEpoch1"], [[1, "hello  ", 1], [2, "later", 1]])
        self.assertEqual(out["exactMatch"], {"id": 1, "text": "hello  ", "epoch": 1})
        self.assertTrue(out["hasAfterExact"])
        self.assertFalse(out["noUntimed"])
        self.assertEqual(out["fallbackTimed"], {"text": "later", "epoch": 1})
        self.assertEqual(out["dropped"], ["other epoch"])
        self.assertEqual(out["finalSnapshot"]["pendingCount"], 0)
        self.assertContains("transcript dependency missing: pendingMatchKey", out["missingError"])
        self.assertTrue(out["frozen"])

    def test_older_load_runtime_owns_state_currentness_and_ui_projection(self) -> None:
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const ctx = {{ window: {{}}, AbortController }};
            const vm = require("vm");
            vm.createContext(ctx);
            vm.runInContext({json.dumps(transcript_source)}, ctx);
            let now = 1000;
            const olderWrap = {{ style: {{ display: "" }} }};
            const olderButton = {{ disabled: false, textContent: "" }};
            const olderError = {{ style: {{ display: "" }} }};
            const olderErrorText = {{ textContent: "" }};
            const runtime = ctx.window.CodoxearTranscript.createOlderLoadRuntime({{
              olderWrap,
              olderButton,
              olderError,
              olderErrorText,
              AbortControllerCtor: AbortController,
              nowMs: () => now,
              autoCooldownMs: 450,
            }});
            const initial = runtime.snapshot();
            runtime.setState({{ hasMore: true, isLoading: false }});
            const visible = {{ wrap: olderWrap.style.display, disabled: olderButton.disabled, text: olderButton.textContent }};
            const firstAuto = runtime.markAutoTrigger();
            const secondAuto = runtime.markAutoTrigger();
            now += 500;
            const thirdAuto = runtime.markAutoTrigger();
            const load = runtime.beginLoad({{ cancelOnScroll: true }});
            const loading = {{ snapshot: runtime.snapshot(), wrap: olderWrap.style.display, disabled: olderButton.disabled, text: olderButton.textContent }};
            const currentBeforeInvalidate = runtime.isCurrent(load);
            const cancelBeforeInvalidate = runtime.shouldCancelOnScroll();
            runtime.invalidate();
            const currentAfterInvalidate = runtime.isCurrent(load);
            const afterInvalidate = runtime.snapshot();
            runtime.showError();
            const errorShown = {{ display: olderError.style.display, text: olderErrorText.textContent }};
            runtime.setState({{ hasMore: false, isLoading: false }});
            const afterHide = {{ snapshot: runtime.snapshot(), wrap: olderWrap.style.display, error: olderError.style.display, errorText: olderErrorText.textContent }};
            let missingError = "";
            try {{ ctx.window.CodoxearTranscript.createOlderLoadRuntime({{ olderWrap }}); }} catch (err) {{ missingError = err && err.message ? err.message : String(err); }}
            process.stdout.write(JSON.stringify({{
              initial,
              visible,
              firstAuto,
              secondAuto,
              thirdAuto,
              loading,
              currentBeforeInvalidate,
              cancelBeforeInvalidate,
              currentAfterInvalidate,
              afterInvalidate,
              errorShown,
              afterHide,
              missingError,
              frozen: Object.isFrozen(runtime),
            }}));
            """
        )
        out = _run_node(js)
        self.assertEqual(out["initial"], {"hasMore": False, "isLoading": False, "requestId": 0, "cancelOnScroll": True, "hasController": False})
        self.assertEqual(out["visible"], {"wrap": "flex", "disabled": False, "text": "Load older messages"})
        self.assertTrue(out["firstAuto"])
        self.assertFalse(out["secondAuto"])
        self.assertTrue(out["thirdAuto"])
        self.assertTrue(out["loading"]["snapshot"]["isLoading"])
        self.assertTrue(out["loading"]["snapshot"]["hasController"])
        self.assertEqual(out["loading"]["text"], "Loading...")
        self.assertTrue(out["currentBeforeInvalidate"])
        self.assertTrue(out["cancelBeforeInvalidate"])
        self.assertFalse(out["currentAfterInvalidate"])
        self.assertFalse(out["afterInvalidate"]["isLoading"])
        self.assertFalse(out["afterInvalidate"]["hasController"])
        self.assertEqual(out["errorShown"], {"display": "flex", "text": "Couldn’t load older messages."})
        self.assertEqual(out["afterHide"]["wrap"], "none")
        self.assertEqual(out["afterHide"]["error"], "none")
        self.assertEqual(out["afterHide"]["errorText"], "")
        self.assertContains("transcript dependency missing: olderButton", out["missingError"])
        self.assertTrue(out["frozen"])

    def test_loaded_chat_search_runtime_owns_open_query_matches_and_index(self) -> None:
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const ctx = {{ window: {{}} }};
            const vm = require("vm");
            vm.createContext(ctx);
            vm.runInContext({json.dumps(transcript_source)}, ctx);
            const runtime = ctx.window.CodoxearTranscript.createLoadedChatSearchRuntime();
            const rowA = {{ dataset: {{}}, id: "a" }};
            const rowB = {{ dataset: {{}}, id: "b" }};
            const rowC = {{ dataset: {{}}, id: "c" }};
            const initial = runtime.snapshot();
            runtime.setOpen(true);
            const query = runtime.setQuery("  Needle ");
            const first = runtime.setMatches([rowA, rowB], {{ preserveCurrent: false }});
            const focused = runtime.focusIndex(1);
            const preserved = runtime.setMatches([rowB, rowC], {{ preserveCurrent: true }});
            const targetIndex = runtime.ensureTargetRow(rowA, "Needle", (x, y) => x.id.localeCompare(y.id));
            runtime.setLoadingOlder(true);
            const loading = runtime.snapshot();
            runtime.reset();
            const reset = runtime.snapshot();
            process.stdout.write(JSON.stringify({{
              initial,
              query,
              first: {{ index: first.index, ids: first.matches.map((r) => r.id) }},
              focused: {{ index: focused.index, row: focused.row.id, ids: focused.matches.map((r) => r.id) }},
              preserved: {{ index: preserved.index, ids: preserved.matches.map((r) => r.id) }},
              targetIndex,
              rowAForcedQuery: rowA.dataset.searchForcedQuery,
              loading: {{ open: loading.open, query: loading.query, index: loading.index, ids: loading.matches.map((r) => r.id), loadingOlder: loading.loadingOlder }},
              reset,
              frozen: Object.isFrozen(runtime),
            }}));
            """
        )
        out = _run_node(js)
        self.assertEqual(out["initial"], {"open": False, "query": "", "matches": [], "index": -1, "loadingOlder": False})
        self.assertEqual(out["query"], "needle")
        self.assertEqual(out["first"], {"index": 0, "ids": ["a", "b"]})
        self.assertEqual(out["focused"], {"index": 1, "row": "b", "ids": ["a", "b"]})
        self.assertEqual(out["preserved"], {"index": 0, "ids": ["b", "c"]})
        self.assertEqual(out["targetIndex"], 0)
        self.assertEqual(out["rowAForcedQuery"], "needle")
        self.assertEqual(out["loading"], {"open": True, "query": "needle", "index": 0, "ids": ["a", "b", "c"], "loadingOlder": True})
        self.assertEqual(out["reset"], {"open": False, "query": "", "matches": [], "index": -1, "loadingOlder": False})
        self.assertTrue(out["frozen"])

    def test_chat_search_all_runtime_owns_debounce_currentness_and_result_state(self) -> None:
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const ctx = {{ window: {{}}, AbortController }};
            const vm = require("vm");
            vm.createContext(ctx);
            vm.runInContext({json.dumps(transcript_source)}, ctx);
            const timers = [];
            const cleared = [];
            const runtime = ctx.window.CodoxearTranscript.createChatSearchAllRuntime({{
              setTimeout: (fn, ms) => {{ const timer = {{ fn, ms, id: timers.length + 1 }}; timers.push(timer); return timer; }},
              clearTimeout: (timer) => cleared.push(timer && timer.id),
              AbortControllerCtor: AbortController,
              debounceMs: 300,
            }});
            const empty = runtime.schedule("", () => {{ throw new Error("empty should not run"); }});
            const scheduled = runtime.schedule(" query ", (q) => {{ ctx.ranQuery = q; }});
            const scheduledSnapshot = runtime.snapshot();
            timers[timers.length - 1].fn();
            const request1 = runtime.beginRequest();
            const current1 = runtime.isCurrent(request1);
            const request2 = runtime.beginRequest();
            const oldCurrentAfterSecond = runtime.isCurrent(request1);
            const completedOld = runtime.completeRequest(request1, {{ count: 99, truncated: true, hint: "stale" }});
            const completedNew = runtime.completeRequest(request2, {{ count: "5", truncated: true, hint: "first match" }});
            const afterComplete = runtime.snapshot();
            runtime.finishRequest(request2);
            const afterFinish = runtime.snapshot();
            const request3 = runtime.beginRequest();
            runtime.failRequest(request3);
            const afterFail = runtime.snapshot();
            runtime.schedule("later", () => {{ ctx.laterRan = true; }});
            const beforeDispose = runtime.snapshot();
            runtime.dispose();
            const afterDispose = runtime.snapshot();
            let missingError = "";
            try {{ ctx.window.CodoxearTranscript.createChatSearchAllRuntime({{ setTimeout: () => {{}} }}); }} catch (err) {{ missingError = err && err.message ? err.message : String(err); }}
            process.stdout.write(JSON.stringify({{
              empty,
              scheduled,
              scheduledSnapshot,
              ranQuery: ctx.ranQuery,
              current1,
              oldCurrentAfterSecond,
              completedOld,
              completedNew,
              afterComplete,
              afterFinish,
              afterFail,
              beforeDispose,
              afterDispose,
              cleared,
              missingError,
              frozen: Object.isFrozen(runtime),
            }}));
            """
        )
        out = _run_node(js)
        self.assertFalse(out["empty"]["scheduled"])
        self.assertTrue(out["scheduled"]["scheduled"])
        self.assertTrue(out["scheduledSnapshot"]["hasTimer"])
        self.assertEqual(out["ranQuery"], "query")
        self.assertTrue(out["current1"])
        self.assertFalse(out["oldCurrentAfterSecond"])
        self.assertFalse(out["completedOld"])
        self.assertTrue(out["completedNew"])
        self.assertEqual(out["afterComplete"]["count"], 5)
        self.assertTrue(out["afterComplete"]["truncated"])
        self.assertEqual(out["afterComplete"]["hint"], "first match")
        self.assertFalse(out["afterFinish"]["hasAbort"])
        self.assertIsNone(out["afterFail"]["count"])
        self.assertFalse(out["afterFail"]["truncated"])
        self.assertEqual(out["afterFail"]["hint"], "")
        self.assertTrue(out["beforeDispose"]["hasTimer"])
        self.assertIsNone(out["afterDispose"]["count"])
        self.assertFalse(out["afterDispose"]["hasAbort"])
        self.assertFalse(out["afterDispose"]["hasTimer"])
        self.assertContains(2, out["cleared"])
        self.assertContains("transcript dependency missing: clearTimeout", out["missingError"])
        self.assertTrue(out["frozen"])

    def test_transcript_module_normalizes_and_trims_tail_cache(self) -> None:
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const ctx = {{ window: {{}} }};
            const vm = require("vm");
            vm.createContext(ctx);
            vm.runInContext({json.dumps(transcript_source)}, ctx);
            const tx = ctx.window.CodoxearTranscript;
            const tailCache = new Map();
            const sessionIndex = new Map([["sid", {{ thread_id: "session-thread", log_path: "/session.jsonl" }}]]);
            tx.rememberTailSnapshot(tailCache, "sid", {{ thread_id: "fallback-thread", log_path: "/fallback.jsonl" }}, {{
              transcript_state: "bound",
              thread_id: "tail-thread",
              log_path: "/tail.jsonl",
              live_cursor: "c1",
              has_older: true,
              busy: true,
              queue_len: "2",
              token: {{ pct: 10 }},
              events: [
                {{ role: "system", text: "skip" }},
                {{ role: "user", text: "one", ts: 1, history_cursor: "h1" }},
                {{ role: "assistant", text: "two", message_id: "m2" }},
                {{ role: "assistant", text: "three" }},
              ],
            }}, 2);
            tx.appendTailSnapshotEvents(tailCache, sessionIndex, "sid", [{{ role: "user", text: "four" }}, {{ role: "assistant", text: "" }}], {{ maxEvents: 2, identityData: {{}} }});
            const afterAppend = tailCache.get("sid");
            tx.rememberTailSnapshot(tailCache, "sid", {{ thread_id: "fallback-thread", log_path: "/fallback.jsonl" }}, {{ transcript_state: "pending_bind" }}, 2);
            process.stdout.write(JSON.stringify({{
              afterAppend,
              deleted: !tailCache.has("sid"),
              key: tx.transcriptKey("thread", "/log"),
              failedState: tx.normalizeTranscriptState({{ transcript_state: "failed" }}),
              frozen: Object.isFrozen(tx),
            }}));
            """
        )
        out = _run_node(js)
        self.assertEqual(out["afterAppend"]["threadId"], "session-thread")
        self.assertEqual(out["afterAppend"]["logPath"], "/session.jsonl")
        self.assertEqual([ev["text"] for ev in out["afterAppend"]["events"]], ["three", "four"])
        self.assertEqual(out["afterAppend"]["queueLen"], 2)
        self.assertEqual(out["afterAppend"]["historyCursor"], "h1")
        self.assertTrue(out["afterAppend"]["hasOlder"])
        self.assertTrue(out["afterAppend"]["busy"])
        self.assertTrue(out["deleted"])
        self.assertEqual(out["key"], "thread\n/log")
        self.assertEqual(out["failedState"], "failed")
        self.assertTrue(out["frozen"])

    def test_recovered_failed_tail_can_page_when_history_cursor_exists(self) -> None:
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const ctx = {{ window: {{}} }};
            const vm = require("vm");
            vm.createContext(ctx);
            vm.runInContext({json.dumps(transcript_source)}, ctx);
            const tx = ctx.window.CodoxearTranscript;
            const recoveredTopLevel = {{ transcript_state: "failed", has_older: true, history_cursor: "top-cursor", events: [] }};
            const recoveredEventCursor = {{ transcript_state: "failed", has_older: true, events: [{{ role: "assistant", text: "backend stopped", history_cursor: "row-cursor" }}] }};
            const preLogFailed = {{ transcript_state: "failed", has_older: true, events: [{{ role: "assistant", text: "launch failed" }}] }};
            const noOlder = {{ transcript_state: "failed", has_older: false, history_cursor: "top-cursor", events: [] }};
            process.stdout.write(JSON.stringify({{
              topCursor: tx.historyCursorFromPayload(recoveredTopLevel),
              eventCursor: tx.historyCursorFromPayload(recoveredEventCursor),
              topUsable: tx.hasUsableOlderHistory(recoveredTopLevel),
              eventUsable: tx.hasUsableOlderHistory(recoveredEventCursor),
              preLogUsable: tx.hasUsableOlderHistory(preLogFailed),
              noOlderUsable: tx.hasUsableOlderHistory(noOlder),
              frozen: Object.isFrozen(tx),
            }}));
            """
        )
        out = _run_node(js)

        self.assertEqual(out["topCursor"], "top-cursor")
        self.assertEqual(out["eventCursor"], "row-cursor")
        self.assertTrue(out["topUsable"])
        self.assertTrue(out["eventUsable"])
        self.assertFalse(out["preLogUsable"])
        self.assertFalse(out["noOlderUsable"])
        self.assertTrue(out["frozen"])


    def test_bound_log_identity_change_replaces_rendered_transcript(self) -> None:
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        lifecycle_source = APP_SESSION_LIFECYCLE_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const vm = require("vm");
            const ctx = {{ window: {{}}, console }};
            vm.createContext(ctx);
            vm.runInContext({json.dumps(transcript_source)}, ctx);
            vm.runInContext({json.dumps(APP_SESSION_STATE_JS.read_text(encoding="utf-8"))}, ctx);
            vm.runInContext({json.dumps(APP_SESSION_CATALOG_JS.read_text(encoding="utf-8"))}, ctx);
            vm.runInContext({json.dumps(lifecycle_source)}, ctx);
            const calls = [];
            const state = {{
              generation: 0,
              responses: [
                // A: direct rebind — new log identity, empty new transcript.
                {{ transcript_state: "bound", thread_id: "t2", log_path: "/new.jsonl", events: [], busy: false, queue_len: 0, token: null }},
                // B: ordinary same-log reload — must preserve the DOM.
                {{ transcript_state: "bound", thread_id: "t2", log_path: "/new.jsonl", events: [{{ role: "user", text: "hello" }}], busy: false, queue_len: 0, token: null }},
                // C: transient rebind state — no proof, must preserve the DOM.
                {{ transcript_state: "pending_bind", events: [], busy: false, queue_len: 0, token: null }},
                // D: recovery to the same rendered log — must preserve the DOM.
                {{ transcript_state: "bound", thread_id: "t2", log_path: "/new.jsonl", events: [{{ role: "user", text: "hello" }}], busy: false, queue_len: 0, token: null }},
                // E (after beginRenewal): stale pre-renewal bind — ignored entirely.
                {{ transcript_state: "bound", thread_id: "t2", log_path: "/new.jsonl", events: [{{ role: "user", text: "stale" }}], busy: false, queue_len: 0, token: null }},
                // F: the renewal's fresh bind — replaces the rendered transcript.
                {{ transcript_state: "bound", thread_id: "t3", log_path: "/newer.jsonl", events: [{{ role: "user", text: "renewed" }}, {{ role: "assistant", text: "reply" }}], busy: false, queue_len: 0, token: null }},
              ],
            }};
            const slotRuntime = ctx.window.CodoxearTranscript.createTranscriptSlotRuntime({{
              sessionIndex: new Map([["sid", {{ thread_id: "t1", log_path: "/old.jsonl" }}]]),
            }});
            slotRuntime.updateSlot("sid", {{ transcript_state: "bound", thread_id: "t1", log_path: "/old.jsonl" }});
            const messageFlow = {{
              prepareSessionOpen: () => {{}},
              beginOpenSessionTailRequest: (sessionId, generation) => ({{ sessionId, generation, signal: {{}} }}),
              isOpenSessionTailAbortError: () => false,
              isCurrentOpenSessionTailRequest: () => true,
              finishOpenSessionTailRequest: () => {{}},
              markMessagePollFailure: () => {{}},
              markMessagePollSuccess: () => {{}},
            }};
            const defaults = () => {{}};
            const sessionState = ctx.window.CodoxearSessionState.createSessionState({{ consoleError: () => {{}} }});
            sessionState.set("selected", "sid");
            const sessionCatalog = ctx.window.CodoxearSessionCatalog.createSessionCatalog({{ consoleError: () => {{}} }});
            const options = new Proxy({{
              sessionState, sessionCatalog, backendSupportsFastForDefaults: () => false,
              asyncEpoch: {{
                currentGeneration: () => state.generation,
                nextGeneration: () => ++state.generation,
                incrementGeneration: () => ++state.generation,
              }},
                                          resetTranscriptForSession: () => calls.push("reset-transcript"),
              resetChatRenderState: () => calls.push("reset-chat"),
              getSession: () => ({{ session_id: "sid", busy: false, queue_len: 0, token: null }}),
              isCurrent: () => true,
              beginFileViewerSync: () => false,
              getTailCache: () => null,
              tailCacheMatchesSession: () => false,
              renderTranscriptLoading: () => calls.push("render-loading"),
              messageFlow: () => messageFlow,
              api: async () => state.responses.shift(),
              initPageLimit: () => 60,
              refreshSessions: async () => [],
              isDisposed: () => true,
              messagePollDelayMs: () => 900,
              updateTranscriptSlot: (sessionId, data) => slotRuntime.updateSlot(sessionId, data),
              invalidateOlderLoad: () => calls.push("invalidate-older"),
              renderPendingTranscriptSlot: () => calls.push("render-pending"),
              replaceWith: (events) => calls.push(["replace-with", events.length]),
              renderSessionTail: () => calls.push("render-tail"),
              restoreSessionScrollPosition: () => calls.push("restore-scroll"),
              sessionIdFromHash: () => "",
              sessionSelectable: () => false,
              normalizeAgentBackendName: (value) => value,
              providerChoiceToSettings: () => ({{}}),
              backendSupportsFast: () => false,
              confirmAction: async () => false,
              sleep: async () => {{}},
              isUnattendedOpen: () => false,
              isMobile: () => false,
            }}, {{ get: (target, name) => name in target ? target[name] : defaults }});
            const controller = ctx.window.CodoxearSessionLifecycle.createSessionLifecycleController(options);
            (async () => {{
              await controller.openSession("sid", {{ useCache: false }});
              await controller.openSession("sid", {{ useCache: false }});
              await controller.openSession("sid", {{ useCache: false }});
              await controller.openSession("sid", {{ useCache: false }});
              slotRuntime.beginRenewal("sid");
              await controller.openSession("sid", {{ useCache: false }});
              await controller.openSession("sid", {{ useCache: false }});
              process.stdout.write(JSON.stringify({{ calls, slot: slotRuntime.getSlot("sid") }}));
            }})().catch((error) => {{ console.error(error); process.exit(1); }});
            """
        )
        out = _run_node(js)

        # Replacement renders happen exactly at the two identity boundaries:
        # the direct rebind (empty new tail) and the renewal's fresh bind.
        self.assertEqual(out["calls"].count(["replace-with", 0]), 1)
        self.assertEqual(out["calls"].count(["replace-with", 2]), 1)
        # Same-log reloads, the transient pending_bind, and the ignored stale
        # bind never replace the DOM.
        self.assertEqual(len([call for call in out["calls"] if isinstance(call, list) and call[0] == "replace-with"]), 2)
        self.assertNotIn("render-tail", out["calls"])
        self.assertNotIn("render-pending", out["calls"])
        # Older-load state is invalidated only at the replacement boundaries.
        self.assertEqual(out["calls"].count("invalidate-older"), 2)
        # Replacement renders at the new tail; the old scroll position is not restored.
        self.assertNotIn("restore-scroll", out["calls"])
        # Same-session reloads still avoid the fresh-selection reset paths.
        self.assertNotIn("reset-transcript", out["calls"])
        self.assertNotIn("reset-chat", out["calls"])
        # The slot tracks the renewal's fresh bind honestly.
        self.assertEqual(out["slot"]["state"], "bound")
        self.assertEqual(out["slot"]["key"], "t3\n/newer.jsonl")

    def test_same_session_reload_preserves_rows_unless_forced_to_render_tail(self) -> None:
        lifecycle_source = APP_SESSION_LIFECYCLE_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const vm = require("vm");
            const ctx = {{ window: {{}}, console }};
            vm.createContext(ctx);
            vm.runInContext({json.dumps(APP_SESSION_STATE_JS.read_text(encoding="utf-8"))}, ctx);
            vm.runInContext({json.dumps(APP_SESSION_CATALOG_JS.read_text(encoding="utf-8"))}, ctx);
            vm.runInContext({json.dumps(lifecycle_source)}, ctx);
            const calls = [];
            const state = {{ generation: 0, responses: [new Error("tail unavailable"), {{ transcript_state: "bound", events: [], busy: false, queue_len: 0, token: null }}, {{ transcript_state: "bound", events: [{{ role: "assistant", text: "latest" }}], busy: false, queue_len: 0, token: null }}] }};
            const sessionState = ctx.window.CodoxearSessionState.createSessionState({{ consoleError: () => {{}} }});
            sessionState.set("selected", "sid");
            const messageFlow = {{
              prepareSessionOpen: () => calls.push("prepare"),
              beginOpenSessionTailRequest: (sessionId, generation) => ({{ sessionId, generation, signal: {{}} }}),
              isOpenSessionTailAbortError: () => false,
              isCurrentOpenSessionTailRequest: () => true,
              finishOpenSessionTailRequest: () => {{}},
              markMessagePollFailure: () => calls.push("poll-failure"),
              markMessagePollSuccess: () => calls.push("poll-success"),
            }};
            const defaults = () => {{}};
            const sessionCatalog = ctx.window.CodoxearSessionCatalog.createSessionCatalog({{ consoleError: () => {{}} }});
            const options = new Proxy({{
              sessionState, sessionCatalog, backendSupportsFastForDefaults: () => false,
              asyncEpoch: {{
                currentGeneration: () => state.generation,
                nextGeneration: () => ++state.generation,
                incrementGeneration: () => ++state.generation,
              }},
              prepareSessionOpen: messageFlow.prepareSessionOpen,
                                          setActiveSession: defaults, saveComposerDraft: defaults, loadComposerDraft: defaults,
              closeUnattendedForOtherSession: defaults, persistSelected: defaults, removePersistedSelected: defaults, setSessionHash: defaults,
              resetTranscriptForSession: () => calls.push("reset-transcript"),
              clearTranscriptForRemovedSession: defaults, syncAttachments: defaults, clearAttachments: defaults, syncAttachmentButton: defaults,
              updateQueueBadge: defaults, setStatus: defaults, setContext: defaults, setTyping: defaults,
              resetChatRenderState: () => calls.push("reset-chat"),
              getSession: () => ({{ session_id: "sid", busy: false, queue_len: 0, token: null }}),
              isCurrent: () => true, setTitle: defaults, setNoSessionTitle: defaults, markClickLoad: defaults,
              updateTypingStats: defaults, beginFileViewerSync: () => false, finishFileViewerSync: defaults,
              handleFileViewerSessionUnavailable: defaults, getTailCache: () => null, tailCacheMatchesSession: () => false,
              applyCachedTail: defaults, renderTranscriptLoading: () => calls.push("render-loading"), messageFlow: () => messageFlow,
              api: async () => {{ const response = state.responses.shift(); if (response instanceof Error) throw response; return response; }},
              initPageLimit: () => 60, handleAuthLoss: defaults, refreshSessions: async () => [], isDisposed: () => true,
              kickPoll: defaults, messagePollDelayMs: () => 900,
              updateTranscriptSlot: () => ({{ ignoredStaleBound: false, previous: {{ state: "bound", key: "k1" }}, current: {{ state: "bound", key: "k1" }} }}),
              renderPendingTranscriptSlot: () => calls.push("render-pending"), applySessionRuntimeFromTail: defaults,
              renderSessionTail: () => calls.push("render-tail"), openMessageEventSource: defaults,
              isMobile: () => false, closeSidebar: defaults, updateUnattendedButton: defaults, refreshFileCandidates: defaults,
              isUnattendedOpen: () => false, hideUnattendedMenu: defaults, syncComposerSendButton: defaults, syncQueueSubmitState: defaults,
              saveSessionScrollPosition: defaults, restoreSessionScrollPosition: () => calls.push("restore-scroll"), clearSessionScrollPosition: defaults,
              setActiveTranscriptPending: defaults, deleteTranscriptSession: defaults, dropPendingUserRows: defaults,
              sessionIdFromHash: () => "", rememberPendingHashSession: defaults, sessionSelectable: () => false,
              normalizeAgentBackendName: (value) => value, providerChoiceToSettings: () => ({{}}), backendSupportsFast: () => false,
              setToast: defaults, confirmAction: async () => false, syncRecoveryUiForSession: defaults, sleep: async () => {{}}, consoleError: defaults,
              renderTranscriptLoadError: (_sessionId, _error, options) => calls.push(["render-load-error", Boolean(options.preserveTranscript)]),
            }}, {{ get: (target, name) => name in target ? target[name] : defaults }});
            const controller = ctx.window.CodoxearSessionLifecycle.createSessionLifecycleController(options);
            (async () => {{
              await controller.openSession("sid", {{ useCache: false }});
              await controller.openSession("sid", {{ useCache: false }});
              await controller.openSession("sid", {{ useCache: false, forceRender: true }});
              process.stdout.write(JSON.stringify({{ calls }}));
            }})().catch((error) => {{ console.error(error); process.exit(1); }});
            """
        )
        out = _run_node(js)

        self.assertNotIn("reset-transcript", out["calls"])
        self.assertNotIn("reset-chat", out["calls"])
        self.assertNotIn("render-loading", out["calls"])
        self.assertEqual(out["calls"].count("render-tail"), 1)
        self.assertNotIn("restore-scroll", out["calls"])
        self.assertIn(["render-load-error", True], out["calls"])


        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const ctx = {{ window: {{}} }};
            const vm = require("vm");
            vm.createContext(ctx);
            vm.runInContext({json.dumps(transcript_source)}, ctx);
            const tx = ctx.window.CodoxearTranscript;
            const sessionIndex = new Map([["sid", {{ thread_id: "new-thread", log_path: "/new.jsonl" }}]]);
            const runtime = tx.createTranscriptSlotRuntime({{ sessionIndex, maxTailEvents: 2 }});
            runtime.updateSlot("sid", {{ transcript_state: "bound", thread_id: "old-thread", log_path: "/old.jsonl" }});
            runtime.syncActiveSlot("sid");
            const before = runtime.getSlot("sid");
            const renewal = runtime.beginRenewal("sid");
            runtime.syncActiveSlot("sid");
            const pending = runtime.getSlot("sid");
            const stale = runtime.updateSlot("sid", {{ transcript_state: "bound", thread_id: "old-thread", log_path: "/old.jsonl" }});
            runtime.syncActiveSlot("sid");
            const afterStale = runtime.getSlot("sid");
            const fresh = runtime.updateSlot("sid", {{ transcript_state: "bound", thread_id: "new-thread", log_path: "/new.jsonl" }});
            runtime.syncActiveSlot("sid");
            const afterFresh = runtime.getSlot("sid");
            runtime.setLiveCursor("cursor-1");
            const activeWithCursor = runtime.activeSnapshot();
            runtime.rememberTail("sid", {{ thread_id: "new-thread", log_path: "/new.jsonl" }}, {{
              transcript_state: "bound",
              thread_id: "new-thread",
              log_path: "/new.jsonl",
              live_cursor: "tail-cursor",
              events: [
                {{ role: "user", text: "one" }},
                {{ role: "assistant", text: "two" }},
                {{ role: "assistant", text: "three" }},
              ],
              busy: true,
              queue_len: 1,
              token: {{ pct: 50 }},
            }});
            const cached = runtime.getTailCache("sid");
            runtime.appendTailEvents("sid", [{{ role: "user", text: "four" }}], {{ liveCursor: "live-2", busy: false, queueLen: 2, token: {{ pct: 49 }} }});
            const afterAppend = runtime.getTailCache("sid");
            const matchesSession = runtime.tailCacheMatchesSession(afterAppend, {{ thread_id: "new-thread", log_path: "/new.jsonl" }});
            const beforeDelete = runtime.snapshot();
            runtime.deleteSession("sid");
            const afterDelete = runtime.snapshot();
            runtime.setActiveFailed();
            const failed = runtime.activeSnapshot();
            process.stdout.write(JSON.stringify({{
              before,
              renewal,
              pending,
              staleIgnored: stale.ignoredStaleBound,
              afterStale,
              freshIgnored: fresh.ignoredStaleBound,
              afterFresh,
              active: activeWithCursor,
              cachedTexts: cached.events.map((ev) => ev.text),
              afterAppendTexts: afterAppend.events.map((ev) => ev.text),
              afterAppendCursor: afterAppend.liveCursor,
              afterAppendQueue: afterAppend.queueLen,
              matchesSession,
              beforeDelete,
              afterDelete,
              failed,
              frozen: Object.isFrozen(runtime),
            }}));
            """
        )
        out = _run_node(js)

        self.assertEqual(out["before"]["state"], "bound")
        self.assertEqual(out["before"]["key"], "old-thread\n/old.jsonl")
        self.assertEqual(out["renewal"]["current"]["ignoredKey"], "old-thread\n/old.jsonl")
        self.assertEqual(out["pending"]["state"], "pending_bind")
        self.assertTrue(out["staleIgnored"])
        self.assertEqual(out["afterStale"]["state"], "pending_bind")
        self.assertFalse(out["freshIgnored"])
        self.assertEqual(out["afterFresh"]["state"], "bound")
        self.assertEqual(out["afterFresh"]["key"], "new-thread\n/new.jsonl")
        self.assertEqual(out["active"]["liveCursor"], "cursor-1")
        self.assertEqual(out["cachedTexts"], ["two", "three"])
        self.assertEqual(out["afterAppendTexts"], ["three", "four"])
        self.assertEqual(out["afterAppendCursor"], "live-2")
        self.assertEqual(out["afterAppendQueue"], 2)
        self.assertTrue(out["matchesSession"])
        self.assertEqual(out["beforeDelete"]["slotCount"], 1)
        self.assertEqual(out["beforeDelete"]["tailCacheCount"], 1)
        self.assertEqual(out["afterDelete"]["slotCount"], 0)
        self.assertEqual(out["afterDelete"]["tailCacheCount"], 0)
        self.assertEqual(out["failed"]["state"], "failed")
        self.assertTrue(out["frozen"])

    def test_transient_pending_snapshot_can_rebind_current_log(self) -> None:
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const ctx = {{ window: {{}} }};
            const vm = require("vm");
            vm.createContext(ctx);
            vm.runInContext({json.dumps(transcript_source)}, ctx);
            const runtime = ctx.window.CodoxearTranscript.createTranscriptSlotRuntime();
            runtime.updateSlot("sid", {{ transcript_state: "bound", thread_id: "thread-a", log_path: "/log-a.jsonl" }});
            const pending = runtime.updateSlot("sid", {{ transcript_state: "pending_bind" }});
            const rebound = runtime.updateSlot("sid", {{ transcript_state: "bound", thread_id: "thread-a", log_path: "/log-a.jsonl" }});
            process.stdout.write(JSON.stringify({{ pending, rebound, slot: runtime.getSlot("sid") }}));
            """
        )
        out = _run_node(js)

        self.assertTrue(out["pending"]["resetPending"])
        self.assertIsNone(out["pending"]["current"]["ignoredKey"])
        self.assertFalse(out["rebound"]["ignoredStaleBound"])
        self.assertEqual(out["slot"]["state"], "bound")
        self.assertEqual(out["slot"]["key"], "thread-a\n/log-a.jsonl")

    def test_normalized_transcript_events_filters_dedupes_and_consumes_pending(self) -> None:
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const ctx = {{ window: {{}} }};
            const vm = require("vm");
            vm.createContext(ctx);
            vm.runInContext({json.dumps(transcript_source)}, ctx);
            const tx = ctx.window.CodoxearTranscript;
            const consumed = [];
            const events = [
              null,
              {{ role: "system", text: "skip" }},
              {{ role: "user", text: "one", id: "u1" }},
              {{ role: "user", text: "one duplicate", id: "u1" }},
              {{ role: "assistant", text: "two", id: "a2" }},
              {{ role: "assistant", text: "no key" }},
              {{ role: "assistant", text: "no key again" }},
            ];
            const normalized = tx.normalizedTranscriptEvents(events, {{
              consumePending: true,
              selectedSessionId: "sid",
              eventKey: (ev) => ev.id || "",
              takePendingMatch: (ev, sid, opts) => consumed.push([ev.text, sid, opts.allowUntimedCommit]),
            }});
            const withoutConsume = tx.normalizedTranscriptEvents(events, {{
              eventKey: (ev) => ev.id || "",
            }});
            let missingKey = false;
            try {{ tx.normalizedTranscriptEvents(events, {{}}); }} catch (err) {{ missingKey = /eventKey/.test(String(err && err.message || err)); }}
            let missingTake = false;
            try {{ tx.normalizedTranscriptEvents(events, {{ consumePending: true, eventKey: () => "" }}); }} catch (err) {{ missingTake = /takePendingMatch/.test(String(err && err.message || err)); }}
            process.stdout.write(JSON.stringify({{
              normalizedTexts: normalized.map((ev) => ev.text),
              withoutConsumeTexts: withoutConsume.map((ev) => ev.text),
              consumed,
              missingKey,
              missingTake,
            }}));
            """
        )
        out = _run_node(js)

        self.assertEqual(out["normalizedTexts"], ["one", "two", "no key", "no key again"])
        self.assertEqual(out["withoutConsumeTexts"], ["one", "two", "no key", "no key again"])
        self.assertEqual(out["consumed"], [
            ["one", "sid", False],
            ["one duplicate", "sid", False],
            ["two", "sid", False],
            ["no key", "sid", False],
            ["no key again", "sid", False],
        ])
        self.assertTrue(out["missingKey"])
        self.assertTrue(out["missingTake"])

    def test_click_first_message_metric_flows_from_render_controller_through_tail_render(self) -> None:
        transcript_render_source = APP_TRANSCRIPT_RENDER_JS.read_text(encoding="utf-8")
        message_history_source = APP_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const vm = require("vm");
            const samples = [];
            let clock = 100;
            const view = {{ renders: [], replaceWith(events, options) {{ this.renders.push({{ count: events.length, options }}); }} }};
            const noop = () => {{}};
            const wiring = new Proxy({{}}, {{ get: () => (deps) => deps }});
            const ctx = {{
              window: {{
                CodoxearMessageRows: {{
                  createMessageCopyNavigationRuntime: () => ({{ reset: noop, syncTabStops: noop, setActiveRow: noop, toggleTouchRow: noop, jumpTarget: () => null }}),
                  messageCopyButtonForRow: () => null, activeElementIsCopyButton: () => false,
                  rowSearchText: () => "", compareRowsInDomOrder: () => 0,
                  loadedUserJumpTarget: () => null, firstVisibleMessageRow: () => null,
                  trimRenderedRowTargets: () => [], trimRowsBeforeViewportTargets: () => [],
                }},
                CodoxearTranscript: {{
                  createTranscriptSlotRuntime: () => ({{ activeSnapshot: () => ({{}}), getSlot: () => ({{ epoch: 0 }}), syncActiveSlot: () => ({{ state: "bound" }}), clearLiveCursor: noop, setLiveCursor: noop, updateSlot: () => ({{ resetPending: false }}), beginRenewal: noop, tailCacheMatchesSession: () => false, rememberTail: noop, appendTailEvents: noop, deleteTailCache: noop }}),
                  createTypingRowRuntime: () => ({{ setVisible: noop, setSubagentVisible: noop, updateSubagentGauge: noop, reset: noop, anchor: () => ({{}}) }}),
                  createTranscriptScrollRuntime: () => ({{ enableAutoScroll: noop, markLiveTail: noop, reset: noop, syncVisibleTimeIndicator: noop, snapshot: () => ({{ renderedAtLiveTail: true }}) }}),
                  createTranscriptDomRuntime: () => ({{ rebuildDecorations: noop, trimRenderedRows: noop, trimRowsBeforeViewport: noop, clear: noop }}),
                  createTranscriptEventRuntime: () => ({{ resetRecentEvents: noop, dropPendingUsers: () => [], pendingUsersForSession: () => [], markEventSeen: noop, isDuplicateEvent: () => false, isAdjacentAssistantDuplicateEvent: () => false, takePendingUserMatch: () => null }}),
                  createOlderLoadRuntime: () => ({{ invalidate: noop, snapshot: () => ({{ hasMore: false, isLoading: false }}) }}),
                  normalizedTranscriptEvents: (events) => events, transcriptSnapshotFromData: () => ({{}}),
                }},
                CodoxearTranscriptView: {{ createTranscriptViewController: () => view }},
                CodoxearPendingUser: {{ createPendingUserController: () => ({{ consumePendingUserIfMatches: () => false }}) }},
                CodoxearChatNavigation: {{ createChatNavigationController: () => ({{ syncButtons: noop, jumpToLoadedUserMessage: noop, jumpToLoadedMessage: noop }}) }},
                CodoxearChatSearch: {{ createChatSearchController: () => ({{ isOpen: () => false, open: noop, close: noop, refreshLoaded: noop, step: noop }}) }},
                CodoxearHintMode: {{ createHintModeController: () => ({{ isActive: () => false }}) }},
                CodoxearNavigationPulse: {{ createNavigationPulseController: () => ({{ pulseNavigatedRow: noop }}) }},
                CodoxearModal: {{ createModalKeyboardHandler: () => noop }},
                CodoxearDisplay: {{ ymd: () => "", dayLabel: () => "", time24: () => "", recoveryPromptPreview: () => "" }},
                CodoxearViewport: {{ prefersReducedMotion: () => false }},
                setTimeout: noop, matchMedia: () => ({{ matches: false }}), getSelection: () => ({{ toString: () => "" }}),
              }},
              console,
              AbortController,
            }};
            const transcriptStubs = ctx.window.CodoxearTranscript;
            vm.createContext(ctx);
            vm.runInContext({json.dumps(APP_SESSION_STATE_JS.read_text(encoding="utf-8"))}, ctx);
            vm.runInContext({json.dumps(APP_SESSION_CATALOG_JS.read_text(encoding="utf-8"))}, ctx);
            vm.runInContext({json.dumps(APP_TRANSCRIPT_JS.read_text(encoding="utf-8"))}, ctx);
            ctx.window.CodoxearTranscript = {{ ...ctx.window.CodoxearTranscript, ...transcriptStubs }};
            vm.runInContext({json.dumps(transcript_render_source)}, ctx);
            const sessionState = ctx.window.CodoxearSessionState.createSessionState({{ consoleError: noop }});
            const sessionCatalog = ctx.window.CodoxearSessionCatalog.createSessionCatalog({{ consoleError: noop }});
            const render = ctx.window.CodoxearTranscriptRender.createTranscriptRenderController({{
              currentGeneration: () => 1, sessionCatalog,
              getSessionLifecycleController: () => ({{}}), getSessionRefreshController: () => ({{}}),
              sessionState, isAppDisposed: () => false, getSessionEditController: () => null,
              getQueueController: () => null, isFileViewerOpen: () => false, upgradeCandidateFileRefs: noop,
              isMobile: () => false, refreshSessions: noop, jumpToLatest: noop,
              getHistoryController: () => ({{ olderLoadRuntime: {{ invalidate: noop, resetAutoTrigger: noop }} }}),
              getSendLifecycleController: () => ({{ messageFlowController: {{ updateTypingStatsFromSession: noop }} }}),
              getAttachmentsController: () => ({{ syncAttachButtonState: noop }}),
              CHAT_DOM_WINDOW: 60, CHAT_DOM_WINDOW_WITH_HISTORY_SLACK: 90, INIT_PAGE_LIMIT: 60,
              Node: {{}}, OLDER_CANCEL_PX: 48, OLDER_TOP_TRIGGER_PX: 1, addAppEvent: noop,
              api: async () => ({{}}), appConfirm: {{}}, bottomSentinel: {{}}, chat: {{ scrollTop: 0, clientHeight: 0, scrollBy: noop }}, chatInner: {{ contains: () => false, querySelector: () => null }},
              chatMarkdownHtmlCached: noop, chatSearchAllHintEl: {{}}, chatSearchBar: {{}}, chatSearchBtn: {{}}, chatSearchCloseBtn: {{}}, chatSearchInput: {{}}, chatSearchNextBtn: {{}}, chatSearchPrevBtn: {{}}, chatSearchStatus: {{}}, chatTimeChip: {{}}, codeBlockCopyRuntime: {{ toggleTouchPre: noop }},
              codoxearCodeCopy: {{ codePreFromTarget: () => null }}, codoxearDisplay: ctx.window.CodoxearDisplay, codoxearModal: ctx.window.CodoxearModal, codoxearNavigationPulse: ctx.window.CodoxearNavigationPulse, codoxearPendingUser: ctx.window.CodoxearPendingUser, codoxearViewport: ctx.window.CodoxearViewport,
              confirmApp: async () => false, copyToClipboard: noop, diagViewer: {{}}, document: {{ querySelectorAll: () => [], activeElement: null }}, editViewer: {{}}, el: noop, handleAppAuthLoss: noop, helpViewer: {{}}, iconSvg: noop, isModalTargetOpen: () => false, isTextEntryElement: () => false, jumpBtn: {{ style: {{}} }}, modalIsolationTargets: [], newSessionDialogController: {{ isOpen: () => false }}, nextUserBtn: {{}}, olderWrap: {{}},
              performance: {{ now: () => clock }}, prevUserBtn: {{}}, pushPerfSample: (...sample) => samples.push(sample), queueViewer: {{ style: {{ display: "none" }} }}, refreshQueueViewer: noop, requestAnimationFrame: noop, sendChoice: {{}}, sessionAgentBackend: () => "pi", setTimeout: noop, setToast: noop, textarea: {{ focus: noop }}, window: ctx.window, wiring,
            }});
            vm.runInContext({json.dumps(message_history_source)}, ctx);
            const history = ctx.window.CodoxearMessageHistory.createMessageHistoryController({{
              currentGeneration: () => 1, sessionCatalog, getSessionLifecycleController: () => ({{}}), getSessionRefreshController: () => ({{ refreshSessions: async () => [] }}), getSendLifecycleController: () => ({{ kickPoll: noop }}), getAttachmentsController: () => ({{}}),
              transcript: {{ transcriptView: render.transcriptView, markClickFirstPaint: render.markClickFirstPaint }}, sessionState,
              wiring, olderWrap: {{}}, olderBtn: {{}}, olderError: {{}}, olderErrorText: {{}}, AbortController,
              performance: {{ now: () => clock }}, OLDER_AUTO_COOLDOWN_MS: 450, OLDER_PAGE_LIMIT: 30,
              api: async () => ({{}}), handleAppAuthLoss: noop, syncQueueSubmitState: noop, syncComposerSendButton: noop, updateUnattendedBtnState: noop, updateQueueBadge: noop, sessionLaunchFailed: () => false, confirmApp: async () => false, setToast: noop, codoxearDisplay: ctx.window.CodoxearDisplay, redactedLaunchErrorText: () => "", sessionIdFromHash: () => "", sessionSelectable: () => false,
            }});
            history.renderSessionTail([{{ role: "assistant", text: "unarmed" }}]);
            const beforeArm = samples.slice();
            render.markClickLoad();
            clock = 137;
            history.renderSessionTail([{{ role: "assistant", text: "armed" }}]);
            process.stdout.write(JSON.stringify({{ beforeArm, samples, renders: view.renders }}));
            """
        )
        out = _run_node(js)

        self.assertEqual(out["beforeArm"], [])
        self.assertEqual(out["samples"], [["click_to_first_message_ms", 37]])
        self.assertEqual([render["count"] for render in out["renders"]], [1, 1])

    def test_transcript_render_runtime_owns_window_render_and_history_prepend(self) -> None:
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const ctx = {{ window: {{}} }};
            const vm = require("vm");
            vm.createContext(ctx);
            vm.runInContext({json.dumps(transcript_source)}, ctx);
            const tx = ctx.window.CodoxearTranscript;
            const calls = [];
            const bottom = {{ name: "bottom" }};
            const firstMsg = {{ name: "first", offsetTop: 20, isConnected: true }};
            const root = {{
              insertBefore(node, before) {{ calls.push(["insert", node.children ? node.children.map((row) => row.name) : node.name || "node", before && before.name]); }},
              querySelector: (selector) => selector === ".msg-row:not(.typing-row)" ? firstMsg : null,
            }};
            const scrollRuntime = {{
              shouldStickToBottom: () => true,
              snapshot: () => ({{ renderedAtLiveTail: ctx.renderedAtLiveTail }}),
              syncJumpButton: () => calls.push(["jump"]),
              scheduleScrollToBottom: () => calls.push(["scroll"]),
              markLiveTail: () => calls.push(["markLiveTail"]),
              disableAutoScroll: () => calls.push(["disableAuto"]),
              setRenderedAtLiveTail: (value) => {{ ctx.renderedAtLiveTail = value; calls.push(["liveTail", value]); }},
              setScrollTop: (value) => calls.push(["scrollTop", value]),
            }};
            const domRuntime = {{
              clear: () => calls.push(["clear"]),
              rebuildDecorations: (opts) => calls.push(["rebuild", Boolean(opts.preserveScroll)]),
              trimRenderedRows: (opts) => {{ calls.push(["trim", opts.fromTop, opts.maxRows || null]); ctx.renderedAtLiveTail = Boolean(opts.fromTop); return 1; }},
            }};
            function makeRuntime(eventsForNormalize) {{
              return tx.createTranscriptRenderRuntime({{
                root,
                bottomSentinel: bottom,
                document: {{ createDocumentFragment: () => ({{ children: [], appendChild(row) {{ this.children.push(row); }} }}) }},
                safeMakeRow: (ev) => (calls.push(["make", ev.text]), {{ row: {{ name: ev.text }}, bubble: {{}} }}),
                normalizeEvents: (events, opts) => (calls.push(["normalize", Boolean(opts.consumePending)]), eventsForNormalize || events),
                consumePendingUserIfMatches: () => false,
                isDuplicateEvent: () => false,
                isAdjacentAssistantDuplicateEvent: () => false,
                markEventSeen: (ev) => calls.push(["seen", ev.text]),
                markFirstPaint: () => calls.push(["paint"]),
                renderRecoveryPanel: (sid) => calls.push(["recovery", sid]),
                restorePendingRows: (sid) => calls.push(["restore", sid]),
                resetRecentEvents: () => calls.push(["resetRecent"]),
                setOlderState: (state) => calls.push(["older", state.hasMore, state.isLoading]),
                firstVisibleMessageRow: () => firstMsg,
                getScrollTop: () => 5,
                getSelectedSessionId: () => "sid",
                domRuntime,
                scrollRuntime,
                typingRowRuntime: {{ anchor: () => bottom }},
                historySlackRows: 99,
              }});
            }}
            ctx.renderedAtLiveTail = true;
            const runtime = makeRuntime([{{ role: "user", text: "one", ts: 1 }}, {{ role: "assistant", text: "two", ts: 2 }}]);
            const full = runtime.renderTranscript([{{ role: "user", text: "ignored" }}], {{ preserveScroll: true }});
            const afterFull = calls.slice();
            calls.length = 0;
            const detached = runtime.renderDetachedTranscriptWindow([{{ role: "assistant", text: "det" }}], {{ hasMore: true }});
            const afterDetached = calls.slice();
            calls.length = 0;
            ctx.renderedAtLiveTail = true;
            const prepended = runtime.prependOlderEvents([{{ role: "system", text: "skip" }}, {{ role: "user", text: "old" }}], {{ preserveViewport: true }});
            const afterPrepend = calls.slice();
            calls.length = 0;
            const empty = makeRuntime([]).renderTranscript([], {{ preserveScroll: false }});
            process.stdout.write(JSON.stringify({{ full, detached, prepended, empty, afterFull, afterDetached, afterPrepend, frozen: Object.isFrozen(runtime) }}));
            """
        )
        out = _run_node(js)

        self.assertTrue(out["full"])
        self.assertTrue(out["detached"])
        self.assertTrue(out["prepended"])
        self.assertFalse(out["empty"])
        self.assertEqual(out["afterFull"], [
            ["normalize", True], ["markLiveTail"], ["clear"], ["resetRecent"],
            ["seen", "one"], ["make", "one"], ["seen", "two"], ["make", "two"],
            ["insert", ["one", "two"], "bottom"], ["rebuild", True], ["restore", "sid"],
        ])
        self.assertEqual(out["afterDetached"], [
            ["normalize", False], ["disableAuto"], ["liveTail", False], ["clear"], ["older", True, False], ["resetRecent"],
            ["seen", "one"], ["make", "one"], ["seen", "two"], ["make", "two"],
            ["insert", ["one", "two"], "bottom"], ["rebuild", False], ["scrollTop", 1], ["jump"],
        ])
        self.assertEqual(out["afterPrepend"], [
            ["disableAuto"], ["make", "old"], ["insert", ["old"], "first"], ["trim", False, 99], ["disableAuto"],
            ["rebuild", False], ["scrollTop", 5], ["jump"],
        ])
        self.assertTrue(out["frozen"])

    def test_transcript_dom_runtime_owns_clear_decorate_and_trim_window(self) -> None:
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const ctx = {{ window: {{}} }};
            const vm = require("vm");
            vm.createContext(ctx);
            vm.runInContext({json.dumps(transcript_source)}, ctx);
            const tx = ctx.window.CodoxearTranscript;
            const calls = [];
            function classList(initial) {{
              const values = new Set(String(initial || "").split(/\\s+/).filter(Boolean));
              return {{
                values,
                add: (...names) => names.forEach((name) => values.add(name)),
                remove: (...names) => names.forEach((name) => values.delete(name)),
                contains: (name) => values.has(name),
              }};
            }}
            function makeNode(name, cls = "") {{
              const node = {{
                name,
                attrs: {{ class: cls }},
                children: [],
                dataset: {{}},
                classList: classList(cls),
                isConnected: true,
                appendChild(child) {{ this.children.push(child); return child; }},
                remove() {{
                  const idx = root.children.indexOf(this);
                  if (idx >= 0) root.children.splice(idx, 1);
                  this.isConnected = false;
                }},
              }};
              return node;
            }}
            const older = makeNode("older");
            const bottom = makeNode("bottom");
            const oldSep = makeNode("old-sep", "day-sep");
            const row1 = makeNode("row1", "msg-row user");
            row1.dataset.ts = "86400";
            const row2 = makeNode("row2", "msg-row user");
            row2.dataset.ts = "86520";
            const row3 = makeNode("row3", "msg-row assistant");
            row3.dataset.ts = "172800";
            const root = {{
              children: [older, oldSep, row1, row2, row3, bottom],
              appendChild(node) {{ this.children.push(node); node.isConnected = true; return node; }},
              insertBefore(node, before) {{
                const existing = this.children.indexOf(node);
                if (existing >= 0) this.children.splice(existing, 1);
                const idx = this.children.indexOf(before);
                this.children.splice(idx >= 0 ? idx : this.children.length, 0, node);
                node.isConnected = true;
                return node;
              }},
              querySelectorAll(selector) {{ return selector === ".day-sep" ? this.children.filter((node) => node.classList && node.classList.contains("day-sep")) : []; }},
            }};
            Object.defineProperty(root, "innerHTML", {{ set() {{ this.children = []; }} }});
            function fakeEl(tag, attrs = {{}}) {{
              const node = makeNode(attrs.text || attrs.class || tag, attrs.class || "");
              node.tag = tag;
              node.textContent = attrs.text || "";
              return node;
            }}
            const scrollRuntime = {{
              captureScrollPosition: () => (calls.push(["capture"]), {{ top: 10 }}),
              preserveScrollFrom: (pos) => calls.push(["preserve", pos.top]),
              snapshot: () => ({{ autoScroll: true }}),
              scheduleScrollToBottom: () => calls.push(["scroll"]),
              syncJumpButton: () => calls.push(["jump"]),
              setRenderedAtLiveTail: (value) => calls.push(["liveTail", value]),
            }};
            const runtime = tx.createTranscriptDomRuntime({{
              root,
              olderWrap: older,
              bottomSentinel: bottom,
              el: fakeEl,
              ymd: (date) => `day-${{date.getUTCDate()}}`,
              dayLabel: (date) => `Day ${{date.getUTCDate()}}`,
              getRenderedRows: () => [row1, row2, row3].filter((row) => row.isConnected),
              trimRenderedRowTargets: (rows, fromTop, maxRows, defaultRows) => {{ calls.push(["trim", fromTop, maxRows, defaultRows]); return rows.slice(0, 1); }},
              trimRowsBeforeViewportTargets: (rows, maxRows, defaultRows, viewportTop) => {{ calls.push(["trimViewport", maxRows, defaultRows, viewportTop]); return rows.slice(0, 1); }},
              scrollRuntime,
              defaultWindowRows: 4,
              afterDecorate: () => calls.push(["after"]),
            }});
            runtime.rebuildDecorations({{ preserveScroll: true }});
            const afterDecorate = {{
              children: root.children.map((node) => node.name),
              row2Grouped: row2.classList.contains("grouped"),
              row3Grouped: row3.classList.contains("grouped"),
              row1Connected: row1.isConnected,
              oldSepConnected: oldSep.isConnected,
              sepDays: root.children.filter((node) => node.classList && node.classList.contains("day-sep")).map((node) => node.dataset.day),
              calls: calls.slice(),
            }};
            calls.length = 0;
            const trimmed = runtime.trimRenderedRows({{ fromTop: true, maxRows: 2 }});
            const trimmedViewport = runtime.trimRowsBeforeViewport({{ maxRows: 3, viewportTop: 42 }});
            const afterTrim = {{ trimmed, trimmedViewport, row1Connected: row1.isConnected, row2Connected: row2.isConnected, calls: calls.slice() }};
            runtime.clear();
            let missingRoot = false;
            try {{ tx.createTranscriptDomRuntime({{ olderWrap: older, bottomSentinel: bottom, el: fakeEl, ymd: () => "", dayLabel: () => "", getRenderedRows: () => [], trimRenderedRowTargets: () => [], trimRowsBeforeViewportTargets: () => [], scrollRuntime, afterDecorate: () => {{}} }}); }} catch (err) {{ missingRoot = /root/.test(String(err && err.message || err)); }}
            process.stdout.write(JSON.stringify({{
              afterDecorate,
              afterTrim,
              afterClear: root.children.map((node) => node.name),
              missingRoot,
              frozen: Object.isFrozen(runtime),
            }}));
            """
        )
        out = _run_node(js)

        self.assertFalse(out["afterDecorate"]["oldSepConnected"])
        self.assertTrue(out["afterDecorate"]["row2Grouped"])
        self.assertFalse(out["afterDecorate"]["row3Grouped"])
        self.assertEqual(out["afterDecorate"]["sepDays"], ["day-2", "day-3"])
        self.assertEqual(out["afterDecorate"]["calls"], [["capture"], ["preserve", 10], ["scroll"], ["jump"], ["after"]])
        self.assertEqual(out["afterTrim"]["trimmed"], 1)
        self.assertEqual(out["afterTrim"]["trimmedViewport"], 1)
        self.assertFalse(out["afterTrim"]["row1Connected"])
        self.assertFalse(out["afterTrim"]["row2Connected"])
        self.assertEqual(out["afterTrim"]["calls"], [["trim", True, 2, 4], ["liveTail", True], ["trimViewport", 3, 4, 42]])
        self.assertEqual(out["afterClear"], ["older", "bottom"])
        self.assertTrue(out["missingRoot"])
        self.assertTrue(out["frozen"])

    def test_typing_row_runtime_owns_row_anchor_and_scroll_projection(self) -> None:
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const ctx = {{ window: {{}} }};
            const vm = require("vm");
            vm.createContext(ctx);
            vm.runInContext({json.dumps(transcript_source)}, ctx);
            const tx = ctx.window.CodoxearTranscript;
            const calls = [];
            const bottom = {{ name: "bottom", isConnected: true }};
            const root = {{
              children: [bottom],
              insertBefore(node, before) {{
                const existing = this.children.indexOf(node);
                if (existing >= 0) this.children.splice(existing, 1);
                const idx = this.children.indexOf(before);
                this.children.splice(idx >= 0 ? idx : this.children.length, 0, node);
                node.isConnected = true;
                refreshSiblings();
              }},
            }};
            function refreshSiblings() {{
              for (let i = 0; i < root.children.length; i += 1) root.children[i].nextSibling = root.children[i + 1] || null;
            }}
            function fakeEl(tag, attrs = {{}}, children = []) {{
              const node = {{
                tag,
                attrs,
                children: [],
                dataset: {{}},
                isConnected: false,
                appendChild(child) {{ this.children.push(child); return child; }},
                remove() {{
                  const idx = root.children.indexOf(this);
                  if (idx >= 0) root.children.splice(idx, 1);
                  this.isConnected = false;
                  refreshSiblings();
                }},
              }};
              for (const child of children || []) node.appendChild(child);
              return node;
            }}
            let autoScroll = true;
            const runtime = tx.createTypingRowRuntime({{
              root,
              bottomSentinel: bottom,
              el: fakeEl,
              shouldAutoScroll: () => autoScroll,
              scheduleScrollToBottom: () => calls.push(["scroll"]),
            }});
            const initialAnchor = runtime.anchor().name;
            const shown = runtime.setVisible(true);
            const row = root.children[0];
            const afterShowAnchorIsRow = runtime.anchor() === row;
            runtime.setVisible(true);
            autoScroll = false;
            runtime.setVisible(true);
            const beforeHide = {{ childNames: root.children.map((node) => node.name || node.attrs.class), scrollCalls: calls.slice(), rowNextIsBottom: row.nextSibling === bottom }};
            runtime.setVisible(false);
            const afterHide = {{ snapshot: runtime.snapshot(), anchor: runtime.anchor().name, childNames: root.children.map((node) => node.name || node.attrs.class) }};
            runtime.setVisible(true);
            runtime.reset();
            let missingRoot = false;
            try {{ tx.createTypingRowRuntime({{ bottomSentinel: bottom, el: fakeEl, shouldAutoScroll: () => false, scheduleScrollToBottom: () => {{}} }}); }} catch (err) {{ missingRoot = /root/.test(String(err && err.message || err)); }}
            process.stdout.write(JSON.stringify({{
              initialAnchor,
              shown,
              rowClass: row.attrs.class,
              rowRole: row.dataset.role,
              bubbleClass: row.children[0].attrs.class,
              dotsClass: row.children[0].children[0].attrs.class,
              dotCount: row.children[0].children[0].children.length,
              afterShowAnchorIsRow,
              beforeHide,
              afterHide,
              afterReset: runtime.snapshot(),
              missingRoot,
              frozen: Object.isFrozen(runtime),
            }}));
            """
        )
        out = _run_node(js)

        self.assertEqual(out["initialAnchor"], "bottom")
        self.assertTrue(out["shown"]["connected"])
        self.assertEqual(out["rowClass"], "msg-row assistant typing-row")
        self.assertEqual(out["rowRole"], "assistant")
        self.assertEqual(out["bubbleClass"], "msg assistant typing")
        self.assertEqual(out["dotsClass"], "typingDots")
        self.assertEqual(out["dotCount"], 3)
        self.assertTrue(out["afterShowAnchorIsRow"])
        self.assertEqual(out["beforeHide"]["childNames"], ["msg-row assistant typing-row", "bottom"])
        self.assertEqual(out["beforeHide"]["scrollCalls"], [["scroll"], ["scroll"]])
        self.assertTrue(out["beforeHide"]["rowNextIsBottom"])
        self.assertFalse(out["afterHide"]["snapshot"]["connected"])
        self.assertEqual(out["afterHide"]["anchor"], "bottom")
        self.assertEqual(out["afterHide"]["childNames"], ["bottom"])
        self.assertFalse(out["afterReset"]["connected"])
        self.assertTrue(out["missingRoot"])
        self.assertTrue(out["frozen"])

    def test_transcript_slot_runtime_uses_current_session_lookup_for_tail_append(self) -> None:
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const ctx = {{ window: {{}} }};
            const vm = require("vm");
            vm.createContext(ctx);
            vm.runInContext({json.dumps(transcript_source)}, ctx);
            const tx = ctx.window.CodoxearTranscript;
            let sessions = new Map([["sid", {{ thread_id: "old-thread", log_path: "/old.jsonl" }}]]);
            const runtime = tx.createTranscriptSlotRuntime({{
              getSession: (sessionId) => sessions.get(sessionId) || null,
              maxTailEvents: 4,
            }});
            runtime.rememberTail("sid", sessions.get("sid"), {{
              transcript_state: "bound",
              thread_id: "old-thread",
              log_path: "/old.jsonl",
              events: [{{ role: "assistant", text: "old" }}],
              live_cursor: "old-cursor",
            }});
            sessions = new Map([["sid", {{ thread_id: "new-thread", log_path: "/new.jsonl" }}]]);
            runtime.appendTailEvents("sid", [{{ role: "assistant", text: "new" }}], {{ liveCursor: "new-cursor" }});
            const cache = runtime.getTailCache("sid");
            process.stdout.write(JSON.stringify({{
              threadId: cache.threadId,
              logPath: cache.logPath,
              texts: cache.events.map((ev) => ev.text),
              matchesNew: runtime.tailCacheMatchesSession(cache, sessions.get("sid")),
              matchesOld: runtime.tailCacheMatchesSession(cache, {{ thread_id: "old-thread", log_path: "/old.jsonl" }}),
            }}));
            """
        )
        out = _run_node(js)

        self.assertEqual(out["threadId"], "new-thread")
        self.assertEqual(out["logPath"], "/new.jsonl")
        self.assertEqual(out["texts"], ["old", "new"])
        self.assertTrue(out["matchesNew"])
        self.assertFalse(out["matchesOld"])

    def test_history_request_cursor_comes_from_oldest_rendered_row(self) -> None:
        snippet = _source_between("async function loadOlderMessages({ auto = false, cancelOnScroll = true, forcePreserveViewport = null } = {}) {", "function maybeAutoLoadOlder()")
        js = textwrap.dedent(
            f"""
            const ctx = {{
              window: {{}},
              currentGeneration: () => ctx.pollGen,
              selected: "sid",
              hasOlder: true,
              loadingOlder: false,
              pollGen: 7,
              olderLoadRequestId: 0,
              olderAutoTriggerAt: 0,
              OLDER_AUTO_COOLDOWN_MS: 450,
              olderLoadController: null,
              performance: {{ now: () => 1000 }},
              AbortController,
              encodeURIComponent,
              olderPageLimit: () => 60,
              oldestRenderedHistoryCursor: () => "cursor-oldest-row",
              usableOlderHistoryCursor: (data) => (data && data.has_older ? (data.history_cursor || (Array.isArray(data.events) && data.events.find((ev) => ev && ev.history_cursor)?.history_cursor) || null) : null),
              setOlderState: (state) => {{ ctx.lastOlderState = state; ctx.hasOlder = Boolean(state.hasMore); ctx.loadingOlder = Boolean(state.isLoading); }},
              clearOlderLoadError: () => {{ ctx.clearedOlderError = true; }},
              showOlderLoadError: () => {{ ctx.showedOlderError = true; }},
              prependOlderEvents: (events, opts) => {{ ctx.prepended = {{ events, opts }}; }},
              openSession: async () => {{ throw new Error("should not reopen"); }},
              api: async (url) => {{
                ctx.requestUrl = url;
                return {{ events: [{{ role: "assistant", text: "older" }}], has_older: false }};
              }},
            }};
            ctx.olderLoadSnapshot = () => ctx.olderLoadRuntime.snapshot();
            ctx.hasOlderMessages = () => ctx.olderLoadSnapshot().hasMore;
            ctx.isLoadingOlderMessages = () => ctx.olderLoadSnapshot().isLoading;
            ctx.olderLoadRuntime = {{
              snapshot: () => ({{ hasMore: Boolean(ctx.hasOlder), isLoading: Boolean(ctx.loadingOlder), requestId: ctx.olderLoadRequestId, cancelOnScroll: ctx.olderLoadCancelOnScroll !== false, hasController: Boolean(ctx.olderLoadController) }}),
              markAutoTrigger: () => {{
                const now = ctx.performance.now();
                if (now - ctx.olderAutoTriggerAt < ctx.OLDER_AUTO_COOLDOWN_MS) return false;
                ctx.olderAutoTriggerAt = now;
                return true;
              }},
              beginLoad: ({{ cancelOnScroll = true }} = {{}}) => {{
                ctx.olderLoadRequestId += 1;
                const ctl = new AbortController();
                ctx.olderLoadController = ctl;
                ctx.olderLoadCancelOnScroll = Boolean(cancelOnScroll);
                ctx.setOlderState({{ hasMore: ctx.hasOlder, isLoading: true }});
                return {{ requestId: ctx.olderLoadRequestId, controller: ctl, signal: ctl.signal }};
              }},
              isCurrent: (load) => load && load.requestId === ctx.olderLoadRequestId,
              finishLoad: (load) => {{ if (load && ctx.olderLoadController === load.controller) ctx.olderLoadController = null; ctx.olderLoadCancelOnScroll = true; }},
              invalidate: () => {{ ctx.olderLoadRequestId += 1; ctx.olderLoadController = null; ctx.olderLoadCancelOnScroll = true; if (ctx.loadingOlder) ctx.setOlderState({{ hasMore: ctx.hasOlder, isLoading: false }}); }},
            }};
            vm = require("vm");
            vm.createContext(ctx);
            vm.runInContext({json.dumps(APP_SESSION_STATE_JS.read_text(encoding="utf-8"))}, ctx);
            vm.runInContext({json.dumps(APP_SESSION_CATALOG_JS.read_text(encoding="utf-8"))}, ctx);
            ctx.sessionState = ctx.window.CodoxearSessionState.createSessionState({{ consoleError: () => {{}} }});
            ctx.sessionState.set("selected", "sid");
            vm.runInContext({json.dumps(snippet)}, ctx);
            ctx.loadOlderMessages({{ auto: false }}).then(() => {{
              process.stdout.write(JSON.stringify({{
                requestUrl: ctx.requestUrl,
                lastOlderState: ctx.lastOlderState,
                prepended: ctx.prepended,
              }}));
            }});
            """
        )
        out = _run_node(js)

        self.assertContains("cursor=cursor-oldest-row", out["requestUrl"])
        self.assertEqual(out["lastOlderState"], {"hasMore": False, "isLoading": False})
        self.assertEqual(out["prepended"]["events"][0]["text"], "older")

    def test_history_failure_preserves_has_older_and_shows_retry_error(self) -> None:
        snippet = _source_between("async function loadOlderMessages({ auto = false, cancelOnScroll = true, forcePreserveViewport = null } = {}) {", "function maybeAutoLoadOlder()")
        js = textwrap.dedent(
            f"""
            const ctx = {{
              window: {{}},
              currentGeneration: () => ctx.pollGen,
              selected: "sid",
              hasOlder: true,
              loadingOlder: false,
              pollGen: 7,
              olderLoadRequestId: 0,
              olderAutoTriggerAt: 0,
              OLDER_AUTO_COOLDOWN_MS: 450,
              olderLoadController: null,
              performance: {{ now: () => 1000 }},
              AbortController,
              encodeURIComponent,
              olderPageLimit: () => 60,
              oldestRenderedHistoryCursor: () => "cursor-oldest-row",
              usableOlderHistoryCursor: (data) => (data && data.has_older ? (data.history_cursor || (Array.isArray(data.events) && data.events.find((ev) => ev && ev.history_cursor)?.history_cursor) || null) : null),
              setOlderState: (state) => {{ ctx.lastOlderState = state; ctx.hasOlder = Boolean(state.hasMore); ctx.loadingOlder = Boolean(state.isLoading); }},
              clearOlderLoadError: () => {{ ctx.clearedOlderError = true; }},
              showOlderLoadError: () => {{ ctx.showedOlderError = true; }},
              prependOlderEvents: () => {{ ctx.prepended = true; }},
              openSession: async () => {{ throw new Error("should not reopen"); }},
              api: async () => {{ const err = new Error("unavailable"); err.status = 503; throw err; }},
            }};
            ctx.olderLoadSnapshot = () => ctx.olderLoadRuntime.snapshot();
            ctx.hasOlderMessages = () => ctx.olderLoadSnapshot().hasMore;
            ctx.isLoadingOlderMessages = () => ctx.olderLoadSnapshot().isLoading;
            ctx.olderLoadRuntime = {{
              snapshot: () => ({{ hasMore: Boolean(ctx.hasOlder), isLoading: Boolean(ctx.loadingOlder), requestId: ctx.olderLoadRequestId, cancelOnScroll: ctx.olderLoadCancelOnScroll !== false, hasController: Boolean(ctx.olderLoadController) }}),
              markAutoTrigger: () => {{
                const now = ctx.performance.now();
                if (now - ctx.olderAutoTriggerAt < ctx.OLDER_AUTO_COOLDOWN_MS) return false;
                ctx.olderAutoTriggerAt = now;
                return true;
              }},
              beginLoad: ({{ cancelOnScroll = true }} = {{}}) => {{
                ctx.olderLoadRequestId += 1;
                const ctl = new AbortController();
                ctx.olderLoadController = ctl;
                ctx.olderLoadCancelOnScroll = Boolean(cancelOnScroll);
                ctx.setOlderState({{ hasMore: ctx.hasOlder, isLoading: true }});
                return {{ requestId: ctx.olderLoadRequestId, controller: ctl, signal: ctl.signal }};
              }},
              isCurrent: (load) => load && load.requestId === ctx.olderLoadRequestId,
              finishLoad: (load) => {{ if (load && ctx.olderLoadController === load.controller) ctx.olderLoadController = null; ctx.olderLoadCancelOnScroll = true; }},
              invalidate: () => {{ ctx.olderLoadRequestId += 1; ctx.olderLoadController = null; ctx.olderLoadCancelOnScroll = true; if (ctx.loadingOlder) ctx.setOlderState({{ hasMore: ctx.hasOlder, isLoading: false }}); }},
            }};
            vm = require("vm");
            vm.createContext(ctx);
            vm.runInContext({json.dumps(APP_SESSION_STATE_JS.read_text(encoding="utf-8"))}, ctx);
            vm.runInContext({json.dumps(APP_SESSION_CATALOG_JS.read_text(encoding="utf-8"))}, ctx);
            ctx.sessionState = ctx.window.CodoxearSessionState.createSessionState({{ consoleError: () => {{}} }});
            ctx.sessionState.set("selected", "sid");
            vm.runInContext({json.dumps(snippet)}, ctx);
            ctx.loadOlderMessages({{ auto: false }}).then(() => {{
              process.stdout.write(JSON.stringify({{
                lastOlderState: ctx.lastOlderState,
                showedOlderError: Boolean(ctx.showedOlderError),
                prepended: Boolean(ctx.prepended),
              }}));
            }});
            """
        )
        out = _run_node(js)

        self.assertEqual(out["lastOlderState"], {"hasMore": True, "isLoading": False})
        self.assertTrue(out["showedOlderError"])
        self.assertFalse(out["prepended"])

    def test_history_401_triggers_auth_loss_without_retry_error(self) -> None:
        snippet = _source_between("async function loadOlderMessages({ auto = false, cancelOnScroll = true, forcePreserveViewport = null } = {}) {", "function maybeAutoLoadOlder()")
        js = textwrap.dedent(
            f"""
            const ctx = {{
              window: {{}},
              currentGeneration: () => ctx.pollGen,
              selected: "sid",
              hasOlder: true,
              loadingOlder: false,
              pollGen: 7,
              olderLoadRequestId: 0,
              olderAutoTriggerAt: 0,
              OLDER_AUTO_COOLDOWN_MS: 450,
              olderLoadController: null,
              performance: {{ now: () => 1000 }},
              AbortController,
              encodeURIComponent,
              olderPageLimit: () => 60,
              oldestRenderedHistoryCursor: () => "cursor-oldest-row",
              usableOlderHistoryCursor: (data) => (data && data.has_older ? (data.history_cursor || (Array.isArray(data.events) && data.events.find((ev) => ev && ev.history_cursor)?.history_cursor) || null) : null),
              setOlderState: (state) => {{ ctx.lastOlderState = state; ctx.hasOlder = Boolean(state.hasMore); ctx.loadingOlder = Boolean(state.isLoading); }},
              clearOlderLoadError: () => {{ ctx.clearedOlderError = true; }},
              showOlderLoadError: () => {{ ctx.showedOlderError = true; }},
              handleAppAuthLoss: () => {{ ctx.authLoss = true; }},
              prependOlderEvents: () => {{ ctx.prepended = true; }},
              openSession: async () => {{ throw new Error("should not reopen"); }},
              api: async () => {{ const err = new Error("unauthorized"); err.status = 401; throw err; }},
            }};
            ctx.olderLoadSnapshot = () => ctx.olderLoadRuntime.snapshot();
            ctx.hasOlderMessages = () => ctx.olderLoadSnapshot().hasMore;
            ctx.isLoadingOlderMessages = () => ctx.olderLoadSnapshot().isLoading;
            ctx.olderLoadRuntime = {{
              snapshot: () => ({{ hasMore: Boolean(ctx.hasOlder), isLoading: Boolean(ctx.loadingOlder), requestId: ctx.olderLoadRequestId, cancelOnScroll: ctx.olderLoadCancelOnScroll !== false, hasController: Boolean(ctx.olderLoadController) }}),
              markAutoTrigger: () => {{
                const now = ctx.performance.now();
                if (now - ctx.olderAutoTriggerAt < ctx.OLDER_AUTO_COOLDOWN_MS) return false;
                ctx.olderAutoTriggerAt = now;
                return true;
              }},
              beginLoad: ({{ cancelOnScroll = true }} = {{}}) => {{
                ctx.olderLoadRequestId += 1;
                const ctl = new AbortController();
                ctx.olderLoadController = ctl;
                ctx.olderLoadCancelOnScroll = Boolean(cancelOnScroll);
                ctx.setOlderState({{ hasMore: ctx.hasOlder, isLoading: true }});
                return {{ requestId: ctx.olderLoadRequestId, controller: ctl, signal: ctl.signal }};
              }},
              isCurrent: (load) => load && load.requestId === ctx.olderLoadRequestId,
              finishLoad: (load) => {{ if (load && ctx.olderLoadController === load.controller) ctx.olderLoadController = null; ctx.olderLoadCancelOnScroll = true; }},
              invalidate: () => {{ ctx.olderLoadRequestId += 1; ctx.olderLoadController = null; ctx.olderLoadCancelOnScroll = true; if (ctx.loadingOlder) ctx.setOlderState({{ hasMore: ctx.hasOlder, isLoading: false }}); }},
            }};
            vm = require("vm");
            vm.createContext(ctx);
            vm.runInContext({json.dumps(APP_SESSION_STATE_JS.read_text(encoding="utf-8"))}, ctx);
            vm.runInContext({json.dumps(APP_SESSION_CATALOG_JS.read_text(encoding="utf-8"))}, ctx);
            ctx.sessionState = ctx.window.CodoxearSessionState.createSessionState({{ consoleError: () => {{}} }});
            ctx.sessionState.set("selected", "sid");
            vm.runInContext({json.dumps(snippet)}, ctx);
            ctx.loadOlderMessages({{ auto: false }}).then(() => {{
              process.stdout.write(JSON.stringify({{
                authLoss: Boolean(ctx.authLoss),
                showedOlderError: Boolean(ctx.showedOlderError),
                prepended: Boolean(ctx.prepended),
              }}));
            }});
            """
        )
        out = _run_node(js)

        self.assertTrue(out["authLoss"])
        self.assertFalse(out["showedOlderError"])
        self.assertFalse(out["prepended"])

    def test_stale_history_401_still_triggers_auth_loss(self) -> None:
        snippet = _source_between("async function loadOlderMessages({ auto = false, cancelOnScroll = true, forcePreserveViewport = null } = {}) {", "function maybeAutoLoadOlder()")
        js = textwrap.dedent(
            f"""
            const ctx = {{
              window: {{}},
              currentGeneration: () => ctx.pollGen,
              selected: "sid",
              hasOlder: true,
              loadingOlder: false,
              pollGen: 7,
              olderLoadRequestId: 0,
              olderAutoTriggerAt: 0,
              OLDER_AUTO_COOLDOWN_MS: 450,
              olderLoadController: null,
              performance: {{ now: () => 1000 }},
              AbortController,
              encodeURIComponent,
              olderPageLimit: () => 60,
              oldestRenderedHistoryCursor: () => "cursor-oldest-row",
              usableOlderHistoryCursor: (data) => (data && data.has_older ? (data.history_cursor || (Array.isArray(data.events) && data.events.find((ev) => ev && ev.history_cursor)?.history_cursor) || null) : null),
              setOlderState: (state) => {{ ctx.lastOlderState = state; ctx.hasOlder = Boolean(state.hasMore); ctx.loadingOlder = Boolean(state.isLoading); }},
              clearOlderLoadError: () => {{ ctx.clearedOlderError = true; }},
              showOlderLoadError: () => {{ ctx.showedOlderError = true; }},
              handleAppAuthLoss: () => {{ ctx.authLoss = true; }},
              prependOlderEvents: () => {{ ctx.prepended = true; }},
              openSession: async () => {{ throw new Error("should not reopen"); }},
              api: async () => {{ ctx.olderLoadRequestId += 1; const err = new Error("unauthorized"); err.status = 401; throw err; }},
            }};
            ctx.olderLoadSnapshot = () => ctx.olderLoadRuntime.snapshot();
            ctx.hasOlderMessages = () => ctx.olderLoadSnapshot().hasMore;
            ctx.isLoadingOlderMessages = () => ctx.olderLoadSnapshot().isLoading;
            ctx.olderLoadRuntime = {{
              snapshot: () => ({{ hasMore: Boolean(ctx.hasOlder), isLoading: Boolean(ctx.loadingOlder), requestId: ctx.olderLoadRequestId, cancelOnScroll: ctx.olderLoadCancelOnScroll !== false, hasController: Boolean(ctx.olderLoadController) }}),
              markAutoTrigger: () => {{
                const now = ctx.performance.now();
                if (now - ctx.olderAutoTriggerAt < ctx.OLDER_AUTO_COOLDOWN_MS) return false;
                ctx.olderAutoTriggerAt = now;
                return true;
              }},
              beginLoad: ({{ cancelOnScroll = true }} = {{}}) => {{
                ctx.olderLoadRequestId += 1;
                const ctl = new AbortController();
                ctx.olderLoadController = ctl;
                ctx.olderLoadCancelOnScroll = Boolean(cancelOnScroll);
                ctx.setOlderState({{ hasMore: ctx.hasOlder, isLoading: true }});
                return {{ requestId: ctx.olderLoadRequestId, controller: ctl, signal: ctl.signal }};
              }},
              isCurrent: (load) => load && load.requestId === ctx.olderLoadRequestId,
              finishLoad: (load) => {{ if (load && ctx.olderLoadController === load.controller) ctx.olderLoadController = null; ctx.olderLoadCancelOnScroll = true; }},
              invalidate: () => {{ ctx.olderLoadRequestId += 1; ctx.olderLoadController = null; ctx.olderLoadCancelOnScroll = true; if (ctx.loadingOlder) ctx.setOlderState({{ hasMore: ctx.hasOlder, isLoading: false }}); }},
            }};
            vm = require("vm");
            vm.createContext(ctx);
            vm.runInContext({json.dumps(APP_SESSION_STATE_JS.read_text(encoding="utf-8"))}, ctx);
            vm.runInContext({json.dumps(APP_SESSION_CATALOG_JS.read_text(encoding="utf-8"))}, ctx);
            ctx.sessionState = ctx.window.CodoxearSessionState.createSessionState({{ consoleError: () => {{}} }});
            ctx.sessionState.set("selected", "sid");
            vm.runInContext({json.dumps(snippet)}, ctx);
            ctx.loadOlderMessages({{ auto: false }}).then(() => {{
              process.stdout.write(JSON.stringify({{
                authLoss: Boolean(ctx.authLoss),
                showedOlderError: Boolean(ctx.showedOlderError),
                prepended: Boolean(ctx.prepended),
              }}));
            }});
            """
        )
        out = _run_node(js)

        self.assertTrue(out["authLoss"])
        self.assertFalse(out["showedOlderError"])
        self.assertFalse(out["prepended"])

    def test_live_delta_does_not_splice_into_history_window(self) -> None:
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const ctx = {{ window: {{}} }};
            const vm = require("vm");
            vm.createContext(ctx);
            vm.runInContext({json.dumps(transcript_source)}, ctx);
            const calls = [];
            const bottom = {{}};
            const scrollRuntime = {{
              snapshot: () => ({{ renderedAtLiveTail: false }}),
              shouldStickToBottom: () => false,
              syncJumpButton: () => calls.push(["jump"]),
              scheduleScrollToBottom: () => calls.push(["scroll"]),
              markLiveTail: () => calls.push(["markLiveTail"]),
              disableAutoScroll: () => calls.push(["disableAuto"]),
              setRenderedAtLiveTail: (value) => calls.push(["liveTail", value]),
              setScrollTop: (value) => calls.push(["scrollTop", value]),
            }};
            const runtime = ctx.window.CodoxearTranscript.createTranscriptRenderRuntime({{
              root: {{ insertBefore: () => calls.push(["insert"]), querySelector: () => null }},
              bottomSentinel: bottom,
              document: {{ createDocumentFragment: () => ({{ appendChild: () => {{}} }}) }},
              safeMakeRow: () => (calls.push(["make"]), {{ row: {{}}, bubble: {{}} }}),
              normalizeEvents: () => [],
              consumePendingUserIfMatches: () => false,
              isDuplicateEvent: () => false,
              isAdjacentAssistantDuplicateEvent: () => false,
              markEventSeen: () => calls.push(["seen"]),
              markFirstPaint: () => calls.push(["paint"]),
              renderRecoveryPanel: () => calls.push(["recovery"]),
              restorePendingRows: () => calls.push(["restore"]),
              resetRecentEvents: () => calls.push(["resetRecent"]),
              setOlderState: () => calls.push(["older"]),
              firstVisibleMessageRow: () => null,
              getScrollTop: () => 0,
              getSelectedSessionId: () => "sid",
              domRuntime: {{ clear: () => calls.push(["clear"]), rebuildDecorations: () => calls.push(["rebuild"]), trimRenderedRows: () => calls.push(["trim"]) }},
              scrollRuntime,
              typingRowRuntime: {{ anchor: () => bottom }},
              historySlackRows: 3,
            }});
            runtime.appendEvent({{ role: "assistant", text: "new tail", ts: 2 }});
            process.stdout.write(JSON.stringify({{ calls }}));
            """
        )
        out = _run_node(js)

        self.assertEqual(out["calls"], [["seen"], ["jump"]])

    def test_live_delta_dedupes_adjacent_assistant_text_across_polls(self) -> None:
        identity_source = APP_MESSAGE_IDENTITY_JS.read_text(encoding="utf-8")
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const ctx = {{ window: {{}} }};
            const vm = require("vm");
            vm.createContext(ctx);
            vm.runInContext({json.dumps(identity_source)}, ctx);
            vm.runInContext({json.dumps(transcript_source)}, ctx);
            const id = ctx.window.CodoxearMessageIdentity;
            const tx = ctx.window.CodoxearTranscript;
            const eventRuntime = tx.createTranscriptEventRuntime({{
              eventKey: id.eventKey,
              pendingMatchKey: id.pendingMatchKey,
              normalizePendingText: id.normalizeTextForPendingMatch,
              assistantDedupeKey: id.chatAssistantDedupeKey,
              maxRecentEventKeys: 320,
            }});
            const calls = [];
            let rows = [{{ dataset: {{ role: "assistant", assistantDedupeKey: "final_response|same final text" }} }}];
            const bottom = {{}};
            const scrollRuntime = {{
              snapshot: () => ({{ renderedAtLiveTail: true }}),
              shouldStickToBottom: () => true,
              syncJumpButton: () => calls.push(["jump"]),
              scheduleScrollToBottom: () => calls.push(["scroll"]),
              markLiveTail: () => calls.push(["markLiveTail"]),
              disableAutoScroll: () => calls.push(["disableAuto"]),
              setRenderedAtLiveTail: (value) => calls.push(["liveTail", value]),
              setScrollTop: (value) => calls.push(["scrollTop", value]),
            }};
            const runtime = tx.createTranscriptRenderRuntime({{
              root: {{ insertBefore: () => calls.push(["insert"]), querySelector: () => null }},
              bottomSentinel: bottom,
              document: {{ createDocumentFragment: () => ({{ appendChild: () => {{}} }}) }},
              safeMakeRow: () => (calls.push(["make"]), {{ row: {{}}, bubble: {{}} }}),
              normalizeEvents: () => [],
              consumePendingUserIfMatches: () => false,
              isDuplicateEvent: (ev) => eventRuntime.isDuplicateEvent(ev),
              isAdjacentAssistantDuplicateEvent: (ev) => eventRuntime.isAdjacentAssistantDuplicateEvent(ev, {{ renderedAtLiveTail: true, rows }}),
              markEventSeen: (ev) => eventRuntime.markEventSeen(ev),
              markFirstPaint: () => calls.push(["paint"]),
              renderRecoveryPanel: () => calls.push(["recovery"]),
              restorePendingRows: () => calls.push(["restore"]),
              resetRecentEvents: () => eventRuntime.resetRecentEvents(),
              setOlderState: () => calls.push(["older"]),
              firstVisibleMessageRow: () => null,
              getScrollTop: () => 0,
              getSelectedSessionId: () => "sid",
              domRuntime: {{ clear: () => calls.push(["clear"]), rebuildDecorations: () => calls.push(["rebuild"]), trimRenderedRows: () => calls.push(["trim"]) }},
              scrollRuntime,
              typingRowRuntime: {{ anchor: () => bottom }},
              historySlackRows: 3,
            }});
            runtime.appendEvent({{ role: "assistant", text: "same final text", message_class: "final_response", ts: 2.4 }});
            const afterDuplicate = {{ calls: calls.slice(), seen: eventRuntime.snapshot().recentEventKeys }};
            calls.length = 0;
            rows = [{{ dataset: {{ role: "user" }} }}];
            runtime.appendEvent({{ role: "assistant", text: "same final text", message_class: "final_response", ts: 3.0 }});
            process.stdout.write(JSON.stringify({{
              afterDuplicate,
              final: {{ calls: calls.slice(), seen: eventRuntime.snapshot().recentEventKeys }},
            }}));
            """
        )
        out = _run_node(js)

        self.assertEqual(out["afterDuplicate"]["calls"], [])
        self.assertEqual(out["afterDuplicate"]["seen"], ["assistant|2400|same final text"])
        self.assertEqual(out["final"]["calls"], [["make"], ["insert"], ["trim"], ["rebuild"], ["paint"], ["scroll"], ["jump"]])
        self.assertEqual(out["final"]["seen"], ["assistant|2400|same final text", "assistant|3000|same final text"])

    def test_composer_resets_typing_counts_for_idle_send_but_not_steer(self) -> None:
        polling_source = APP_POLLING_JS.read_text(encoding="utf-8")
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        message_flow_source = APP_MESSAGE_FLOW_JS.read_text(encoding="utf-8")
        composer_source = APP_COMPOSER_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const vm = require("vm");
            const ctx = {{ window: {{}}, console, Date }};
            vm.createContext(ctx);
            vm.runInContext({json.dumps(polling_source)}, ctx);
            vm.runInContext({json.dumps(transcript_source)}, ctx);
            vm.runInContext({json.dumps(message_flow_source)}, ctx);
            vm.runInContext({json.dumps(APP_SESSION_STATE_JS.read_text(encoding="utf-8"))}, ctx);
            vm.runInContext({json.dumps(APP_SESSION_CATALOG_JS.read_text(encoding="utf-8"))}, ctx);
            vm.runInContext({json.dumps(composer_source)}, ctx);
            {MESSAGE_FLOW_HARNESS_JS}
            const fakeNode = () => ({{
              addEventListener: () => {{}}, removeEventListener: () => {{}},
              setAttribute: () => {{}}, removeAttribute: () => {{}},
              classList: {{ toggle: () => {{}} }}, style: {{}}, value: "",
              scrollHeight: 32, disabled: false, focus: () => {{}}, blur: () => {{}},
            }});
            const noop = () => {{}};
            function runSend(initialRunning) {{
              const nodes = Array.from({{ length: 9 }}, fakeNode);
              const [form, textarea, msgPh, sendBtn, sendChoice, sendChoiceBackdrop, sendChoiceNowBtn, sendChoiceLaterBtn, sendChoiceCancelBtn] = nodes;
              form.requestSubmit = noop;
              const state = {{ sending: false, running: initialRunning, resets: 0 }};
              const messageFlowController = createMessageFlow(state);
              const controller = ctx.window.CodoxearComposer.createComposerController({{
                form, textarea, msgPh, sendBtn, sendChoice, sendChoiceBackdrop,
                sendChoiceNowBtn, sendChoiceLaterBtn, sendChoiceCancelBtn,
                sessionCatalog: state.sessionCatalog,
                sessionLaunchFailed: () => false, getSending: () => state.sending,
                sessionState: state.sessionState, getStagedAttachments: () => [],
                api: async () => ({{}}), setToast: noop, setPollFastUntilMs: noop, kickPoll: noop,
                sendText: (...args) => messageFlowController.sendText(...args),
                enqueueComposerText: async () => false, prepareModalOpen: noop,
                afterModalVisibilityChanged: noop, restoreModalFocus: noop,
                storageGetItem: () => null, storageSetItem: noop, storageRemoveItem: noop,
                getComputedStyle: () => ({{ minHeight: "32" }}), isHTMLElement: () => false, now: () => 1000,
              }});
              return controller.sendText("steer or start").then((ok) => ({{ ok, resets: state.resets }}));
            }}
            Promise.all([runSend(false), runSend(true)]).then(([idle, steer]) => {{
              process.stdout.write(JSON.stringify({{ idle, steer }}));
            }});
            """
        )
        out = _run_node(js)
        self.assertEqual(out["idle"], {"ok": True, "resets": 1})
        self.assertEqual(out["steer"], {"ok": True, "resets": 0})

    def test_new_command_send_failure_does_not_detach_current_transcript(self) -> None:
        polling_source = APP_POLLING_JS.read_text(encoding="utf-8")
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        message_flow_source = APP_MESSAGE_FLOW_JS.read_text(encoding="utf-8")
        composer_source = APP_COMPOSER_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const vm = require("vm");
            const ctx = {{ window: {{}}, console, Date }};
            vm.createContext(ctx);
            vm.runInContext({json.dumps(polling_source)}, ctx);
            vm.runInContext({json.dumps(transcript_source)}, ctx);
            vm.runInContext({json.dumps(message_flow_source)}, ctx);
            vm.runInContext({json.dumps(APP_SESSION_STATE_JS.read_text(encoding="utf-8"))}, ctx);
            vm.runInContext({json.dumps(APP_SESSION_CATALOG_JS.read_text(encoding="utf-8"))}, ctx);
            vm.runInContext({json.dumps(composer_source)}, ctx);
            {MESSAGE_FLOW_HARNESS_JS}
            const fakeNode = () => ({{
              addEventListener: () => {{}}, removeEventListener: () => {{}},
              setAttribute: () => {{}}, removeAttribute: () => {{}},
              classList: {{ toggle: () => {{}} }}, style: {{}}, value: "",
              scrollHeight: 32, disabled: false, focus: () => {{}},
            }});
            const nodes = Array.from({{ length: 9 }}, fakeNode);
            const [form, textarea, msgPh, sendBtn, sendChoice, sendChoiceBackdrop, sendChoiceNowBtn, sendChoiceLaterBtn, sendChoiceCancelBtn] = nodes;
            form.requestSubmit = () => {{}};
            textarea.value = "/new";
            const state = {{ sending: false, detached: 0, renderedPending: 0, deletedCache: false, toast: "" }};
            const noop = () => {{}};
            const messageFlowController = createMessageFlow(state, {{
              getSessionInfo: () => ({{ agent_backend: "codex" }}),
              api: async () => {{ throw new Error("broker down"); }},
              isTranscriptRenewalCommand: () => true,
              deleteTailCache: () => {{ state.deletedCache = true; }},
              beginTranscriptRenewal: () => {{ state.detached += 1; }},
              renderPendingTranscriptSlot: () => {{ state.renderedPending += 1; }},
            }});
            const controller = ctx.window.CodoxearComposer.createComposerController({{
              form, textarea, msgPh, sendBtn, sendChoice, sendChoiceBackdrop,
              sendChoiceNowBtn, sendChoiceLaterBtn, sendChoiceCancelBtn,
              sessionCatalog: state.sessionCatalog,
              sessionLaunchFailed: () => false, getSending: () => state.sending,
              sessionState: state.sessionState, getStagedAttachments: () => [],
              api: async () => ({{}}), setToast: noop, setPollFastUntilMs: noop, kickPoll: noop,
              sendText: (...args) => messageFlowController.sendText(...args),
              enqueueComposerText: async () => false, prepareModalOpen: noop,
              afterModalVisibilityChanged: noop, restoreModalFocus: noop,
              storageGetItem: () => null, storageSetItem: noop, storageRemoveItem: noop,
              getComputedStyle: () => ({{ minHeight: "32" }}), isHTMLElement: () => false, now: () => 1000,
            }});
            controller.sendText("/new").then((ok) => {{
              process.stdout.write(JSON.stringify({{ ok, ...state }}));
            }});
            """
        )
        out = _run_node(js)

        self.assertFalse(out["ok"])
        self.assertEqual(out["detached"], 0)
        self.assertEqual(out["renderedPending"], 0)
        self.assertFalse(out["deletedCache"])
        self.assertEqual(out["toast"], "send error: broker down")

if __name__ == "__main__":
    unittest.main()
