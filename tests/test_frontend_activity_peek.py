"""Behavior contract for the busy typing bubble's activity peek line.

The peek is chrome: one ephemeral line naming the agent's current action.
Wire contract on live deltas and tail payloads: ``activity: {label, running}``
sets/replaces it, ``activity: null`` clears it, and an absent field leaves it
unchanged. The runtime tests execute the real typing-row factory, the live
delta path executes the real message-flow controller against that factory, and
the CSS test parses the cascade.
"""

from __future__ import annotations

import json
import subprocess
import textwrap

from css_tokens import APP_CSS, declarations, parse_stylesheet, selector_text, top_level_rules
from frontend_module_loader import module_path
from test_frontend_message_flow import _run_node as run_message_flow_node

APP_TRANSCRIPT_JS = module_path("app_transcript.js")
APP_MESSAGE_HISTORY_JS = module_path("app_message_history.js")

NODE_FIXTURE = """
function node(attrs = {}, children = []) {
  const out = { ...attrs, children: [], dataset: {}, isConnected: false, parentNode: null };
  out.appendChild = (child) => { out.children.push(child); child.parentNode = out; return child; };
  out.insertBefore = (child, before) => {
    const index = out.children.indexOf(before);
    if (index < 0) out.children.push(child); else out.children.splice(index, 0, child);
    child.parentNode = out;
    child.isConnected = true;
    return child;
  };
  out.remove = () => {
    out.isConnected = false;
    if (out.parentNode) out.parentNode.children = out.parentNode.children.filter((child) => child !== out);
    out.parentNode = null;
  };
  Object.defineProperty(out, "nextSibling", { get: () => out.parentNode ? out.parentNode.children[out.parentNode.children.indexOf(out) + 1] || null : null });
  let text = "";
  Object.defineProperty(out, "textContent", { get: () => text, set: (value) => { text = String(value); if (value === "") out.children = []; } });
  for (const child of children) out.appendChild(child);
  return out;
}
function createTyping() {
  const root = node();
  const bottom = node();
  root.appendChild(bottom);
  bottom.isConnected = true;
  const runtime = ctx.window.CodoxearTranscript.createTypingRowRuntime({
    root, bottomSentinel: bottom,
    el: (_tag, attrs, children) => node(attrs, children),
    shouldAutoScroll: () => false, scheduleScrollToBottom: () => {},
  });
  const peekNode = () => {
    const row = runtime.anchor();
    if (!row || !row.children || !row.children[0]) return null;
    return row.children[0].children.find((child) => child.class === "activityPeekLine") || null;
  };
  const bubbleClasses = () => runtime.anchor().children[0].children.map((child) => child.class);
  return { runtime, peekNode, bubbleClasses };
}
"""


def _run_runtime(body: str) -> dict:
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}}, console }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(APP_TRANSCRIPT_JS.read_text(encoding="utf-8"))}, ctx);
        {NODE_FIXTURE}
        {body}
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, capture_output=True, text=True)
    return json.loads(proc.stdout)


def test_peek_renders_label_with_ellipsis_only_while_running() -> None:
    out = _run_runtime(
        """
        const { runtime, peekNode } = createTyping();
        runtime.setVisible(true);
        const empty = peekNode().textContent;
        runtime.updateActivityPeek({ label: "bash: git diff --stat", running: true });
        const running = peekNode().textContent;
        runtime.updateActivityPeek({ label: "bash: git diff --stat", running: false });
        const finished = peekNode().textContent;
        process.stdout.write(JSON.stringify({ empty, running, finished }));
        """
    )
    assert out == {"empty": "", "running": "bash: git diff --stat…", "finished": "bash: git diff --stat"}


def test_peek_sits_between_header_and_subagent_details() -> None:
    out = _run_runtime(
        """
        const { runtime, bubbleClasses } = createTyping();
        runtime.setVisible(true);
        process.stdout.write(JSON.stringify({ classes: bubbleClasses() }));
        """
    )
    assert out["classes"] == ["subagentActivityHeader", "activityPeekLine", "subagentDetails"]


def test_newer_activity_replaces_and_null_or_malformed_clears() -> None:
    out = _run_runtime(
        """
        const { runtime, peekNode } = createTyping();
        runtime.setVisible(true);
        runtime.updateActivityPeek({ label: "read: a.py", running: true });
        runtime.updateActivityPeek({ label: "edit: b.py", running: true });
        const replaced = peekNode().textContent;
        const lineCount = runtime.anchor().children[0].children.filter((child) => child.class === "activityPeekLine").length;
        runtime.updateActivityPeek(null);
        const nulled = peekNode().textContent;
        const snapshotAfterNull = runtime.snapshot().activity;
        runtime.updateActivityPeek({ label: "read: a.py", running: true });
        runtime.updateActivityPeek({ label: 7, running: true });
        const malformed = peekNode().textContent;
        process.stdout.write(JSON.stringify({ replaced, lineCount, nulled, snapshotAfterNull, malformed }));
        """
    )
    assert out == {"replaced": "edit: b.py…", "lineCount": 1, "nulled": "", "snapshotAfterNull": None, "malformed": ""}


def test_flip_only_signal_drops_the_running_marker_and_keeps_the_label() -> None:
    out = _run_runtime(
        """
        const { runtime, peekNode } = createTyping();
        runtime.setVisible(true);
        runtime.updateActivityPeek({ label: "bash: npm test", running: true });
        const whileRunning = peekNode().textContent;
        runtime.updateActivityPeek({ label: null, running: false });
        const afterFlip = peekNode().textContent;
        const snapshotAfterFlip = runtime.snapshot().activity;
        // A flip with nothing shown is a no-op, not a malformed clear.
        runtime.updateActivityPeek(null);
        runtime.updateActivityPeek({ label: null, running: false });
        const flipWithoutPeek = peekNode().textContent;
        process.stdout.write(JSON.stringify({ whileRunning, afterFlip, snapshotAfterFlip, flipWithoutPeek }));
        """
    )
    assert out == {
        "whileRunning": "bash: npm test…",
        "afterFlip": "bash: npm test",
        "snapshotAfterFlip": {"label": "bash: npm test", "running": False},
        "flipWithoutPeek": "",
    }


def test_peek_set_before_row_exists_renders_at_construction() -> None:
    out = _run_runtime(
        """
        const { runtime, peekNode } = createTyping();
        runtime.updateActivityPeek({ label: "bash: ls", running: true });
        runtime.setVisible(true);
        process.stdout.write(JSON.stringify({ text: peekNode().textContent, snapshot: runtime.snapshot().activity }));
        """
    )
    assert out == {"text": "bash: ls…", "snapshot": {"label": "bash: ls", "running": True}}


def test_every_clear_path_funnels_through_reset_typing_stats() -> None:
    out = _run_runtime(
        """
        const results = {};
        const seed = () => {
          const typing = createTyping();
          typing.runtime.setVisible(true);
          typing.runtime.updateActivityPeek({ label: "bash: make", running: true });
          return typing;
        };
        let t = seed();
        t.runtime.resetTypingStats();
        results.resetTypingStats = t.runtime.snapshot().activity;
        results.resetTypingStatsText = t.peekNode().textContent;
        t = seed();
        t.runtime.setVisible(false);
        results.setVisibleFalse = t.runtime.snapshot().activity;
        t.runtime.setVisible(true);
        results.reopenedText = t.peekNode().textContent;
        t = seed();
        t.runtime.reset();
        results.reset = t.runtime.snapshot().activity;
        t.runtime.setVisible(true);
        results.afterResetText = t.peekNode().textContent;
        process.stdout.write(JSON.stringify(results));
        """
    )
    assert out == {
        "resetTypingStats": None,
        "resetTypingStatsText": "",
        "setVisibleFalse": None,
        "reopenedText": "",
        "reset": None,
        "afterResetText": "",
    }


def test_live_delta_sets_replaces_clears_and_leaves_peek_when_field_absent() -> None:
    out = run_message_flow_node(
        """
        const concrete = (() => {
          const root = { children: [], appendChild() {}, insertBefore() {} };
          const bottom = { isConnected: true };
          return ctx.window.CodoxearTranscript.createTypingRowRuntime({
            root, bottomSentinel: bottom,
            el: () => { let t = ""; return { children: [], dataset: {}, appendChild() {}, set textContent(v) { t = v; }, get textContent() { return t; } }; },
            shouldAutoScroll: () => false, scheduleScrollToBottom: () => {},
          });
        })();
        const harness = createHarness({ typingRowRuntime: concrete });
        const controller = harness.create();
        const live = (cursor, extra) => controller.applyLiveMessageData("sid", 1, {
          transcript_state: "bound", log_path: "/tmp/log.jsonl", live_cursor: cursor, events: [], busy: true, queue_len: 0, ...extra,
        });
        const seen = [];
        (async () => {
          await live("c2", { activity: { label: "bash: git diff --stat", running: true } });
          seen.push(concrete.snapshot().activity);
          await live("c3", {});
          seen.push(concrete.snapshot().activity);
          await live("c4", { activity: { label: "edit: app.js", running: false } });
          seen.push(concrete.snapshot().activity);
          // Same cursor as the stored window is a duplicate and must not rewrite the peek.
          await live("c4", { activity: { label: "stale replay", running: true } });
          seen.push(concrete.snapshot().activity);
          await live("c5", { activity: null });
          seen.push(concrete.snapshot().activity);
          process.stdout.write(JSON.stringify({ seen }));
        })();
        """
    )
    assert out["seen"] == [
        {"label": "bash: git diff --stat", "running": True},
        {"label": "bash: git diff --stat", "running": True},
        {"label": "edit: app.js", "running": False},
        {"label": "edit: app.js", "running": False},
        None,
    ]


def test_tail_payload_seeds_peek_and_absent_field_leaves_it() -> None:
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}}, console }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(APP_TRANSCRIPT_JS.read_text(encoding="utf-8"))}, ctx);
        vm.runInContext({json.dumps(APP_MESSAGE_HISTORY_JS.read_text(encoding="utf-8"))}, ctx);
        const noop = () => {{}};
        const peeks = [];
        const typingRowRuntime = {{ updateActivityPeek: (peek) => peeks.push(peek) }};
        const sessionState = {{ get: () => null, set: noop, applyRuntime: noop }};
        const sessionCatalog = {{ get: () => new Map() }};
        const history = ctx.window.CodoxearMessageHistory.createMessageHistoryController({{
          currentGeneration: () => 1, sessionCatalog,
          getSessionLifecycleController: () => ({{}}), getSessionRefreshController: () => ({{}}),
          getSendLifecycleController: () => ({{}}), getAttachmentsController: () => ({{}}),
          transcript: {{
            typingRowRuntime,
            transcriptView: () => ({{ setHistory: noop }}),
            transcriptSlotRuntime: {{ setLiveCursor: noop, deleteTailCache: noop }},
            syncActiveTranscriptSlot: () => ({{ state: "pending_bind" }}),
            updateTypingStatsFromSession: noop,
          }},
          sessionState, wiring: {{ createOlderLoadOptions: (options) => options }}, olderWrap: {{}}, olderBtn: {{}}, olderError: {{}}, olderErrorText: {{}},
          AbortController, performance: {{ now: () => 0 }}, OLDER_AUTO_COOLDOWN_MS: 450, OLDER_PAGE_LIMIT: 30,
          api: async () => ({{}}), handleAppAuthLoss: noop, sessionLaunchFailed: () => false, confirmApp: async () => false,
          setToast: noop, codoxearDisplay: {{}}, redactedLaunchErrorText: () => "", sessionIdFromHash: () => "", sessionSelectable: () => false,
        }});
        history.applySessionRuntimeFromTail("sid", {{ busy: true, activity: {{ label: "bash: ls", running: true }} }});
        history.applySessionRuntimeFromTail("sid", {{ busy: true }});
        history.applySessionRuntimeFromTail("sid", {{ busy: true, activity: null }});
        process.stdout.write(JSON.stringify({{ peeks }}));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, capture_output=True, text=True)
    assert json.loads(proc.stdout)["peeks"] == [{"label": "bash: ls", "running": True}, None]


def _cascade(selector: str) -> dict[str, str]:
    style: dict[str, str] = {}
    for rule in top_level_rules(parse_stylesheet(APP_CSS)):
        if selector in [part.strip() for part in selector_text(rule).split(",")]:
            style.update(declarations(rule))
    assert style, f"no rule for {selector}"
    return style


def test_peek_line_is_one_clipped_grid_row_that_collapses_when_empty() -> None:
    line = _cascade(".activityPeekLine")
    assert line["grid-column"] == "1 / -1"
    assert line["min-width"] == "0"
    assert line["white-space"] == "nowrap"
    assert line["overflow"] == "hidden"
    assert line["text-overflow"] == "ellipsis"
    assert line["margin-top"] == "var(--space-2)"
    assert line["color"] == "var(--muted)"
    assert _cascade(".activityPeekLine:empty")["display"] == "none"

