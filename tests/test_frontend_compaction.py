from __future__ import annotations

import json
import subprocess

from frontend_module_loader import module_path


def test_compaction_row_and_transcript_paths():
    modules = [module_path(name).read_text() for name in ("app_message_rows.js", "app_transcript.js", "app_conversation_copy.js")]
    harness = r'''
const vm = require("vm");
const assert = require("assert/strict");
const ctx = {window: {}};
vm.createContext(ctx);
for (const source of MODULES) vm.runInContext(source, ctx);
const M = ctx.window.CodoxearMessageRows, T = ctx.window.CodoxearTranscript;
class Node {
  constructor(tag, attrs = {}) {
    this.tag = tag; this.attrs = attrs; this.dataset = {}; this.style = {}; this.children = [];
    this.ownText = attrs.text || ""; this.parentNode = null; this.offsetTop = 0; this.offsetHeight = 20;
    this.classList = {values: new Set((attrs.class || "").split(/\s+/).filter(Boolean)),
      contains: (x) => this.classList.values.has(x),
      add: (...xs) => xs.forEach(x => this.classList.values.add(x)),
      remove: (...xs) => xs.forEach(x => this.classList.values.delete(x))};
  }
  get textContent() { return this.ownText + this.children.map(x => x.textContent).join(""); }
  set textContent(text) { this.ownText = text; this.children = []; }
  set innerHTML(html) { this.children.forEach(x => x.parentNode = null); this.children = []; this.ownText = html; }
  get isConnected() { return this === root || Boolean(this.parentNode && this.parentNode.isConnected); }
  get nextSibling() { return this.parentNode?.children[this.parentNode.children.indexOf(this) + 1] || null; }
  appendChild(child) { return this.insertBefore(child, null); }
  insertBefore(child, anchor) {
    if (child.tag === "fragment") { for (const node of [...child.children]) this.insertBefore(node, anchor); return child; }
    child.remove(); const i = anchor ? this.children.indexOf(anchor) : this.children.length;
    this.children.splice(i, 0, child); child.parentNode = this; return child;
  }
  remove() { if (this.parentNode) this.parentNode.children.splice(this.parentNode.children.indexOf(this), 1); this.parentNode = null; }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  querySelectorAll(selector) {
    const matches = (node) => {
      if (selector === "button") return node.tag === "button";
      if (selector === ".msg-row[data-message-id]") return node.classList.contains("msg-row") && Boolean(node.dataset.messageId);
      if (selector === ".msg-row:not(.typing-row)") return node.classList.contains("msg-row") && !node.classList.contains("typing-row");
      return selector.startsWith(".") && node.classList.contains(selector.slice(1));
    };
    return this.children.flatMap(node => [...(matches(node) ? [node] : []), ...node.querySelectorAll(selector)]);
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
}
const root = new Node("root");
const el = (tag, attrs) => new Node(tag, attrs);
const deps = {el, time24: () => "21:59", chatMarkdownHtmlCached: text => text,
  upgradeCandidateFileRefs: () => {}, iconSvg: () => "", copyToClipboard: () => {},
  setToast: () => {}, chatAssistantDedupeKey: T.chatAssistantDedupeKey,
  setTimeout: () => {}, consoleError: (...args) => {throw new Error(args.join(" "));}};
const divider = {role: "system", kind: "compaction", message_id: "compaction:100", history_cursor: "h100",
  ts: 1794585540, tokens_before: 758000, tokens_after: 47000, kept_messages: 3, trigger: "auto",
  summary: "NEVER DISPLAY THIS", text: "NEVER DISPLAY THIS"};
function make(event) { return M.safeMakeRow(event, {ts: event.ts, pending: false}, deps).row; }
const row = make(divider);
assert.equal(row.attrs.role, "separator"); assert.equal(row.attrs["aria-label"], "Context compacted");
assert.equal(row.dataset.role, "system"); assert.equal(row.dataset.messageId, divider.message_id);
assert.equal(row.dataset.historyCursor, "h100");
assert.equal(row.querySelector(".compactionTokens").textContent, "758k → 47k");
assert.equal(row.querySelector(".compactionMeta").textContent, "auto · kept last 3 messages · 21:59");
assert.equal(row.querySelectorAll("button").length, 0); assert.equal(row.querySelector(".md"), null);
assert(!row.textContent.includes("NEVER")); assert.equal(M.rowSearchText(row), "");
const bare = make({role: "system", kind: "compaction"});
assert.equal(bare.textContent, "Context compacted"); assert.equal(bare.querySelector(".compactionMeta"), null);
const beforeOnly = make({role: "system", kind: "compaction", tokens_before: 1200000, kept_messages: 1});
assert.equal(beforeOnly.querySelector(".compactionTokens").textContent, "1.2M");
assert.equal(beforeOnly.querySelector(".compactionMeta").textContent, "kept last 1 message");
assert.equal(make({kind: "compaction", tokens_before: 0, tokens_after: 999}).querySelector(".compactionTokens").textContent, "0 → 999");
const clean = T.normalizeTailEvent(divider);
assert.equal(clean.kind, "compaction"); assert.equal(clean.tokens_after, 47000);
assert(!("summary" in clean)); assert(!("text" in clean));
const identity = T.createTranscriptEventRuntime({eventKey: T.eventKey, pendingMatchKey: T.pendingMatchKey,
  normalizePendingText: T.normalizeTextForPendingMatch, assistantDedupeKey: T.chatAssistantDedupeKey, maxRecentEventKeys: 320});
const bottom = new Node("bottom"), older = new Node("older"); root.appendChild(older); root.appendChild(bottom);
let decorations = 0;
const scroll = {captureScrollPosition: () => ({}), preserveScrollFrom: () => {}, snapshot: () => ({renderedAtLiveTail: true, autoScroll: false}),
  scheduleScrollToBottom: () => {}, syncJumpButton: () => {}, setRenderedAtLiveTail: () => {}, shouldStickToBottom: () => true,
  markLiveTail: () => {}, disableAutoScroll: () => {}, setScrollTop: () => {}};
const dom = T.createTranscriptDomRuntime({root, olderWrap: older, bottomSentinel: bottom, el,
  ymd: () => "2026-11-13", dayLabel: () => "Today", getRenderedRows: () => M.renderedMessageRows(root),
  trimRenderedRowTargets: M.trimRenderedRowTargets, trimRowsBeforeViewportTargets: M.trimRowsBeforeViewportTargets,
  afterDecorate: () => decorations++, scrollRuntime: scroll, defaultWindowRows: 100});
const runtime = T.createTranscriptRenderRuntime({root, bottomSentinel: bottom, document: {createDocumentFragment: () => new Node("fragment")},
  safeMakeRow: (event, opts) => M.safeMakeRow(event, opts, deps),
  normalizeEvents: (events, opts) => T.normalizedTranscriptEvents(events, {...opts, eventKey: T.eventKey, takePendingMatch: () => null}),
  consumePendingUserIfMatches: () => false, isDuplicateEvent: identity.isDuplicateEvent,
  isAdjacentAssistantDuplicateEvent: ev => identity.isAdjacentAssistantDuplicateEvent(ev, {renderedAtLiveTail: true, rows: M.renderedMessageRows(root)}),
  markEventSeen: identity.markEventSeen, markFirstPaint: () => {}, restorePendingRows: () => {}, resetRecentEvents: identity.resetRecentEvents,
  setOlderState: () => {}, firstVisibleMessageRow: () => null, getScrollTop: () => 0, getSelectedSessionId: () => "s1",
  domRuntime: dom, scrollRuntime: scroll, typingRowRuntime: {anchor: () => bottom}, historySlackRows: 100});
const asst = {role: "assistant", text: "before", ts: divider.ts - 10, message_id: "a1"};
const after = {...asst, text: "after", ts: divider.ts + 10, message_id: "a2"};
assert(runtime.renderTranscript([asst, divider, after]));
assert.equal(root.querySelectorAll(".compaction-row").length, 1);
assert.equal(root.querySelectorAll(".day-sep").length, 1);
assert(!root.querySelectorAll(".assistant").some(n => n.classList.contains("grouped")));
assert.equal(M.renderedMessageRows(root).length, 3);
assert.equal(M.loadedCopyMessageRows(root).length, 2);
assert(!runtime.appendEvent(divider)); // replay of a canonical ID is inert
const live = {...divider, message_id: "compaction:200", ts: divider.ts + 20};
assert(runtime.appendEvent(live)); assert.equal(root.querySelectorAll(".compaction-row").length, 2);
const olderDivider = {...divider, message_id: "compaction:50", ts: divider.ts - 20};
assert(runtime.prependOlderEvents([olderDivider, divider]));
assert.equal(root.querySelectorAll(".compaction-row").length, 3);
assert.equal(M.oldestRenderedHistoryCursor(M.renderedMessageRows(root)), "h100");
assert(runtime.renderTranscript([after])); assert.equal(root.querySelectorAll(".compaction-row").length, 0);
assert(runtime.renderDetachedTranscriptWindow([divider, after]));
assert.equal(root.querySelectorAll(".compaction-row").length, 1);
const copied = ctx.window.CodoxearConversationCopy.formatConversationForCopyResult([asst, divider, after]);
assert.equal(copied.messageCount, 2); assert(!copied.text.includes("compacted")); assert(!copied.text.includes("NEVER"));
const cache = new Map();
T.rememberTailSnapshot(cache, "s1", {thread_id: "t", log_path: "/log"}, {transcript_state: "bound", thread_id: "t", log_path: "/log", events: [divider]}, 100);
assert.equal(cache.get("s1").events[0].kind, "compaction");
assert(decorations >= 5);
console.log(JSON.stringify({divider: row.textContent, replayDeduped: true, appendPrependReplace: true, messageCount: copied.messageCount}));
'''
    script = "const MODULES = " + json.dumps(modules) + ";\n" + harness
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
    output = json.loads(result.stdout)
    assert output["replayDeduped"] and output["appendPrependReplace"]
    assert output["messageCount"] == 2
