"""Behavioral VM coverage for the shared session-card content tree.

The touch and fine-pointer variants intentionally differ only in action exposure:
touch binds the swipe gesture, while desktop adds `.desktop` and exposes actions
on hover.  The title, badges, metadata, and card border owner must be the same
DOM structure in both variants.
"""

import json
import subprocess
import textwrap
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


HARNESS = textwrap.dedent(
    r"""
    const fs = require("fs");
    const vm = require("vm");
    const source = fs.readFileSync("codoxear/static/app_sessions.js", "utf8");

    function classes(node) {
      return String(node.attrs.class || "").split(/\s+/).filter(Boolean).sort();
    }
    function node(tag, attrs = {}, children = []) {
      const listeners = {};
      const value = {
        tag, attrs, children: [], dataset: {}, style: {}, parentNode: null,
        appendChild(child) { if (child) { child.parentNode = this; this.children.push(child); } return child; },
        remove() { if (!this.parentNode) return; const i = this.parentNode.children.indexOf(this); if (i >= 0) this.parentNode.children.splice(i, 1); this.parentNode = null; },
        addEventListener(type, listener) { (listeners[type] ||= []).push(listener); },
        listeners,
        classList: {
          add(name) { if (!classes(value).includes(name)) value.attrs.class = `${value.attrs.class || ""} ${name}`.trim(); },
          remove(name) { value.attrs.class = classes(value).filter((entry) => entry !== name).join(" "); },
        },
      };
      Object.defineProperty(value, "childElementCount", { get: () => value.children.length });
      Object.defineProperty(value, "innerHTML", { get: () => "", set: () => { value.children = []; } });
      children.forEach((child) => value.appendChild(child));
      return value;
    }
    function el(tag, attrs, children) { return node(tag, attrs || {}, Array.isArray(children) ? children : []); }
    function byClass(root, name) {
      if (classes(root).includes(name)) return root;
      for (const child of root.children) { const found = byClass(child, name); if (found) return found; }
      return null;
    }
    function topology(root) {
      return { tag: root.tag, classes: classes(root), children: root.children.map(topology) };
    }
    function textByClass(root, name) {
      const found = byClass(root, name);
      function text(node) { return `${node.attrs.text || ""}${node.children.map(text).join("")}`; }
      return found ? text(found) : "";
    }

    const ctx = { window: {}, performance: { now: () => 0 } };
    vm.createContext(ctx);
    vm.runInContext(source, ctx);

    function renderVariant(swipeActions) {
      const wrap = node("div");
      const selections = [];
      const sidebarOpen = [];
      const controller = ctx.window.CodoxearSessions.createSessionsController({
        sessionsWrap: wrap, sidebarEmptyHint: node("div"), el, iconSvg: (name) => `<svg>${name}</svg>`,
        sidebarRenderSignature: () => `${swipeActions}`, sidebarSessionEntries: (sessions) => sessions.map((session) => ({ type: "session", session })),
        sessionDisplayName: () => "Unified card", sessionLaunchFailed: () => false, sessionLaunchPending: () => false,
        redactedLaunchErrorText: () => "", fmtRelativeAge: () => "2m", sidebarEffortCode: () => "hi", sidebarModelText: () => "model-2026",
        baseName: () => "codoxear", sessionIsFast: () => true, agentBackendLogoPath: () => "/pi.svg", agentBackendDisplayName: () => "Pi",
        sessionAgentBackend: () => "pi", sessionLaunchIcon: () => "terminal", sessionLaunchLabel: () => "Terminal session",
        confirmAction: async () => true, api: async () => ({}), clearDeletedSessionClientState: () => {}, refreshSessions: async () => {},
        setToast: () => {}, openEditSession: () => {}, duplicateSession: async () => {}, selectSession: (id) => selections.push(id),
        setSidebarOpen: (open) => sidebarOpen.push(open), now: () => 0,
      });
      controller.renderSessions([{
        session_id: "s1", busy: true, queue_len: 2, unread_count: 1, cwd: "/work/codoxear", git_branch: "main", agent_backend: "pi", owned: false,
      }], { selectedId: "s1", swipeActions });
      const card = wrap.children[0];
      const swipe = byClass(card, "sessionSwipe");
      const content = byClass(card, "sessionContent");
      const title = textByClass(card, "titleText");
      const badges = byClass(card, "sessionBadges").children.map((badge) => badge.attrs.text);
      const metadata = textByClass(card, "metaText");
      card.onclick();
      return {
        cardClasses: classes(card), cardTopology: topology(card), contentTopology: topology(content),
        swipeX: Object.hasOwn(content.dataset, "swipeX") ? content.dataset.swipeX : null,
        swipeListeners: Object.keys(content.listeners).sort(), title, badges, metadata, selections, sidebarOpen,
        actionLabels: swipe.children.slice(0, 2).flatMap((actions) => actions.children.map((button) => button.attrs["aria-label"])),
      };
    }

    process.stdout.write(JSON.stringify({ touch: renderVariant(true), desktop: renderVariant(false) }));
    """
)


def run_harness() -> dict:
    result = subprocess.run(
        ["node", "-e", HARNESS], cwd=REPO, capture_output=True, text=True, check=False, timeout=20
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_session_card_content_tree_is_identical_across_reveal_branches() -> None:
    result = run_harness()
    touch = result["touch"]
    desktop = result["desktop"]

    assert touch["cardTopology"]["children"] == desktop["cardTopology"]["children"]
    assert touch["contentTopology"] == desktop["contentTopology"]
    assert touch["title"] == desktop["title"] == "Unified card"
    assert touch["badges"] == desktop["badges"] == ["queue 2", "unread 1"]
    assert touch["metadata"] == desktop["metadata"] == "2m | model-2026 ·hi | codoxear | main"
    assert touch["actionLabels"] == desktop["actionLabels"] == ["Delete session", "Edit conversation", "Duplicate session"]

    assert "desktop" not in touch["cardClasses"]
    assert "desktop" in desktop["cardClasses"]
    assert touch["swipeX"] == "0"
    assert touch["swipeListeners"] == ["pointercancel", "pointerdown", "pointermove", "pointerup"]
    assert desktop["swipeX"] is None
    assert desktop["swipeListeners"] == []
    assert touch["sidebarOpen"] == [False]
    assert desktop["sidebarOpen"] == []
    assert touch["selections"] == desktop["selections"] == ["s1"]
