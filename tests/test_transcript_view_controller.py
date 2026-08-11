"""Behavioral state-machine tests for the transcript view authority."""

from __future__ import annotations
from frontend_module_loader import module_path

import json
import subprocess
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VIEW_JS = module_path("app_transcript_view.js")


def run_node(script: str) -> dict:
    result = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def test_transcript_view_serializes_append_prepend_and_replace() -> None:
    script = textwrap.dedent(
        """
        const vm = require("vm");
        const source = __SOURCE__;
        const calls = [];
        let atBottom = true;
        const scroll = {
          snapshot: () => ({ renderedAtLiveTail: true }),
          shouldStickToBottom: () => atBottom,
          enableAutoScroll: () => calls.push("enable"),
          markLiveTail: () => calls.push("tail"),
          scheduleScrollToBottom: () => calls.push("bottom"),
          handleScroll: () => calls.push("scroll"),
        };
        const root = {
          querySelectorAll: () => [],
          insertBefore: () => {},
        };
        const ctx = { window: {}, console };
        vm.createContext(ctx);
        vm.runInContext(source, ctx);
        const controller = ctx.window.CodoxearTranscriptView.createTranscriptViewController({
          root,
          bottomSentinel: {},
          document: {},
          el: () => ({ appendChild: () => {}, dataset: {} }),
          messageRows: {
            safeMakeRow: () => ({ row: {} }),
            renderedMessageRows: () => [], loadedUserMessageRows: () => [], loadedCopyMessageRows: () => [],
            rowSearchText: () => "", clearChatSearchMarks: () => {}, applyChatSearchMarks: () => {},
            oldestRenderedHistoryCursor: () => null, firstVisibleMessageRow: () => null,
          },
          transcript: {
            createTranscriptRenderRuntime: () => ({
              appendEvent: (event) => { calls.push(["append", event.text]); return true; },
              prependOlderEvents: (events) => { calls.push(["prepend", events.map((event) => event.text)]); return true; },
              renderTranscript: (events) => { calls.push(["replace", events.map((event) => event.text)]); return true; },
              renderDetachedTranscriptWindow: () => true,
            }),
          },
          getSelectedSessionId: () => "session",
          getMessageRowDeps: () => ({}),
          policyRuntime: { domRuntime: {}, scrollRuntime: scroll, setOlderState: (state) => calls.push(["older", state]), getScrollTop: () => 44 },
          renderRuntime: { normalizeEvents: () => [], consumePendingUserIfMatches: () => false, isDuplicateEvent: () => false,
            isAdjacentAssistantDuplicateEvent: () => false, markEventSeen: () => {}, markFirstPaint: () => {},
            restorePendingRows: () => {}, resetRecentEvents: () => {}, firstVisibleMessageRow: () => null,
            getScrollTop: () => 44, getSelectedSessionId: () => "session", domRuntime: {}, scrollRuntime: scroll,
            typingRowRuntime: { anchor: () => ({}) }, setOlderState: () => {}, historySlackRows: 1 },
        });
        controller.setHistory({ cursor: "cursor-1", nextHasMore: true });
        controller.appendEvents([{ text: "live" }]);
        atBottom = false;
        controller.observeScroll("handleScroll");
        const began = controller.beginOlderLoad();
        controller.appendEvents([{ text: "queued" }]);
        controller.prependEvents([{ text: "older" }], { cursor: "cursor-0", nextHasMore: true });
        const browsing = controller.state();
        controller.replaceWith([{ text: "latest" }]);
        process.stdout.write(JSON.stringify({ calls, began, browsing, final: controller.state() }));
        """
    ).replace("__SOURCE__", json.dumps(VIEW_JS.read_text(encoding="utf-8")))
    result = run_node(script)

    assert result["began"] is True
    assert result["browsing"] == {
        "state": "BROWSING",
        "scrollTop": 44,
        "renderedAtLiveTail": True,
        "hasMore": True,
    }
    assert result["final"]["state"] == "LIVE"
    assert result["final"]["hasMore"] is False
    assert ["append", "live"] in result["calls"]
    assert ["prepend", ["older"]] in result["calls"]
    assert ["append", "queued"] in result["calls"]
    assert ["replace", ["latest"]] in result["calls"]
