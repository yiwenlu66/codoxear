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
        let renderOptions = null;
        let safeRowDeps = null;
        const rowDependencyBag = {
          el: () => ({}), chatMarkdownHtmlCached: () => "", upgradeCandidateFileRefs: () => {},
          time24: () => "", iconSvg: () => "", copyToClipboard: () => {}, setToast: () => {},
          chatAssistantDedupeKey: () => "", setTimeout: () => {}, consoleError: () => {},
        };
        const renderDependencyBag = {
          normalizeEvents: () => [], consumePendingUserIfMatches: () => false, isDuplicateEvent: () => false,
          isAdjacentAssistantDuplicateEvent: () => false, markEventSeen: () => {}, markFirstPaint: () => {},
          restorePendingRows: () => {}, resetRecentEvents: () => {}, setOlderState: () => {},
          firstVisibleMessageRow: () => null, getScrollTop: () => 44, getSelectedSessionId: () => "session",
          domRuntime: {}, scrollRuntime: null, typingRowRuntime: { anchor: () => ({}) }, historySlackRows: 1,
        };
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
            safeMakeRow: (event, rowOptions, deps) => { safeRowDeps = deps; return { row: {} }; },
            renderedMessageRows: () => [], loadedUserMessageRows: () => [], loadedCopyMessageRows: () => [],
            rowSearchText: () => "", clearChatSearchMarks: () => {}, applyChatSearchMarks: () => {},
            oldestRenderedHistoryCursor: () => null, firstVisibleMessageRow: () => null,
          },
          transcript: {
            createTranscriptRenderRuntime: (options) => {
              renderOptions = options;
              return {
                appendEvent: (event) => { calls.push(["append", event.text]); return true; },
                prependOlderEvents: (events) => { calls.push(["prepend", events.map((event) => event.text)]); return true; },
                renderTranscript: (events) => { calls.push(["replace", events.map((event) => event.text)]); return true; },
                renderDetachedTranscriptWindow: () => true,
              };
            },
          },
          getSelectedSessionId: () => "session",
          getMessageRowDeps: () => rowDependencyBag,
          policyRuntime: { domRuntime: {}, scrollRuntime: scroll, setOlderState: (state) => calls.push(["older", state]), getScrollTop: () => 44 },
          renderRuntime: {
            normalizeEvents: renderDependencyBag.normalizeEvents,
            consumePendingUserIfMatches: renderDependencyBag.consumePendingUserIfMatches,
            isDuplicateEvent: renderDependencyBag.isDuplicateEvent,
            isAdjacentAssistantDuplicateEvent: renderDependencyBag.isAdjacentAssistantDuplicateEvent,
            markEventSeen: renderDependencyBag.markEventSeen,
            markFirstPaint: renderDependencyBag.markFirstPaint,
            restorePendingRows: renderDependencyBag.restorePendingRows,
            resetRecentEvents: renderDependencyBag.resetRecentEvents,
            setOlderState: renderDependencyBag.setOlderState,
            firstVisibleMessageRow: renderDependencyBag.firstVisibleMessageRow,
            getScrollTop: renderDependencyBag.getScrollTop,
            getSelectedSessionId: renderDependencyBag.getSelectedSessionId,
            domRuntime: renderDependencyBag.domRuntime,
            scrollRuntime: scroll,
            typingRowRuntime: renderDependencyBag.typingRowRuntime,
            historySlackRows: renderDependencyBag.historySlackRows,
          },
        });
        renderOptions.safeMakeRow({ role: "assistant", text: "row" }, {});
        controller.setHistory({ cursor: "cursor-1", nextHasMore: true });
        controller.appendEvents([{ text: "live" }]);
        atBottom = false;
        controller.observeScroll("handleScroll");
        const began = controller.beginOlderLoad();
        controller.appendEvents([{ text: "queued" }]);
        controller.prependEvents([{ text: "older" }], { cursor: "cursor-0", nextHasMore: true });
        const browsing = controller.state();
        controller.replaceWith([{ text: "latest" }]);
        const renderKeys = [
          "normalizeEvents", "consumePendingUserIfMatches", "isDuplicateEvent", "isAdjacentAssistantDuplicateEvent",
          "markEventSeen", "markFirstPaint", "restorePendingRows", "resetRecentEvents", "setOlderState",
          "firstVisibleMessageRow", "getScrollTop", "getSelectedSessionId", "domRuntime", "scrollRuntime",
          "typingRowRuntime", "historySlackRows",
        ];
        const rowKeys = [
          "el", "chatMarkdownHtmlCached", "upgradeCandidateFileRefs", "time24", "iconSvg", "copyToClipboard",
          "setToast", "chatAssistantDedupeKey", "setTimeout", "consoleError",
        ];
        process.stdout.write(JSON.stringify({
          calls, began, browsing, final: controller.state(),
          renderRuntimeContract: renderKeys.every((key) => renderOptions[key] === (key === "scrollRuntime" ? scroll : renderDependencyBag[key])) &&
            renderOptions.root === root && renderOptions.bottomSentinel !== null && typeof renderOptions.safeMakeRow === "function",
          messageRowContract: rowKeys.every((key) => safeRowDeps[key] === rowDependencyBag[key]) && safeRowDeps.selectedSessionId === "session",
        }));
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
    assert result["renderRuntimeContract"] is True
    assert result["messageRowContract"] is True
    assert ["append", "live"] in result["calls"]
    assert ["prepend", ["older"]] in result["calls"]
    assert ["append", "queued"] in result["calls"]
    assert ["replace", ["latest"]] in result["calls"]


def _make_view_controller_script(body: str) -> str:
    """Minimal harness: a transcript view pinned at the bottom with no
    scrollable overflow (scrollHeight == clientHeight), so no scroll-driven
    LIVE -> BROWSING transition can ever fire."""
    return textwrap.dedent(
        """
        const vm = require("vm");
        const source = __SOURCE__;
        const calls = [];
        const scroll = {
          snapshot: () => ({ renderedAtLiveTail: true }),
          shouldStickToBottom: () => true,
          enableAutoScroll: () => {},
          disableAutoScroll: () => {},
          markLiveTail: () => {},
          scheduleScrollToBottom: () => {},
          handleScroll: () => {},
        };
        const root = { querySelectorAll: () => [], insertBefore: () => {} };
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
              appendEvent: () => true,
              prependOlderEvents: (events) => { calls.push(["prepend", events.length]); return true; },
              renderTranscript: () => true,
              renderDetachedTranscriptWindow: () => true,
            }),
          },
          getSelectedSessionId: () => "session",
          getMessageRowDeps: () => ({}),
          policyRuntime: { domRuntime: {}, scrollRuntime: scroll, setOlderState: () => {}, getScrollTop: () => 0 },
          renderRuntime: { scrollRuntime: scroll, typingRowRuntime: { anchor: () => ({}) } },
        });
        __BODY__
        """
    ).replace("__SOURCE__", json.dumps(VIEW_JS.read_text(encoding="utf-8"))).replace("__BODY__", body)


def test_begin_older_load_is_allowed_from_live_state() -> None:
    """A transcript shorter than the viewport stays in LIVE forever (it can
    never scroll into BROWSING). An explicit older-page request must still
    begin, otherwise the 'Load older messages' button is permanently dead."""
    script = _make_view_controller_script(
        """
        controller.setHistory({ cursor: "cursor-1", nextHasMore: true });
        const initialState = controller.state().state;
        const began = controller.beginOlderLoad();
        const loadingState = controller.state().state;
        const prepended = controller.prependEvents([{ text: "older" }], { cursor: "cursor-0", nextHasMore: true });
        const finalState = controller.state().state;
        controller.setHistory({ cursor: null, nextHasMore: false });
        const beganWithoutMore = controller.beginOlderLoad();
        process.stdout.write(JSON.stringify({ initialState, began, loadingState, prepended, finalState, beganWithoutMore, calls }));
        """
    )
    result = run_node(script)

    assert result["initialState"] == "LIVE"
    assert result["began"] is True
    assert result["loadingState"] == "LOADING_OLDER"
    assert result["prepended"] is True
    assert result["finalState"] == "BROWSING"
    assert result["beganWithoutMore"] is False
    assert ["prepend", 1] in result["calls"]
