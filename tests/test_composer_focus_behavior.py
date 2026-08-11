"""Behavioral regression coverage for composer focus and model-refresh contracts."""

from __future__ import annotations
from frontend_module_loader import module_path

import json
from pathlib import Path
import subprocess
import textwrap


ROOT = Path(__file__).resolve().parents[1]
COMPOSER_SOURCE = (module_path("app_composer.js")).read_text(encoding="utf-8")
POLLING_SOURCE = (module_path("app_polling.js")).read_text(encoding="utf-8")
TRANSCRIPT_SOURCE = (module_path("app_transcript.js")).read_text(encoding="utf-8")
MESSAGE_FLOW_SOURCE = (module_path("app_message_flow.js")).read_text(encoding="utf-8")


def _run_node(script: str) -> dict:
    result = subprocess.run(["node", "-e", script], cwd=ROOT, text=True, capture_output=True, check=False, timeout=20)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_composer_escape_blurs_only_without_an_open_dialog_and_successful_send_blurs() -> None:
    script = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}}, console }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(COMPOSER_SOURCE)}, ctx);

        function node() {{
          const listeners = new Map();
          return {{
            style: {{}}, value: "", textContent: "", title: "", disabled: false, scrollHeight: 32, children: [],
            classList: {{ toggle() {{}} }},
            addEventListener(type, handler) {{ listeners.set(type, handler); }},
            removeEventListener(type) {{ listeners.delete(type); }},
            emit(type, event) {{ listeners.get(type)(event); }},
            setAttribute() {{}}, removeAttribute() {{}}, appendChild(child) {{ this.children.push(child); return child; }},
            focus() {{}}, blur() {{ this.blurCount = (this.blurCount || 0) + 1; }},
          }};
        }}
        function harness(dialogOpen) {{
          const form = node(); form.requestSubmit = () => {{}};
          const textarea = node();
          const sent = [];
          ctx.window.CodoxearComposer.createComposerController({{
            form, textarea, msgPh: node(), sendBtn: node(), sendChoice: node(), sendChoiceBackdrop: node(),
            sendChoiceNowBtn: node(), sendChoiceLaterBtn: node(), sendChoiceCancelBtn: node(),
            getSelected: () => "sid", getSessionInfo: () => ({{ session_id: "sid", agent_backend: "pi" }}),
            sessionLaunchFailed: () => false, getSending: () => false, getCurrentRunning: () => false,
            getStagedAttachments: () => [], isModalOpen: () => dialogOpen, api: async () => ({{}}),
            setToast: () => {{}}, setPollFastUntilMs: () => {{}}, kickPoll: () => {{}},
            sendText: async (text) => {{ sent.push(text); return true; }}, enqueueComposerText: async () => true,
            prepareModalOpen: () => {{}}, afterModalVisibilityChanged: () => {{}}, restoreModalFocus: () => {{}},
            storageGetItem: () => null, storageSetItem: () => {{}}, storageRemoveItem: () => {{}},
            requestFrame: (callback) => callback(), getComputedStyle: () => ({{ minHeight: "32px" }}),
            activeElement: () => null, isHTMLElement: () => false,
          }});
          return {{ form, textarea, sent }};
        }}
        (async () => {{
          const normal = harness(false);
          const escaped = {{ prevented: false, stopped: false }};
          normal.textarea.emit("keydown", {{ key: "Escape", preventDefault() {{ escaped.prevented = true; }}, stopPropagation() {{ escaped.stopped = true; }} }});
          normal.textarea.value = "send this";
          await normal.form.onsubmit({{ preventDefault() {{}} }});
          const modal = harness(true);
          const modalEscape = {{ prevented: false, stopped: false }};
          modal.textarea.emit("keydown", {{ key: "Escape", preventDefault() {{ modalEscape.prevented = true; }}, stopPropagation() {{ modalEscape.stopped = true; }} }});
          process.stdout.write(JSON.stringify({{
            escape: {{ blurCount: normal.textarea.blurCount, ...escaped }},
            send: {{ sent: normal.sent, value: normal.textarea.value }},
            modalEscape: {{ blurCount: modal.textarea.blurCount || 0, ...modalEscape }},
          }}));
        }})().catch((error) => {{ console.error(error); process.exit(1); }});
        """
    )

    result = _run_node(script)

    # The first blur is Esc; the second is confirmed delivery. The modal keeps
    # Esc available to its dialog-level handler instead of swallowing it here.
    assert result == {
        "escape": {"blurCount": 2, "prevented": True, "stopped": True},
        "send": {"sent": ["send this"], "value": ""},
        "modalEscape": {"blurCount": 0, "prevented": False, "stopped": False},
    }


def test_model_command_refreshes_session_listing_again_after_backend_applies_change() -> None:
    script = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}}, console, Date }};
        vm.createContext(ctx);
        for (const source of {json.dumps([POLLING_SOURCE, TRANSCRIPT_SOURCE, MESSAGE_FLOW_SOURCE])}) vm.runInContext(source, ctx);
        const timers = [];
        let refreshes = 0;
        let sending = false;
        const noop = () => {{}};
        const typingRowRuntime = {{
          snapshot: () => ({{ stats: {{ thinking: 0, thinkingTokens: 0, tools: 0 }} }}),
          updateTypingStats: noop, updateSubagentGauge: noop, resetTypingStats: noop,
        }};
        const controller = ctx.window.CodoxearMessageFlow.createMessageFlowController({{
          getSelected: () => "sid", getGeneration: () => 1, isAppDisposed: () => false,
          getTurnOpen: () => false, setTurnOpen: noop,
          getSessionInfo: () => ({{ session_id: "sid", agent_backend: "pi" }}), patchSessionInfo: noop, sessionLaunchFailed: () => false,
          api: async () => ({{ queued: false, queue_len: 0 }}), resolveAppUrl: (path) => path, handleAppAuthLoss: noop,
          refreshSessions: async () => {{ refreshes += 1; }}, openSession: async () => null, clearSelectedSessionAfterRemoval: noop,
          activeTranscriptSnapshot: () => ({{ state: "bound", liveCursor: "cursor", logPath: "/tmp/log" }}),
          updateSessionTranscriptSlot: () => ({{ ignoredStaleBound: false, current: {{ state: "bound" }} }}),
          renderPendingTranscriptSlot: noop, renderSessionTail: noop, applySessionRuntimeFromTail: noop,
          resetChatRenderState: noop, setAttachCount: noop, setLiveCursor: noop, appendEvent: noop, appendTailSnapshotEvents: noop,
          setStatus: noop, setContext: noop, setTyping: noop, setSubagentsRunning: noop, updateSessionTitle: noop,
          initPageLimit: () => 24, typingRowRuntime,
          getSending: () => sending, setSending: (value) => {{ sending = value; }}, getCurrentRunning: () => false, setCurrentRunning: noop,
          getStagedAttachments: () => [], normalizedStagedAttachments: () => [], setSelectedSessionPendingAttachment: noop,
          syncSendButtonState: noop, syncAttachButtonState: noop, syncQueueSubmitState: noop, syncRecoveryUiForSession: noop,
          confirmAction: async () => false, setToast: noop, isTranscriptRenewalCommand: () => false, nextLocalEchoId: () => 1,
          renderedAtLiveTail: () => true, clearTranscriptDom: noop, clearRenderedTranscriptRange: noop, setOlderState: noop,
          getSessionTranscriptSlot: () => ({{ epoch: 0 }}), addPendingUser: noop, deleteTailCache: noop, beginTranscriptRenewal: noop,
          clearLiveCursor: noop, invalidateOlderLoad: noop, dropPendingUser: noop, removePendingUserRow: noop, hasPendingForSession: () => false,
          visibilityState: () => "visible", navigatorValue: () => ({{ onLine: true }}), EventSource: null, AbortController: null,
          setTimeout: (callback, delay) => {{ timers.push({{ callback, delay }}); return timers.length; }}, clearTimeout: noop,
          now: () => 1000, consoleWarn: noop, consoleError: noop,
        }});
        (async () => {{
          const sent = await controller.sendText("/model configured/model");
          const beforeDelayed = refreshes;
          const delayed = timers.filter((timer) => timer.delay === 1500);
          delayed.forEach((timer) => timer.callback());
          await Promise.resolve();
          process.stdout.write(JSON.stringify({{ sent, beforeDelayed, refreshes, delays: timers.map((timer) => timer.delay), delayedCount: delayed.length }}));
        }})().catch((error) => {{ console.error(error); process.exit(1); }});
        """
    )

    assert _run_node(script) == {"sent": True, "beforeDelayed": 1, "refreshes": 2, "delays": [0, 1500], "delayedCount": 1}
