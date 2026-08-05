"""Behavioral pin for steering a running session from the send-choice dialog."""

from __future__ import annotations

import json
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
POLLING_SOURCE = (ROOT / "codoxear" / "static" / "app_polling.js").read_text(encoding="utf-8")
TRANSCRIPT_SOURCE = (ROOT / "codoxear" / "static" / "app_transcript.js").read_text(encoding="utf-8")
MESSAGE_FLOW_SOURCE = (ROOT / "codoxear" / "static" / "app_message_flow.js").read_text(encoding="utf-8")
COMPOSER_SOURCE = (ROOT / "codoxear" / "static" / "app_composer.js").read_text(encoding="utf-8")


def _run_node(script: str) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".js", encoding="utf-8") as script_file:
        script_file.write(script)
        script_file.flush()
        completed = subprocess.run(
            ["node", script_file.name],
            check=True,
            capture_output=True,
            text=True,
        )
    return json.loads(completed.stdout)


def test_send_now_steers_busy_session_via_confirmed_send_without_interrupting() -> None:
    """A running turn opens a choice first; Send now then uses only ``/send``."""
    script = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}}, console, Date }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(POLLING_SOURCE)}, ctx);
        vm.runInContext({json.dumps(TRANSCRIPT_SOURCE)}, ctx);
        vm.runInContext({json.dumps(MESSAGE_FLOW_SOURCE)}, ctx);
        vm.runInContext({json.dumps(COMPOSER_SOURCE)}, ctx);

        function node() {{
          return {{
            addEventListener: () => {{}}, removeEventListener: () => {{}},
            setAttribute: () => {{}}, removeAttribute: () => {{}},
            classList: {{ toggle: () => {{}} }}, style: {{}}, value: "", textContent: "",
            scrollHeight: 32, disabled: false, focus: () => {{}}, blur: () => {{}},
          }};
        }}

        const state = {{ apiCalls: [], queueCalls: [], busyChecks: [], sending: false }};
        const noop = () => {{}};
        const typingRowRuntime = {{
          snapshot: () => ({{ stats: {{ thinking: 0, thinkingTokens: 0, thinkingMode: "blocks", tools: 0 }} }}),
          updateTypingStats: noop, updateSubagentGauge: noop, resetTypingStats: noop,
        }};
        const messageFlow = ctx.window.CodoxearMessageFlow.createMessageFlowController({{
          getSelected: () => "busy-session", getGeneration: () => 1, isAppDisposed: () => false,
          getTurnOpen: () => true, setTurnOpen: noop,
          getSessionInfo: () => ({{ session_id: "busy-session", agent_backend: "pi" }}), patchSessionInfo: noop,
          sessionLaunchFailed: () => false,
          api: async (path, options) => {{
            state.apiCalls.push({{ path, method: options.method, body: options.body }});
            if (path.includes("interrupt")) throw new Error("Send now must not interrupt the running turn");
            return {{ queued: false, queue_len: 0, busy: true }};
          }},
          resolveAppUrl: (path) => `http://example.test${{path}}`, handleAppAuthLoss: noop, refreshSessions: async () => [],
          openSession: async () => null, clearSelectedSessionAfterRemoval: noop,
          activeTranscriptSnapshot: () => ({{ state: "bound", liveCursor: "c1", logPath: "/tmp/log.jsonl" }}),
          updateSessionTranscriptSlot: () => ({{ ignoredStaleBound: false, current: {{ state: "bound" }} }}),
          renderPendingTranscriptSlot: noop, renderSessionTail: noop, applySessionRuntimeFromTail: noop,
          resetChatRenderState: noop, setAttachCount: noop, setLiveCursor: noop, appendEvent: noop,
          appendTailSnapshotEvents: noop, setStatus: noop, setContext: noop, setTyping: noop,
          setSubagentsRunning: noop, updateSessionTitle: noop, initPageLimit: () => 60, typingRowRuntime,
          getSending: () => state.sending, setSending: (value) => {{ state.sending = Boolean(value); }},
          getCurrentRunning: () => {{ state.busyChecks.push(true); return true; }}, setCurrentRunning: noop,
          getStagedAttachments: () => [], normalizedStagedAttachments: () => [], setSelectedSessionPendingAttachment: noop,
          syncSendButtonState: noop, syncAttachButtonState: noop, syncQueueSubmitState: noop, syncRecoveryUiForSession: noop,
          confirmAction: async () => false, setToast: noop, isTranscriptRenewalCommand: () => false,
          nextLocalEchoId: () => 1, renderedAtLiveTail: () => true, clearTranscriptDom: noop,
          clearRenderedTranscriptRange: noop, setOlderState: noop, getSessionTranscriptSlot: () => ({{ epoch: 0 }}),
          addPendingUser: noop, deleteTailCache: noop, beginTranscriptRenewal: noop, clearLiveCursor: noop,
          invalidateOlderLoad: noop, dropPendingUser: noop, removePendingUserRow: noop, hasPendingForSession: () => false,
          visibilityState: () => "visible", navigatorValue: () => ({{ onLine: true }}), EventSource: null,
          AbortController: null, setTimeout: () => 0, clearTimeout: noop, now: () => 1000,
          consoleWarn: noop, consoleError: noop,
        }});

        const [form, textarea, msgPh, sendBtn, sendChoice, sendChoiceBackdrop, nowBtn, laterBtn, cancelBtn] = Array.from({{ length: 9 }}, node);
        form.requestSubmit = noop;
        const composer = ctx.window.CodoxearComposer.createComposerController({{
          form, textarea, msgPh, sendBtn, sendChoice, sendChoiceBackdrop,
          sendChoiceNowBtn: nowBtn, sendChoiceLaterBtn: laterBtn, sendChoiceCancelBtn: cancelBtn,
          getSelected: () => "busy-session", getSessionInfo: () => ({{ agent_backend: "pi" }}), sessionLaunchFailed: () => false,
          getSending: () => state.sending,
          getCurrentRunning: () => {{ state.busyChecks.push(true); return true; }}, getStagedAttachments: () => [],
          api: async () => ({{}}), setToast: noop, setPollFastUntilMs: noop, kickPoll: noop,
          sendText: (...args) => messageFlow.sendText(...args),
          enqueueComposerText: async (...args) => {{ state.queueCalls.push(args); return true; }},
          prepareModalOpen: noop, afterModalVisibilityChanged: noop, restoreModalFocus: noop,
          storageGetItem: () => null, storageSetItem: noop, storageRemoveItem: noop,
          getComputedStyle: () => ({{ minHeight: "32px" }}), isHTMLElement: () => false,
          activeElement: () => null, requestFrame: (callback) => callback(), now: () => 1000,
        }});

        (async () => {{
          textarea.value = "steer the current turn";
          await form.onsubmit({{ preventDefault: noop }});
          const dialogBeforeChoice = sendChoice.style.display;
          await nowBtn.onclick();
          process.stdout.write(JSON.stringify({{
            dialogBeforeChoice,
            dialogAfterChoice: sendChoice.style.display,
            apiCalls: state.apiCalls,
            queueCalls: state.queueCalls,
            busyChecks: state.busyChecks.length,
            value: textarea.value,
          }}));
          composer.dispose();
        }})().catch((error) => {{ console.error(error); process.exit(1); }});
        """
    )

    result = _run_node(script)

    assert result == {
        "dialogBeforeChoice": "flex",
        "dialogAfterChoice": "none",
        "apiCalls": [
            {
                "path": "/api/sessions/busy-session/send",
                "method": "POST",
                "body": {"text": "steer the current turn", "allow_pending_attachment": False},
            }
        ],
        "queueCalls": [],
        "busyChecks": 2,
        "value": "",
    }
