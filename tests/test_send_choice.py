from __future__ import annotations
from frontend_module_loader import module_path

import json
import subprocess
import textwrap
import tempfile
import threading
from pathlib import Path
from typing import Any

from codoxear.queue_store import QueueStore
from codoxear.session_model import Session
from codoxear.session_queue import SessionQueueCoordinator


ROOT = Path(__file__).resolve().parents[1]
POLLING_SOURCE = (module_path("app_polling.js")).read_text(encoding="utf-8")
TRANSCRIPT_SOURCE = (module_path("app_transcript.js")).read_text(encoding="utf-8")
MESSAGE_FLOW_SOURCE = (module_path("app_message_flow.js")).read_text(encoding="utf-8")
SESSION_STATE_SOURCE = (module_path("app_session_state.js")).read_text(encoding="utf-8")
SESSION_CATALOG_SOURCE = module_path("app_session_catalog.js").read_text(encoding="utf-8")
COMPOSER_SOURCE = (module_path("app_composer.js")).read_text(encoding="utf-8")


class NotReady(Exception):
    pass


class InjectionError(Exception):
    pass


class CommitUnknown(Exception):
    pass


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


def test_busy_send_choice_routes_now_later_and_cancel_through_distinct_actions() -> None:
    """The busy composer offers one immediate send, one queued send, and a no-op cancel."""
    script = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}}, console, Date }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(POLLING_SOURCE)}, ctx);
        vm.runInContext({json.dumps(TRANSCRIPT_SOURCE)}, ctx);
        vm.runInContext({json.dumps(MESSAGE_FLOW_SOURCE)}, ctx);
        vm.runInContext({json.dumps(SESSION_CATALOG_SOURCE)}, ctx);
        vm.runInContext({json.dumps(SESSION_STATE_SOURCE)}, ctx);
        vm.runInContext({json.dumps(COMPOSER_SOURCE)}, ctx);

        function node() {{
          return {{
            addEventListener: () => {{}}, removeEventListener: () => {{}},
            setAttribute: () => {{}}, removeAttribute: () => {{}},
            classList: {{ toggle: () => {{}} }}, style: {{}}, value: "", textContent: "",
            scrollHeight: 32, disabled: false, focus: () => {{}}, blur: () => {{}},
          }};
        }}
        function createMessageFlow(state, sessionState, sessionCatalog) {{
          const noop = () => {{}};
          const typingRowRuntime = {{
            snapshot: () => ({{ stats: {{ thinking: 0, thinkingTokens: 0, thinkingMode: "blocks", tools: 0 }} }}),
            updateTypingStats: noop, updateSubagentGauge: noop, resetTypingStats: noop,
          }};
          return ctx.window.CodoxearMessageFlow.createMessageFlowController({{
            sessionState, sessionCatalog,
            currentGeneration: () => 1, isAppDisposed: () => false,

            sessionLaunchFailed: () => false,
            api: async (path, options) => {{ state.apiCalls.push({{ path, body: options.body }}); return {{ queued: false, queue_len: 0, busy: true }}; }},
            resolveAppUrl: (path) => `http://example.test${{path}}`, handleAppAuthLoss: noop, refreshSessions: async () => [],
            openSession: async () => null, clearSelectedSessionAfterRemoval: noop,
            activeTranscriptSnapshot: () => ({{ state: "bound", liveCursor: "c1", logPath: "/tmp/log.jsonl" }}),
            updateSessionTranscriptSlot: () => ({{ ignoredStaleBound: false, current: {{ state: "bound" }} }}),
            renderPendingTranscriptSlot: noop, renderSessionTail: noop, applySessionRuntimeFromTail: noop,
            resetChatRenderState: noop, setAttachCount: noop, setLiveCursor: noop, appendEvent: noop,
            appendTailSnapshotEvents: noop, setStatus: noop, setContext: noop, setTyping: noop,
            setSubagentsRunning: noop, updateSessionTitle: noop, initPageLimit: () => 60, typingRowRuntime,

            getCurrentRunning: () => true, setCurrentRunning: noop,
            getStagedAttachments: () => [], normalizedStagedAttachments: () => [], setSelectedSessionPendingAttachment: noop,
            syncSendButtonState: noop, syncAttachButtonState: noop, syncQueueSubmitState: noop, syncRecoveryUiForSession: noop,
            confirmAction: async () => false, setToast: (message) => state.toasts.push(message),
            isTranscriptRenewalCommand: () => false, nextLocalEchoId: () => 1, renderedAtLiveTail: () => true,
            clearTranscriptDom: noop, clearRenderedTranscriptRange: noop, setOlderState: noop,
            getSessionTranscriptSlot: () => ({{ epoch: 0 }}), addPendingUser: noop, deleteTailCache: noop,
            beginTranscriptRenewal: noop, clearLiveCursor: noop, invalidateOlderLoad: noop,
            dropPendingUser: noop, removePendingUserRow: noop, hasPendingForSession: () => false,
            visibilityState: () => "visible", navigatorValue: () => ({{ onLine: true }}), EventSource: null,
            AbortController: null, setTimeout: () => 0, clearTimeout: noop, now: () => 1000,
            consoleWarn: noop, consoleError: noop,
          }});
        }}
        function harness() {{
          const nodes = Array.from({{ length: 9 }}, node);
          const [form, textarea, msgPh, sendBtn, sendChoice, sendChoiceBackdrop, sendChoiceNowBtn, sendChoiceLaterBtn, sendChoiceCancelBtn] = nodes;
          form.requestSubmit = () => {{}};
          const state = {{ sending: false, apiCalls: [], queueCalls: [], toasts: [] }};
          const noop = () => {{}};
          const sessionState = ctx.window.CodoxearSessionState.createSessionState({{ consoleError: noop }});
          sessionState.applyRuntime({{ selected: "busy-session", running: true, turnOpen: true }});
          const sessionCatalog = ctx.window.CodoxearSessionCatalog.createSessionCatalog({{ consoleError: noop }});
          sessionCatalog.set("latestSessions", [{{ session_id: "busy-session", agent_backend: "pi" }}]);
          const messageFlowController = createMessageFlow(state, sessionState, sessionCatalog);
          const controller = ctx.window.CodoxearComposer.createComposerController({{
            form, textarea, msgPh, sendBtn, sendChoice, sendChoiceBackdrop,
            sendChoiceNowBtn, sendChoiceLaterBtn, sendChoiceCancelBtn,
            sessionCatalog,
            sessionLaunchFailed: () => false, getSending: () => state.sending,
            sessionState, getStagedAttachments: () => [],
            api: async () => ({{}}), setToast: noop, setPollFastUntilMs: noop, kickPoll: noop,
            sendText: (...args) => messageFlowController.sendText(...args),
            enqueueComposerText: async (text, options) => {{ state.queueCalls.push({{ text, sid: options.sid }}); return true; }},
            prepareModalOpen: noop, afterModalVisibilityChanged: noop, restoreModalFocus: noop,
            storageGetItem: () => null, storageSetItem: noop, storageRemoveItem: noop,
            getComputedStyle: () => ({{ minHeight: "32px" }}), isHTMLElement: () => false,
            activeElement: () => null, requestFrame: (callback) => callback(), now: () => 1000,
          }});
          return {{ nodes: {{ form, textarea, sendChoice, sendChoiceBackdrop, sendChoiceNowBtn, sendChoiceLaterBtn, sendChoiceCancelBtn }}, state, controller }};
        }}
        async function submit(h, text) {{
          h.nodes.textarea.value = text;
          await h.nodes.form.onsubmit({{ preventDefault: () => {{}} }});
          return {{ dialog: h.nodes.sendChoice.style.display, backdrop: h.nodes.sendChoiceBackdrop.style.display }};
        }}
        (async () => {{
          const now = harness();
          const nowOpen = await submit(now, "send immediately");
          await now.nodes.sendChoiceNowBtn.onclick();

          const later = harness();
          const laterOpen = await submit(later, "send later");
          await later.nodes.sendChoiceLaterBtn.onclick();

          const cancel = harness();
          const cancelOpen = await submit(cancel, "keep draft");
          cancel.nodes.sendChoiceCancelBtn.onclick();

          process.stdout.write(JSON.stringify({{
            nowOpen, now: {{ apiCalls: now.state.apiCalls, queueCalls: now.state.queueCalls, dialog: now.nodes.sendChoice.style.display, value: now.nodes.textarea.value }},
            laterOpen, later: {{ apiCalls: later.state.apiCalls, queueCalls: later.state.queueCalls, dialog: later.nodes.sendChoice.style.display, value: later.nodes.textarea.value }},
            cancelOpen, cancel: {{ apiCalls: cancel.state.apiCalls, queueCalls: cancel.state.queueCalls, dialog: cancel.nodes.sendChoice.style.display, value: cancel.nodes.textarea.value }},
          }}));
        }})().catch((error) => {{ console.error(error); process.exit(1); }});
        """
    )

    result = _run_node(script)

    assert result["nowOpen"] == {"dialog": "flex", "backdrop": "block"}
    assert result["now"] == {
        "apiCalls": [{"path": "/api/sessions/busy-session/send", "body": {"text": "send immediately", "allow_pending_attachment": False}}],
        "queueCalls": [],
        "dialog": "none",
        "value": "",
    }
    assert result["laterOpen"] == {"dialog": "flex", "backdrop": "block"}
    assert result["later"] == {
        "apiCalls": [],
        "queueCalls": [{"text": "send later", "sid": "busy-session"}],
        "dialog": "none",
        "value": "",
    }
    assert result["cancelOpen"] == {"dialog": "flex", "backdrop": "block"}
    assert result["cancel"] == {
        "apiCalls": [],
        "queueCalls": [],
        "dialog": "none",
        "value": "keep draft",
    }

def _busy_session() -> Session:
    return Session(
        session_id="busy-session",
        thread_id="thread-1",
        broker_pid=2,
        codex_pid=1,
        agent_backend="pi",
        owned=True,
        start_ts=0.0,
        cwd="/repo",
        log_path=None,
        sock_path=Path("/tmp/busy-session.sock"),
        busy=True,
        sync_send_supported=True,
    )


def test_enqueue_when_busy_persists_prompt_without_calling_send(tmp_path: Path) -> None:
    """Send after current is an explicit queue write while the active turn remains busy."""
    session = _busy_session()
    sessions = {session.session_id: session}
    queues: dict[str, list[dict[str, Any]]] = {}
    sent: list[str] = []
    coordinator = SessionQueueCoordinator(
        lock=threading.Lock(),
        sessions=lambda: sessions,
        queues=lambda: queues,
        queue_store=lambda: QueueStore(tmp_path / "queues.json"),
        commit_unknown_sends=lambda: {},
        save_queues=lambda: None,
        input_lock_for_session=lambda _session_id: threading.RLock(),
        remote_ready=lambda _session_id, _log_path: False,
        send=lambda _session_id, text, **_kwargs: sent.append(text) or {"queued": False, "queue_len": 0},
        not_ready_error=NotReady,
        retryable_send_errors=(NotReady, InjectionError),
        commit_unknown_error=CommitUnknown,
        queue_idle_grace_seconds=5.0,
    )

    response = coordinator.enqueue(session.session_id, "run this after the current response")

    assert response["queued"] is True
    assert response["queue_len"] == 1
    assert response["item"]["text"] == "run this after the current response"
    queued_items = coordinator.list_local(session.session_id)
    assert len(queued_items) == 1
    assert queued_items[0]["id"] == response["item"]["id"]
    assert queued_items[0]["text"] == response["item"]["text"]
    assert sent == []
