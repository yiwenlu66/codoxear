from __future__ import annotations

import json
import subprocess
import textwrap
import threading
from pathlib import Path
from typing import Any

from codoxear.queue_store import QueueStore
from codoxear.session_model import Session
from codoxear.session_queue import SessionQueueCoordinator


ROOT = Path(__file__).resolve().parents[1]
COMPOSER_SOURCE = (ROOT / "codoxear" / "static" / "app_composer.js").read_text(encoding="utf-8")


class NotReady(Exception):
    pass


class InjectionError(Exception):
    pass


class CommitUnknown(Exception):
    pass


def _run_node(script: str) -> dict[str, Any]:
    completed = subprocess.run(
        ["node", "-e", script],
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
        vm.runInContext({json.dumps(COMPOSER_SOURCE)}, ctx);

        function node() {{
          return {{
            addEventListener: () => {{}}, removeEventListener: () => {{}},
            setAttribute: () => {{}}, removeAttribute: () => {{}},
            classList: {{ toggle: () => {{}} }}, style: {{}}, value: "", textContent: "",
            scrollHeight: 32, disabled: false, focus: () => {{}}, blur: () => {{}},
          }};
        }}
        function harness() {{
          const nodes = Array.from({{ length: 9 }}, node);
          const [form, textarea, msgPh, sendBtn, sendChoice, sendChoiceBackdrop, sendChoiceNowBtn, sendChoiceLaterBtn, sendChoiceCancelBtn] = nodes;
          form.requestSubmit = () => {{}};
          const state = {{ sending: false, apiCalls: [], queueCalls: [], toasts: [] }};
          const noop = () => {{}};
          const controller = ctx.window.CodoxearComposer.createComposerController({{
            form, textarea, msgPh, sendBtn, sendChoice, sendChoiceBackdrop,
            sendChoiceNowBtn, sendChoiceLaterBtn, sendChoiceCancelBtn,
            getSelected: () => "busy-session", getSessionInfo: () => ({{ agent_backend: "pi" }}),
            patchSessionInfo: noop, sessionLaunchFailed: () => false,
            getSending: () => state.sending, setSending: (value) => {{ state.sending = value; }},
            getCurrentRunning: () => true, setCurrentRunning: noop, setTurnOpen: noop, resetTypingStats: noop,
            getStagedAttachments: () => [], normalizedStagedAttachments: () => [],
            setSelectedSessionPendingAttachment: noop, setAttachCount: noop, syncAttachButtonState: noop,
            syncQueueSubmitState: noop, syncRecoveryUiForSession: noop, confirmAction: async () => false,
            api: async (path, options) => {{ state.apiCalls.push({{ path, body: options.body }}); return {{ queued: false, queue_len: 0, busy: true }}; }},
            setToast: (message) => state.toasts.push(message), handleAppAuthLoss: noop, refreshSessions: async () => [],
            setPollFastUntilMs: noop, kickPoll: noop, isTranscriptRenewalCommand: () => false,
            nextLocalEchoId: () => 1, renderedAtLiveTail: () => true, clearTranscriptDom: noop,
            clearRenderedTranscriptRange: noop, setOlderState: noop, getSessionTranscriptSlot: () => ({{ epoch: 0 }}),
            addPendingUser: noop, appendEvent: noop, deleteTailCache: noop, beginTranscriptRenewal: noop,
            clearLiveCursor: noop, invalidateOlderLoad: noop, renderPendingTranscriptSlot: noop,
            dropPendingUser: noop, removePendingUserRow: noop, hasPendingForSession: () => false,
            enqueueComposerText: async (text, options) => {{ state.queueCalls.push({{ text, sid: options.sid }}); return true; }},
            prepareModalOpen: noop, afterModalVisibilityChanged: noop, restoreModalFocus: noop,
            storageGetItem: () => null, storageSetItem: noop, storageRemoveItem: noop,
            getComputedStyle: () => ({{ minHeight: "32" }}), isHTMLElement: () => false,
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
