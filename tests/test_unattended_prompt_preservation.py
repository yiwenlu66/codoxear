"""Restart persistence contract for queued prompts and unattended settings.

The composer draft is browser-owned localStorage (``codexweb.draft.<session_id>``),
so it has no server-side file or SessionManager state to reload. This test covers
the server-owned half of the same restart scenario: a queued staged prompt and
its unattended configuration survive a fresh manager/store bootstrap. Browser
validation exercises the localStorage draft separately.
"""

from __future__ import annotations
from frontend_module_loader import module_path

import json
import subprocess
import textwrap
import tempfile
import threading
from pathlib import Path

from codoxear.session_model import Session
from codoxear.session_store import SessionStore
from codoxear.session_store import SessionStorePaths
from codoxear.session_unattended_config import SessionUnattendedConfigCoordinator
from codoxear.unattended import unattended_config_key


ROOT = Path(__file__).resolve().parents[1]
APP_POLLING_JS = module_path("app_polling.js")
APP_TRANSCRIPT_JS = module_path("app_transcript.js")
APP_MESSAGE_FLOW_JS = module_path("app_message_flow.js")
APP_SESSION_STATE_JS = module_path("app_session_state.js")
APP_COMPOSER_JS = module_path("app_composer.js")


def test_composer_draft_is_session_scoped_across_controller_recreation() -> None:
    """The browser-owned composer draft survives reload and clears by session."""
    polling_source = APP_POLLING_JS.read_text(encoding="utf-8")
    transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
    message_flow_source = APP_MESSAGE_FLOW_JS.read_text(encoding="utf-8")
    session_state_source = APP_SESSION_STATE_JS.read_text(encoding="utf-8")
    composer_source = APP_COMPOSER_JS.read_text(encoding="utf-8")
    script = textwrap.dedent(
        f"""
        const vm = require("vm");
        class Node {{
          constructor() {{ this.listeners = {{}}; this.style = {{}}; this.attributes = {{}}; this.value = ""; this.textContent = ""; this.disabled = false; this.scrollHeight = 32; this.children = []; this.classList = {{ toggle() {{}} }}; }}
          addEventListener(type, fn) {{ (this.listeners[type] ||= []).push(fn); }}
          removeEventListener(type, fn) {{ this.listeners[type] = (this.listeners[type] || []).filter((candidate) => candidate !== fn); }}
          dispatch(type, event = {{}}) {{ for (const fn of this.listeners[type] || []) fn(event); }}
          setAttribute(name, value) {{ this.attributes[name] = String(value); }}
          removeAttribute(name) {{ delete this.attributes[name]; }}
          appendChild(node) {{ this.children.push(node); return node; }}
          set innerHTML(value) {{ this.children = []; }} get innerHTML() {{ return ""; }}
          focus() {{}} blur() {{}}
        }}
        const storage = new Map();
        let selected = "sid-1";
        const form = new Node(); form.requestSubmit = () => {{}};
        const textarea = new Node();
        const nodes = Array.from({{ length: 8 }}, () => new Node());
        const [msgPh, sendBtn, sendChoice, sendChoiceBackdrop, nowBtn, laterBtn, cancelBtn, modelPicker] = nodes;
        const ctx = {{ window: {{}}, document: {{ activeElement: null, createElement: () => new Node() }}, console, Date, Set, Object, String, Number, Promise }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(polling_source)}, ctx);
        vm.runInContext({json.dumps(transcript_source)}, ctx);
        vm.runInContext({json.dumps(message_flow_source)}, ctx);
        vm.runInContext({json.dumps(session_state_source)}, ctx);
        vm.runInContext({json.dumps(composer_source)}, ctx);
        const noop = () => {{}};
        function createMessageFlow() {{
          const typingRowRuntime = {{
            snapshot: () => ({{ stats: {{ thinking: 0, thinkingTokens: 0, thinkingMode: "blocks", tools: 0 }} }}),
            updateTypingStats: noop, updateSubagentGauge: noop, resetTypingStats: noop,
          }};
          return ctx.window.CodoxearMessageFlow.createMessageFlowController({{
            sessionState: ctx.window.CodoxearSessionState.createSessionState({{ consoleError: () => {{}} }}),
            getSelected: () => selected, getGeneration: () => 1, isAppDisposed: () => false,
            getTurnOpen: () => false, setTurnOpen: noop,
            getSessionInfo: () => ({{ session_id: selected, agent_backend: "pi" }}), patchSessionInfo: noop,
            sessionLaunchFailed: () => false, api: async () => ({{ queued: false, queue_len: 0 }}),
            resolveAppUrl: (path) => `http://example.test${{path}}`, handleAppAuthLoss: noop,
            refreshSessions: async () => [], openSession: async () => null, clearSelectedSessionAfterRemoval: noop,
            activeTranscriptSnapshot: () => ({{ state: "bound", liveCursor: "c1", logPath: "/tmp/log.jsonl" }}),
            updateSessionTranscriptSlot: () => ({{ ignoredStaleBound: false, current: {{ state: "bound" }} }}),
            renderPendingTranscriptSlot: noop, renderSessionTail: noop, applySessionRuntimeFromTail: noop,
            resetChatRenderState: noop, setAttachCount: noop, setLiveCursor: noop, appendEvent: noop,
            appendTailSnapshotEvents: noop, setStatus: noop, setContext: noop, setTyping: noop,
            setSubagentsRunning: noop, updateSessionTitle: noop, initPageLimit: () => 60, typingRowRuntime,
            getSending: () => false, setSending: noop, getCurrentRunning: () => false, setCurrentRunning: noop,
            getStagedAttachments: () => [], normalizedStagedAttachments: () => [], setSelectedSessionPendingAttachment: noop,
            syncSendButtonState: noop, syncAttachButtonState: noop, syncQueueSubmitState: noop, syncRecoveryUiForSession: noop,
            confirmAction: async () => false, setToast: noop, isTranscriptRenewalCommand: () => false,
            nextLocalEchoId: () => 1, renderedAtLiveTail: () => true, clearTranscriptDom: noop,
            clearRenderedTranscriptRange: noop, setOlderState: noop, getSessionTranscriptSlot: () => ({{ epoch: 0 }}),
            addPendingUser: noop, deleteTailCache: noop, beginTranscriptRenewal: noop, clearLiveCursor: noop,
            invalidateOlderLoad: noop, dropPendingUser: noop, removePendingUserRow: noop, hasPendingForSession: () => false,
            visibilityState: () => "visible", navigatorValue: () => ({{ onLine: true }}), EventSource: null,
            AbortController: null, setTimeout: () => 0, clearTimeout: noop, now: () => 1,
            consoleWarn: noop, consoleError: noop,
          }});
        }}
        const makeController = () => {{
          const messageFlowController = createMessageFlow();
          return ctx.window.CodoxearComposer.createComposerController({{
          sessionState: ctx.window.CodoxearSessionState.createSessionState({{ consoleError: () => {{}} }}),
          form, textarea, msgPh, sendBtn, sendChoice, sendChoiceBackdrop,
          sendChoiceNowBtn: nowBtn, sendChoiceLaterBtn: laterBtn, sendChoiceCancelBtn: cancelBtn, modelPicker,
          getSelected: () => selected, getSessionInfo: () => ({{}}), patchSessionInfo: noop, sessionLaunchFailed: () => false,
          getSending: () => false, setSending: noop, getCurrentRunning: () => false, setCurrentRunning: noop, setTurnOpen: noop, resetTypingStats: noop,
          getStagedAttachments: () => [], normalizedStagedAttachments: () => [], setSelectedSessionPendingAttachment: noop, setAttachCount: noop,
          syncAttachButtonState: noop, syncQueueSubmitState: noop, syncRecoveryUiForSession: noop, confirmAction: async () => false,
          api: async () => ({{}}), setToast: noop, handleAppAuthLoss: noop, refreshSessions: async () => [], setPollFastUntilMs: noop, kickPoll: noop,
          isTranscriptRenewalCommand: () => false, nextLocalEchoId: () => "local", renderedAtLiveTail: () => true,
          clearTranscriptDom: noop, clearRenderedTranscriptRange: noop, setOlderState: noop, getSessionTranscriptSlot: () => ({{ epoch: 0 }}),
          addPendingUser: noop, appendEvent: noop, deleteTailCache: noop, beginTranscriptRenewal: noop, clearLiveCursor: noop,
          invalidateOlderLoad: noop, renderPendingTranscriptSlot: noop, dropPendingUser: noop, removePendingUserRow: noop,
          enqueueComposerText: async () => true, prepareModalOpen: noop, afterModalVisibilityChanged: noop,
          restoreModalFocus: noop, sendText: (...args) => messageFlowController.sendText(...args),
          storageGetItem: (key) => storage.get(key) || null, storageSetItem: (key, value) => storage.set(key, value),
          storageRemoveItem: (key) => storage.delete(key), getComputedStyle: () => ({{ minHeight: "32px" }}), requestFrame: (fn) => fn(),
          activeElement: () => textarea, isHTMLElement: () => true, now: () => 1,
        }});
        }};
        const first = makeController();
        textarea.value = "draft survives restart";
        textarea.dispatch("input");
        const wrote = storage.get("codexweb.draft.sid-1");
        first.dispose();
        textarea.value = "";
        const second = makeController();
        second.loadSessionDraft("sid-1");
        const restored = textarea.value;
        selected = "sid-2";
        textarea.value = "other session";
        textarea.dispatch("input");
        selected = "sid-1";
        second.clearComposer();
        process.stdout.write(JSON.stringify({{ wrote, restored, sid1: storage.has("codexweb.draft.sid-1"), sid2: storage.get("codexweb.draft.sid-2") }}));
        """
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".js", encoding="utf-8") as script_file:
        script_file.write(script)
        script_file.flush()
        result = subprocess.run(["node", script_file.name], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr or result.stdout
    assert json.loads(result.stdout) == {
        "wrote": "draft survives restart",
        "restored": "draft survives restart",
        "sid1": False,
        "sid2": "other session",
    }


DEFAULT_IDLE_MINUTES = 5
DEFAULT_MAX_INJECTIONS = 10


def _session(*, session_id: str, thread_id: str, log_path: Path) -> Session:
    return Session(
        session_id=session_id,
        thread_id=thread_id,
        broker_pid=1,
        codex_pid=1,
        agent_backend="codex",
        owned=False,
        start_ts=1.0,
        cwd="/workspace",
        log_path=log_path,
        sock_path=log_path.with_suffix(".sock"),
        sync_send_supported=True,
        key_write_errors_supported=True,
    )


def _store(root: Path) -> SessionStore:
    return SessionStore(
        paths=SessionStorePaths(
            aliases=root / "session_aliases.json",
            sidebar_meta=root / "session_sidebar.json",
            hidden_sessions=root / "hidden_sessions.json",
            files=root / "session_files.json",
            queues=root / "session_queues.json",
            pending_attachments=root / "pending_attachments.json",
            staged_attachments=root / "staged_attachments.json",
            commit_unknown_sends=root / "commit_unknown_sends.json",
            recent_cwds=root / "recent_cwds.json",
            unattended=root / "unattended.json",
            uploads_root=root / "uploads",
        ),
        file_history_max=10,
        recent_cwd_max=10,
        unattended_default_idle_minutes=DEFAULT_IDLE_MINUTES,
        unattended_default_max_injections=DEFAULT_MAX_INJECTIONS,
        clean_alias=lambda value: value,
        clean_priority_offset=lambda value: value,
        clean_snooze_until=lambda value: value,
        clean_dependency_session_id=lambda value: value,
        clean_recent_cwd=lambda value: value,
        clean_commit_unknown_send_record=lambda value: value,
    )


class _SessionManagerHarness:
    """Minimal SessionManager persistence boundary with a real SessionStore.

    Each instance deliberately creates a new store and loads it from disk. That
    is the relevant boundary for a server restart; a live broker keeps its
    socket-derived ``session_id`` while the manager process is recreated.
    """

    def __init__(self, *, root: Path, sessions: dict[str, Session]) -> None:
        self.store = _store(root)
        self.store.load_persistent_state()
        self.sessions = sessions
        self.lock = threading.Lock()
        self.input_locks: dict[str, threading.RLock] = {}
        self.unattended_last_injected: dict[str, float] = {}

    def input_lock_for_session(self, session_id: str) -> threading.RLock:
        return self.input_locks.setdefault(session_id, threading.RLock())

    def unattended_config(self) -> SessionUnattendedConfigCoordinator:
        return SessionUnattendedConfigCoordinator(
            lock=self.lock,
            sessions=lambda: self.sessions,
            unattended=lambda: self.store.unattended,
            unattended_last_injected=lambda: self.unattended_last_injected,
            input_lock_for_session=self.input_lock_for_session,
            save_unattended=lambda: self.store.save_unattended(self.store.unattended),
            clean_unattended_cooldown_minutes=self.store.unattended_store.clean_cooldown_minutes,
            clean_unattended_remaining_injections=self.store.unattended_store.clean_remaining_injections,
        )


def test_unattended_restart_preserves_staged_prompt_queue_and_thread_config(tmp_path: Path) -> None:
    """A fresh server manager reloads the queued prompt and unattended intent.

    The original session id remains the queue key across a server-only restart;
    the resumed session id changes below to prove unattended configuration is
    recovered by the stable backend thread identity rather than broker identity.
    """
    app_dir = tmp_path / ".local" / "share" / "codoxear"
    original = _session(
        session_id="broker-before-restart",
        thread_id="thread-continues",
        log_path=app_dir / "sessions" / "thread-continues.jsonl",
    )
    before_restart = _SessionManagerHarness(root=app_dir, sessions={original.session_id: original})

    staged_prompt = "Continue from the failing unattended restart test."
    queued, queue_len = before_restart.store.queue_store.append(
        before_restart.store.queues,
        original.session_id,
        staged_prompt,
    )
    before_restart.store.save_queues(before_restart.store.queues)
    saved_config = before_restart.unattended_config().set(
        original.session_id,
        enabled=True,
        request="Finish the queued validation and report the result.",
        cooldown_minutes=7,
        remaining_injections=3,
    )

    assert queue_len == 1
    assert (app_dir / "session_queues.json").is_file()
    assert (app_dir / "unattended.json").is_file()

    # New SessionStore instance = a new SessionManager process after restart.
    # The original broker session is rediscovered with its unchanged socket id.
    restarted = _SessionManagerHarness(root=app_dir, sessions={original.session_id: original})
    assert restarted.store.queue_store.list_items(restarted.store.queues, original.session_id) == [
        {**queued, "sending": False, "commit_unknown": False}
    ]

    # A broker/session replacement for the same backend thread receives the
    # persisted unattended config through its thread-scoped storage key.
    resumed = _session(
        session_id="broker-after-restart",
        thread_id=original.thread_id,
        log_path=original.log_path,
    )
    restarted.sessions[resumed.session_id] = resumed
    assert unattended_config_key(original) == unattended_config_key(resumed) == "thread:thread-continues"
    assert restarted.unattended_config().get(resumed.session_id) == saved_config
