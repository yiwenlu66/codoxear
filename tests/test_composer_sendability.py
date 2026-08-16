"""Behavioral coverage for the focused composer, attachment, and queue controllers.

Each assertion runs the shipped controller source in a Node VM.  The harness
models the browser nodes and controller boundaries, so the tests observe button
state and delivery decisions instead of asserting implementation text.
"""

from __future__ import annotations
from frontend_module_loader import module_path

import json
import subprocess
import textwrap
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
COMPOSER_SOURCE = (module_path("app_composer.js")).read_text(encoding="utf-8")
ATTACHMENTS_SOURCE = (module_path("app_attachments.js")).read_text(encoding="utf-8")
QUEUE_SOURCE = (module_path("app_queue.js")).read_text(encoding="utf-8")
MODAL_SOURCE = (module_path("app_modal.js")).read_text(encoding="utf-8")
SESSION_HELPERS_SOURCE = (module_path("app_session_helpers.js")).read_text(encoding="utf-8")
SESSION_STATE_SOURCE = (module_path("app_session_state.js")).read_text(encoding="utf-8")


def run_controller_harness() -> dict[str, Any]:
    script = "(async () => {\n" + textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{
          HTMLElement: function HTMLElement() {{}},
          Event: function Event(type, options) {{ this.type = type; this.options = options; }},
          Uint8Array,
          console,
          document: {{ activeElement: null, querySelector: () => null, documentElement: {{}}, body: {{}} }},
          window: {{ innerWidth: 1200, innerHeight: 800, addEventListener() {{}}, removeEventListener() {{}} }},
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(MODAL_SOURCE)}, ctx, {{ filename: "app_modal.js" }});
        vm.runInContext({json.dumps(SESSION_HELPERS_SOURCE)}, ctx, {{ filename: "app_session_helpers.js" }});
        vm.runInContext({json.dumps(SESSION_STATE_SOURCE)}, ctx, {{ filename: "app_session_state.js" }});
        vm.runInContext({json.dumps(COMPOSER_SOURCE)}, ctx, {{ filename: "app_composer.js" }});
        vm.runInContext({json.dumps(ATTACHMENTS_SOURCE)}, ctx, {{ filename: "app_attachments.js" }});
        vm.runInContext({json.dumps(QUEUE_SOURCE)}, ctx, {{ filename: "app_queue.js" }});

        function node(extra = {{}}) {{
          const listeners = new Map();
          const classes = new Set();
          return {{
            style: {{}}, value: "", textContent: "", title: "", disabled: false,
            scrollHeight: 32, selectionStart: 0, selectionEnd: 0, children: [], attrs: {{}}, files: [],
            classList: {{
              toggle(name, enabled) {{ if (enabled) classes.add(name); else classes.delete(name); }},
              add(name) {{ classes.add(name); }}, remove(name) {{ classes.delete(name); }},
            }},
            addEventListener(type, handler) {{ listeners.set(type, handler); }},
            removeEventListener(type) {{ listeners.delete(type); }},
            emit(type, event = {{}}) {{ return listeners.get(type)?.(event); }},
            setAttribute(name, value) {{ this.attrs[name] = String(value); }},
            getAttribute(name) {{ return this.attrs[name]; }},
            removeAttribute(name) {{ delete this.attrs[name]; }},
            appendChild(child) {{ this.children.push(child); return child; }},
            dispatchEvent(event) {{ return this.emit(event.type, event); }},
            click() {{ this.clicked = true; }},
            focus() {{}}, blur() {{}},
            set innerHTML(value) {{ this.children = []; }}, get innerHTML() {{ return ""; }},
            ...extra,
          }};
        }}
        const noop = () => {{}};

        function composerHarness({{ running = false, attachments = [] }} = {{}}) {{
          const form = node({{ requestSubmit: noop }});
          const textarea = node();
          const sendBtn = node();
          const sendChoiceLaterBtn = node();
          const calls = [];
          const sessionState = ctx.window.CodoxearSessionState.createSessionState({{ consoleError: noop }});
          sessionState.applyRuntime({{ running }});
          const controller = ctx.window.CodoxearComposer.createComposerController({{
            form, textarea, msgPh: node(), sendBtn, sendChoice: node(), sendChoiceBackdrop: node(),
            sendChoiceNowBtn: node(), sendChoiceLaterBtn, sendChoiceCancelBtn: node(),
            getSelected: () => "sid", getSessionInfo: () => ({{ session_id: "sid", launch_state: "ready" }}),
            sessionLaunchFailed: () => false, getSending: () => false, sessionState,
            getStagedAttachments: () => attachments, api: async () => ({{}}), setToast: noop,
            setPollFastUntilMs: noop, kickPoll: noop,
            sendText: async (text) => {{ calls.push(["send", text]); return true; }},
            enqueueComposerText: async (text) => {{ calls.push(["queue", text]); return true; }},
            prepareModalOpen: noop, afterModalVisibilityChanged: noop, restoreModalFocus: noop,
            storageGetItem: () => null, storageSetItem: noop, storageRemoveItem: noop,
            requestFrame: (callback) => callback(), getComputedStyle: () => ({{ minHeight: "32px" }}),
            activeElement: () => null, isHTMLElement: () => false,
          }});
          return {{ form, textarea, sendBtn, sendChoiceLaterBtn, calls, controller }};
        }}

        const empty = composerHarness();
        await empty.form.onsubmit({{ preventDefault: noop }});

        const text = composerHarness();
        text.textarea.value = "send this";
        text.textarea.emit("input");
        await text.form.onsubmit({{ preventDefault: noop }});

        const queuedWithAttachment = composerHarness({{ running: true, attachments: [{{ id: "a1" }}] }});
        queuedWithAttachment.textarea.value = "wait";
        await queuedWithAttachment.form.onsubmit({{ preventDefault: noop }});

        const attachBtn = node();
        const attachmentController = ctx.window.CodoxearAttachments.createAttachmentsController({{
          attachBtn, imgInput: node(), composer: node(), textarea: node(),
          getSelected: () => "sid", getSessionInfo: () => ({{ session_id: "sid", launch_state: "ready" }}),
          patchSessionInfo: noop, getSending: () => false, sessionLaunchFailed: () => false,
          sessionHasUnknownSend: () => false, sessionIsOrphanRecovery: () => false,
          sessionHasOrphanQueueRecovery: () => false, api: async () => ({{}}), setToast: noop,
          handleAppAuthLoss: noop, refreshSessions: async () => {{}}, setPollFastUntilMs: noop, kickPoll: noop,
          resizeComposer: noop, getTray: () => node(), el: () => node(), fmtBytes: (n) => `${{n}} B`,
          safeAttachmentStem: (name) => name, isLikelyHeic: () => false, looksLikeImage: () => false,
          b64FromBytes: () => "", dataTransferHasFiles: () => false,
          extractFilesFromClipboardData: () => [], extractFilesFromDropData: () => [],
          addEventListener: (target, type, handler) => target.addEventListener(type, handler), uploadMaxBytes: 1024,
        }});
        attachmentController.setStagedAttachments([{{ id: "a1", display_name: "report.txt" }}]);
        attachmentController.syncAttachButtonState();

        const queueBtn = node();
        const queueController = ctx.window.CodoxearQueue.createQueueController({{
          queueBackdrop: node(), queueCloseBtn: node(), queueList: node(), queueEmpty: node(), queueViewer: node(), queueBtn,
          sessionState: ctx.window.CodoxearSessionState.createSessionState({{ consoleError: noop }}),
          getSelected: () => "sid", getSessionInfo: () => ({{ session_id: "sid", launch_state: "ready", queue_len: 0 }}),
          isAppDisposed: () => false, api: async () => ({{}}), setToast: noop, clearCommitUnknownSend: async () => {{}},
          refreshSessions: async () => {{}}, updateQueueBadge: noop, syncRecoveryUiForSession: noop,
          kickPoll: noop, setPollFastUntilMs: noop, handleAppAuthLoss: noop,
          prepareModalOpen: noop, afterModalVisibilityChanged: noop, el: () => node(), iconSvg: () => "",
          recoveryPanelFocusFallback: () => null, confirmAction: async () => true,
          requestFrame: (callback) => callback(), setTimeout: () => 0, clearTimeout: noop,
        }});
        queueController.syncQueueSubmitState();

        process.stdout.write(JSON.stringify({{
          empty: {{ disabled: empty.sendBtn.disabled, calls: empty.calls }},
          text: {{ disabled: text.sendBtn.disabled, calls: text.calls }},
          queuedWithAttachment: {{ queueDisabled: queuedWithAttachment.sendChoiceLaterBtn.disabled }},
          stagedAttachment: {{ attachDisabled: attachBtn.disabled }},
          emptyQueue: {{ disabled: queueBtn.disabled }},
        }}));
        """
    ) + "\n})().catch((error) => { console.error(error); process.exit(1); });\n"
    completed = subprocess.run(
        ["node", "-"],
        check=True,
        input=script,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_empty_draft_cannot_send_even_though_the_composer_stays_editable() -> None:
    result = run_controller_harness()

    assert result["empty"] == {"disabled": False, "calls": []}


def test_text_draft_enables_send_and_reaches_the_send_boundary() -> None:
    result = run_controller_harness()

    assert result["text"] == {"disabled": False, "calls": [["send", "send this"]]}


def test_staged_attachment_disables_the_send_after_current_choice() -> None:
    result = run_controller_harness()

    assert result["queuedWithAttachment"] == {"queueDisabled": True}


def test_empty_queue_remains_openable_and_staged_attachment_allows_more_files() -> None:
    result = run_controller_harness()

    assert result["emptyQueue"] == {"disabled": False}
    assert result["stagedAttachment"] == {"attachDisabled": False}


if __name__ == "__main__":
    raise SystemExit("Run with pytest")
