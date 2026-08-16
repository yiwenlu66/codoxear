from frontend_module_loader import module_path
import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "codoxear" / "static"


def run_vm(body: str) -> dict:
    sources = {
        "TOAST": module_path("app_application_composition.js"),
        "MODAL": module_path("app_modal.js"),
        "HELPERS": module_path("app_session_helpers.js"),
        "QUEUE": module_path("app_queue.js"),
        "SESSION_STATE": module_path("app_session_state.js"),
        "DIAGNOSTICS": module_path("app_diagnostics.js"),
        "VOICE_HELPERS": module_path("app_voice_helpers.js"),
        "VOICE": module_path("app_voice.js"),
        "FILE_VIEWER_SCRIPTS": [
            module_path(name)
            for name in (
                "app_file_candidates.js", "app_file_candidate_state.js", "app_file_viewer_operations.js", "app_file_download.js", "app_file_viewer_lifecycle.js", "app_file_viewer_panel.js",
                "app_file_unsaved_dialog.js", "app_file_paste_dialog.js", "app_file_pdf.js", "app_file_video.js",
                "app_file_mode.js", "app_file_render_surface.js", "app_file_viewer_controller.js", "app_file_viewer.js",
            )
        ],
    }
    declarations = "\n".join(
        f"const {name}_SOURCE = {json.dumps(path.read_text(encoding='utf-8'))};"
        for name, path in sources.items()
        if name != "FILE_VIEWER_SCRIPTS"
    )
    viewer_sources = json.dumps([path.read_text(encoding="utf-8") for path in sources["FILE_VIEWER_SCRIPTS"]])
    script = f"""
        const vm = require("vm");
        {declarations}
        const timers = [];
        function node(id) {{
          return {{
            id, textContent: "", value: "", checked: false, disabled: false, open: false,
            style: {{}}, _attrs: {{}}, _children: [],
            classList: {{ add() {{}}, remove() {{}}, toggle() {{}}, contains() {{ return false; }} }},
            setAttribute(name, value) {{ this._attrs[name] = String(value); }},
            getAttribute(name) {{ return this._attrs[name]; }},
            removeAttribute(name) {{ delete this._attrs[name]; }},
            appendChild(child) {{ this._children.push(child); return child; }},
            addEventListener() {{}}, removeEventListener() {{}}, matches() {{ return false; }},
            focus() {{}}, play() {{ return Promise.resolve(); }}, pause() {{}}, load() {{}},
            showModal() {{ this.open = true; }}, close() {{ this.open = false; }},
          }};
        }}
        const ctx = {{
          window: {{ isSecureContext: false }},
          document: {{ activeElement: null, contains: () => true }},
          navigator: {{ userAgent: "X11 Linux x86_64" }},
          HTMLElement: function HTMLElement() {{}},
          console,
          setTimeout: (fn, ms) => {{ timers.push({{ fn, ms }}); return timers.length; }},
          clearTimeout() {{}}, setInterval: () => 0, clearInterval() {{}},
        }};
        vm.createContext(ctx);
        [MODAL_SOURCE, HELPERS_SOURCE, TOAST_SOURCE, SESSION_STATE_SOURCE, QUEUE_SOURCE, DIAGNOSTICS_SOURCE, VOICE_HELPERS_SOURCE, VOICE_SOURCE]
          .forEach((source) => vm.runInContext(source, ctx));
        {viewer_sources}.forEach((source) => vm.runInContext(source, ctx));
        {body}
    """
    proc = subprocess.run(
        ["node"],
        input=textwrap.dedent(script),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode:
        raise AssertionError(proc.stderr)
    return json.loads(proc.stdout)


class TestFrontendToastRouting(unittest.TestCase):
    def test_voice_file_diagnostics_and_queue_triggers_share_one_auto_dismissing_toast(self) -> None:
        result = run_vm(
            r'''
            (async () => {
              const toast = node("shared-toast");
              const toastController = ctx.window.CodoxearToast.createToastController({
                toast,
                setTimeout: (fn, ms) => { timers.push({ fn, ms }); return timers.length; },
              });
              const routes = [];
              const notify = (source) => (text) => {
                const visible = toastController.show(text);
                routes.push({ source, text: visible, target: toast.id });
                return visible;
              };

              let selected = "s1";
              const sessionState = ctx.window.CodoxearSessionState.createSessionState({ consoleError: () => {} });
              const queueController = ctx.window.CodoxearQueue.createQueueController({
                queueBackdrop: node("queueBackdrop"), queueCloseBtn: node("queueCloseBtn"),
                queueList: node("queueList"), queueEmpty: node("queueEmpty"), queueViewer: node("queueViewer"), queueBtn: node("queueBtn"),
                getSelected: () => selected, getSessionInfo: () => ({ launch_state: "ready" }), sessionState, isAppDisposed: () => false,
                api: async () => ({ queued: true, queue_len: 1 }), setToast: notify("queue"),
                clearCommitUnknownSend: async () => true, refreshSessions: async () => {}, updateQueueBadge: () => {},
                syncRecoveryUiForSession: () => {}, kickPoll: () => {}, setPollFastUntilMs: () => {}, handleAppAuthLoss: () => {},
                prepareModalOpen: () => {}, afterModalVisibilityChanged: () => {},
                el: (tag, attrs = {}, children = []) => Object.assign(node(tag), { tag, textContent: attrs.text || "", _children: children }),
                iconSvg: () => "", recoveryPanelFocusFallback: () => null, confirmAction: async () => false,
                requestFrame: (fn) => fn(), setTimeout: () => 0, clearTimeout: () => {}, now: () => 1000,
              });
              await queueController.enqueueComposerText("queued message");

              const diagnosticsController = ctx.window.CodoxearDiagnostics.createDiagnosticsController({
                diagBackdrop: node("diagBackdrop"), diagViewer: node("diagViewer"), diagContent: node("diagContent"), diagStatus: node("diagStatus"),
                diagCloseBtn: node("diagCloseBtn"), diagCopyConversationBtn: node("diagCopyConversationBtn"), diagCopyBtn: node("diagCopyBtn"),
                getSelected: () => selected, getSessionInfo: () => ({}), api: async () => ({}), setToast: notify("diagnostics"),
                copyToClipboard: async () => true, copyConversation: async () => true, recoveryDetailsText: () => "", redactedLaunchErrorText: () => "",
                sessionLaunchLabel: () => "", agentBackendDisplayName: () => "", diagnosticsProviderDisplay: () => "", diagnosticsCopyText: () => "",
                fmtTs: () => "", fmtRelativeAge: () => "", formatPriorityOffset: () => "", prepareModalOpen: () => {}, afterModalVisibilityChanged: () => {},
                el: (tag, attrs = {}, children = []) => Object.assign(node(tag), { tag, textContent: attrs.text || "", _children: children }), uiVersion: "test", requestFrame: (fn) => fn(),
              });
              await diagnosticsController.onCopyClick({ preventDefault() {}, stopPropagation() {} });

              const fileReferences = ctx.window.CodoxearFileViewer.createFileReferenceRuntime({
                selectedSessionId: () => selected, sessionById: () => null, sessionRelativePath: () => "", listFromFilesField: () => [], listFromFileRecords: () => [],
                normalizeFileApiPath: (value) => String(value || ""), normalizeLineNumber: () => null, api: async () => ({}),
                el: (tag) => node(tag), setToast: notify("file-viewer"), parseLocalFileRef: () => null, showFileViewer: async () => {}, sessions: () => [], selectSession: async () => {}, openDirectorySession: () => {},
              });
              await fileReferences.openReference({ path: "not-a-file-reference", literal: false });

              const voiceNodes = {
                announceBtn: node("announceBtn"), notificationBtn: node("notificationBtn"), liveAudio: node("liveAudio"), voiceSettingsBackdrop: node("voiceSettingsBackdrop"),
                voiceSettingsCloseBtn: node("voiceSettingsCloseBtn"), voiceSettingsStatus: node("voiceSettingsStatus"), voiceBaseUrlInput: node("voiceBaseUrlInput"),
                voiceApiKeyInput: node("voiceApiKeyInput"), voiceClearApiKeyToggle: node("voiceClearApiKeyToggle"), narrationSettingToggle: node("narrationSettingToggle"),
                voiceSettingsViewer: node("voiceSettingsViewer"), voiceSettingsCancelBtn: node("voiceSettingsCancelBtn"), voiceSettingsSaveBtn: node("voiceSettingsSaveBtn"),
              };
              const voiceController = ctx.window.CodoxearVoice.createVoiceController(Object.assign(voiceNodes, {
                isAppDisposed: () => false, api: async () => ({}), setToast: notify("voice"), handleAppAuthLoss: () => {}, prepareModalOpen: () => {}, afterModalVisibilityChanged: () => {},
                resolveAppUrl: (value) => value, versionedShellAssetPath: (value) => value, storageGetItem: () => null, storageSetItem: () => {}, storageRemoveItem: () => {},
                windowTarget: { isSecureContext: false }, navigatorTarget: { userAgent: "X11 Linux x86_64" }, documentTarget: { activeElement: null, contains: () => true },
                Notification: undefined, requestFrame: (fn) => fn(), setTimeout: () => 0, clearTimeout: () => {}, setInterval: () => 0, clearInterval: () => {},
              }));
              await voiceNodes.notificationBtn.onclick({ preventDefault() {}, stopPropagation() {} });

              const lastTimer = timers[timers.length - 1];
              const beforeDismiss = toast.textContent;
              lastTimer.fn();
              process.stdout.write(JSON.stringify({
                routes,
                sharedTarget: routes.length === 4 && routes.every((route) => route.target === "shared-toast"),
                messages: routes.map((route) => route.text),
                dismissDelay: lastTimer.ms,
                beforeDismiss,
                afterDismiss: toast.textContent,
              }));
            })().catch((error) => { console.error(error && error.stack || error); process.exit(1); });
            '''
        )
        self.assertTrue(result["sharedTarget"])
        self.assertEqual([route["source"] for route in result["routes"]], ["queue", "diagnostics", "file-viewer", "voice"])
        self.assertEqual(result["messages"], ["queued (1)", "details not loaded", "unsupported file reference", "notification error: notifications require HTTPS or localhost"])
        self.assertEqual(result["dismissDelay"], 2200)
        self.assertEqual(result["beforeDismiss"], "notification error: notifications require HTTPS or localhost")
        self.assertEqual(result["afterDismiss"], "")


if __name__ == "__main__":
    unittest.main()
