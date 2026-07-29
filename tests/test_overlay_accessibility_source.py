import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODAL = ROOT / "codoxear" / "static" / "app_modal.js"
VOICE = ROOT / "codoxear" / "static" / "app_voice.js"
VOICE_HELPERS = ROOT / "codoxear" / "static" / "app_voice_helpers.js"


def run_modal_behavior() -> dict:
    scripts = {name: path.read_text(encoding="utf-8") for name, path in {"modal": MODAL, "helpers": VOICE_HELPERS, "voice": VOICE}.items()}
    program = textwrap.dedent(
        f"""
        const vm = require("vm");
        const calls = [];
        class HTMLElement {{ constructor() {{ this.isConnected = true; }} }}
        function node() {{ return {{ style: {{}}, value: "", checked: false, textContent: "", disabled: false,
          classList: {{ add() {{}}, remove() {{}}, toggle() {{}} }}, attrs: {{}},
          setAttribute(k,v) {{ this.attrs[k] = String(v); }}, getAttribute(k) {{ return this.attrs[k]; }},
          removeAttribute(k) {{ delete this.attrs[k]; }}, addEventListener() {{}}, matches() {{ return false; }}, focus() {{ calls.push("focus"); }},
          showModal() {{ this.open = true; calls.push("showModal"); }}, close() {{ this.open = false; calls.push("close"); }} }}; }}
        const opener = new HTMLElement(); opener.focus = () => calls.push("restore-focus");
        const dialog = node(); const backdrop = node();
        const ctx = {{ HTMLElement, window: {{}}, document: {{ activeElement: opener, contains: (el) => el === opener }},
          navigator: {{ userAgent: "X11" }}, requestAnimationFrame: (fn) => fn(), setTimeout: () => 1, clearTimeout() {{}}, setInterval: () => 1, clearInterval() {{}} }};
        vm.createContext(ctx);
        for (const code of [{json.dumps(scripts['modal'])}, {json.dumps(scripts['helpers'])}, {json.dumps(scripts['voice'])}]) vm.runInContext(code, ctx);
        const app = {{ attrs: {{}}, toggleAttribute(k,on) {{ if (on) this.attrs[k] = ""; else delete this.attrs[k]; }}, setAttribute(k,v) {{ this.attrs[k] = v; }}, removeAttribute(k) {{ delete this.attrs[k]; }} }};
        const active = ctx.window.CodoxearModal.syncModalIsolation(app, [{{style: {{display:"none"}}}}, {{style: {{display:"flex"}}}}]);
        const deps = {{ announceBtn: node(), notificationBtn: node(), liveAudio: node(), voiceSettingsBackdrop: backdrop, voiceSettingsCloseBtn: node(), voiceSettingsStatus: node(), voiceBaseUrlInput: node(), voiceApiKeyInput: node(), voiceClearApiKeyToggle: node(), narrationSettingToggle: node(), voiceSettingsViewer: dialog, voiceSettingsCancelBtn: node(), voiceSettingsSaveBtn: node(),
          isAppDisposed: () => false, api: async () => ({{}}), setToast() {{}}, handleAppAuthLoss() {{}}, prepareModalOpen: () => calls.push("prepare"), afterModalVisibilityChanged: () => calls.push("visibility"), resolveAppUrl: (v) => v, versionedShellAssetPath: (v) => v, storageGetItem: () => null, storageSetItem() {{}}, storageRemoveItem() {{}}, requestFrame: (fn) => fn(), setTimeout: () => 1, clearTimeout() {{}}, setInterval: () => 1, clearInterval() {{}} }};
        const controller = ctx.window.CodoxearVoice.createVoiceController(deps);
        controller.showVoiceSettingsDialog();
        const shown = {{ open: controller.isSettingsOpen(), display: dialog.style.display, backdrop: backdrop.style.display, nativeOpen: dialog.open }};
        controller.hideVoiceSettingsDialog();
        process.stdout.write(JSON.stringify({{ active, attrs: app.attrs, shown, hidden: {{ open: controller.isSettingsOpen(), display: dialog.style.display, backdrop: backdrop.style.display, nativeOpen: dialog.open }}, calls }}));
        """
    )
    result = subprocess.run(["node", "-e", program], check=True, text=True, capture_output=True)
    return json.loads(result.stdout)


class TestOverlayAccessibilityBehavior(unittest.TestCase):
    def test_modal_show_hide_isolates_background_and_restores_focus(self) -> None:
        result = run_modal_behavior()
        self.assertTrue(result["active"])
        self.assertEqual(result["attrs"], {"inert": "", "aria-hidden": "true"})
        self.assertEqual(result["shown"], {"open": True, "display": "flex", "backdrop": "block", "nativeOpen": True})
        self.assertEqual(result["hidden"], {"open": False, "display": "none", "backdrop": "none", "nativeOpen": False})
        self.assertContains("prepare", result["calls"])
        self.assertContains("showModal", result["calls"])
        self.assertContains("close", result["calls"])
        self.assertContains("restore-focus", result["calls"])


if __name__ == "__main__":
    unittest.main()
