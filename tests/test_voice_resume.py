import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VOICE_SOURCE = (ROOT / "codoxear" / "static" / "app_voice.js").read_text(encoding="utf-8")
VOICE_HELPERS_SOURCE = (ROOT / "codoxear" / "static" / "app_voice_helpers.js").read_text(encoding="utf-8")
MODAL_SOURCE = (ROOT / "codoxear" / "static" / "app_modal.js").read_text(encoding="utf-8")


def run_voice_resume_harness() -> dict:
    script = textwrap.dedent(
        f"""
        const vm = require("node:vm");
        const server = {{ listeners: new Set(), heartbeatCalls: [] }};
        const storage = new Map([["codoxear.announcementEnabled", "1"], ["codoxear.announcementClientId", "voice-client"]]);

        function node() {{
          return {{
            style: {{}}, value: "", checked: false, textContent: "", open: false,
            classList: {{ toggle() {{}}, add() {{}}, remove() {{}} }},
            setAttribute() {{}}, removeAttribute() {{}}, addEventListener() {{}}, removeEventListener() {{}},
            matches: () => false, load() {{}}, pause() {{}}, play: async () => {{}}, canPlayType: () => "",
          }};
        }}
        function makeController(ctx) {{
          const dom = {{
            announceBtn: node(), notificationBtn: node(), liveAudio: node(), voiceSettingsBackdrop: node(),
            voiceSettingsCloseBtn: node(), voiceSettingsStatus: node(), voiceBaseUrlInput: node(),
            voiceApiKeyInput: node(), voiceClearApiKeyToggle: node(), narrationSettingToggle: node(),
            voiceSettingsViewer: node(), voiceSettingsCancelBtn: node(), voiceSettingsSaveBtn: node(),
          }};
          return ctx.window.CodoxearVoice.createVoiceController({{
            ...dom,
            isAppDisposed: () => false,
            api: async (url, options = {{}}) => {{
              if (url === "/api/audio/listener") {{
                const body = options.body || {{}};
                if (body.enabled) server.listeners.add(body.client_id);
                else server.listeners.delete(body.client_id);
                server.heartbeatCalls.push({{ clientId: body.client_id, enabled: body.enabled }});
                return {{ active_listener_count: server.listeners.size }};
              }}
              if (url === "/api/settings/voice") return {{
                tts_enabled_for_narration: false, tts_enabled_for_final_response: true,
                tts_base_url: "https://api.openai.com/v1", has_tts_api_key: true,
                audio: {{ queue_depth: 0, segment_count: 0, last_error: "", stream_url: "/api/audio/live.m3u8", active_listener_count: server.listeners.size }},
                notifications: {{ enabled_devices: 0, total_devices: 0, vapid_public_key: "" }},
              }};
              return {{}};
            }},
            setToast() {{}}, handleAppAuthLoss() {{}}, prepareModalOpen() {{}}, afterModalVisibilityChanged() {{}},
            resolveAppUrl: (path) => path, versionedShellAssetPath: (path) => path,
            storageGetItem: (key) => storage.has(key) ? storage.get(key) : null,
            storageSetItem: (key, value) => storage.set(key, String(value)),
            storageRemoveItem: (key) => storage.delete(key),
            requestFrame: (fn) => fn(), setTimeout: () => 1, clearTimeout() {{}}, setInterval: () => 1, clearInterval() {{}},
          }});
        }}

        const ctx = {{
          HTMLElement: function HTMLElement() {{}},
          window: {{ isSecureContext: true }}, navigator: {{ userAgent: "X11" }},
          document: {{ activeElement: null, contains: () => true }}, console,
        }};
        vm.createContext(ctx);
        for (const source of [{json.dumps(MODAL_SOURCE)}, {json.dumps(VOICE_HELPERS_SOURCE)}, {json.dumps(VOICE_SOURCE)}]) vm.runInContext(source, ctx);

        (async () => {{
          // The persisted opt-in immediately registers before session selection;
          // opening a different session does not replace this global listener.
          const firstPage = makeController(ctx);
          await Promise.resolve();
          const beforeNewSession = server.listeners.size;
          const openedSessionId = "new-session";
          const afterNewSession = server.listeners.size;

          // A restarted server loses its in-memory listener registry. The next
          // voice-settings snapshot reasserts the persisted listener.
          server.listeners.clear();
          await firstPage.loadVoiceSettings();
          const afterServerRestart = server.listeners.size;

          // A full page reload constructs a new controller against the same
          // local storage and registers again without any selected-session state.
          server.listeners.clear();
          const reloadedPage = makeController(ctx);
          await Promise.resolve();
          process.stdout.write(JSON.stringify({{
            beforeNewSession, openedSessionId, afterNewSession, afterServerRestart,
            afterPageReload: server.listeners.size,
            heartbeatCalls: server.heartbeatCalls,
            enabledAfterReload: reloadedPage.voiceAnnouncementsEnabled(),
          }}));
        }})().catch((error) => {{ console.error(error.stack || error); process.exit(1); }});
        """
    )
    completed = subprocess.run(["node", "-e", script], check=True, text=True, capture_output=True)
    return json.loads(completed.stdout)


class TestVoiceResume(unittest.TestCase):
    def test_persisted_announcement_listener_reconnects_across_sessions_reloads_and_server_restart(self) -> None:
        result = run_voice_resume_harness()

        self.assertEqual(result["beforeNewSession"], 1)
        self.assertEqual(result["openedSessionId"], "new-session")
        self.assertEqual(result["afterNewSession"], 1)
        self.assertEqual(result["afterServerRestart"], 1)
        self.assertEqual(result["afterPageReload"], 1)
        self.assertTrue(result["enabledAfterReload"])
        self.assertGreaterEqual(
            sum(1 for call in result["heartbeatCalls"] if call == {"clientId": "voice-client", "enabled": True}),
            3,
        )


if __name__ == "__main__":
    unittest.main()
