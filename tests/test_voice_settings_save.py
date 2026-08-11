from frontend_module_loader import module_path
import json
import subprocess
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VOICE_SOURCE = (module_path("app_voice.js")).read_text(encoding="utf-8")
VOICE_HELPERS_SOURCE = (module_path("app_voice_helpers.js")).read_text(encoding="utf-8")
MODAL_SOURCE = (module_path("app_modal.js")).read_text(encoding="utf-8")


def run_voice_settings_save_harness() -> dict:
    script = textwrap.dedent(
        f"""
        const vm = require("node:vm");
        const calls = [];
        const audioEvents = [];
        const storage = new Map();
        const feedItems = [
          {{ message_id: "fresh", session_id: "session-a", session_display_name: "Agent", notification_text: "Needs attention", updated_ts: 1 }},
        ];

        function node(extra = {{}}) {{
          const listeners = new Map();
          return Object.assign({{
            style: {{}}, dataset: {{}}, value: "", checked: false, textContent: "", title: "", disabled: false, open: false,
            classList: {{ toggle() {{}}, add() {{}}, remove() {{}} }},
            setAttribute() {{}}, removeAttribute() {{}}, append() {{}}, appendChild(child) {{ return child; }}, replaceChildren() {{}},
            matches: () => false,
            addEventListener(type, listener) {{ listeners.set(type, listener); }},
            removeEventListener(type) {{ listeners.delete(type); }},
            load() {{}}, pause() {{}}, play() {{ return Promise.resolve(); }},
          }}, extra);
        }}

        class NotificationCtor {{
          constructor(title, options) {{ calls.push(["desktop-notification", title, options.body]); }}
        }}
        NotificationCtor.permission = "granted";

        class AudioContextCtor {{
          constructor() {{ this.state = "suspended"; this.currentTime = 0; this.destination = {{}}; }}
          resume() {{ audioEvents.push("resume"); this.state = "running"; return Promise.resolve(); }}
          createOscillator() {{
            return {{
              frequency: {{}}, connect() {{}},
              start() {{ audioEvents.push("oscillator-start"); }}, stop() {{ audioEvents.push("oscillator-stop"); }},
            }};
          }}
          createGain() {{
            return {{ gain: {{ setValueAtTime() {{}}, exponentialRampToValueAtTime() {{}} }}, connect() {{}} }};
          }}
          close() {{ return Promise.resolve(); }}
        }}

        const dom = {{
          announceBtn: node(), notificationBtn: node(), notificationPanel: node(), notificationList: node(), notificationEmpty: node(),
          notificationClearBtn: node(), notificationEnableBtn: node(), liveAudio: node(), voiceSettingsBackdrop: node(), voiceSettingsCloseBtn: node(),
          voiceSettingsStatus: node(), voiceBaseUrlInput: node(), voiceApiKeyInput: node(), voiceClearApiKeyToggle: node(), narrationSettingToggle: node(),
          voiceSettingsViewer: node(), voiceSettingsCancelBtn: node(), voiceSettingsSaveBtn: node(),
        }};
        const documentTarget = {{ activeElement: null, contains: () => true, createElement: () => node() }};
        const ctx = {{
          HTMLElement: function HTMLElement() {{}}, Notification: NotificationCtor, console,
          window: {{ isSecureContext: true }}, navigator: {{ userAgent: "X11 Linux" }}, document: documentTarget,
        }};
        vm.createContext(ctx);
        for (const source of [
          {json.dumps(MODAL_SOURCE)},
          {json.dumps(VOICE_HELPERS_SOURCE)},
          {json.dumps(VOICE_SOURCE)},
        ]) vm.runInContext(source, ctx);

        const controller = ctx.window.CodoxearVoice.createVoiceController({{
          ...dom,
          AudioContext: AudioContextCtor,
          Notification: NotificationCtor,
          windowTarget: ctx.window,
          navigatorTarget: ctx.navigator,
          documentTarget,
          isAppDisposed: () => false,
          api: async (url, options = {{}}) => {{
            const body = options.body ? JSON.parse(JSON.stringify(options.body)) : null;
            calls.push(["api", url, body]);
            if (url === "/api/settings/voice") {{
              return {{
                ...(body || {{
                  tts_enabled_for_narration: false,
                  tts_enabled_for_final_response: true,
                  tts_base_url: "https://api.openai.com/v1",
                  tts_api_key: "",
                  tts_api_key_clear: false,
                }}),
                has_tts_api_key: false,
                audio: {{ queue_depth: 0, segment_count: 0, last_error: "", stream_url: "/api/audio/live.m3u8" }},
                notifications: {{ enabled_devices: 0, total_devices: 0, vapid_public_key: "" }},
              }};
            }}
            if (url === "/api/settings/unattended-prompt") return {{ prompt: "", default_prompt: "" }};
            if (url === "/api/notifications/subscription") return {{ subscriptions: [] }};
            if (url.indexOf("/api/notifications/feed") === 0) return {{ items: feedItems.splice(0) }};
            throw new Error(`unexpected API call: ${{url}}`);
          }},
          setToast() {{}}, handleAppAuthLoss() {{}}, prepareModalOpen() {{}}, afterModalVisibilityChanged() {{}},
          resolveAppUrl: (path) => path, versionedShellAssetPath: (path) => path,
          storageGetItem: (key) => storage.get(key) || null,
          storageSetItem: (key, value) => storage.set(key, String(value)),
          storageRemoveItem: (key) => storage.delete(key),
          requestFrame: (callback) => callback(), setTimeout: () => 1, clearTimeout() {{}}, setInterval: () => 1, clearInterval() {{}},
        }});

        (async () => {{
          await controller.loadVoiceSettings();
          dom.voiceBaseUrlInput.value = "https://voice.example/v1";
          dom.narrationSettingToggle.onchange({{ target: {{ checked: true }} }});
          await dom.voiceSettingsSaveBtn.onclick();

          await dom.notificationEnableBtn.onclick();
          await controller.pollNotificationFeed();

          const settingsSave = calls.find((entry) => entry[0] === "api" && entry[1] === "/api/settings/voice" && entry[2]);
          process.stdout.write(JSON.stringify({{
            settingsSave: settingsSave && settingsSave[2],
            audioEvents,
            desktopNotification: calls.find((entry) => entry[0] === "desktop-notification") || null,
          }}));
        }})().catch((error) => {{ console.error(error.stack || error); process.exit(1); }});
        """
    )
    completed = subprocess.run(["node", "-e", script], check=True, text=True, capture_output=True)
    return json.loads(completed.stdout)


def test_voice_settings_save_posts_narration_change_and_primes_notification_audio() -> None:
    result = run_voice_settings_save_harness()

    assert result["settingsSave"] == {
        "tts_enabled_for_narration": True,
        "tts_enabled_for_final_response": True,
        "tts_base_url": "https://voice.example/v1",
        "tts_api_key": "",
        "tts_api_key_clear": False,
    }
    assert result["desktopNotification"] == ["desktop-notification", "Agent", "Needs attention"]
    assert result["audioEvents"] == ["resume", "oscillator-start", "oscillator-stop"]
