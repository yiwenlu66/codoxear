import json
import subprocess
import textwrap
from pathlib import Path

from codoxear.rollout_delivery import _extract_delivery_messages
from codoxear.voice_projection import notification_feed_since


ROOT = Path(__file__).resolve().parents[1]
VOICE_SOURCE = (ROOT / "codoxear" / "static" / "app_voice.js").read_text(encoding="utf-8")
VOICE_HELPERS_SOURCE = (ROOT / "codoxear" / "static" / "app_voice_helpers.js").read_text(encoding="utf-8")
MODAL_SOURCE = (ROOT / "codoxear" / "static" / "app_modal.js").read_text(encoding="utf-8")


def run_voice_vm() -> dict:
    script = textwrap.dedent(
        f"""
        const vm = require("node:vm");
        const notifications = [];
        const sound = {{ starts: 0, stops: 0, resumes: 0 }};
        const storage = new Map();
        const feeds = [
          [{{ message_id: "old", session_id: "s1", session_display_name: "Agent", notification_text: "Earlier intercom", updated_ts: 1 }}],
          [{{ message_id: "fresh", session_id: "s1", session_display_name: "Agent", notification_text: "Needs attention", updated_ts: 2 }}],
        ];
        class NotificationCtor {{
          constructor(title, options) {{ notifications.push({{ title, body: options.body }}); }}
          static requestPermission() {{ NotificationCtor.permission = "granted"; return Promise.resolve("granted"); }}
        }}
        NotificationCtor.permission = "default";
        class AudioContextCtor {{
          constructor() {{ this.state = "suspended"; this.currentTime = 0; this.destination = {{}}; }}
          resume() {{ this.state = "running"; sound.resumes += 1; return Promise.resolve(); }}
          createOscillator() {{ return {{ frequency: {{}}, connect() {{}}, start() {{ sound.starts += 1; }}, stop() {{ sound.stops += 1; }} }}; }}
          createGain() {{ return {{ gain: {{ setValueAtTime() {{}}, exponentialRampToValueAtTime() {{}} }}, connect() {{}} }}; }}
          close() {{ return Promise.resolve(); }}
        }}
        function node() {{
          return {{
            style: {{}}, dataset: {{}}, textContent: "", title: "", disabled: false, open: false, value: "", checked: false,
            _children: [], _attrs: {{}},
            classList: {{ toggle() {{}}, add() {{}}, remove() {{}} }},
            setAttribute(name, value) {{ this._attrs[name] = String(value); }}, removeAttribute(name) {{ delete this._attrs[name]; }},
            append(...children) {{ this._children.push(...children); }}, appendChild(child) {{ this._children.push(child); return child; }},
            replaceChildren(...children) {{ this._children = children; }},
            addEventListener() {{}}, removeEventListener() {{}}, matches() {{ return false; }},
            load() {{}}, pause() {{}}, play() {{ return Promise.resolve(); }},
          }};
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
        for (const source of [{json.dumps(MODAL_SOURCE)}, {json.dumps(VOICE_HELPERS_SOURCE)}, {json.dumps(VOICE_SOURCE)}]) vm.runInContext(source, ctx);
        const controller = ctx.window.CodoxearVoice.createVoiceController({{
          ...dom, Notification: NotificationCtor, AudioContext: AudioContextCtor, windowTarget: ctx.window, navigatorTarget: ctx.navigator, documentTarget,
          isAppDisposed: () => false,
          api: async (url) => {{
            if (url.includes("/api/notifications/feed")) return {{ items: feeds.shift() || [] }};
            if (url.includes("/api/notifications/subscription")) return {{ subscriptions: [] }};
            return {{ notifications: {{}}, audio: {{}} }};
          }},
          setToast() {{}}, handleAppAuthLoss() {{}}, prepareModalOpen() {{}}, afterModalVisibilityChanged() {{}}, resolveAppUrl: (x) => x, versionedShellAssetPath: (x) => x,
          storageGetItem: (key) => storage.get(key) || null, storageSetItem: (key, value) => storage.set(key, String(value)), storageRemoveItem: (key) => storage.delete(key),
          requestFrame: (fn) => fn(), setTimeout: () => 1, clearTimeout() {{}}, setInterval: () => 1, clearInterval() {{}}, focusSessionFromNotification() {{}},
        }});
        (async () => {{
          await dom.notificationEnableBtn.onclick();
          await controller.pollNotificationFeed({{ prime: true }});
          await controller.pollNotificationFeed();
          const beforeRead = dom.notificationBtn.dataset.unread;
          await dom.notificationBtn.onclick({{ preventDefault() {{}}, stopPropagation() {{}} }});
          process.stdout.write(JSON.stringify({{
            browserNotifications: notifications,
            sound,
            beforeRead,
            afterRead: dom.notificationBtn.dataset.unread,
            panelOpen: dom.notificationPanel.style.display,
            panelRows: dom.notificationList._children.length,
            permission: NotificationCtor.permission,
          }}));
        }})().catch((error) => {{ console.error(error.stack || error); process.exit(1); }});
        """
    )
    result = subprocess.run(["node", "-e", script], check=True, text=True, capture_output=True)
    return json.loads(result.stdout)


def test_intercom_delivery_is_a_notification_feed_item() -> None:
    messages = _extract_delivery_messages(
        [
            {
                "type": "custom_message",
                "customType": "intercom_message",
                "id": "intercom-1",
                "timestamp": "2026-08-04T12:00:00.000Z",
                "content": "Subagent needs attention in run 12345678-1234-1234-1234-123456789012",
            }
        ]
    )

    assert len(messages) == 1
    assert messages[0].message_class == "intercom"
    assert messages[0].text.startswith("Subagent needs attention")
    feed = notification_feed_since(
        {
            messages[0].message_id: {
                "message_id": messages[0].message_id,
                "message_class": "intercom",
                "session_id": "s1",
                "session_display_name": "Agent",
                "notification_text": messages[0].text,
                "summary_status": "skipped",
                "updated_ts": 4.0,
            }
        },
        since_ts=0,
    )
    assert feed == [
        {
            "message_id": messages[0].message_id,
            "session_id": "s1",
            "session_display_name": "Agent",
            "notification_text": messages[0].text,
            "updated_ts": 4.0,
        }
    ]


def test_notification_panel_tracks_feed_alerts_with_sound_and_marks_read() -> None:
    result = run_voice_vm()

    assert result["permission"] == "granted"
    assert result["browserNotifications"] == [{"title": "Agent", "body": "Needs attention"}]
    assert result["sound"] == {"starts": 1, "stops": 1, "resumes": 1}
    assert result["beforeRead"] == "1"
    assert result["afterRead"] == "0"
    assert result["panelOpen"] == "flex"
    assert result["panelRows"] == 2
