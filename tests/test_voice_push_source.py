import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VOICE_PUSH = ROOT / "codoxear" / "voice_push.py"
VOICE_PUSH_STATE = ROOT / "codoxear" / "voice_push_state.py"
VOICE_OPENAI_CLIENT = ROOT / "codoxear" / "voice_openai_client.py"
VOICE_HLS = ROOT / "codoxear" / "voice_hls.py"
VOICE_WEBPUSH = ROOT / "codoxear" / "voice_webpush.py"
VOICE_PERSISTENCE = ROOT / "codoxear" / "voice_persistence.py"
VOICE_PROJECTION = ROOT / "codoxear" / "voice_projection.py"
VOICE_ROUTES = ROOT / "codoxear" / "voice_routes.py"
SERVICE_WORKER = ROOT / "codoxear" / "static" / "service-worker.js"
APP_JS = ROOT / "codoxear" / "static" / "app.js"


def js_function(source: str, name: str) -> str:
    raw_start = source.index(f"function {name}")
    start = raw_start - len("async ") if source[max(0, raw_start - len("async ")) : raw_start] == "async " else raw_start
    params_end = source.index(")", raw_start)
    brace = source.index("{", params_end)
    depth = 0
    for idx in range(brace, len(source)):
        ch = source[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return source[start : idx + 1]
    raise AssertionError(f"could not extract {name}")


def eval_desktop_notification_clickthrough() -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    snippet = "\n".join(js_function(source, name) for name in ["focusSessionFromDesktopNotification", "showDesktopNotification"])
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const notifications = [];
        let focusCalls = 0;
        let closeCalls = 0;
        let prevented = 0;
        const ctx = {{
          deliveredDesktopNotificationIds: new Set(),
          desktopNotificationsEnabled: () => true,
          sessionHash: "",
          selectCalls: [],
          window: {{ focus: () => {{ focusCalls += 1; }} }},
          Notification: class {{
            constructor(title, options) {{
              this.title = title;
              this.options = options;
              this.close = () => {{ closeCalls += 1; }};
              notifications.push(this);
            }}
          }},
          Date: {{ now: () => 12345 }},
          sessionIdFromHash: () => ctx.sessionHash,
          setSessionHash: (sid) => {{ ctx.sessionHash = sid; }},
          selectSessionFromHash: async (options) => {{ ctx.selectCalls.push(options); }},
          handleAppAuthLoss: () => {{ ctx.authLost = true; }},
          console,
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet)}, ctx);
        ctx.showDesktopNotification({{ messageId: "m1", title: "Session A", body: "  hello   world  ", sessionId: "s2" }});
        notifications[0].onclick({{ preventDefault: () => {{ prevented += 1; }} }});
        process.stdout.write(JSON.stringify({{
          count: notifications.length,
          title: notifications[0].title,
          body: notifications[0].options.body,
          tag: notifications[0].options.tag,
          delivered: ctx.deliveredDesktopNotificationIds.has("m1"),
          focusCalls,
          closeCalls,
          prevented,
          sessionHash: ctx.sessionHash,
          selectCalls: ctx.selectCalls,
        }}));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


class TestVoicePushSource(unittest.TestCase):
    def test_summary_prompts_cover_final_and_narration_targets(self) -> None:
        source = VOICE_OPENAI_CLIENT.read_text(encoding="utf-8")
        self.assertIn("about 30 words", source)
        self.assertIn("roughly 24 to 36 words", source)
        self.assertIn("about 15 words", source)
        self.assertIn("roughly 12 to 18 words", source)

    def test_keepalive_silence_is_enabled(self) -> None:
        hls_source = VOICE_HLS.read_text(encoding="utf-8")
        self.assertIn("HLS_KEEPALIVE_SECONDS", hls_source)
        self.assertIn("append_silence", hls_source)
        self.assertIn("anullsrc", hls_source)

    def test_voice_pool_includes_full_verified_set(self) -> None:
        state_source = VOICE_PUSH_STATE.read_text(encoding="utf-8")
        self.assertIn('"cedar"', state_source)
        self.assertIn('"marin"', state_source)
        self.assertIn('"verse"', state_source)

    def test_voice_push_facade_uses_extracted_runtime_modules(self) -> None:
        source = VOICE_PUSH.read_text(encoding="utf-8")
        self.assertIn("from .voice_hls import MergedHLSStream", source)
        self.assertIn("from .voice_openai_client import OpenAICompatibleClient", source)
        self.assertIn("from .voice_webpush import send_web_push_notifications", source)
        self.assertIn("from .voice_persistence import load_voice_settings", source)
        self.assertIn("from .voice_projection import voice_settings_snapshot_payload", source)
        self.assertIn("from .voice_task_queue import enqueue_announcement_task", source)
        self.assertIn("from .voice_ledger import set_ledger_fields_many", source)
        self.assertNotIn("from pywebpush import webpush", source)
        self.assertNotIn("from py_vapid import Vapid", source)

    def test_notification_text_is_canonical_backend_field(self) -> None:
        source = VOICE_PUSH.read_text(encoding="utf-8")
        projection_source = VOICE_PROJECTION.read_text(encoding="utf-8")
        self.assertIn('"notification_text"', source)
        self.assertIn('row.get("notification_text")', projection_source)
        self.assertNotIn('"preview_text": notification_text', source)
        sw_source = SERVICE_WORKER.read_text(encoding="utf-8")
        self.assertIn("payload.notification_text", sw_source)

    def test_voice_settings_redact_api_key_and_preserve_blank_save(self) -> None:
        voice_source = VOICE_PUSH.read_text(encoding="utf-8")
        voice_state_source = VOICE_PUSH_STATE.read_text(encoding="utf-8")
        voice_webpush_source = VOICE_WEBPUSH.read_text(encoding="utf-8")
        voice_persistence_source = VOICE_PERSISTENCE.read_text(encoding="utf-8")
        voice_projection_source = VOICE_PROJECTION.read_text(encoding="utf-8")
        voice_route_source = VOICE_ROUTES.read_text(encoding="utf-8")
        app_source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("def _chmod_private_file(path: Path) -> None:", voice_state_source)
        self.assertIn("os.chmod(path, 0o600)", voice_state_source)
        self.assertIn("save_voice_settings(self._settings_path, payload)", voice_source)
        self.assertIn("_chmod_private_file(path)", voice_persistence_source)
        self.assertIn("ensure_vapid_public_key(self._vapid_private_key_path)", voice_source)
        self.assertIn("_chmod_private_file(path)", voice_webpush_source)
        self.assertIn("def settings_snapshot(self, *, redact_secrets: bool = False)", voice_source)
        self.assertIn('settings["tts_api_key"] = ""', voice_projection_source)
        self.assertIn('settings["has_tts_api_key"] = has_tts_api_key', voice_projection_source)
        self.assertIn("preserve_blank_api_key: bool = False", voice_source)
        self.assertIn('obj["tts_api_key"] = str(self._voice_settings.get("tts_api_key") or "")', voice_source)
        self.assertIn("settings_snapshot(redact_secrets=True)", voice_route_source)
        self.assertIn("set_settings(obj, preserve_blank_api_key=True, redact_response=True)", voice_route_source)
        self.assertIn('has_tts_api_key: false', app_source)
        self.assertIn('id: "voiceClearApiKeyToggle"', app_source)
        self.assertIn("function voiceSettingsDialogOpen()", app_source)
        self.assertIn("function syncVoiceSettingsFormFromState()", app_source)
        self.assertIn("if (!voiceSettingsDialogOpen()) syncVoiceSettingsFormFromState();", app_source)
        self.assertIn("syncVoiceSettingsFormFromState();\n          if (!voiceSettingsViewer.open) voiceSettingsViewer.showModal();", app_source)
        self.assertIn('voiceApiKeyInput.value = "";', app_source)
        self.assertIn('voiceApiKeyInput.placeholder = voiceSettings.has_tts_api_key ? "Saved API key (leave blank to keep)" : "Enter API key";', app_source)
        self.assertIn('tts_api_key: clearApiKey ? "" : String(voiceApiKeyInput.value || "").trim()', app_source)
        self.assertIn('tts_api_key_clear: clearApiKey', app_source)
        self.assertNotIn('voiceApiKeyInput.value = String(voiceSettings.tts_api_key || "")', app_source)

    def test_notification_click_awaits_navigation_before_focus(self) -> None:
        sw_source = SERVICE_WORKER.read_text(encoding="utf-8")
        self.assertIn('clients.matchAll({ type: "window", includeUncontrolled: true }).then(async (windows) => {', sw_source)
        self.assertIn('if ("navigate" in client) {', sw_source)
        self.assertIn("const navigated = await client.navigate(target);", sw_source)
        self.assertIn("return (navigated || client).focus();", sw_source)
        self.assertNotIn("client.navigate(target);\n          return client.focus();", sw_source)

    def test_desktop_notification_click_focuses_origin_session(self) -> None:
        result = eval_desktop_notification_clickthrough()
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["title"], "Session A")
        self.assertEqual(result["body"], "hello world")
        self.assertEqual(result["tag"], "m1")
        self.assertTrue(result["delivered"])
        self.assertEqual(result["focusCalls"], 1)
        self.assertEqual(result["closeCalls"], 1)
        self.assertEqual(result["prevented"], 1)
        self.assertEqual(result["sessionHash"], "s2")
        self.assertEqual(result["selectCalls"], [{"refreshIfMissing": True, "deferIfMissing": True}])
        app_source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("function focusSessionFromDesktopNotification(sessionId)", app_source)
        self.assertIn("if (sessionIdFromHash() !== sid) setSessionHash(sid);", app_source)
        self.assertIn("selectSessionFromHash({ refreshIfMissing: true, deferIfMissing: true })", app_source)
        self.assertIn("const notification = new Notification(safeTitle", app_source)
        self.assertIn("notification.onclick = (event) => {", app_source)
        self.assertIn("focusSessionFromDesktopNotification(sid);", app_source)
        self.assertIn("sessionId: item && item.session_id", app_source)
        self.assertIn("showDesktopNotification({ messageId, title, body, sessionId });", app_source)
        self.assertIn("scheduleDesktopNotificationResolve({ ...ev, session_id: sessionId });", app_source)

    def test_hashchange_refreshes_and_defers_missing_notification_target(self) -> None:
        app_source = APP_JS.read_text(encoding="utf-8")
        start = app_source.index("function rememberPendingHashSession")
        end = app_source.index("// Unattended menu state, async load/save orchestration", start)
        block = app_source[start:end]
        self.assertIn('let pendingHashSessionId = "";', app_source)
        self.assertIn("let pendingHashSessionSelectInFlight = false;", app_source)
        self.assertIn("function maybeSelectPendingHashSession()", block)
        self.assertIn("if (sessionIdFromHash() !== sid)", block)
        self.assertIn("const session = sessionIndex.get(sid);", block)
        self.assertIn("if (!sessionSelectable(session)) return;", block)
        self.assertIn("pendingHashSessionSelectInFlight = true;", block)
        self.assertIn("void selectSession(sid)", block)
        self.assertIn("async function selectSessionFromHash({ refreshIfMissing = false, deferIfMissing = false } = {})", block)
        self.assertIn("const sid = sessionIdFromHash();", block)
        self.assertIn("let session = sessionIndex.get(sid);", block)
        self.assertIn("if (!session && refreshIfMissing) {", block)
        self.assertIn("await refreshSessions();", block)
        self.assertIn("if (e && e.status === 401) handleAppAuthLoss();", block)
        self.assertIn("session = sessionIndex.get(sid);", block)
        self.assertIn("if (!sessionSelectable(session)) {\n            if (deferIfMissing) rememberPendingHashSession(sid);\n            return;\n          }", block)
        self.assertIn("rememberPendingHashSession(\"\");\n          await selectSession(sid);", block)
        self.assertIn("maybeSelectPendingHashSession();\n          return sessions;", app_source)
        self.assertIn('addAppEvent(window, "hashchange", async () => {\n                await selectSessionFromHash({ refreshIfMissing: true, deferIfMissing: true });\n              });', app_source)


if __name__ == "__main__":
    unittest.main()
