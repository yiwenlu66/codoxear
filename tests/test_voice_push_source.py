import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VOICE_PUSH = ROOT / "codoxear" / "voice_push.py"
SERVICE_WORKER = ROOT / "codoxear" / "static" / "service-worker.js"
APP_JS = ROOT / "codoxear" / "static" / "app.js"


class TestVoicePushSource(unittest.TestCase):
    def test_summary_prompts_cover_final_and_narration_targets(self) -> None:
        source = VOICE_PUSH.read_text(encoding="utf-8")
        self.assertIn("about 30 words", source)
        self.assertIn("roughly 24 to 36 words", source)
        self.assertIn("about 15 words", source)
        self.assertIn("roughly 12 to 18 words", source)

    def test_keepalive_silence_is_enabled(self) -> None:
        source = VOICE_PUSH.read_text(encoding="utf-8")
        self.assertIn("HLS_KEEPALIVE_SECONDS", source)
        self.assertIn("append_silence", source)
        self.assertIn("anullsrc", source)

    def test_voice_pool_includes_full_verified_set(self) -> None:
        source = VOICE_PUSH.read_text(encoding="utf-8")
        self.assertIn('"cedar"', source)
        self.assertIn('"marin"', source)
        self.assertIn('"verse"', source)

    def test_notification_text_is_canonical_backend_field(self) -> None:
        source = VOICE_PUSH.read_text(encoding="utf-8")
        self.assertIn('"notification_text"', source)
        self.assertIn('row.get("notification_text")', source)
        self.assertNotIn('"preview_text": notification_text', source)
        sw_source = SERVICE_WORKER.read_text(encoding="utf-8")
        self.assertIn("payload.notification_text", sw_source)

    def test_voice_settings_redact_api_key_and_preserve_blank_save(self) -> None:
        voice_source = VOICE_PUSH.read_text(encoding="utf-8")
        server_source = (ROOT / "codoxear" / "server.py").read_text(encoding="utf-8")
        app_source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("def _chmod_private_file(path: Path) -> None:", voice_source)
        self.assertIn("os.chmod(path, 0o600)", voice_source)
        self.assertIn("_chmod_private_file(self._settings_path)", voice_source)
        self.assertIn("_chmod_private_file(self._vapid_private_key_path)", voice_source)
        self.assertIn("def settings_snapshot(self, *, redact_secrets: bool = False)", voice_source)
        self.assertIn('settings["tts_api_key"] = ""', voice_source)
        self.assertIn('settings["has_tts_api_key"] = has_tts_api_key', voice_source)
        self.assertIn("preserve_blank_api_key: bool = False", voice_source)
        self.assertIn('obj["tts_api_key"] = str(self._voice_settings.get("tts_api_key") or "")', voice_source)
        self.assertIn("settings_snapshot(redact_secrets=True)", server_source)
        self.assertIn("set_settings(obj, preserve_blank_api_key=True, redact_response=True)", server_source)
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

    def test_hashchange_refreshes_and_defers_missing_notification_target(self) -> None:
        app_source = APP_JS.read_text(encoding="utf-8")
        start = app_source.index("function rememberPendingHashSession")
        end = app_source.index("function parseUnattendedDraftInt", start)
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
