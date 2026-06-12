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
