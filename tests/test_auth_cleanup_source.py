import re
import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


def render_app_block() -> str:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("function renderApp()")
    end = source.index("(async function boot()", start)
    return source[start:end]


class TestAuthCleanupSource(unittest.TestCase):
    def test_render_boundaries_cleanup_previous_app(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        login_start = source.index("function renderLogin(onAuthed)")
        login_end = source.index("function renderApp()", login_start)
        login_block = source[login_start:login_end]
        app = render_app_block()

        self.assertIn("let activeAppCleanup = null;", source)
        self.assertIn("function cleanupActiveApp()", source)
        self.assertIn("cleanupActiveApp();\n        const root = $(\"#root\");", login_block)
        self.assertIn("cleanupActiveApp();\n\t        const root = $(\"#root\");", app)
        self.assertIn("activeAppCleanup = cleanupApp;", app)
        self.assertIn("} finally {\n              if (appDisposed) return;", app)

    def test_cleanup_stops_polling_timers_controllers_and_listeners(self) -> None:
        app = render_app_block()
        cleanup_start = app.index("function cleanupApp()")
        cleanup_end = app.index("function handleAppAuthLoss()", cleanup_start)
        cleanup = app[cleanup_start:cleanup_end]

        self.assertIn("appDisposed = true;", cleanup)
        self.assertIn("sessionsPollingEnabled = false;", cleanup)
        self.assertIn("secondaryPollingEnabled = false;", cleanup)
        self.assertIn("stopMessagePolling();", cleanup)
        self.assertIn("stopAllPolling();", cleanup)
        for name in [
            "newSessionResumeLoadTimer",
            "voiceSaveTimer",
            "unattendedSaveTimer",
            "liveAudioRetryTimer",
            "fileSearchTimer",
            "iosViewportGuardTimer",
        ]:
            self.assertIn(f"if ({name}) clearTimeout({name});", cleanup)
            self.assertIn(f"{name} = null;", cleanup)
        for name in ["chatSearchAllAbort", "olderLoadController", "fileOpenAbortController", "fileSearchAbort"]:
            self.assertIn(f"abortController({name});", cleanup)
            self.assertIn(f"{name} = null;", cleanup)
        self.assertIn("desktopNotificationTimers.forEach((timer) => clearTimeout(timer));", cleanup)
        self.assertIn("stopAnnouncementHeartbeat();", cleanup)
        self.assertIn("stopLiveAudioWatchdog();", cleanup)
        self.assertIn("resetLiveAudioState();", cleanup)
        self.assertIn("while (appEventCleanups.length)", cleanup)
        self.assertIn("cleanup();", cleanup)
        self.assertIn("apiEtags.clear();", cleanup)

    def test_auth_loss_and_logout_use_shared_cleanup(self) -> None:
        app = render_app_block()
        self.assertIn("function handleAppAuthLoss() {\n          cleanupApp();\n          renderLogin(renderApp);\n        }", app)
        self.assertIn("if (e && e.status === 401) {\n              handleAppAuthLoss();\n              return;\n            }", app)
        logout_start = app.index('$("#logoutBtnSide").onclick = async () => {')
        logout_end = app.index("toggleSidebarBtn.onclick", logout_start)
        logout = app[logout_start:logout_end]
        self.assertIn("finally {\n            cleanupApp();\n            renderLogin(renderApp);\n          }", logout)

    def test_app_global_listeners_are_cleanup_tracked(self) -> None:
        app = render_app_block()
        self.assertIn("function addAppEvent(target, type, handler, options)", app)
        self.assertIn("appEventCleanups.push(() => target.removeEventListener(type, handler, options));", app)
        self.assertIn('addAppEvent(window, "hashchange"', app)
        self.assertIn('addAppEvent(window, "beforeunload"', app)
        self.assertIn('addAppEvent(document, "visibilitychange"', app)
        self.assertIn('addAppEvent(document, "keydown", handleFileTouchSelectionKeydown, true);', app)
        self.assertIn('addAppEvent(window.visualViewport, "resize", onViewportShift);', app)
        self.assertNotRegex(app, re.compile(r"\b(?:window|document)(?:\.visualViewport)?\.addEventListener\s*\("))
        self.assertNotIn("__codexwebUnattendedGlobalHandlers", app)


if __name__ == "__main__":
    unittest.main()
