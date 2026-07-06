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
        self.assertIn("if (newSessionController) newSessionController.disposeResumeLoadTimer();", cleanup)
        # Voice cleanup (voice save timer, live-audio retry timer, desktop
        # notification resolve timers, announcement heartbeat, live-audio
        # watchdog, HLS state, and all voice/notification handlers) is owned by
        # the CodoxearVoice controller; app.js only delegates through
        # voiceController.dispose().
        self.assertIn("if (voiceController) voiceController.dispose();", cleanup)
        for removed in [
            "if (voiceSaveTimer) clearTimeout(voiceSaveTimer);",
            "voiceSaveTimer = null;",
            "if (liveAudioRetryTimer) clearTimeout(liveAudioRetryTimer);",
            "liveAudioRetryTimer = null;",
            "desktopNotificationTimers.forEach((timer) => clearTimeout(timer));",
            "desktopNotificationTimers.clear();",
            "stopAnnouncementHeartbeat();",
            "stopLiveAudioWatchdog();",
            "resetLiveAudioState();",
        ]:
            self.assertNotIn(removed, cleanup)
        self.assertIn("if (iosViewportGuardTimer) clearTimeout(iosViewportGuardTimer);", cleanup)
        # Search cleanup is now owned by the CodoxearChatSearch controller;
        # app.js only delegates through chatSearchController.dispose().
        self.assertIn("if (chatSearchController) chatSearchController.dispose();", cleanup)
        self.assertNotIn("chatSearchAllRuntime.dispose();", cleanup)
        self.assertNotIn("chatSearchAllAbort", cleanup)
        self.assertNotIn("chatSearchAllTimer", cleanup)
        self.assertIn("olderLoadRuntime.invalidate();", cleanup)
        self.assertNotIn("olderLoadController", cleanup)
        self.assertIn("fileViewerController.abortPendingFileOpenTransport();", cleanup)
        self.assertIn("filePickerSearchState.dispose();", cleanup)
        # Unattended cleanup is now owned by the CodoxearUnattended controller;
        # app.js only delegates through unattendedController.dispose().
        self.assertIn("if (unattendedController) unattendedController.dispose();", cleanup)
        for removed in [
            "unattendedSaveTimers.forEach((timer) => clearTimeout(timer));",
            "unattendedSaveTimers.clear();",
            "unattendedSavePending.clear();",
            "unattendedSaveInFlight.clear();",
        ]:
            self.assertNotIn(removed, cleanup)
        # Queue cleanup is now owned by the CodoxearQueue controller; app.js only
        # delegates through queueController.dispose().
        self.assertIn("if (queueController) queueController.dispose();", cleanup)
        for removed in [
            "queueUpdateTimers.forEach((timer) => clearTimeout(timer));",
            "queueMutationLocks.clear();",
            "queuePendingDeletes.clear();",
        ]:
            self.assertNotIn(removed, cleanup)
        self.assertIn("while (appEventCleanups.length)", cleanup)
        self.assertIn("cleanup();", cleanup)
        self.assertIn("clearApiCache();", cleanup)

    def test_hide_new_session_dialog_disposes_resume_load_timer(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        hide_start = source.index("function hideNewSessionDialog()")
        hide_end = source.index("function launchPresetProviderChoice(s)", hide_start)
        hide_block = source[hide_start:hide_end]
        self.assertIn("if (newSessionController) newSessionController.disposeResumeLoadTimer();", hide_block)
        # The dispose call must run unconditionally on close, before menu/state reset.
        dispose_idx = hide_block.index("if (newSessionController) newSessionController.disposeResumeLoadTimer();")
        status_idx = hide_block.index("newSessionStatus.textContent = \"\";")
        self.assertLess(dispose_idx, status_idx)

    def test_auth_loss_and_logout_use_shared_cleanup(self) -> None:
        app = render_app_block()
        self.assertIn("function handleAppAuthLoss() {\n          if (appDisposed) return;\n          cleanupApp();\n          renderLogin(renderApp);\n        }", app)
        self.assertIn("if (e && e.status === 401) {\n              handleAppAuthLoss();\n              return;\n            }", app)
        logout_start = app.index('$("#logoutBtnSide").onclick = async () => {')
        logout_end = app.index("toggleSidebarBtn.onclick", logout_start)
        logout = app[logout_start:logout_end]
        self.assertIn("finally {\n            if (appDisposed) return;\n            cleanupApp();\n            renderLogin(renderApp);\n          }", logout)

    def test_send_and_queue_api_401_uses_global_auth_loss(self) -> None:
        app = render_app_block()
        send_start = app.index("async function sendText(")
        send_end = app.index("form.onsubmit = async", send_start)
        send_block = app[send_start:send_end]
        send_catch = send_block[send_block.index("} catch (e2) {") : send_block.index("} finally {", send_block.index("} catch (e2) {"))]
        self.assertLess(send_catch.index("if (e2 && e2.status === 401)"), send_catch.index("const commitUnknown = Boolean"))
        self.assertLess(send_catch.index("if (e2 && e2.status === 401)"), send_catch.index("setToast(`send error:"))
        self.assertIn("handleAppAuthLoss();\n              return false;", send_catch)
        self.assertIn("if (clearErr && clearErr.status === 401)", send_block)
        self.assertGreaterEqual(send_block.count("if (e && e.status === 401) handleAppAuthLoss();"), 3)
        self.assertIn("if (refreshErr && refreshErr.status === 401) handleAppAuthLoss();", app)
        attachment_catch_start = app.index("} catch (e) {", app.index("/inject_file"))
        attachment_catch = app[attachment_catch_start : app.index("function clearComposer", attachment_catch_start)]
        self.assertLess(attachment_catch.index("if (e && e.status === 401)"), attachment_catch.index("failures.push("))
        self.assertIn("handleAppAuthLoss();\n                return false;", attachment_catch)

        clear_start = app.index("async function clearCommitUnknownSend(")
        clear_end = app.index("async function refreshSessions", clear_start)
        clear_block = app[clear_start:clear_end]
        clear_catch = clear_block[clear_block.index("} catch (e) {") :]
        self.assertLess(clear_catch.index("if (e && e.status === 401)"), clear_catch.index("setToast(`clear unknown send error:"))

        # Queue enqueue/delete/move/update/refresh 401 ordering now lives in the
        # CodoxearQueue controller module (see test_frontend_queue_module_source).
        # app.js keeps only the delegating wrappers and the dispose hook; assert
        # those contracts here while leaving the detailed ordering to the module.
        self.assertIn("return queueController.enqueueComposerText(raw, opts);", app)
        self.assertIn("return queueController.refreshQueueViewer();", app)
        for removed in [
            "async function deleteQueueItem(",
            "async function moveQueueItem(",
            "function scheduleQueueUpdate(",
            "function renderQueueList() {",
        ]:
            self.assertNotIn(removed, app)

    def test_async_poll_results_stop_after_cleanup(self) -> None:
        app = render_app_block()
        voice_source = (Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app_voice.js").read_text(encoding="utf-8")
        refresh_start = app.index("async function refreshSessions()")
        refresh_end = app.index("function appendEvent", refresh_start)
        refresh_block = app[refresh_start:refresh_end]
        # Voice feed/subscription polling now lives in the CodoxearVoice
        # controller module; assert the disposed-guards moved with it and that
        # app.js delegates the public entry points.
        self.assertIn('const data = await api("/api/sessions");\n          if (appDisposed) return latestSessions;', refresh_block)
        self.assertIn("if (isAppDisposed() || !desktopNotificationsEnabled()) return;", voice_source)
        self.assertIn("if (isAppDisposed()) return;", voice_source)
        self.assertIn("if (isAppDisposed()) return data;", voice_source)
        self.assertIn("async function syncNotificationState(serverSnapshot) {", voice_source)
        self.assertIn('snapshot = await api("/api/notifications/subscription");', voice_source)
        self.assertIn('let endpoint = "";', voice_source)
        self.assertIn("const reg = await ensureVoiceServiceWorker();", voice_source)
        self.assertIn("const sub = await reg.pushManager.getSubscription();", voice_source)
        self.assertIn("if (isAppDisposed()) return;", voice_source)
        self.assertIn("return voiceController.pollNotificationFeed(opts);", app)
        self.assertIn("return voiceController.syncNotificationState(serverSnapshot);", app)
        self.assertIn("return voiceController.loadVoiceSettings();", app)

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
