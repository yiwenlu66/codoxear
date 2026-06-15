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
            "liveAudioRetryTimer",
            "fileSearchTimer",
            "chatSearchAllTimer",
            "iosViewportGuardTimer",
        ]:
            self.assertIn(f"if ({name}) clearTimeout({name});", cleanup)
            self.assertIn(f"{name} = null;", cleanup)
        for name in ["chatSearchAllAbort", "olderLoadController", "fileOpenAbortController", "fileSearchAbort"]:
            self.assertIn(f"abortController({name});", cleanup)
            self.assertIn(f"{name} = null;", cleanup)
        self.assertIn("unattendedSaveTimers.forEach((timer) => clearTimeout(timer));", cleanup)
        self.assertIn("unattendedSaveTimers.clear();", cleanup)
        self.assertIn("unattendedSavePending.clear();", cleanup)
        self.assertIn("unattendedSaveInFlight.clear();", cleanup)
        self.assertIn("queueUpdateTimers.forEach((timer) => clearTimeout(timer));", cleanup)
        self.assertIn("queueUpdateTimers.clear();", cleanup)
        self.assertIn("queueMutationLocks.clear();", cleanup)
        self.assertIn("queuePendingDeletes.clear();", cleanup)
        self.assertIn("desktopNotificationTimers.forEach((timer) => clearTimeout(timer));", cleanup)
        self.assertIn("stopAnnouncementHeartbeat();", cleanup)
        self.assertIn("stopLiveAudioWatchdog();", cleanup)
        self.assertIn("resetLiveAudioState();", cleanup)
        self.assertIn("while (appEventCleanups.length)", cleanup)
        self.assertIn("cleanup();", cleanup)
        self.assertIn("clearApiCache();", cleanup)

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
        self.assertLess(attachment_catch.index("if (e && e.status === 401)"), attachment_catch.index("if (selected === sid)"))

        clear_start = app.index("async function clearCommitUnknownSend(")
        clear_end = app.index("async function refreshSessions", clear_start)
        clear_block = app[clear_start:clear_end]
        clear_catch = clear_block[clear_block.index("} catch (e) {") :]
        self.assertLess(clear_catch.index("if (e && e.status === 401)"), clear_catch.index("setToast(`clear unknown send error:"))

        enqueue_start = app.index("async function enqueueComposerText(")
        enqueue_end = app.index("async function deleteQueueItem", enqueue_start)
        enqueue_block = app[enqueue_start:enqueue_end]
        enqueue_catch = enqueue_block[enqueue_block.index("} catch (e) {") : enqueue_block.index("} finally {", enqueue_block.index("} catch (e) {"))]
        self.assertLess(enqueue_catch.index("if (e && e.status === 401)"), enqueue_catch.index("setToast(`queue error:"))
        self.assertIn("handleAppAuthLoss();\n              return false;", enqueue_catch)

        delete_start = app.index("async function deleteQueueItem(")
        delete_end = app.index("async function moveQueueItem", delete_start)
        delete_block = app[delete_start:delete_end]
        delete_catch = delete_block[delete_block.index("} catch (e) {") : delete_block.index("} finally {", delete_block.index("} catch (e) {"))]
        self.assertLess(delete_catch.index("if (e && e.status === 401)"), delete_catch.index("await refreshQueueViewer();"))
        self.assertLess(delete_catch.index("if (e && e.status === 401)"), delete_catch.index("setToast(`queue delete error:"))

        move_start = app.index("async function moveQueueItem(")
        move_end = app.index("function scheduleQueueUpdate", move_start)
        move_block = app[move_start:move_end]
        move_catch = move_block[move_block.index("} catch (e) {") : move_block.index("} finally {", move_block.index("} catch (e) {"))]
        self.assertLess(move_catch.index("if (e && e.status === 401)"), move_catch.index("setToast(`queue move error:"))

        update_start = app.index("function scheduleQueueUpdate(")
        update_end = app.index("function renderQueueList", update_start)
        update_block = app[update_start:update_end]
        update_catch = update_block[update_block.index("} catch (e) {") : update_block.index("} finally {", update_block.index("} catch (e) {"))]
        self.assertLess(update_catch.index("if (appDisposed) return;"), update_catch.index("if (e && e.status === 401)"))
        self.assertLess(update_catch.index("if (e && e.status === 401)"), update_catch.index("setToast(`queue update error:"))
        self.assertIn("if (appDisposed) return;", update_block)
        self.assertIn("if (appDisposed) return;\n              queueLastEditMs = 0;", update_block)
        self.assertIn("if (!appDisposed && queuePendingDeletes.has(itemKey))", update_block)

        refresh_start = app.index("async function refreshQueueViewer()")
        refresh_end = app.index("function showQueueViewer({ opener = null } = {})", refresh_start)
        refresh_block = app[refresh_start:refresh_end]
        refresh_catch = refresh_block[refresh_block.index("} catch (e) {") :]
        self.assertLess(refresh_catch.index("if (e && e.status === 401)"), refresh_catch.index("Queue unavailable:"))
        self.assertLess(refresh_catch.index("if (e && e.status === 401)"), refresh_catch.index("setToast(`queue load error:"))

    def test_async_poll_results_stop_after_cleanup(self) -> None:
        app = render_app_block()
        refresh_start = app.index("async function refreshSessions()")
        refresh_end = app.index("function appendEvent", refresh_start)
        refresh_block = app[refresh_start:refresh_end]
        voice_start = app.index("async function pollNotificationFeed")
        voice_end = app.index("async function enableNotificationsOnDevice", voice_start)
        voice_block = app[voice_start:voice_end]

        self.assertIn('const data = await api("/api/sessions");\n          if (appDisposed) return latestSessions;', refresh_block)
        self.assertIn("if (appDisposed || !desktopNotificationsEnabled()) return;", voice_block)
        self.assertIn("if (appDisposed) return;", voice_block)
        self.assertIn("if (appDisposed || !desktopNotificationsEnabled()) {", voice_block)
        self.assertIn("if (appDisposed) return data;", voice_block)
        self.assertIn("async function syncNotificationState(serverSnapshot) {\n          if (appDisposed) return;", voice_block)
        self.assertIn('snapshot = await api("/api/notifications/subscription");', voice_block)
        self.assertIn("if (appDisposed) return;\n          let endpoint = \"\";", voice_block)
        self.assertIn("const reg = await ensureVoiceServiceWorker();\n              if (appDisposed) return;", voice_block)
        self.assertIn("const sub = await reg.pushManager.getSubscription();\n              if (appDisposed) return;", voice_block)

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
