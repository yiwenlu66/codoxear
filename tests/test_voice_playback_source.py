import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


class TestVoicePlaybackSource(unittest.TestCase):
    def test_live_audio_support_is_checked_before_play(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("function browserSupportsLiveAudioPlayback() {")
        end = source.index("function base64UrlToUint8Array", start)
        block = source[start:end]
        self.assertIn('"application/vnd.apple.mpegurl"', block)
        self.assertIn('"audio/mpegurl"', block)
        self.assertIn("liveAudio.canPlayType(kind)", block)
        self.assertIn("function liveAudioHasReadySegments()", block)

        play_start = source.index("async function startLiveAudioPlayback({ resetSource = false } = {}) {")
        play_end = source.index("function describeLiveAudioStartError(error) {", play_start)
        play_block = source[play_start:play_end]
        self.assertIn("if (!browserSupportsLiveAudioPlayback())", play_block)
        self.assertIn("if (!liveAudioHasReadySegments())", play_block)
        self.assertIn("use Safari or an installed iOS PWA", play_block)
        self.assertIn("wait for the first announcement and try again", play_block)
        self.assertIn("if (resetSource) {", play_block)
        self.assertIn("resetLiveAudioState();", play_block)
        self.assertIn("if (resetSource || liveAudio.currentSrc !== nextSrc)", play_block)

        err_start = source.index("function describeLiveAudioStartError(error) {")
        err_end = source.index("announceBtn.onclick", err_start)
        err_block = source[err_start:err_end]
        self.assertIn("if (/unsupported/i.test(message))", err_block)
        self.assertIn("no live audio segments are available yet", err_block)

    def test_live_audio_watchdog_detects_non_progress_and_hard_resets(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("LIVE_AUDIO_WATCHDOG_MS", source)
        self.assertIn("LIVE_AUDIO_STALL_GRACE_MS", source)
        self.assertIn("LIVE_AUDIO_RESTART_THROTTLE_MS", source)
        self.assertIn("function markLiveAudioProgress()", source)
        self.assertIn("function resetLiveAudioState()", source)
        self.assertIn("function noteLiveAudioPotentialStall(_reason = \"\")", source)
        self.assertIn("function queueLiveAudioHardRestart(_reason = \"\")", source)
        self.assertIn("function runLiveAudioWatchdog()", source)
        self.assertIn("currentTime > liveAudioLastCurrentTime + 0.05", source)
        self.assertIn("queueLiveAudioHardRestart(\"watchdog\")", source)
        self.assertIn("scheduleLiveAudioRetry(150, { resetSource: true });", source)
        self.assertIn("function startLiveAudioWatchdog()", source)
        self.assertIn("function stopLiveAudioWatchdog()", source)
        self.assertIn("function resumeAnnouncementRuntime({ resetSource = false } = {})", source)
        self.assertIn("startAnnouncementHeartbeat();", source)
        self.assertIn('liveAudio.addEventListener("timeupdate"', source)
        self.assertIn('liveAudio.addEventListener("waiting"', source)
        self.assertIn('liveAudio.addEventListener("stalled"', source)
        self.assertIn('liveAudio.addEventListener("suspend"', source)
        self.assertIn('document.addEventListener("visibilitychange"', source)
        self.assertIn('window.addEventListener("pageshow"', source)
        self.assertIn('window.addEventListener("online"', source)
        self.assertIn('window.addEventListener("focus"', source)
        self.assertIn('resumeAnnouncementRuntime({ resetSource: false });', source)
        self.assertIn('resumeAnnouncementRuntime({ resetSource: true });', source)

    def test_voice_controls_attempt_to_arm_live_audio_from_user_gesture(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("async function maybeAutoStartLiveAudioFromGesture({ resetSource = false } = {})", source)
        self.assertIn("await maybeAutoStartLiveAudioFromGesture({ resetSource: true });", source)
        self.assertIn("announceBtn auto-start failed", source)

    def test_notification_transport_is_split_between_desktop_and_mobile(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("desktop_supported", source)
        self.assertIn("push_supported", source)
        self.assertIn("function isMobileNotificationDevice()", source)
        self.assertIn("function notificationDeviceClass()", source)
        self.assertIn("function pushNotificationsEnabledForCurrentDevice()", source)
        self.assertIn('if (notificationDeviceClass() === "mobile") {', source)
        self.assertIn('return pushNotificationsEnabledForCurrentDevice() ? "push" : "none";', source)
        self.assertIn('return activeNotificationTransport() === "desktop";', source)
        self.assertIn('notification error: ${err && err.message ? err.message : "unknown error"}', source)
        self.assertIn("notifications require HTTPS or localhost", source)
        self.assertIn("mobile notifications require web push in an installed HTTPS web app", source)
        self.assertIn('if (notificationDeviceClass() === "desktop") {', source)
        self.assertIn('device_class: notificationDeviceClass()', source)
        self.assertIn("maybeShowDesktopNotification(ev)", source)
        self.assertIn("scheduleDesktopNotificationResolve(ev)", source)
        self.assertIn("/api/notifications/message?message_id=", source)
        self.assertIn("async function pollNotificationFeed({ prime = false } = {})", source)
        self.assertIn("/api/notifications/feed?since=", source)
        self.assertIn("if (!desktopNotificationsEnabled()) return;", source)


if __name__ == "__main__":
    unittest.main()
