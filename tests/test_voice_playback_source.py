import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_VOICE_JS = ROOT / "codoxear" / "static" / "app_voice.js"
APP_VOICE_HELPERS_JS = ROOT / "codoxear" / "static" / "app_voice_helpers.js"
INDEX_HTML = ROOT / "codoxear" / "static" / "index.html"


def eval_voice_helpers() -> dict:
    source = APP_VOICE_HELPERS_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx);
        const helpers = ctx.window.CodoxearVoiceHelpers;
        function canPlay(result) {{ return {{ canPlayType: () => result }}; }}
        const decoded = Array.from(helpers.base64UrlToUint8Array("SGVsbG8td29ybGQ", (value) => Buffer.from(value, "base64").toString("binary")));
        process.stdout.write(JSON.stringify({{
          frozen: Object.isFrozen(helpers),
          nativeProbably: helpers.browserSupportsNativeLiveAudioPlayback(canPlay("probably")),
          nativeMaybe: helpers.browserSupportsNativeLiveAudioPlayback(canPlay("maybe")),
          nativeNo: helpers.browserSupportsNativeLiveAudioPlayback(canPlay("")),
          nativeMissingAudio: helpers.browserSupportsNativeLiveAudioPlayback(null),
          mseYes: helpers.browserSupportsMseLiveAudioPlayback({{ Hls: {{ isSupported: () => true }} }}),
          mseNo: helpers.browserSupportsMseLiveAudioPlayback({{ Hls: {{ isSupported: () => false }} }}),
          preferVendorApple: helpers.shouldPreferNativeLiveAudioPlayback(canPlay("probably"), {{ vendor: "Apple Computer, Inc.", userAgent: "whatever" }}),
          preferWebKitNotChrome: helpers.shouldPreferNativeLiveAudioPlayback(canPlay("maybe"), {{ vendor: "", userAgent: "Mozilla AppleWebKit Safari" }}),
          preferChromeFalse: helpers.shouldPreferNativeLiveAudioPlayback(canPlay("maybe"), {{ vendor: "", userAgent: "AppleWebKit Chrome" }}),
          livePlaybackViaMse: helpers.browserSupportsLiveAudioPlayback(canPlay(""), {{ Hls: {{ isSupported: () => true }} }}),
          decodedText: String.fromCharCode(...decoded),
          androidMobile: helpers.isMobileNotificationDevice({{ userAgent: "Mozilla Android Mobile" }}),
          ipadDesktopUa: helpers.isMobileNotificationDevice({{ userAgent: "Macintosh", maxTouchPoints: 5 }}),
          desktopClass: helpers.notificationDeviceClass({{ userAgent: "X11 Linux", maxTouchPoints: 0 }}),
          mobileClass: helpers.notificationDeviceClass({{ userAgent: "iPhone" }}),
        }}));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


class TestVoicePlaybackSource(unittest.TestCase):
    def test_voice_helpers_runtime_behavior(self) -> None:
        result = eval_voice_helpers()
        self.assertTrue(result["frozen"])
        self.assertTrue(result["nativeProbably"])
        self.assertTrue(result["nativeMaybe"])
        self.assertFalse(result["nativeNo"])
        self.assertFalse(result["nativeMissingAudio"])
        self.assertTrue(result["mseYes"])
        self.assertFalse(result["mseNo"])
        self.assertTrue(result["preferVendorApple"])
        self.assertTrue(result["preferWebKitNotChrome"])
        self.assertFalse(result["preferChromeFalse"])
        self.assertTrue(result["livePlaybackViaMse"])
        self.assertEqual(result["decodedText"], "Hello-world")
        self.assertTrue(result["androidMobile"])
        self.assertTrue(result["ipadDesktopUa"])
        self.assertEqual(result["desktopClass"], "desktop")
        self.assertEqual(result["mobileClass"], "mobile")

    def test_voice_helpers_load_before_app(self) -> None:
        source = INDEX_HTML.read_text(encoding="utf-8")
        self.assertIn('app_voice_helpers.js?v=__CODOXEAR_ASSET_VERSION__', source)
        self.assertIn('app_voice.js?v=__CODOXEAR_ASSET_VERSION__', source)
        self.assertLess(source.index('app_clipboard.js?v=__CODOXEAR_ASSET_VERSION__'), source.index('app_voice_helpers.js?v=__CODOXEAR_ASSET_VERSION__'))
        self.assertLess(source.index('app_voice_helpers.js?v=__CODOXEAR_ASSET_VERSION__'), source.index('app_voice.js?v=__CODOXEAR_ASSET_VERSION__'))
        self.assertLess(source.index('app_voice.js?v=__CODOXEAR_ASSET_VERSION__'), source.index('app.js?v=__CODOXEAR_ASSET_VERSION__'))

    def test_live_audio_support_is_checked_before_play(self) -> None:
        source = APP_VOICE_JS.read_text(encoding="utf-8")
        app_source = APP_JS.read_text(encoding="utf-8")
        helper_source = APP_VOICE_HELPERS_JS.read_text(encoding="utf-8")
        # app.js delegates voice to the CodoxearVoice controller and validates
        # both the helper module and the controller module fail-loud at load.
        self.assertIn("const codoxearVoice = window.CodoxearVoice;", app_source)
        self.assertIn('throw new Error("Codoxear voice controller failed to load")', app_source)
        self.assertIn("const codoxearVoiceHelpers = window.CodoxearVoiceHelpers;", source)
        self.assertIn('throw new Error("Codoxear voice helpers failed to load")', source)
        self.assertIn('typeof codoxearVoiceHelpers.browserSupportsNativeLiveAudioPlayback !== "function"', source)
        self.assertIn('typeof codoxearVoiceHelpers.browserSupportsMseLiveAudioPlayback !== "function"', source)
        self.assertIn('typeof codoxearVoiceHelpers.shouldPreferNativeLiveAudioPlayback !== "function"', source)
        self.assertIn('typeof codoxearVoiceHelpers.browserSupportsLiveAudioPlayback !== "function"', source)
        self.assertIn('typeof codoxearVoiceHelpers.base64UrlToUint8Array !== "function"', source)
        self.assertIn('typeof codoxearVoiceHelpers.isMobileNotificationDevice !== "function"', source)
        self.assertIn('typeof codoxearVoiceHelpers.notificationDeviceClass !== "function"', source)
        self.assertIn("browserSupportsMseLiveAudioPlayback(windowTarget)", source)
        self.assertIn("shouldPreferNativeLiveAudioPlayback(liveAudio, navigatorTarget)", source)
        self.assertIn("browserSupportsLiveAudioPlayback(liveAudio, windowTarget)", source)
        self.assertIn("base64UrlToUint8Array(publicKey, atob)", source)
        self.assertIn('"application/vnd.apple.mpegurl"', helper_source)
        self.assertIn('"audio/mpegurl"', helper_source)
        self.assertIn("liveAudio.canPlayType(kind)", helper_source)
        self.assertIn("function browserSupportsMseLiveAudioPlayback(windowLike)", helper_source)
        self.assertIn("const HlsCtor = windowLike && windowLike.Hls;", helper_source)
        self.assertIn("HlsCtor.isSupported()", helper_source)
        self.assertIn("function shouldPreferNativeLiveAudioPlayback(liveAudio, navigatorLike)", helper_source)
        self.assertIn("navigatorLike && navigatorLike.vendor", helper_source)
        self.assertIn("navigatorLike && navigatorLike.userAgent", helper_source)
        self.assertIn("function base64UrlToUint8Array(value, atobFunc)", helper_source)
        # The stale pad-based base64 decode stays out of app.js now that the
        # voice wrapper helpers moved entirely into the controller module.
        self.assertNotIn("const pad = \"=\".repeat((4 - (raw.length % 4 || 4)) % 4);", app_source)
        start = source.index("async function ensureLiveAudioPlaybackSource(nextSrc, { resetSource = false } = {})")
        end = source.index("async function sendAnnouncementHeartbeat", start)
        block = source[start:end]
        self.assertIn("function ensureLiveAudioPlaybackSource(nextSrc, { resetSource = false } = {})", block)
        self.assertIn("const hls = new HlsCtor();", block)
        self.assertIn("hls.attachMedia(liveAudio);", block)
        self.assertIn("hls.loadSource(nextSrc);", block)
        self.assertIn("destroyLiveAudioHls();", block)
        self.assertIn("if (shouldPreferNativeLiveAudioPlayback(liveAudio, navigatorTarget))", block)
        self.assertIn("function liveAudioHasReadySegments()", source)

        play_start = source.index("async function startLiveAudioPlayback({ resetSource = false } = {}) {")
        play_end = source.index("function describeLiveAudioStartError(error) {", play_start)
        play_block = source[play_start:play_end]
        self.assertIn("if (!browserSupportsLiveAudioPlayback(liveAudio, windowTarget))", play_block)
        self.assertIn("if (!liveAudioHasReadySegments())", play_block)
        self.assertIn("this browser does not support HLS audio playback in this app", play_block)
        self.assertIn("wait for the first announcement and try again", play_block)
        self.assertIn("await ensureLiveAudioPlaybackSource(nextSrc, { resetSource });", play_block)

        err_start = source.index("function describeLiveAudioStartError(error) {")
        err_end = source.index("function showVoiceSettingsDialog() {", err_start)
        err_block = source[err_start:err_end]
        self.assertIn("if (/unsupported/i.test(message))", err_block)
        self.assertIn("this browser does not support HLS audio playback in this app", err_block)
        self.assertIn("no live audio segments are available yet", err_block)

    def test_live_audio_watchdog_detects_non_progress_and_hard_resets(self) -> None:
        source = APP_VOICE_JS.read_text(encoding="utf-8")
        app_source = APP_JS.read_text(encoding="utf-8")
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
        self.assertIn('addEvent(liveAudio, "timeupdate"', source)
        self.assertIn('addEvent(liveAudio, "waiting"', source)
        self.assertIn('addEvent(liveAudio, "stalled"', source)
        self.assertIn('addEvent(liveAudio, "suspend"', source)
        self.assertIn('addAppEvent(document, "visibilitychange"', app_source)
        self.assertIn('addAppEvent(window, "pageshow"', app_source)
        self.assertIn('addAppEvent(window, "online"', app_source)
        self.assertIn('addAppEvent(window, "focus"', app_source)
        self.assertIn('resumeAnnouncementRuntime({ resetSource: false });', app_source)
        self.assertIn('resumeAnnouncementRuntime({ resetSource: true });', app_source)

    def test_voice_controls_attempt_to_arm_live_audio_from_user_gesture(self) -> None:
        source = APP_VOICE_JS.read_text(encoding="utf-8")
        self.assertIn("async function maybeAutoStartLiveAudioFromGesture({ resetSource = false } = {})", source)
        self.assertIn("await maybeAutoStartLiveAudioFromGesture({ resetSource: true });", source)
        self.assertIn("announceBtn auto-start failed", source)

    def test_notification_transport_is_split_between_desktop_and_mobile(self) -> None:
        source = APP_VOICE_JS.read_text(encoding="utf-8")
        self.assertIn("desktop_supported", source)
        self.assertIn("push_supported", source)
        helper_source = APP_VOICE_HELPERS_JS.read_text(encoding="utf-8")
        self.assertIn("function isMobileNotificationDevice(navigatorLike)", helper_source)
        self.assertIn("function notificationDeviceClass(navigatorLike)", helper_source)
        self.assertIn("notificationDeviceClass(navigatorTarget)", source)
        self.assertIn("function pushNotificationsEnabledForCurrentDevice()", source)
        self.assertIn('if (deviceNotificationClass() === "mobile") {', source)
        self.assertIn('return pushNotificationsEnabledForCurrentDevice() ? "push" : "none";', source)
        self.assertIn('return activeNotificationTransport() === "desktop";', source)
        self.assertIn('notification error: ${err && err.message ? err.message : "unknown error"}', source)
        self.assertIn("notifications require HTTPS or localhost", source)
        self.assertIn("mobile notifications require web push in an installed HTTPS web app", source)
        self.assertIn('if (deviceNotificationClass() === "desktop") {', source)
        self.assertIn('device_class: deviceNotificationClass()', source)
        # The dead per-message desktop notification resolver path had no
        # runtime caller and was removed during the voice module extraction;
        # the live desktop/mobile transport split is exercised behaviorally in
        # test_frontend_voice_module_source.
        self.assertNotIn("maybeShowDesktopNotification", source)
        self.assertNotIn("scheduleDesktopNotificationResolve", source)
        self.assertNotIn("/api/notifications/message?message_id=", source)
        self.assertIn("async function pollNotificationFeed({ prime = false } = {})", source)
        self.assertIn("/api/notifications/feed?since=", source)
        self.assertIn("if (isAppDisposed() || !desktopNotificationsEnabled()) return;", source)


if __name__ == "__main__":
    unittest.main()
