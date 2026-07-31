(function () {
  "use strict";

  function browserSupportsNativeLiveAudioPlayback(liveAudio) {
    if (!liveAudio || typeof liveAudio.canPlayType !== "function") return false;
    return ["application/vnd.apple.mpegurl", "audio/mpegurl"].some((kind) => {
      const result = liveAudio.canPlayType(kind);
      return result === "probably" || result === "maybe";
    });
  }

  function browserSupportsMseLiveAudioPlayback(windowLike) {
    const HlsCtor = windowLike && windowLike.Hls;
    return !!(HlsCtor && typeof HlsCtor.isSupported === "function" && HlsCtor.isSupported());
  }

  function shouldPreferNativeLiveAudioPlayback(liveAudio, navigatorLike) {
    if (!browserSupportsNativeLiveAudioPlayback(liveAudio)) return false;
    const vendor = String((navigatorLike && navigatorLike.vendor) || "");
    const ua = String((navigatorLike && navigatorLike.userAgent) || "");
    if (/Apple/i.test(vendor)) return true;
    return /AppleWebKit/i.test(ua) && !/(?:Chrom(?:e|ium)|CriOS|Edg|OPR|Firefox|FxiOS)/i.test(ua);
  }

  function browserSupportsLiveAudioPlayback(liveAudio, windowLike) {
    return browserSupportsNativeLiveAudioPlayback(liveAudio) || browserSupportsMseLiveAudioPlayback(windowLike);
  }

  function base64UrlToUint8Array(value, atobFunc) {
    const raw = String(value || "");
    const pad = "=".repeat((4 - (raw.length % 4 || 4)) % 4);
    const base64 = (raw + pad).replace(/-/g, "+").replace(/_/g, "/");
    const data = atobFunc(base64);
    const out = new Uint8Array(data.length);
    for (let i = 0; i < data.length; i += 1) out[i] = data.charCodeAt(i);
    return out;
  }

  function isMobileNotificationDevice(navigatorLike) {
    const ua = (navigatorLike && navigatorLike.userAgent) || "";
    if (/Android|iPhone|iPad|iPod|Mobile/i.test(ua)) return true;
    if (/Macintosh/i.test(ua) && Number((navigatorLike && navigatorLike.maxTouchPoints) || 0) > 1) return true;
    return false;
  }

  function notificationDeviceClass(navigatorLike) {
    return isMobileNotificationDevice(navigatorLike) ? "mobile" : "desktop";
  }

  window.CodoxearVoiceHelpers = Object.freeze({
    browserSupportsNativeLiveAudioPlayback,
    browserSupportsMseLiveAudioPlayback,
    shouldPreferNativeLiveAudioPlayback,
    browserSupportsLiveAudioPlayback,
    base64UrlToUint8Array,
    isMobileNotificationDevice,
    notificationDeviceClass,
  });
})();
