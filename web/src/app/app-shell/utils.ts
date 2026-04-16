import type { VoiceSettingsResponse } from "../../lib/types";

export const DEFAULT_VOICE_SETTINGS: VoiceSettingsResponse = {
  tts_enabled_for_narration: false,
  tts_enabled_for_final_response: true,
  tts_base_url: "",
  tts_api_key: "",
  audio: {
    active_listener_count: 0,
    queue_depth: 0,
    segment_count: 0,
    stream_url: "/api/audio/live.m3u8",
  },
  notifications: {
    enabled_devices: 0,
    total_devices: 0,
    vapid_public_key: "",
  },
};

export function shortSessionId(sessionId: string) {
  const match = sessionId.match(/^([0-9a-f]{8})[0-9a-f-]{20,}$/i);
  return match ? match[1] : sessionId.slice(0, 8);
}

export function readLocalToggle(key: string) {
  if (typeof window === "undefined") return false;
  return window.localStorage.getItem(key) === "1";
}

export function writeLocalToggle(key: string, value: boolean) {
  if (typeof window === "undefined") return;
  if (value) {
    window.localStorage.setItem(key, "1");
  } else {
    window.localStorage.removeItem(key);
  }
}

export function readLocalToggleDefaultOn(key: string) {
  if (typeof window === "undefined") return true;
  return window.localStorage.getItem(key) !== "0";
}

export function getAnnouncementClientId() {
  if (typeof window === "undefined") return "announcement-client";
  const key = "codoxear.announcementClientId";
  const current = window.localStorage.getItem(key);
  if (current) return current;
  const next = typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
    ? crypto.randomUUID()
    : `announcement-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  window.localStorage.setItem(key, next);
  return next;
}

export function base64UrlToUint8Array(value: string) {
  const raw = String(value || "");
  const pad = "=".repeat((4 - (raw.length % 4 || 4)) % 4);
  const base64 = (raw + pad).replace(/-/g, "+").replace(/_/g, "/");
  const data = atob(base64);
  const out = new Uint8Array(data.length);
  for (let index = 0; index < data.length; index += 1) {
    out[index] = data.charCodeAt(index);
  }
  return out;
}

export function isStalePushSubscriptionEndpoint(endpoint: string | null | undefined) {
  const raw = typeof endpoint === "string" ? endpoint.trim() : "";
  if (!raw) return false;
  try {
    return new URL(raw).hostname.endsWith(".invalid");
  } catch {
    return false;
  }
}

export function isMobileNotificationDevice() {
  if (typeof navigator === "undefined") return false;
  const ua = navigator.userAgent || "";
  if (/Android|iPhone|iPad|iPod|Mobile/i.test(ua)) return true;
  if (/Macintosh/i.test(ua) && Number(navigator.maxTouchPoints || 0) > 1) return true;
  return false;
}

export function notificationDeviceClass() {
  return isMobileNotificationDevice() ? "mobile" : "desktop";
}

export function shouldUseMobileWorkspaceSheet() {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") return false;
  return window.matchMedia("(max-width: 880px)").matches;
}

export function shouldPreferNativeHlsPlayback() {
  if (typeof navigator === "undefined") return false;
  const ua = navigator.userAgent || "";
  const vendor = navigator.vendor || "";
  const isAppleVendor = /Apple/i.test(vendor);
  const isSafariEngine = /Safari/i.test(ua) && !/Chrom(e|ium)|Edg|OPR|CriOS|FxiOS/i.test(ua);
  return isAppleVendor || isSafariEngine;
}

export function mergeVoiceSettings(value: VoiceSettingsResponse | null | undefined): VoiceSettingsResponse {
  return {
    ...DEFAULT_VOICE_SETTINGS,
    ...(value || {}),
    audio: {
      ...DEFAULT_VOICE_SETTINGS.audio,
      ...((value && value.audio) || {}),
    },
    notifications: {
      ...DEFAULT_VOICE_SETTINGS.notifications,
      ...((value && value.notifications) || {}),
    },
  };
}

export function replySoundTextKey(sessionId: string, row: Record<string, unknown>) {
  const sessionKey = String(sessionId || "").trim();
  if (!sessionKey) return "";
  const text = typeof row.notification_text === "string"
    ? row.notification_text
    : typeof row.text === "string"
      ? row.text
      : "";
  const normalizedText = text.replace(/\s+/g, " ").trim();
  return normalizedText ? `session:${sessionKey}:text:${normalizedText}` : "";
}
