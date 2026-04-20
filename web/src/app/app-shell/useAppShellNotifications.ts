import { useCallback, useEffect, useRef, useState } from "preact/hooks";
import { api } from "../../lib/api";
import type { NotificationSubscriptionStateResponse, VoiceSettingsResponse } from "../../lib/types";
import { toPublicAssetUrl } from "../../lib/publicAssetUrl";
import {
  base64UrlToUint8Array,
  isStalePushSubscriptionEndpoint,
  notificationDeviceClass,
  readLocalToggle,
  readLocalToggleDefaultOn,
  replySoundTextKey,
  writeLocalToggle,
} from "./utils";

const NOTIFICATION_MESSAGE_RETRY_MS = 15000;
const FINAL_NOTIFICATION_SUMMARY_STATUSES = new Set(["sent", "skipped", "error"]);
const REPLY_SOUND_TEXT_DEDUPE_MS = 30000;
const NOTIFICATION_FEED_POLL_MS = 5000;
const SSE_NOTIFICATION_FALLBACK_MS = 60000;

function isDocumentVisible() {
  if (typeof document === "undefined") {
    return true;
  }
  return document.visibilityState !== "hidden";
}

type NotificationMessageLookupState = {
  retryAfter: number;
  terminal: boolean;
};

interface UseAppShellNotificationsOptions {
  activeSessionId: string | null;
  activeTitle: string;
  bySessionId: Record<string, unknown[]>;
  playReplyBeep(): void;
  realtimeConnected?: boolean;
  suppressedReplySoundSessionIdsRef: { current: Set<string> };
  voiceSettings: VoiceSettingsResponse;
}

export function useAppShellNotifications({
  activeSessionId,
  activeTitle,
  bySessionId,
  playReplyBeep,
  realtimeConnected = false,
  suppressedReplySoundSessionIdsRef,
  voiceSettings,
}: UseAppShellNotificationsOptions) {
  const [notificationsEnabled, setNotificationsEnabled] = useState(() => readLocalToggle("codoxear.notificationEnabled"));
  const [replySoundEnabled, setReplySoundEnabled] = useState(() => readLocalToggleDefaultOn("codoxear.replySoundEnabled"));
  const [notificationPermission, setNotificationPermission] = useState(() => (
    typeof Notification === "undefined" ? "unsupported" : Notification.permission
  ));
  const [pushNotificationsEnabled, setPushNotificationsEnabled] = useState(false);
  const [pageVisible, setPageVisible] = useState(isDocumentVisible);

  const playReplyBeepRef = useRef(playReplyBeep);
  const notificationFeedCursorRef = useRef(Date.now() / 1000);
  const deliveredNotificationIdsRef = useRef(new Set<string>());
  const resolvingNotificationIdsRef = useRef(new Set<string>());
  const notificationLookupStateRef = useRef(new Map<string, NotificationMessageLookupState>());
  const notificationEndpointRef = useRef("");
  const seenFinalResponseKeysRef = useRef(new Set<string>());
  const playedReplySoundKeysRef = useRef(new Set<string>());
  const playedReplySoundTextKeysRef = useRef(new Map<string, number>());
  const finalResponseBeepPrimedRef = useRef(false);

  useEffect(() => {
    playReplyBeepRef.current = playReplyBeep;
  }, [playReplyBeep]);

  useEffect(() => {
    const handleVisibilityChange = () => {
      setPageVisible(isDocumentVisible());
    };
    document.addEventListener("visibilitychange", handleVisibilityChange);
    return () => document.removeEventListener("visibilitychange", handleVisibilityChange);
  }, []);

  useEffect(() => {
    writeLocalToggle("codoxear.notificationEnabled", notificationsEnabled);
  }, [notificationsEnabled]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem("codoxear.replySoundEnabled", replySoundEnabled ? "1" : "0");
  }, [replySoundEnabled]);

  const ensureVoiceServiceWorker = async () => {
    if (!("serviceWorker" in navigator)) {
      throw new Error("service workers are not available");
    }
    return navigator.serviceWorker.register(toPublicAssetUrl("service-worker.js"));
  };

  const syncNotificationSubscriptionState = async (
    snapshot?: NotificationSubscriptionStateResponse | null,
    endpointOverride?: string,
  ) => {
    if (notificationDeviceClass() !== "mobile" || !("serviceWorker" in navigator) || typeof PushManager === "undefined") {
      notificationEndpointRef.current = "";
      setPushNotificationsEnabled(false);
      return;
    }
    let endpoint = String(endpointOverride || "").trim();
    if (!endpoint) {
      const registration = await ensureVoiceServiceWorker();
      const subscription = await registration.pushManager.getSubscription();
      endpoint = subscription && typeof subscription.endpoint === "string" ? subscription.endpoint : "";
    }
    notificationEndpointRef.current = endpoint;
    const state = snapshot ?? await api.getNotificationSubscriptionState();
    const current = endpoint ? state.subscriptions.find((item) => item && item.endpoint === endpoint) : null;
    setPushNotificationsEnabled(Boolean(current && current.notifications_enabled));
  };

  useEffect(() => {
    if (notificationDeviceClass() !== "mobile") return;
    syncNotificationSubscriptionState().catch(() => undefined);
  }, [voiceSettings.notifications?.vapid_public_key]);

  const prunePlayedReplySoundTextKeys = (nowTs: number) => {
    for (const [key, ts] of playedReplySoundTextKeysRef.current.entries()) {
      if ((nowTs - ts) > REPLY_SOUND_TEXT_DEDUPE_MS) {
        playedReplySoundTextKeysRef.current.delete(key);
      }
    }
  };

  const rememberPlayedReplySound = (sessionId: string, row: Record<string, unknown>) => {
    const messageId = typeof row.message_id === "string" ? row.message_id.trim() : "";
    if (messageId) {
      playedReplySoundKeysRef.current.add(`id:${messageId}`);
    }
    const nowTs = Date.now();
    prunePlayedReplySoundTextKeys(nowTs);
    const textKey = replySoundTextKey(sessionId, row);
    if (textKey) {
      playedReplySoundTextKeysRef.current.set(textKey, nowTs);
    }
  };

  const hasPlayedReplySound = (sessionId: string, row: Record<string, unknown>, key: string) => {
    if (playedReplySoundKeysRef.current.has(key)) {
      return true;
    }
    const nowTs = Date.now();
    prunePlayedReplySoundTextKeys(nowTs);
    const textKey = replySoundTextKey(sessionId, row);
    return textKey ? playedReplySoundTextKeysRef.current.has(textKey) : false;
  };

  const finalResponseEventKey = (row: Record<string, unknown>) => {
    const messageId = typeof row.message_id === "string" ? row.message_id.trim() : "";
    if (messageId) return `id:${messageId}`;
    const ts = typeof row.ts === "number" ? row.ts : 0;
    const text = typeof row.notification_text === "string"
      ? row.notification_text
      : typeof row.text === "string"
        ? row.text
        : "";
    const normalizedText = text.replace(/\s+/g, " ").trim();
    return normalizedText ? `text:${ts}:${normalizedText}` : "";
  };

  const showDesktopNotification = (title: string, body: string, messageId?: string) => {
    if (notificationDeviceClass() !== "desktop" || notificationPermission !== "granted" || typeof Notification === "undefined") {
      return;
    }
    const trimmedBody = body.replace(/\s+/g, " ").trim();
    if (!trimmedBody) return;
    const id = String(messageId || "").trim();
    if (id && deliveredNotificationIdsRef.current.has(id)) return;
    new Notification(title.trim() || "Session", {
      body: trimmedBody.length <= 180 ? trimmedBody : `${trimmedBody.slice(0, 179).trimEnd()}...`,
      tag: id || `desktop:${Date.now()}`,
    });
    if (id) deliveredNotificationIdsRef.current.add(id);
  };

  const refreshNotificationFeed = useCallback(async (prime = false) => {
    const desktopNotificationsEnabled = (
      notificationsEnabled
      && notificationPermission === "granted"
      && typeof Notification !== "undefined"
      && notificationDeviceClass() === "desktop"
    );
    if (!pageVisible || (!replySoundEnabled && !desktopNotificationsEnabled)) {
      return;
    }
    const response = await api.getNotificationsFeed(notificationFeedCursorRef.current);
    let maxSeen = notificationFeedCursorRef.current;
    for (const item of response.items || []) {
      const updatedTs = Number(item.updated_ts || 0);
      if (updatedTs > maxSeen) maxSeen = updatedTs;
      if (prime) continue;
      const messageId = typeof item.message_id === "string" ? item.message_id.trim() : "";
      const replySoundKey = messageId ? `id:${messageId}` : "";
      const replySoundSessionId = typeof (item as any).session_id === "string"
        ? String((item as any).session_id)
        : String(item.session_display_name || "");
      const replySoundRow = {
        message_id: messageId,
        notification_text: item.notification_text,
      } satisfies Record<string, unknown>;
      if (replySoundEnabled && replySoundKey && !hasPlayedReplySound(replySoundSessionId, replySoundRow, replySoundKey)) {
        playReplyBeepRef.current();
        rememberPlayedReplySound(replySoundSessionId, replySoundRow);
      }
      if (desktopNotificationsEnabled) {
        showDesktopNotification(
          String(item.session_display_name || "Session"),
          String(item.notification_text || ""),
          item.message_id,
        );
      }
    }
    notificationFeedCursorRef.current = maxSeen;
  }, [notificationPermission, notificationsEnabled, pageVisible, replySoundEnabled]);

  useEffect(() => {
    notificationFeedCursorRef.current = Date.now() / 1000;
    void refreshNotificationFeed(true);
    const intervalId = window.setInterval(() => {
      void refreshNotificationFeed(false);
    }, realtimeConnected ? SSE_NOTIFICATION_FALLBACK_MS : NOTIFICATION_FEED_POLL_MS);
    return () => {
      window.clearInterval(intervalId);
    };
  }, [realtimeConnected, refreshNotificationFeed]);

  useEffect(() => {
    const nextSeen = new Set<string>();
    for (const sessionId of Object.keys(bySessionId)) {
      const events = Array.isArray(bySessionId[sessionId]) ? bySessionId[sessionId] : [];
      const suppressReplySound = suppressedReplySoundSessionIdsRef.current.has(sessionId);
      for (const event of events) {
        if (!event || typeof event !== "object") continue;
        const row = event as Record<string, unknown>;
        if (row.role !== "assistant" || row.pending === true || row.message_class !== "final_response") continue;
        const key = finalResponseEventKey(row);
        if (!key) continue;
        nextSeen.add(key);
        if (
          !suppressReplySound
          && finalResponseBeepPrimedRef.current
          && replySoundEnabled
          && !seenFinalResponseKeysRef.current.has(key)
          && !hasPlayedReplySound(sessionId, row, key)
        ) {
          playReplyBeepRef.current();
          rememberPlayedReplySound(sessionId, row);
        }
      }
    }
    seenFinalResponseKeysRef.current = nextSeen;
    finalResponseBeepPrimedRef.current = true;
  }, [bySessionId, replySoundEnabled, suppressedReplySoundSessionIdsRef]);

  useEffect(() => {
    if (
      notificationDeviceClass() !== "desktop"
      || !notificationsEnabled
      || notificationPermission !== "granted"
      || !activeSessionId
    ) {
      return;
    }

    const events = Array.isArray(bySessionId[activeSessionId]) ? bySessionId[activeSessionId] : [];
    for (const event of events) {
      if (!event || typeof event !== "object") continue;
      const row = event as Record<string, unknown>;
      if (row.role !== "assistant") continue;
      if (row.pending === true) continue;
      if (row.message_class !== "final_response") continue;
      const messageId = typeof row.message_id === "string" ? row.message_id : "";
      const notificationText = typeof row.notification_text === "string"
        ? row.notification_text
        : typeof row.text === "string"
          ? row.text
          : "";
      if (notificationText) {
        showDesktopNotification(activeTitle, notificationText, messageId);
        continue;
      }
      if (!messageId) {
        continue;
      }
      const lookupState = notificationLookupStateRef.current.get(messageId);
      if (lookupState?.terminal || (lookupState && lookupState.retryAfter > Date.now()) || resolvingNotificationIdsRef.current.has(messageId)) {
        continue;
      }
      resolvingNotificationIdsRef.current.add(messageId);
      api.getNotificationMessage(messageId)
        .then((response) => {
          const text = String(response.notification_text || "").trim();
          const status = String(response.summary_status || "").trim();
          if (text && (!status || FINAL_NOTIFICATION_SUMMARY_STATUSES.has(status))) {
            notificationLookupStateRef.current.set(messageId, {
              retryAfter: Number.POSITIVE_INFINITY,
              terminal: true,
            });
            showDesktopNotification(activeTitle, text, messageId);
            return;
          }

          notificationLookupStateRef.current.set(messageId, {
            retryAfter: status && FINAL_NOTIFICATION_SUMMARY_STATUSES.has(status)
              ? Number.POSITIVE_INFINITY
              : Date.now() + NOTIFICATION_MESSAGE_RETRY_MS,
            terminal: Boolean(status && FINAL_NOTIFICATION_SUMMARY_STATUSES.has(status)),
          });
        })
        .catch(() => undefined)
        .finally(() => {
          resolvingNotificationIdsRef.current.delete(messageId);
        });
    }
  }, [activeSessionId, activeTitle, bySessionId, notificationPermission, notificationsEnabled]);

  const notificationLabel = notificationsEnabled
    ? notificationDeviceClass() === "mobile"
      ? pushNotificationsEnabled
        ? "Notifications on (push)"
        : "Notifications pending"
      : notificationPermission === "granted" || notificationPermission === "unsupported"
        ? "Notifications on"
        : "Notifications pending"
    : "Notifications off";

  const toggleNotifications = async () => {
    const next = !notificationsEnabled;
    if (!next) {
      if (notificationDeviceClass() === "mobile" && notificationEndpointRef.current) {
        const snapshot = await api.toggleNotificationSubscription(notificationEndpointRef.current, false);
        await syncNotificationSubscriptionState(snapshot);
      }
      setNotificationsEnabled(false);
      return;
    }
    if (typeof Notification !== "undefined" && Notification.permission !== "granted") {
      const permission = await Notification.requestPermission();
      setNotificationPermission(permission);
      if (permission !== "granted") {
        setNotificationsEnabled(false);
        return;
      }
    }
    if (typeof Notification !== "undefined" && Notification.permission === "granted") {
      setNotificationPermission("granted");
    }
    if (notificationDeviceClass() === "mobile") {
      if (typeof PushManager === "undefined") {
        setNotificationsEnabled(false);
        return;
      }
      const registration = await ensureVoiceServiceWorker();
      const subscriptionState = await api.getNotificationSubscriptionState();
      let subscription = await registration.pushManager.getSubscription();
      const subscriptionEndpoint = typeof subscription?.endpoint === "string" ? subscription.endpoint : "";
      const currentSubscription = subscriptionEndpoint
        ? subscriptionState.subscriptions.find((item) => item && item.endpoint === subscriptionEndpoint)
        : null;
      if (isStalePushSubscriptionEndpoint(subscription?.endpoint) || (subscription && !currentSubscription)) {
        try {
          await subscription?.unsubscribe?.();
        } catch {
          // Ignore stale subscription cleanup failures and continue with a fresh subscribe.
        }
        subscription = null;
      }
      const publicKey = String(voiceSettings.notifications?.vapid_public_key || "").trim();
      if (!subscription) {
        if (!publicKey) {
          setNotificationsEnabled(false);
          return;
        }
        subscription = await registration.pushManager.subscribe({
          userVisibleOnly: true,
          applicationServerKey: base64UrlToUint8Array(publicKey),
        });
      }
      const snapshot = await api.upsertNotificationSubscription({
        subscription: subscription.toJSON(),
        user_agent: navigator.userAgent,
        device_label: "current-device",
        device_class: notificationDeviceClass(),
      });
      const endpoint = typeof subscription.endpoint === "string" ? subscription.endpoint : "";
      await syncNotificationSubscriptionState(snapshot, endpoint);
    }
    setNotificationsEnabled(true);
  };

  return {
    notificationLabel,
    notificationsEnabled,
    pushNotificationsEnabled,
    refreshNotificationFeed: () => refreshNotificationFeed(false),
    replySoundEnabled,
    setReplySoundEnabled,
    toggleNotifications,
  };
}
