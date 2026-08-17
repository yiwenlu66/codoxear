import * as CodoxearVoiceHelpers from "./app_voice_helpers.js";

const base64UrlToUint8Array = CodoxearVoiceHelpers.base64UrlToUint8Array;
const notificationDeviceClass = CodoxearVoiceHelpers.notificationDeviceClass;
const NOTIFICATION_PANEL_MAX_ITEMS = 100;

function requireFunction(value, name) {
  if (typeof value !== "function") throw new TypeError(`notification runtime dependency missing: ${name}`);
  return value;
}

function requireNode(value, name) {
  if (!value || typeof value !== "object" || !value.style) {
    throw new TypeError(`notification runtime dependency missing: ${name}`);
  }
  return value;
}

function createNotificationDom(root, elValue, iconSvgValue, voiceHostValue) {
  const el = requireFunction(elValue, "el");
  const iconSvg = requireFunction(iconSvgValue, "iconSvg");
  const voiceHost = requireNode(voiceHostValue, "voiceHost");
  if (!root || typeof root.appendChild !== "function") throw new TypeError("notification DOM dependency missing: root");

  const notificationBtn = el("button", {
    id: "notificationBtn",
    class: "icon-btn",
    title: "Notifications",
    "aria-label": "Notifications",
    "aria-controls": "notificationPanel",
    "aria-expanded": "false",
    type: "button",
    html: iconSvg("bell"),
  });
  const notificationPanel = el("section", {
    id: "notificationPanel",
    class: "notificationPanel",
    role: "dialog",
    "aria-label": "Notifications",
  });
  const notificationClearBtn = el("button", {
    id: "notificationClearBtn",
    class: "text-btn",
    type: "button",
    text: "Mark read",
  });
  const notificationPanelHeader = el("div", { class: "notificationPanelHeader" }, [
    el("div", { class: "title", text: "Notifications" }),
    notificationClearBtn,
  ]);
  const notificationEnableBtn = el("button", {
    id: "notificationEnableBtn",
    class: "text-btn",
    type: "button",
    text: "Enable browser alerts",
  });
  const notificationEmpty = el("div", {
    id: "notificationEmpty",
    class: "muted",
    text: "No notifications",
  });
  const notificationList = el("div", {
    id: "notificationList",
    class: "notificationList",
    role: "list",
  });
  notificationPanel.append(notificationPanelHeader, notificationEnableBtn, notificationEmpty, notificationList);
  notificationPanel.style.display = "none";
  if (typeof voiceHost.insertBefore === "function") voiceHost.insertBefore(notificationBtn, voiceHost.firstChild || null);
  else voiceHost.appendChild(notificationBtn);
  root.appendChild(notificationPanel);

  return Object.freeze({
    notificationBtn,
    notificationPanel,
    notificationList,
    notificationEmpty,
    notificationClearBtn,
    notificationEnableBtn,
  });
}

function createNotificationRuntime(options = {}) {
  if (!options || typeof options !== "object") throw new TypeError("notification runtime dependency missing: options");

  const {
    notificationBtn,
    notificationPanel,
    notificationList,
    notificationEmpty,
    notificationClearBtn,
    notificationEnableBtn,
  } = createNotificationDom(options.root, options.el, options.iconSvg, options.voiceHost);
  const isAppDisposed = requireFunction(options.isAppDisposed, "isAppDisposed");
  const api = requireFunction(options.api, "api");
  const setToast = requireFunction(options.setToast, "setToast");
  const handleAppAuthLoss = requireFunction(options.handleAppAuthLoss, "handleAppAuthLoss");
  const resolveAppUrl = requireFunction(options.resolveAppUrl, "resolveAppUrl");
  const versionedShellAssetPath = requireFunction(options.versionedShellAssetPath, "versionedShellAssetPath");
  const storageGetItem = requireFunction(options.storageGetItem, "storageGetItem");
  const storageSetItem = requireFunction(options.storageSetItem, "storageSetItem");
  const storageRemoveItem = requireFunction(options.storageRemoveItem, "storageRemoveItem");
  const eventBindings = options.eventBindings;
  if (!eventBindings || typeof eventBindings.on !== "function") {
    throw new TypeError("notification runtime dependency missing: eventBindings");
  }
  const focusSessionFromNotification = typeof options.focusSessionFromNotification === "function"
    ? options.focusSessionFromNotification
    : null;

  const windowTarget = options.windowTarget || window;
  const navigatorTarget = options.navigatorTarget || (typeof navigator !== "undefined" ? navigator : null);
  const documentTarget = options.documentTarget || document;
  const NotificationCtor = typeof options.Notification !== "undefined"
    ? options.Notification
    : (typeof Notification !== "undefined" ? Notification : undefined);
  const AudioContextCtor = typeof options.AudioContext !== "undefined"
    ? options.AudioContext
    : (windowTarget.AudioContext || windowTarget.webkitAudioContext || null);
  const clearTimeoutFn = typeof options.clearTimeout === "function" ? options.clearTimeout : clearTimeout;

  let localNotificationEnabled = storageGetItem("codoxear.notificationEnabled") === "1";
  const desktopNotificationTimers = new Map();
  const deliveredDesktopNotificationIds = new Set();
  let notificationFeedSinceTs = 0;
  const notificationItems = new Map();
  const readNotificationIds = new Set();
  let notificationPanelOpen = false;
  let notificationAudioContext = null;
  let notificationState = {
    desktop_supported: false,
    push_supported: false,
    permission: NotificationCtor ? NotificationCtor.permission : "unsupported",
    desktop_enabled: false,
    endpoint: "",
    notifications_enabled: false,
    subscriptions: [],
    vapid_public_key: "",
  };
  let swRegistration = null;
  let disposed = false;

  function deviceNotificationClass() {
    return notificationDeviceClass(navigatorTarget);
  }

  function enabledLocally() {
    return !!localNotificationEnabled;
  }

  function setNotificationEnabledLocal(enabled) {
    localNotificationEnabled = !!enabled;
    if (localNotificationEnabled) storageSetItem("codoxear.notificationEnabled", "1");
    else storageRemoveItem("codoxear.notificationEnabled");
    render();
  }

  function setDesktopNotificationsEnabled(enabled) {
    if (enabled) storageSetItem("codoxear.desktopNotificationsEnabled", "1");
    else storageRemoveItem("codoxear.desktopNotificationsEnabled");
    notificationState.desktop_enabled = !!enabled;
  }

  function pushNotificationsEnabledForCurrentDevice() {
    return !!(
      localNotificationEnabled &&
      deviceNotificationClass() === "mobile" &&
      notificationState.push_supported &&
      notificationState.permission === "granted" &&
      notificationState.notifications_enabled &&
      notificationState.endpoint
    );
  }

  function activeNotificationTransport() {
    if (!localNotificationEnabled) return "none";
    if (deviceNotificationClass() === "mobile") {
      return pushNotificationsEnabledForCurrentDevice() ? "push" : "none";
    }
    if (
      notificationState.desktop_supported &&
      notificationState.permission === "granted" &&
      notificationState.desktop_enabled
    ) {
      return "desktop";
    }
    return "none";
  }

  function desktopNotificationsEnabled() {
    return activeNotificationTransport() === "desktop";
  }

  function focusOriginSession(sessionId) {
    const sid = String(sessionId || "").trim();
    try {
      if (typeof windowTarget.focus === "function") windowTarget.focus();
    } catch {}
    if (sid && focusSessionFromNotification) focusSessionFromNotification(sid);
  }

  function showDesktopNotification({ messageId, title, body, sessionId }) {
    if (!desktopNotificationsEnabled()) return false;
    const id = String(messageId || "").trim();
    if (id && deliveredDesktopNotificationIds.has(id)) return false;
    const sid = String(sessionId || "").trim();
    const safeTitle = String(title || "Session").trim() || "Session";
    const safeBody = String(body || "").replace(/\s+/g, " ").trim();
    if (!safeBody) return false;
    try {
      const notification = new NotificationCtor(safeTitle, {
        body: safeBody.length <= 180 ? safeBody : `${safeBody.slice(0, 179).trimEnd()}...`,
        tag: id || `desktop:${Date.now()}`,
      });
      if (sid) {
        notification.onclick = (event) => {
          if (event && typeof event.preventDefault === "function") event.preventDefault();
          try {
            if (typeof notification.close === "function") notification.close();
          } catch {}
          focusOriginSession(sid);
        };
      }
      if (id) deliveredDesktopNotificationIds.add(id);
      return true;
    } catch (error) {
      console.error("desktop notification failed", error);
      return false;
    }
  }

  async function primeNotificationSound() {
    if (!AudioContextCtor) return;
    if (!notificationAudioContext) notificationAudioContext = new AudioContextCtor();
    if (notificationAudioContext.state === "suspended" && typeof notificationAudioContext.resume === "function") {
      await notificationAudioContext.resume();
    }
  }

  function playNotificationSound() {
    if (!notificationAudioContext || notificationAudioContext.state === "suspended") return;
    try {
      const oscillator = notificationAudioContext.createOscillator();
      const gain = notificationAudioContext.createGain();
      oscillator.frequency.value = 740;
      gain.gain.setValueAtTime(0.05, notificationAudioContext.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, notificationAudioContext.currentTime + 0.14);
      oscillator.connect(gain);
      gain.connect(notificationAudioContext.destination);
      oscillator.start();
      oscillator.stop(notificationAudioContext.currentTime + 0.14);
    } catch (error) {
      console.error("notification sound failed", error);
    }
  }

  function sortedNotificationItems() {
    return Array.from(notificationItems.values()).sort((left, right) => {
      const byUpdated = Number(right.updated_ts || 0) - Number(left.updated_ts || 0);
      return byUpdated || String(right.message_id || "").localeCompare(String(left.message_id || ""));
    });
  }

  function trimNotificationItems() {
    for (const item of sortedNotificationItems().slice(NOTIFICATION_PANEL_MAX_ITEMS)) {
      notificationItems.delete(item.message_id);
      readNotificationIds.delete(item.message_id);
    }
  }

  function markNotificationRead(messageId) {
    const id = String(messageId || "").trim();
    if (id) readNotificationIds.add(id);
  }

  function markAllNotificationsRead() {
    notificationItems.forEach((_item, messageId) => markNotificationRead(messageId));
  }

  function unreadNotificationCount() {
    let count = 0;
    notificationItems.forEach((_item, messageId) => {
      if (!readNotificationIds.has(messageId)) count += 1;
    });
    return count;
  }

  function render() {
    notificationState.permission = NotificationCtor ? NotificationCtor.permission : "unsupported";
    // The panel widget has one render authority. Unread state owns the active
    // affordance whenever the panel exists; this preserves the former final
    // write from renderNotificationPanel without a competing enablement write.
    const unread = unreadNotificationCount();
    const suffix = unread ? `, ${unread} unread` : "";
    notificationBtn.dataset.unread = String(unread);
    notificationBtn.setAttribute("aria-label", `Notifications${suffix}`);
    notificationBtn.title = `Notifications${suffix}`;
    notificationBtn.classList.toggle("active", unread > 0);
    notificationBtn.setAttribute("aria-expanded", notificationPanelOpen ? "true" : "false");
    notificationPanel.style.display = notificationPanelOpen ? "flex" : "none";
    notificationEmpty.style.display = notificationItems.size ? "none" : "block";
    notificationClearBtn.disabled = unread === 0;
    notificationEnableBtn.textContent = enabledLocally() ? "Disable browser alerts" : "Enable browser alerts";
    notificationList.replaceChildren();
    for (const item of sortedNotificationItems()) {
      const row = documentTarget.createElement("button");
      row.type = "button";
      row.className = "notificationItem";
      row.setAttribute("role", "listitem");
      row.classList.toggle("unread", !readNotificationIds.has(item.message_id));
      const title = documentTarget.createElement("span");
      title.className = "notificationItemTitle";
      title.textContent = String(item.session_display_name || "Session");
      const body = documentTarget.createElement("span");
      body.className = "notificationItemBody";
      body.textContent = String(item.notification_text || "");
      row.append(title, body);
      row.onclick = () => {
        markNotificationRead(item.message_id);
        notificationPanelOpen = false;
        render();
        focusOriginSession(item.session_id);
      };
      notificationList.appendChild(row);
    }
  }

  async function pollFeed({ prime = false } = {}) {
    if (disposed || isAppDisposed()) return;
    let maxSeen = notificationFeedSinceTs;
    try {
      const data = await api(`/api/notifications/feed?since=${encodeURIComponent(notificationFeedSinceTs)}`);
      if (disposed || isAppDisposed()) return;
      const items = Array.isArray(data.items) ? data.items : [];
      for (const item of items) {
        const messageId = String(item && item.message_id ? item.message_id : "").trim();
        if (!messageId) continue;
        const updatedTs = Number(item && item.updated_ts ? item.updated_ts : 0);
        if (updatedTs > maxSeen) maxSeen = updatedTs;
        const alreadyKnown = notificationItems.has(messageId);
        notificationItems.set(messageId, { ...item, message_id: messageId });
        if (prime || notificationPanelOpen) markNotificationRead(messageId);
        if (!prime && !alreadyKnown && desktopNotificationsEnabled()) {
          showDesktopNotification({
            messageId,
            title: item && item.session_display_name,
            body: item && item.notification_text,
            sessionId: item && item.session_id,
          });
          playNotificationSound();
        }
      }
      trimNotificationItems();
    } catch (error) {
      if (error && error.status === 401) {
        handleAppAuthLoss();
        return;
      }
      console.error("notification feed poll failed", error);
      return;
    }
    notificationFeedSinceTs = maxSeen;
    render();
  }

  async function ensureServiceWorker() {
    if (!navigatorTarget || !("serviceWorker" in navigatorTarget) || !("PushManager" in windowTarget) || !NotificationCtor) {
      throw new Error("push notifications are not supported in this browser");
    }
    if (!swRegistration) {
      swRegistration = await navigatorTarget.serviceWorker.register(resolveAppUrl(versionedShellAssetPath("/service-worker.js")), {
        scope: resolveAppUrl("/"),
      });
    }
    return swRegistration;
  }

  async function syncState(snapshot) {
    if (disposed || isAppDisposed()) return;
    notificationState.desktop_supported = !!(windowTarget.isSecureContext && NotificationCtor);
    notificationState.push_supported = !!(
      notificationState.desktop_supported && navigatorTarget && "serviceWorker" in navigatorTarget && "PushManager" in windowTarget
    );
    notificationState.permission = NotificationCtor ? NotificationCtor.permission : "unsupported";
    notificationState.desktop_enabled = storageGetItem("codoxear.desktopNotificationsEnabled") === "1";
    let nextSnapshot = snapshot;
    if (!nextSnapshot) {
      try {
        nextSnapshot = await api("/api/notifications/subscription");
      } catch (error) {
        if (!(error && error.status === 404)) throw error;
      }
    }
    if (disposed || isAppDisposed()) return;
    const snapshotObject = nextSnapshot && typeof nextSnapshot === "object" ? nextSnapshot : {};
    notificationState.vapid_public_key = String(snapshotObject.vapid_public_key || "");
    let endpoint = "";
    if (deviceNotificationClass() === "mobile" && notificationState.push_supported) {
      try {
        const registration = await ensureServiceWorker();
        if (disposed || isAppDisposed()) return;
        const subscription = await registration.pushManager.getSubscription();
        if (disposed || isAppDisposed()) return;
        endpoint = subscription && typeof subscription.endpoint === "string" ? subscription.endpoint : "";
      } catch (error) {
        console.error("load push subscription failed", error);
      }
    }
    const subscriptions = Array.isArray(snapshotObject.subscriptions) ? snapshotObject.subscriptions : [];
    const current = endpoint ? subscriptions.find((item) => item && item.endpoint === endpoint) : null;
    notificationState.endpoint = endpoint;
    notificationState.subscriptions = subscriptions;
    notificationState.notifications_enabled = !!(current && current.notifications_enabled);
    render();
  }

  async function enableNotificationsOnDevice() {
    if (!notificationState.desktop_supported) throw new Error("notifications require HTTPS or localhost");
    if (NotificationCtor && NotificationCtor.permission !== "granted") {
      const permission = await NotificationCtor.requestPermission();
      if (permission !== "granted") throw new Error(`notification permission ${permission}`);
    }
    if (deviceNotificationClass() === "desktop") {
      setDesktopNotificationsEnabled(true);
      await syncState();
      return;
    }
    if (!notificationState.push_supported) {
      throw new Error("mobile notifications require web push in an installed HTTPS web app");
    }
    const registration = await ensureServiceWorker();
    const publicKey = notificationState.vapid_public_key;
    if (!publicKey) throw new Error("missing VAPID public key");
    let subscription = await registration.pushManager.getSubscription();
    if (!subscription) {
      subscription = await registration.pushManager.subscribe({
        userVisibleOnly: true,
        applicationServerKey: base64UrlToUint8Array(publicKey, atob),
      });
    }
    const nextSnapshot = await api("/api/notifications/subscription", {
      method: "POST",
      body: {
        subscription: subscription.toJSON(),
        user_agent: navigatorTarget ? navigatorTarget.userAgent : "",
        device_label: "current-device",
        device_class: deviceNotificationClass(),
      },
    });
    await syncState(nextSnapshot);
  }

  async function toggleCurrentDeviceNotifications(enabled) {
    if (!notificationState.desktop_supported) throw new Error("notifications require HTTPS or localhost");
    if (deviceNotificationClass() === "desktop") {
      setDesktopNotificationsEnabled(enabled);
      await syncState();
      return;
    }
    if (!notificationState.push_supported) {
      throw new Error("mobile notifications require web push in an installed HTTPS web app");
    }
    if (!notificationState.endpoint && enabled) {
      await enableNotificationsOnDevice();
      return;
    }
    if (!notificationState.endpoint) {
      await syncState();
      return;
    }
    const nextSnapshot = await api("/api/notifications/subscription/toggle", {
      method: "POST",
      body: { endpoint: notificationState.endpoint, enabled: !!enabled },
    });
    await syncState(nextSnapshot);
  }

  eventBindings.on(notificationBtn, "click", async (event) => {
    event.preventDefault();
    event.stopPropagation();
    notificationPanelOpen = !notificationPanelOpen;
    if (notificationPanelOpen) {
      markAllNotificationsRead();
      render();
      await pollFeed();
      return;
    }
    render();
  });

  if (notificationClearBtn) {
    eventBindings.on(notificationClearBtn, "click", () => {
      markAllNotificationsRead();
      render();
    });
  }

  if (notificationEnableBtn) {
    eventBindings.on(notificationEnableBtn, "click", async () => {
      try {
        await syncState();
        if (enabledLocally()) {
          await toggleCurrentDeviceNotifications(false);
          setNotificationEnabledLocal(false);
        } else {
          await primeNotificationSound();
          setNotificationEnabledLocal(true);
          await enableNotificationsOnDevice();
        }
      } catch (error) {
        console.error("notification toggle failed", error);
        setNotificationEnabledLocal(false);
        setToast(`notification error: ${error && error.message ? error.message : "unknown error"}`);
      }
      render();
    });
  }

  render();

  function dispose() {
    if (disposed) return;
    disposed = true;
    desktopNotificationTimers.forEach((timer) => clearTimeoutFn(timer));
    desktopNotificationTimers.clear();
    deliveredDesktopNotificationIds.clear();
    if (notificationAudioContext && typeof notificationAudioContext.close === "function") {
      void notificationAudioContext.close();
    }
    notificationAudioContext = null;
    notificationItems.clear();
    readNotificationIds.clear();
    notificationPanelOpen = false;
    swRegistration = null;
    if (notificationList && typeof notificationList.replaceChildren === "function") notificationList.replaceChildren();
  }

  return Object.freeze({ enabledLocally, syncState, pollFeed, dispose });
}

export { createNotificationRuntime };
