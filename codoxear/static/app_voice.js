import * as CodoxearModal from "./app_modal.js";
import * as CodoxearVoiceHelpers from "./app_voice_helpers.js";


// Voice / Settings / Notifications / Announcement orchestration authority.
  //
  // Owns every piece of voice/notification/announcement state that used to live
  // as app.js locals (voiceSaveTimer, voiceSettings, localAnnouncementEnabled,
  // localNotificationEnabled, desktopNotificationTimers,
  // deliveredDesktopNotificationIds, notificationFeedSinceTs,
  // announcementClientId/heartbeat timer, liveAudio retry/watchdog timers and
  // HLS state, notificationState and service-worker registration state) plus
  // every voice/settings/notification/announcement function that used to live
  // in app.js (announcement toggle, notification transport projection, desktop
  // + push notification enable/toggle, notification feed polling, voice
  // settings load/save, live-audio source/playback/watchdog orchestration, the
  // voice settings dialog show/hide/focus behavior) and the announceBtn /
  // notificationBtn / liveAudio / voice-settings-dialog event handlers.
  //
  // app.js supplies the shell root and voice host; this module constructs and
  // owns the announce/notification buttons, hidden live-audio node, and voice
  // settings dialog through createVoiceDom(options), then injects those nodes
  // into createVoiceController(options). App-level runtime accessors (api,
  // toasts, auth loss, modal open/close coordination, storage, url/version
  // helpers, timer/animation-frame/window/navigator/document targets) remain
  // injected so the controller has no hidden coupling to app.js globals and
  // can be exercised in a VM with fakes.
  //
  // Pure helpers (browserSupports*/base64UrlToUint8Array/isMobileNotificationDevice/
  // notificationDeviceClass) come from CodoxearVoiceHelpers; the modal
  // open-state + focus-restore helpers come from CodoxearModal.



  const browserSupportsNativeLiveAudioPlayback = CodoxearVoiceHelpers.browserSupportsNativeLiveAudioPlayback;
  const browserSupportsMseLiveAudioPlayback = CodoxearVoiceHelpers.browserSupportsMseLiveAudioPlayback;
  const shouldPreferNativeLiveAudioPlayback = CodoxearVoiceHelpers.shouldPreferNativeLiveAudioPlayback;
  const browserSupportsLiveAudioPlayback = CodoxearVoiceHelpers.browserSupportsLiveAudioPlayback;
  const base64UrlToUint8Array = CodoxearVoiceHelpers.base64UrlToUint8Array;
  const isMobileNotificationDevice = CodoxearVoiceHelpers.isMobileNotificationDevice;
  const notificationDeviceClass = CodoxearVoiceHelpers.notificationDeviceClass;
  const isModalTargetOpen = CodoxearModal.isModalTargetOpen;
  const restoreModalFocus = CodoxearModal.restoreModalFocus;

  const LIVE_AUDIO_WATCHDOG_MS = 2500;
  const LIVE_AUDIO_STALL_GRACE_MS = 12000;
  const LIVE_AUDIO_RESTART_THROTTLE_MS = 4000;
  const ANNOUNCEMENT_HEARTBEAT_INTERVAL_MS = 15000;
  const VOICE_SAVE_DEBOUNCE_MS = 250;
  const NOTIFICATION_PANEL_MAX_ITEMS = 100;

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`voice controller dependency missing: ${name}`);
    return value;
  }

  function requireNode(value, name) {
    if (!value || typeof value !== "object" || !value.style) throw new TypeError(`voice controller dependency missing: ${name}`);
    return value;
  }

  function createVoiceDom(options = {}) {
    if (!options || typeof options !== "object") throw new TypeError("voice DOM dependency missing: options");
    const root = options.root;
    const el = requireFunction(options.el, "el");
    const iconSvg = requireFunction(options.iconSvg, "iconSvg");
    const voiceHost = requireNode(options.voiceHost, "voiceHost");
    if (!root || typeof root.appendChild !== "function") throw new TypeError("voice DOM dependency missing: root");

    const announceBtn = el("button", { id: "announceBtn", class: "icon-btn", title: "Voice announcements", "aria-label": "Voice announcements", type: "button", html: iconSvg("volume") });
    const notificationBtn = el("button", { id: "notificationBtn", class: "icon-btn", title: "Notifications", "aria-label": "Notifications", "aria-controls": "notificationPanel", "aria-expanded": "false", type: "button", html: iconSvg("bell") });
    const notificationPanel = el("section", { id: "notificationPanel", class: "notificationPanel", role: "dialog", "aria-label": "Notifications" });
    const notificationPanelHeader = el("div", { class: "notificationPanelHeader" }, [
      el("div", { class: "title", text: "Notifications" }),
      el("button", { id: "notificationClearBtn", class: "text-btn", type: "button", text: "Mark read" }),
    ]);
    const notificationEnableBtn = el("button", { id: "notificationEnableBtn", class: "text-btn", type: "button", text: "Enable browser alerts" });
    const notificationEmpty = el("div", { id: "notificationEmpty", class: "muted", text: "No notifications" });
    const notificationList = el("div", { id: "notificationList", class: "notificationList", role: "list" });
    notificationPanel.append(notificationPanelHeader, notificationEnableBtn, notificationEmpty, notificationList);
    notificationPanel.style.display = "none";
    voiceHost.appendChild(notificationBtn);
    voiceHost.appendChild(announceBtn);

    const liveAudio = el("audio", { id: "liveAudio", preload: "none", playsinline: "true" });
    liveAudio.style.display = "none";
    const voiceSettingsBackdrop = el("div", { class: "modalBackdrop", id: "voiceSettingsBackdrop" });
    const voiceSettingsCloseBtn = el("button", {
      id: "voiceSettingsCloseBtn",
      class: "icon-btn",
      title: "Close",
      "aria-label": "Close",
      type: "button",
      html: iconSvg("x"),
    });
    const voiceSettingsStatus = el("div", { class: "muted", id: "voiceSettingsStatus", text: "" });
    const voiceBaseUrlInput = el("input", { id: "voiceBaseUrlInput", type: "text", autocomplete: "off", spellcheck: "false" });
    const voiceApiKeyInput = el("input", { id: "voiceApiKeyInput", type: "password", autocomplete: "off", spellcheck: "false" });
    const voiceClearApiKeyToggle = el("input", { id: "voiceClearApiKeyToggle", type: "checkbox" });
    const narrationSettingToggle = el("input", { id: "narrationSettingToggle", type: "checkbox" });
    const unattendedPromptInput = el("textarea", {
      id: "unattendedPromptInput",
      rows: "14",
      spellcheck: "true",
      "aria-describedby": "unattendedPromptHint",
    });
    const unattendedPromptResetBtn = el("button", { id: "unattendedPromptResetBtn", class: "text-btn", type: "button", text: "Reset to default" });
    const voiceSettingsCancelBtn = el("button", { id: "voiceSettingsCancelBtn", type: "button", text: "Cancel" });
    const voiceSettingsSaveBtn = el("button", { id: "voiceSettingsSaveBtn", class: "primary", type: "button", text: "Save" });
    const voiceSettingsViewer = el("dialog", { class: "formViewer formDialog", id: "voiceSettingsViewer", "aria-label": "Settings" }, [
      el("div", { class: "queueHeader" }, [
        el("div", { class: "title", text: "Settings" }),
        el("div", { class: "actions" }, [voiceSettingsCloseBtn]),
      ]),
      voiceSettingsStatus,
      el("div", { class: "formBody" }, [
        el("label", { class: "field" }, [
          el("span", { class: "fieldLabel", text: "OpenAI-compatible API base URL" }),
          voiceBaseUrlInput,
          el("span", { class: "fieldHint", text: "Used for both summarization and speech." }),
        ]),
        el("label", { class: "field" }, [
          el("span", { class: "fieldLabel", text: "OpenAI-compatible API key" }),
          voiceApiKeyInput,
          el("span", { class: "fieldHint", text: "Leave blank to keep the saved key." }),
        ]),
        el("div", { class: "field" }, [
          el("label", { class: "voiceToggleRow" }, [voiceClearApiKeyToggle, el("span", { text: "Clear saved API key" })]),
        ]),
        el("div", { class: "field" }, [
          el("label", { class: "voiceToggleRow" }, [narrationSettingToggle, el("span", { text: "Announce narration messages" })]),
        ]),
        el("div", { class: "field" }, [
          el("span", { class: "fieldLabel", text: "Unattended mode prompt" }),
          unattendedPromptInput,
          el("span", { class: "fieldHint", id: "unattendedPromptHint", text: "Sent when unattended mode resumes an idle session. Reset then Save to restore the built-in constitution." }),
          unattendedPromptResetBtn,
        ]),
      ]),
      el("div", { class: "formActions" }, [voiceSettingsCancelBtn, voiceSettingsSaveBtn]),
    ]);
    root.appendChild(notificationPanel);
    root.appendChild(liveAudio);
    root.appendChild(voiceSettingsBackdrop);
    root.appendChild(voiceSettingsViewer);
    return Object.freeze({
      announceBtn,
      notificationBtn,
      notificationPanel,
      notificationList,
      notificationEmpty,
      notificationClearBtn,
      notificationEnableBtn,
      liveAudio,
      voiceSettingsBackdrop,
      voiceSettingsCloseBtn,
      voiceSettingsStatus,
      voiceBaseUrlInput,
      voiceApiKeyInput,
      voiceClearApiKeyToggle,
      narrationSettingToggle,
      unattendedPromptInput,
      unattendedPromptResetBtn,
      voiceSettingsViewer,
      voiceSettingsCancelBtn,
      voiceSettingsSaveBtn,
    });
  }

  function createVoiceController(options = {}) {
    if (!options || typeof options !== "object") throw new TypeError("voice controller dependency missing: options");

    // DOM nodes (created and owned by app.js).
    const announceBtn = requireNode(options.announceBtn, "announceBtn");
    const notificationBtn = requireNode(options.notificationBtn, "notificationBtn");
    const notificationPanel = options.notificationPanel || null;
    const notificationList = options.notificationList || null;
    const notificationEmpty = options.notificationEmpty || null;
    const notificationClearBtn = options.notificationClearBtn || null;
    const notificationEnableBtn = options.notificationEnableBtn || null;
    const hasNotificationPanel = Boolean(
      notificationPanel && notificationList && notificationEmpty && notificationClearBtn && notificationEnableBtn
    );
    const liveAudio = requireNode(options.liveAudio, "liveAudio");
    const voiceSettingsBackdrop = requireNode(options.voiceSettingsBackdrop, "voiceSettingsBackdrop");
    const voiceSettingsCloseBtn = requireNode(options.voiceSettingsCloseBtn, "voiceSettingsCloseBtn");
    const voiceSettingsStatus = requireNode(options.voiceSettingsStatus, "voiceSettingsStatus");
    const voiceBaseUrlInput = requireNode(options.voiceBaseUrlInput, "voiceBaseUrlInput");
    const voiceApiKeyInput = requireNode(options.voiceApiKeyInput, "voiceApiKeyInput");
    const voiceClearApiKeyToggle = requireNode(options.voiceClearApiKeyToggle, "voiceClearApiKeyToggle");
    const narrationSettingToggle = requireNode(options.narrationSettingToggle, "narrationSettingToggle");
    const unattendedPromptInput = options.unattendedPromptInput ? requireNode(options.unattendedPromptInput, "unattendedPromptInput") : null;
    const unattendedPromptResetBtn = options.unattendedPromptResetBtn ? requireNode(options.unattendedPromptResetBtn, "unattendedPromptResetBtn") : null;
    const voiceSettingsViewer = requireNode(options.voiceSettingsViewer, "voiceSettingsViewer");
    const voiceSettingsCancelBtn = requireNode(options.voiceSettingsCancelBtn, "voiceSettingsCancelBtn");
    const voiceSettingsSaveBtn = requireNode(options.voiceSettingsSaveBtn, "voiceSettingsSaveBtn");

    // App-level runtime state accessors and effects.
    const isAppDisposed = requireFunction(options.isAppDisposed, "isAppDisposed");
    const api = requireFunction(options.api, "api");
    const setToast = requireFunction(options.setToast, "setToast");
    const handleAppAuthLoss = requireFunction(options.handleAppAuthLoss, "handleAppAuthLoss");
    const prepareModalOpen = requireFunction(options.prepareModalOpen, "prepareModalOpen");
    const afterModalVisibilityChanged = requireFunction(options.afterModalVisibilityChanged, "afterModalVisibilityChanged");
    const resolveAppUrl = requireFunction(options.resolveAppUrl, "resolveAppUrl");
    const versionedShellAssetPath = requireFunction(options.versionedShellAssetPath, "versionedShellAssetPath");
    const storageGetItem = requireFunction(options.storageGetItem, "storageGetItem");
    const storageSetItem = requireFunction(options.storageSetItem, "storageSetItem");
    const storageRemoveItem = requireFunction(options.storageRemoveItem, "storageRemoveItem");
    const focusSessionFromNotification = typeof options.focusSessionFromNotification === "function" ? options.focusSessionFromNotification : null;

    // Injectable browser targets (default to the real globals).
    const windowTarget = options.windowTarget || window;
    const navigatorTarget = options.navigatorTarget || (typeof navigator !== "undefined" ? navigator : null);
    const documentTarget = options.documentTarget || document;
    const NotificationCtor = typeof options.Notification !== "undefined" ? options.Notification : (typeof Notification !== "undefined" ? Notification : undefined);
    const cryptoRef = typeof options.crypto !== "undefined" ? options.crypto : (typeof windowTarget.crypto !== "undefined" ? windowTarget.crypto : undefined);
    const AudioContextCtor = typeof options.AudioContext !== "undefined"
      ? options.AudioContext
      : (windowTarget.AudioContext || windowTarget.webkitAudioContext || null);

    const requestFrame = typeof options.requestFrame === "function" ? options.requestFrame : (typeof requestAnimationFrame === "function" ? requestAnimationFrame : null);
    const setTimeoutFn = typeof options.setTimeout === "function" ? options.setTimeout : setTimeout;
    const clearTimeoutFn = typeof options.clearTimeout === "function" ? options.clearTimeout : clearTimeout;
    const setIntervalFn = typeof options.setInterval === "function" ? options.setInterval : setInterval;
    const clearIntervalFn = typeof options.clearInterval === "function" ? options.clearInterval : clearInterval;

    function deviceNotificationClass() {
      return notificationDeviceClass(navigatorTarget);
    }

    // --- Voice / notification / announcement state owned by this controller ---

    let voiceSaveTimer = null;
    let voiceSettings = {
      tts_enabled_for_narration: false,
      tts_enabled_for_final_response: true,
      tts_base_url: "https://api.openai.com/v1",
      tts_api_key: "",
      audio: { queue_depth: 0, segment_count: 0, last_error: "", stream_url: "/api/audio/live.m3u8" },
      notifications: { enabled_devices: 0, total_devices: 0, vapid_public_key: "" },
      has_tts_api_key: false,
    };
    let localAnnouncementEnabled = storageGetItem("codoxear.announcementEnabled") === "1";
    let localNotificationEnabled = storageGetItem("codoxear.notificationEnabled") === "1";
    const desktopNotificationTimers = new Map();
    const deliveredDesktopNotificationIds = new Set();
    let notificationFeedSinceTs = 0;
    const notificationItems = new Map();
    const readNotificationIds = new Set();
    let notificationPanelOpen = false;
    let notificationAudioContext = null;
    const announcementClientId = (() => {
      const key = "codoxear.announcementClientId";
      const current = storageGetItem(key);
      if (current) return current;
      const next =
        cryptoRef && typeof cryptoRef.randomUUID === "function"
          ? cryptoRef.randomUUID()
          : `ann-${Date.now()}-${Math.random().toString(16).slice(2)}`;
      storageSetItem(key, next);
      return next;
    })();
    let announcementHeartbeatTimer = null;
    let liveAudioRetryTimer = null;
    let liveAudioWatchdogTimer = null;
    let notificationState = {
      desktop_supported: false,
      push_supported: false,
      permission: NotificationCtor ? NotificationCtor.permission : "unsupported",
      desktop_enabled: false,
      endpoint: "",
      notifications_enabled: false,
      subscriptions: [],
    };
    let unattendedPrompt = { prompt: "", default_prompt: "" };
    let liveAudioStarted = false;
    let liveAudioErrorState = false;
    let liveAudioSourceUrl = "";
    let liveAudioHls = null;
    let liveAudioLastProgressTs = 0;
    let liveAudioLastCurrentTime = 0;
    let liveAudioSuspectSinceTs = 0;
    let liveAudioLastRestartTs = 0;
    let swRegistration = null;
    // Canonical voice-settings dialog open flag. The controller owns this; the
    // <dialog>.open / style state is a fallback only, so callers must not rely
    // on style.display alone to decide whether the dialog is open.
    let settingsOpen = false;
    let voiceSettingsReturnFocusEl = null;

    // Controller-owned event listener lifecycle (cleared by dispose).
    const eventCleanups = [];
    function addEvent(target, type, handler, options) {
      if (!target || typeof target.addEventListener !== "function") return handler;
      target.addEventListener(type, handler, options);
      eventCleanups.push(() => {
        try {
          target.removeEventListener(type, handler, options);
        } catch (_error) {}
      });
      return handler;
    }

    function voiceAnnouncementsEnabled() {
      return !!localAnnouncementEnabled;
    }

    function notificationsEnabledLocally() {
      return !!localNotificationEnabled;
    }

    function isSettingsOpen() {
      if (settingsOpen) return true;
      if (voiceSettingsViewer && voiceSettingsViewer.open) return true;
      return false;
    }

    function currentVoiceStreamUrl() {
      const streamUrl =
        voiceSettings && voiceSettings.audio && typeof voiceSettings.audio.stream_url === "string" && voiceSettings.audio.stream_url
          ? voiceSettings.audio.stream_url
          : "/api/audio/live.m3u8";
      return resolveAppUrl(streamUrl);
    }

    function hasAnnouncementCredentials() {
      return Boolean(
        String(voiceSettings.tts_base_url || "").trim() &&
          (String(voiceSettings.tts_api_key || "").trim() || voiceSettings.has_tts_api_key)
      );
    }

    function liveAudioHasReadySegments() {
      const audio = voiceSettings && voiceSettings.audio ? voiceSettings.audio : {};
      return Number(audio.segment_count || 0) > 0;
    }

    function liveAudioHasErrorState() {
      const audioLastError = voiceSettings && voiceSettings.audio ? String(voiceSettings.audio.last_error || "").trim() : "";
      return Boolean(audioLastError || liveAudioErrorState);
    }

    function destroyLiveAudioHls() {
      const current = liveAudioHls;
      liveAudioHls = null;
      if (!current || typeof current.destroy !== "function") return;
      try {
        current.destroy();
      } catch (e) {
        console.error("destroy live hls failed", e);
      }
    }

    async function ensureLiveAudioPlaybackSource(nextSrc, { resetSource = false } = {}) {
      if (resetSource) {
        resetLiveAudioState();
      }
      if (shouldPreferNativeLiveAudioPlayback(liveAudio, navigatorTarget)) {
        destroyLiveAudioHls();
        if (resetSource || liveAudioSourceUrl !== nextSrc || liveAudio.currentSrc !== nextSrc) {
          liveAudio.src = nextSrc;
          liveAudioSourceUrl = nextSrc;
        }
        return;
      }
      if (!browserSupportsMseLiveAudioPlayback(windowTarget)) {
        throw new Error("this browser does not support HLS audio playback in this app");
      }
      const HlsCtor = windowTarget.Hls;
      const needsReload = resetSource || !liveAudioHls || liveAudioSourceUrl !== nextSrc;
      if (!needsReload) return;
      destroyLiveAudioHls();
      liveAudio.removeAttribute("src");
      liveAudio.load();
      liveAudioSourceUrl = nextSrc;
      const hls = new HlsCtor();
      liveAudioHls = hls;
      hls.on(HlsCtor.Events.ERROR, (_event, data) => {
        if (!data) return;
        console.error("live hls error", data.type, data.details, data);
        if (!data.fatal || liveAudioHls !== hls) return;
        switch (data.type) {
          case HlsCtor.ErrorTypes.NETWORK_ERROR:
            hls.startLoad();
            return;
          case HlsCtor.ErrorTypes.MEDIA_ERROR:
            hls.recoverMediaError();
            return;
          default:
            destroyLiveAudioHls();
            liveAudioStarted = false;
            liveAudioErrorState = true;
            liveAudioSuspectSinceTs = 0;
            updateVoiceUi();
            scheduleLiveAudioRetry(1200, { resetSource: true });
        }
      });
      await new Promise((resolve, reject) => {
        let settled = false;
        const cleanup = () => {
          hls.off(HlsCtor.Events.MEDIA_ATTACHED, onAttached);
          hls.off(HlsCtor.Events.MANIFEST_PARSED, onManifestParsed);
          hls.off(HlsCtor.Events.ERROR, onInitError);
        };
        const settle = (fn, value) => {
          if (settled) return;
          settled = true;
          cleanup();
          fn(value);
        };
        const onAttached = () => {
          hls.loadSource(nextSrc);
        };
        const onManifestParsed = () => {
          settle(resolve);
        };
        const onInitError = (_event, data) => {
          if (!data || !data.fatal) return;
          settle(reject, new Error(data.details || data.type || "failed to load HLS stream"));
        };
        hls.on(HlsCtor.Events.MEDIA_ATTACHED, onAttached);
        hls.on(HlsCtor.Events.MANIFEST_PARSED, onManifestParsed);
        hls.on(HlsCtor.Events.ERROR, onInitError);
        hls.attachMedia(liveAudio);
      });
    }

    async function sendAnnouncementHeartbeat(enabled) {
      try {
        await api("/api/audio/listener", {
          method: "POST",
          body: {
            client_id: announcementClientId,
            enabled: !!enabled,
          },
        });
      } catch (e) {
        console.error("announcement heartbeat failed", e);
      }
    }

    function stopAnnouncementHeartbeat() {
      if (announcementHeartbeatTimer) clearIntervalFn(announcementHeartbeatTimer);
      announcementHeartbeatTimer = null;
      void sendAnnouncementHeartbeat(false);
    }

    function markLiveAudioProgress() {
      liveAudioLastProgressTs = Date.now();
      liveAudioSuspectSinceTs = 0;
      liveAudioLastCurrentTime = Number(liveAudio.currentTime || 0);
    }

    function resetLiveAudioState() {
      destroyLiveAudioHls();
      try {
        liveAudio.pause();
      } catch (_error) {}
      liveAudio.removeAttribute("src");
      liveAudio.load();
      liveAudioStarted = false;
      liveAudioSourceUrl = "";
      liveAudioLastProgressTs = 0;
      liveAudioLastCurrentTime = 0;
      liveAudioSuspectSinceTs = 0;
    }

    function noteLiveAudioPotentialStall(_reason = "") {
      if (!localAnnouncementEnabled) return;
      if (!liveAudioStarted || liveAudio.paused || liveAudio.ended) return;
      if (!liveAudioHasReadySegments()) return;
      if (!liveAudioSuspectSinceTs) liveAudioSuspectSinceTs = Date.now();
    }

    function queueLiveAudioHardRestart(_reason = "") {
      if (!localAnnouncementEnabled) return;
      if (!browserSupportsLiveAudioPlayback(liveAudio, windowTarget)) return;
      if (!liveAudioHasReadySegments()) return;
      const now = Date.now();
      if (now - liveAudioLastRestartTs < LIVE_AUDIO_RESTART_THROTTLE_MS) return;
      liveAudioLastRestartTs = now;
      liveAudioStarted = false;
      liveAudioSuspectSinceTs = 0;
      updateVoiceUi();
      scheduleLiveAudioRetry(150, { resetSource: true });
    }

    function runLiveAudioWatchdog() {
      if (!localAnnouncementEnabled) return;
      if (!browserSupportsLiveAudioPlayback(liveAudio, windowTarget)) return;
      if (!liveAudioHasReadySegments()) return;
      const now = Date.now();
      const currentTime = Number(liveAudio.currentTime || 0);
      if (currentTime > liveAudioLastCurrentTime + 0.05) {
        markLiveAudioProgress();
        return;
      }
      liveAudioLastCurrentTime = currentTime;
      if (!liveAudioStarted || liveAudio.paused || liveAudio.ended) {
        liveAudioSuspectSinceTs = 0;
        return;
      }
      if (!liveAudioSuspectSinceTs) liveAudioSuspectSinceTs = now;
      const baselineTs = Math.max(liveAudioLastProgressTs || 0, liveAudioSuspectSinceTs || 0);
      if (!baselineTs) return;
      if (now - baselineTs < LIVE_AUDIO_STALL_GRACE_MS) return;
      queueLiveAudioHardRestart("watchdog");
    }

    function stopLiveAudioWatchdog() {
      if (liveAudioWatchdogTimer) clearIntervalFn(liveAudioWatchdogTimer);
      liveAudioWatchdogTimer = null;
    }

    function startLiveAudioWatchdog() {
      runLiveAudioWatchdog();
      if (liveAudioWatchdogTimer) clearIntervalFn(liveAudioWatchdogTimer);
      liveAudioWatchdogTimer = setIntervalFn(() => {
        runLiveAudioWatchdog();
      }, LIVE_AUDIO_WATCHDOG_MS);
    }

    function startAnnouncementHeartbeat() {
      void sendAnnouncementHeartbeat(true);
      if (announcementHeartbeatTimer) clearIntervalFn(announcementHeartbeatTimer);
      announcementHeartbeatTimer = setIntervalFn(() => {
        void sendAnnouncementHeartbeat(true);
      }, ANNOUNCEMENT_HEARTBEAT_INTERVAL_MS);
    }

    function setAnnouncementEnabled(enabled) {
      localAnnouncementEnabled = !!enabled;
      if (localAnnouncementEnabled) storageSetItem("codoxear.announcementEnabled", "1");
      else storageRemoveItem("codoxear.announcementEnabled");
      if (!localAnnouncementEnabled) {
        stopAnnouncementHeartbeat();
        stopLiveAudioWatchdog();
        if (liveAudioRetryTimer) clearTimeoutFn(liveAudioRetryTimer);
        liveAudioRetryTimer = null;
        resetLiveAudioState();
      } else {
        startAnnouncementHeartbeat();
        startLiveAudioWatchdog();
      }
      updateVoiceUi();
    }

    function setNotificationEnabledLocal(enabled) {
      localNotificationEnabled = !!enabled;
      if (localNotificationEnabled) storageSetItem("codoxear.notificationEnabled", "1");
      else storageRemoveItem("codoxear.notificationEnabled");
      updateVoiceUi();
    }

    function resumeAnnouncementRuntime({ resetSource = false } = {}) {
      if (!localAnnouncementEnabled) return;
      startAnnouncementHeartbeat();
      startLiveAudioWatchdog();
      if (!liveAudioStarted && browserSupportsLiveAudioPlayback(liveAudio, windowTarget) && liveAudioHasReadySegments()) {
        scheduleLiveAudioRetry(150, { resetSource });
      }
    }

    function scheduleLiveAudioRetry(delayMs = 1200, { resetSource = true } = {}) {
      if (!localAnnouncementEnabled) return;
      if (liveAudioRetryTimer) clearTimeoutFn(liveAudioRetryTimer);
      liveAudioRetryTimer = setTimeoutFn(async () => {
        liveAudioRetryTimer = null;
        if (!localAnnouncementEnabled) return;
        try {
          await startLiveAudioPlayback({ resetSource });
        } catch (e) {
          console.error("live audio retry failed", e);
        }
      }, delayMs);
    }

    async function maybeAutoStartLiveAudioFromGesture({ resetSource = false } = {}) {
      if (!localAnnouncementEnabled) return;
      if (!browserSupportsLiveAudioPlayback(liveAudio, windowTarget)) return;
      if (!liveAudioHasReadySegments()) return;
      try {
        await startLiveAudioPlayback({ resetSource: resetSource || liveAudio.ended });
      } catch (e) {
        console.error("auto-start live audio failed", e);
      }
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

    function focusSessionFromDesktopNotification(sessionId) {
      const sid = String(sessionId || "").trim();
      try {
        if (typeof windowTarget.focus === "function") windowTarget.focus();
      } catch {}
      if (!sid) return;
      if (focusSessionFromNotification) focusSessionFromNotification(sid);
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
            focusSessionFromDesktopNotification(sid);
          };
        }
        if (id) deliveredDesktopNotificationIds.add(id);
        return true;
      } catch (e) {
        console.error("desktop notification failed", e);
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
      } catch (e) {
        console.error("notification sound failed", e);
      }
    }

    function sortedNotificationItems() {
      return Array.from(notificationItems.values()).sort((a, b) => {
        const byUpdated = Number(b.updated_ts || 0) - Number(a.updated_ts || 0);
        return byUpdated || String(b.message_id || "").localeCompare(String(a.message_id || ""));
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

    function renderNotificationPanel() {
      if (!hasNotificationPanel) return;
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
      const alertsEnabled = notificationsEnabledLocally();
      notificationEnableBtn.textContent = alertsEnabled ? "Disable browser alerts" : "Enable browser alerts";
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
          renderNotificationPanel();
          focusSessionFromDesktopNotification(item.session_id);
        };
        notificationList.appendChild(row);
      }
    }

    async function pollNotificationFeed({ prime = false } = {}) {
      if (isAppDisposed()) return;
      let maxSeen = notificationFeedSinceTs;
      try {
        const data = await api(`/api/notifications/feed?since=${encodeURIComponent(notificationFeedSinceTs)}`);
        if (isAppDisposed()) return;
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
      } catch (e) {
        if (e && e.status === 401) {
          handleAppAuthLoss();
          return;
        }
        console.error("notification feed poll failed", e);
        return;
      }
      notificationFeedSinceTs = maxSeen;
      renderNotificationPanel();
    }

    function syncVoiceSettingsFormFromState() {
      if (voiceBaseUrlInput) voiceBaseUrlInput.value = String(voiceSettings.tts_base_url || "");
      if (voiceApiKeyInput && !voiceApiKeyInput.matches(":focus")) {
        voiceApiKeyInput.value = "";
        voiceApiKeyInput.placeholder = voiceSettings.has_tts_api_key ? "Saved API key (leave blank to keep)" : "Enter API key";
      }
      if (voiceClearApiKeyToggle) voiceClearApiKeyToggle.checked = false;
      if (narrationSettingToggle) narrationSettingToggle.checked = !!voiceSettings.tts_enabled_for_narration;
      if (unattendedPromptInput && !unattendedPromptInput.matches(":focus")) {
        unattendedPromptInput.value = String(unattendedPrompt.prompt || unattendedPrompt.default_prompt || "");
      }
    }

    async function loadUnattendedPrompt() {
      const data = await api("/api/settings/unattended-prompt");
      if (isAppDisposed()) return data;
      if (!data || typeof data !== "object" || typeof data.prompt !== "string" || typeof data.default_prompt !== "string") {
        throw new Error("invalid unattended prompt response");
      }
      unattendedPrompt = { prompt: data.prompt, default_prompt: data.default_prompt };
      if (!isSettingsOpen()) syncVoiceSettingsFormFromState();
      return data;
    }

    async function saveUnattendedPrompt() {
      if (!unattendedPromptInput) return null;
      const data = await api("/api/settings/unattended-prompt", { method: "POST", body: { prompt: unattendedPromptInput.value } });
      if (!data || typeof data !== "object" || typeof data.prompt !== "string" || typeof data.default_prompt !== "string") {
        throw new Error("invalid unattended prompt response");
      }
      unattendedPrompt = { prompt: data.prompt, default_prompt: data.default_prompt };
      return data;
    }

    function updateVoiceUi() {
      const announcementsOn = voiceAnnouncementsEnabled();
      // Only surface the audio error state when the user has actually enabled
      // voice. Server-side pipeline errors (e.g. "tts_api_key is required",
      // keepalive sweep failures) are irrelevant to users who never opted in —
      // showing them as a red button is a category error.
      const audioError = announcementsOn && liveAudioHasErrorState();
      announceBtn.classList.toggle("active", announcementsOn);
      const announceBase = announcementsOn ? "Announcements on" : "Announcements off";
      announceBtn.title = audioError ? `${announceBase} (audio error)` : announceBase;
      announceBtn.setAttribute("aria-label", announceBtn.title);
      announceBtn.classList.toggle("error", audioError);
      notificationBtn.classList.toggle("active", notificationsEnabledLocally());
      const transport = activeNotificationTransport();
      notificationBtn.title = notificationsEnabledLocally()
        ? transport === "push"
          ? "Notifications on (push)"
          : transport === "desktop"
            ? "Notifications on"
            : "Notifications pending"
        : "Notifications off";
      notificationBtn.setAttribute("aria-label", notificationBtn.title);
      notificationState.permission = NotificationCtor ? NotificationCtor.permission : "unsupported";
      renderNotificationPanel();
      if (!isSettingsOpen()) syncVoiceSettingsFormFromState();
    }

    async function refreshBackgroundState({ force = false, primeNotifications = false } = {}) {
      const announcementsEnabled = voiceAnnouncementsEnabled();
      const notificationsEnabled = notificationsEnabledLocally();
      if (!force && !announcementsEnabled && !notificationsEnabled) return false;

      // `/api/settings/voice` includes the subscription snapshot. A single
      // request refreshes both state authorities when a tab becomes visible.
      const snapshot = await loadVoiceSettings();
      await syncNotificationState(snapshot);
      if (notificationsEnabled) await pollNotificationFeed({ prime: primeNotifications });
      return true;
    }

    async function loadVoiceSettings() {
      const data = await api("/api/settings/voice");
      if (isAppDisposed()) return data;
      if (!data || typeof data !== "object") throw new Error("invalid voice settings response");
      voiceSettings = {
        ...voiceSettings,
        ...data,
        audio: data && typeof data.audio === "object" && data.audio ? data.audio : voiceSettings.audio,
        notifications: data && typeof data.notifications === "object" && data.notifications ? data.notifications : voiceSettings.notifications,
      };
      if (liveAudioStarted && liveAudioSourceUrl !== currentVoiceStreamUrl()) {
        void ensureLiveAudioPlaybackSource(currentVoiceStreamUrl(), { resetSource: true }).catch((e) => {
          console.error("reload live audio source failed", e);
        });
      }
      updateVoiceUi();
      // The server keeps active listeners only in memory. Reassert the
      // persisted browser opt-in whenever a fresh settings snapshot arrives:
      // this reconnects a tab after a server restart even when the page itself
      // never received a visibility/focus event.
      if (localAnnouncementEnabled) resumeAnnouncementRuntime({ resetSource: false });
      return data;
    }

    async function saveVoiceSettings() {
      const clearApiKey = !!(voiceClearApiKeyToggle && voiceClearApiKeyToggle.checked);
      const payload = {
        tts_enabled_for_narration: !!voiceSettings.tts_enabled_for_narration,
        tts_enabled_for_final_response: true,
        tts_base_url: String(voiceBaseUrlInput.value || voiceSettings.tts_base_url || "").trim(),
        tts_api_key: clearApiKey ? "" : String(voiceApiKeyInput.value || "").trim(),
        tts_api_key_clear: clearApiKey,
      };
      const data = await api("/api/settings/voice", { method: "POST", body: payload });
      if (!data || typeof data !== "object") throw new Error("invalid voice settings response");
      voiceSettings = {
        ...voiceSettings,
        ...data,
        audio: data && typeof data.audio === "object" && data.audio ? data.audio : voiceSettings.audio,
        notifications: data && typeof data.notifications === "object" && data.notifications ? data.notifications : voiceSettings.notifications,
      };
      updateVoiceUi();
      return data;
    }

    function scheduleVoiceSave() {
      if (voiceSaveTimer) clearTimeoutFn(voiceSaveTimer);
      voiceSaveTimer = setTimeoutFn(async () => {
        voiceSaveTimer = null;
        try {
          await saveVoiceSettings();
        } catch (e) {
          console.error("save voice settings failed", e);
          setToast(`voice settings error: ${e && e.message ? e.message : "unknown error"}`);
          try {
            await loadVoiceSettings();
          } catch (_error) {}
        }
      }, VOICE_SAVE_DEBOUNCE_MS);
    }

    async function ensureVoiceServiceWorker() {
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

    async function syncNotificationState(serverSnapshot) {
      if (isAppDisposed()) return;
      notificationState.desktop_supported = !!(windowTarget.isSecureContext && NotificationCtor);
      notificationState.push_supported = !!(notificationState.desktop_supported && navigatorTarget && "serviceWorker" in navigatorTarget && "PushManager" in windowTarget);
      notificationState.permission = NotificationCtor ? NotificationCtor.permission : "unsupported";
      notificationState.desktop_enabled = storageGetItem("codoxear.desktopNotificationsEnabled") === "1";
      let snapshot = serverSnapshot;
      if (!snapshot) {
        try {
          snapshot = await api("/api/notifications/subscription");
        } catch (e) {
          if (!(e && e.status === 404)) throw e;
        }
      }
      if (isAppDisposed()) return;
      let endpoint = "";
      if (deviceNotificationClass() === "mobile" && notificationState.push_supported) {
        try {
          const reg = await ensureVoiceServiceWorker();
          if (isAppDisposed()) return;
          const sub = await reg.pushManager.getSubscription();
          if (isAppDisposed()) return;
          endpoint = sub && typeof sub.endpoint === "string" ? sub.endpoint : "";
        } catch (e) {
          console.error("load push subscription failed", e);
        }
      }
      const subscriptions = snapshot && Array.isArray(snapshot.subscriptions) ? snapshot.subscriptions : [];
      const current = endpoint ? subscriptions.find((item) => item && item.endpoint === endpoint) : null;
      notificationState.endpoint = endpoint;
      notificationState.subscriptions = subscriptions;
      notificationState.notifications_enabled = !!(current && current.notifications_enabled);
      updateVoiceUi();
    }

    async function enableNotificationsOnDevice() {
      if (!notificationState.desktop_supported) {
        throw new Error("notifications require HTTPS or localhost");
      }
      if (NotificationCtor && NotificationCtor.permission !== "granted") {
        const permission = await NotificationCtor.requestPermission();
        if (permission !== "granted") {
          throw new Error(`notification permission ${permission}`);
        }
      }
      if (deviceNotificationClass() === "desktop") {
        setDesktopNotificationsEnabled(true);
        await syncNotificationState();
        return;
      }
      if (!notificationState.push_supported) {
        throw new Error("mobile notifications require web push in an installed HTTPS web app");
      }
      const reg = await ensureVoiceServiceWorker();
      const publicKey = voiceSettings && voiceSettings.notifications ? voiceSettings.notifications.vapid_public_key : "";
      if (!publicKey) throw new Error("missing VAPID public key");
      let sub = await reg.pushManager.getSubscription();
      if (!sub) {
        sub = await reg.pushManager.subscribe({
          userVisibleOnly: true,
          applicationServerKey: base64UrlToUint8Array(publicKey, atob),
        });
      }
      const snapshot = await api("/api/notifications/subscription", {
        method: "POST",
        body: {
          subscription: sub.toJSON(),
          user_agent: navigatorTarget ? navigatorTarget.userAgent : "",
          device_label: "current-device",
          device_class: deviceNotificationClass(),
        },
      });
      await syncNotificationState(snapshot);
    }

    async function toggleCurrentDeviceNotifications(enabled) {
      if (!notificationState.desktop_supported) {
        throw new Error("notifications require HTTPS or localhost");
      }
      if (deviceNotificationClass() === "desktop") {
        setDesktopNotificationsEnabled(enabled);
        await syncNotificationState();
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
        await syncNotificationState();
        return;
      }
      const snapshot = await api("/api/notifications/subscription/toggle", {
        method: "POST",
        body: {
          endpoint: notificationState.endpoint,
          enabled: !!enabled,
        },
      });
      await syncNotificationState(snapshot);
    }

    async function startLiveAudioPlayback({ resetSource = false } = {}) {
      if (!browserSupportsLiveAudioPlayback(liveAudio, windowTarget)) {
        throw new Error("this browser does not support HLS audio playback in this app");
      }
      if (!liveAudioHasReadySegments()) {
        throw new Error("no live audio segments are available yet; wait for the first announcement and try again");
      }
      const nextSrc = currentVoiceStreamUrl();
      await ensureLiveAudioPlaybackSource(nextSrc, { resetSource });
      await liveAudio.play();
      liveAudioStarted = true;
      liveAudioErrorState = false;
      markLiveAudioProgress();
      updateVoiceUi();
    }

    function describeLiveAudioStartError(error) {
      const message = error && error.message ? String(error.message) : "";
      if (/unsupported/i.test(message)) {
        if (!browserSupportsLiveAudioPlayback(liveAudio, windowTarget)) {
          return "this browser does not support HLS audio playback in this app";
        }
        if (!liveAudioHasReadySegments()) {
          return "no live audio segments are available yet; wait for the first announcement and try again";
        }
      }
      return message || "unknown error";
    }

    function showVoiceSettingsDialog() {
      prepareModalOpen();
      voiceSettingsReturnFocusEl = documentTarget.activeElement instanceof HTMLElement ? documentTarget.activeElement : null;
      voiceSettingsBackdrop.style.display = "block";
      voiceSettingsViewer.style.display = "flex";
      settingsOpen = true;
      updateVoiceUi();
      syncVoiceSettingsFormFromState();
      void loadUnattendedPrompt().then(() => {
        if (isSettingsOpen()) syncVoiceSettingsFormFromState();
      }).catch((e) => {
        console.error("load unattended prompt failed", e);
        voiceSettingsStatus.textContent = `unattended prompt error: ${e && e.message ? e.message : "unknown error"}`;
      });
      if (typeof voiceSettingsViewer.showModal === "function" && !voiceSettingsViewer.open) voiceSettingsViewer.showModal();
      afterModalVisibilityChanged();
    }

    function hideVoiceSettingsDialog() {
      const wasOpen = isModalTargetOpen(voiceSettingsViewer) || settingsOpen;
      const focusTarget = voiceSettingsReturnFocusEl;
      voiceSettingsReturnFocusEl = null;
      voiceSettingsBackdrop.style.display = "none";
      voiceSettingsViewer.style.display = "none";
      voiceSettingsStatus.textContent = "";
      settingsOpen = false;
      if (typeof voiceSettingsViewer.close === "function" && voiceSettingsViewer.open) voiceSettingsViewer.close();
      afterModalVisibilityChanged();
      if (wasOpen && focusTarget && documentTarget.contains(focusTarget) && typeof focusTarget.focus === "function") {
        const restore = () => isModalTargetOpen(voiceSettingsViewer) || settingsOpen;
        restoreModalFocus(focusTarget, restore, requestFrame);
      }
    }

    // --- Event handler wiring (owned by this controller) ---

    announceBtn.onclick = async (e) => {
      e.preventDefault();
      e.stopPropagation();
      const next = !voiceAnnouncementsEnabled();
      if (next && !hasAnnouncementCredentials()) {
        voiceSettingsStatus.textContent = "Set the OpenAI-compatible API base URL and API key before enabling announcements.";
        showVoiceSettingsDialog();
        return;
      }
      setAnnouncementEnabled(next);
      if (!next) return;
      try {
        await maybeAutoStartLiveAudioFromGesture({ resetSource: true });
      } catch (err) {
        console.error("announceBtn auto-start failed", err);
        setToast(`audio start error: ${describeLiveAudioStartError(err)}`);
      }
    };
    notificationBtn.onclick = async (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (!hasNotificationPanel) {
        try {
          await enableNotificationsOnDevice();
        } catch (err) {
          console.error("notification toggle failed", err);
          setNotificationEnabledLocal(false);
          setToast(`notification error: ${err && err.message ? err.message : "unknown error"}`);
        }
        return;
      }
      notificationPanelOpen = !notificationPanelOpen;
      if (notificationPanelOpen) {
        markAllNotificationsRead();
        renderNotificationPanel();
        await pollNotificationFeed();
        return;
      }
      renderNotificationPanel();
    };
    if (notificationClearBtn) {
      notificationClearBtn.onclick = () => {
        markAllNotificationsRead();
        renderNotificationPanel();
      };
    }
    if (notificationEnableBtn) {
      notificationEnableBtn.onclick = async () => {
        try {
          await syncNotificationState();
          if (notificationsEnabledLocally()) {
            await toggleCurrentDeviceNotifications(false);
            setNotificationEnabledLocal(false);
          } else {
            await primeNotificationSound();
            setNotificationEnabledLocal(true);
            await enableNotificationsOnDevice();
          }
        } catch (err) {
          console.error("notification toggle failed", err);
          setNotificationEnabledLocal(false);
          setToast(`notification error: ${err && err.message ? err.message : "unknown error"}`);
        }
        renderNotificationPanel();
      };
    }
    addEvent(liveAudio, "error", () => {
      liveAudioStarted = false;
      liveAudioErrorState = true;
      liveAudioSuspectSinceTs = 0;
      updateVoiceUi();
      scheduleLiveAudioRetry(1200, { resetSource: true });
    });
    addEvent(liveAudio, "playing", () => {
      liveAudioStarted = true;
      liveAudioErrorState = false;
      markLiveAudioProgress();
      updateVoiceUi();
    });
    addEvent(liveAudio, "timeupdate", () => {
      markLiveAudioProgress();
    });
    addEvent(liveAudio, "waiting", () => {
      noteLiveAudioPotentialStall("waiting");
      runLiveAudioWatchdog();
    });
    addEvent(liveAudio, "stalled", () => {
      noteLiveAudioPotentialStall("stalled");
      runLiveAudioWatchdog();
    });
    addEvent(liveAudio, "suspend", () => {
      noteLiveAudioPotentialStall("suspend");
      runLiveAudioWatchdog();
    });
    addEvent(liveAudio, "ended", () => {
      liveAudioStarted = false;
      liveAudioSuspectSinceTs = 0;
      updateVoiceUi();
      scheduleLiveAudioRetry(500, { resetSource: true });
    });
    addEvent(liveAudio, "pause", () => {
      liveAudioStarted = false;
      liveAudioSuspectSinceTs = 0;
      updateVoiceUi();
    });
    narrationSettingToggle.onchange = (e) => {
      voiceSettings.tts_enabled_for_narration = Boolean(e.target.checked);
      scheduleVoiceSave();
    };
    if (unattendedPromptResetBtn) {
      unattendedPromptResetBtn.onclick = () => {
        if (unattendedPromptInput) unattendedPromptInput.value = unattendedPrompt.default_prompt;
      };
    }
    voiceSettingsCloseBtn.onclick = hideVoiceSettingsDialog;
    voiceSettingsCancelBtn.onclick = hideVoiceSettingsDialog;
    voiceSettingsBackdrop.onclick = hideVoiceSettingsDialog;
    addEvent(voiceSettingsViewer, "cancel", (e) => {
      e.preventDefault();
      hideVoiceSettingsDialog();
    });
    // The controller owns the browser's voice runtime. Restoring a visible tab
    // reasserts the listener and restarts a paused announcement stream even if
    // the session/SSE layer has not yet selected or refreshed a conversation.
    addEvent(documentTarget, "visibilitychange", () => {
      if (isAppDisposed() || documentTarget.visibilityState !== "visible") return;
      resumeAnnouncementRuntime({ resetSource: false });
    });
    voiceSettingsSaveBtn.onclick = async () => {
      try {
        voiceSettingsStatus.textContent = "Saving...";
        await saveVoiceSettings();
        await saveUnattendedPrompt();
        await syncNotificationState();
        voiceSettingsStatus.textContent = "";
        hideVoiceSettingsDialog();
      } catch (e) {
        console.error("save voice settings failed", e);
        voiceSettingsStatus.textContent = `save error: ${e && e.message ? e.message : "unknown error"}`;
      }
    };

    function dispose() {
      if (voiceSaveTimer) clearTimeoutFn(voiceSaveTimer);
      voiceSaveTimer = null;
      if (liveAudioRetryTimer) clearTimeoutFn(liveAudioRetryTimer);
      liveAudioRetryTimer = null;
      stopAnnouncementHeartbeat();
      stopLiveAudioWatchdog();
      desktopNotificationTimers.forEach((timer) => clearTimeoutFn(timer));
      desktopNotificationTimers.clear();
      deliveredDesktopNotificationIds.clear();
      resetLiveAudioState();
      while (eventCleanups.length) {
        const cleanup = eventCleanups.pop();
        try {
          cleanup();
        } catch (_error) {}
      }
      settingsOpen = false;
      voiceSettingsReturnFocusEl = null;
      announceBtn.onclick = null;
      notificationBtn.onclick = null;
      if (notificationClearBtn) notificationClearBtn.onclick = null;
      if (notificationEnableBtn) notificationEnableBtn.onclick = null;
      narrationSettingToggle.onchange = null;
      if (unattendedPromptResetBtn) unattendedPromptResetBtn.onclick = null;
      voiceSettingsCloseBtn.onclick = null;
      voiceSettingsCancelBtn.onclick = null;
      voiceSettingsBackdrop.onclick = null;
      voiceSettingsSaveBtn.onclick = null;
      if (notificationAudioContext && typeof notificationAudioContext.close === "function") {
        void notificationAudioContext.close();
      }
      notificationAudioContext = null;
      notificationItems.clear();
      readNotificationIds.clear();
      notificationPanelOpen = false;
      swRegistration = null;
      liveAudioErrorState = false;
    }

    // A browser opt-in survives reloads in local storage while the server's
    // listener registry does not. Register before initial session selection,
    // so an active/new/resumed session cannot finish during that bootstrap gap.
    if (localAnnouncementEnabled) resumeAnnouncementRuntime({ resetSource: false });

    return Object.freeze({
      voiceAnnouncementsEnabled,
      notificationsEnabledLocally,
      isSettingsOpen,
      loadVoiceSettings,
      refreshBackgroundState,
      syncNotificationState,
      pollNotificationFeed,
      resumeAnnouncementRuntime,
      showVoiceSettingsDialog,
      hideVoiceSettingsDialog,
      updateVoiceUi,
      dispose,
    });
  }

export { createVoiceDom, createVoiceController };
