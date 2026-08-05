/* Application composition runtime. This owns the former app.js shell,
 * lifecycle, and controller wiring; app.js intentionally remains only bootstrap. */
(function installCodoxearApplication(global) {
  "use strict";

  function createApplicationController(deps = {}) {
    const window = deps.windowTarget || global;
    const document = deps.documentTarget || window.document;
    if (!window || !document) throw new Error("Codoxear application requires a browser window and document");
    const navigator = deps.navigatorTarget || window.navigator;
    const HTMLElement = deps.HTMLElement || window.HTMLElement;
    const EventSource = deps.EventSource || window.EventSource;
    const AbortController = deps.AbortController || window.AbortController;
    const getComputedStyle = deps.getComputedStyle || window.getComputedStyle.bind(window);
    const requestAnimationFrame = deps.requestAnimationFrame || window.requestAnimationFrame.bind(window);
    const setTimeout = deps.setTimeout || window.setTimeout.bind(window);
    const clearTimeout = deps.clearTimeout || window.clearTimeout.bind(window);

	      const $ = (q) => document.querySelector(q);
	      const UI_VERSION = String(window.CODOXEAR_ASSET_VERSION || "dev");
	      const ATTACH_UPLOAD_MAX_BYTES = (() => {
	        const raw = Number(window.CODOXEAR_ATTACH_MAX_BYTES);
	        if (!Number.isFinite(raw) || raw <= 0) return 16 * 1024 * 1024;
	        return Math.max(1, Math.floor(raw));
	      })();
      const codoxearViewport = window.CodoxearViewport;
      if (
        !codoxearViewport ||
        typeof codoxearViewport.isMobile !== "function" ||
        typeof codoxearViewport.prefersReducedMotion !== "function" ||
        typeof codoxearViewport.useDesktopSessionActions !== "function" ||
        typeof codoxearViewport.useTouchFileEditorControls !== "function" ||
        typeof codoxearViewport.isTextEntryElement !== "function" ||
        typeof codoxearViewport.updateAppHeightVar !== "function"
      )
        throw new Error("Codoxear viewport helpers failed to load");
      function isTextEntryElement(target) {
        return codoxearViewport.isTextEntryElement(target);
      }
      function updateAppHeightVar() {
        return codoxearViewport.updateAppHeightVar();
      }
      updateAppHeightVar();
      window.addEventListener("resize", updateAppHeightVar);
      const codoxearDisplay = window.CodoxearDisplay;
      if (
        !codoxearDisplay ||
        typeof codoxearDisplay.defaultButtonTooltip !== "function" ||
        typeof codoxearDisplay.fmtTs !== "function" ||
        typeof codoxearDisplay.ymd !== "function" ||
        typeof codoxearDisplay.dayLabel !== "function" ||
        typeof codoxearDisplay.time24 !== "function" ||
        typeof codoxearDisplay.fmtBytes !== "function" ||
        typeof codoxearDisplay.baseName !== "function" ||
        typeof codoxearDisplay.shortSessionId !== "function" ||
        typeof codoxearDisplay.sessionDisplayName !== "function" ||
        typeof codoxearDisplay.fmtIdleAge !== "function" ||
        typeof codoxearDisplay.fmtRelativeAge !== "function" ||
        typeof codoxearDisplay.sessionTitleWithId !== "function" ||
        typeof codoxearDisplay.recoveryPromptPreview !== "function" ||
        typeof codoxearDisplay.fuzzyRecentCwdScore !== "function" ||
        typeof codoxearDisplay.compactChatSearchSnippet !== "function" ||
        typeof codoxearDisplay.chatSearchTranscriptHint !== "function" ||
        typeof codoxearDisplay.iconSvg !== "function"
      )
        throw new Error("Codoxear display helpers failed to load");
      function defaultButtonTooltip(attrs = {}, node = null) {
        return codoxearDisplay.defaultButtonTooltip(attrs, node);
      }

      // Voice helpers + the voice/settings/notification/announcement
      // orchestration controller now live in codoxear/static/app_voice.js
      // (loaded after app_voice_helpers.js and before app.js). app.js fails
      // loud here if either module is missing; the controller itself
      // additionally validates every helper API it consumes.
      const codoxearVoiceHelpers = window.CodoxearVoiceHelpers;
      if (
        !codoxearVoiceHelpers ||
        typeof codoxearVoiceHelpers.browserSupportsNativeLiveAudioPlayback !== "function" ||
        typeof codoxearVoiceHelpers.browserSupportsMseLiveAudioPlayback !== "function" ||
        typeof codoxearVoiceHelpers.shouldPreferNativeLiveAudioPlayback !== "function" ||
        typeof codoxearVoiceHelpers.browserSupportsLiveAudioPlayback !== "function" ||
        typeof codoxearVoiceHelpers.base64UrlToUint8Array !== "function" ||
        typeof codoxearVoiceHelpers.isMobileNotificationDevice !== "function" ||
        typeof codoxearVoiceHelpers.notificationDeviceClass !== "function"
      )
        throw new Error("Codoxear voice helpers failed to load");
      const codoxearVoice = window.CodoxearVoice;
      if (!codoxearVoice || typeof codoxearVoice.createVoiceDom !== "function" || typeof codoxearVoice.createVoiceController !== "function")
        throw new Error("Codoxear voice controller failed to load");

      const codoxearDom = window.CodoxearDom;
      if (!codoxearDom || typeof codoxearDom.createElement !== "function") throw new Error("Codoxear DOM helpers failed to load");
      const el = (tag, attrs = {}, children = []) => codoxearDom.createElement(tag, attrs, children, defaultButtonTooltip);
      const codoxearShell = window.CodoxearShell;
      if (!codoxearShell || typeof codoxearShell.createShellDOM !== "function")
        throw new Error("Codoxear shell module failed to load");
      const codoxearSessions = window.CodoxearSessions;
      if (!codoxearSessions || typeof codoxearSessions.createSessionsController !== "function")
        throw new Error("Codoxear sessions controller failed to load");
      const codoxearComposer = window.CodoxearComposer;
      if (!codoxearComposer || typeof codoxearComposer.createComposerController !== "function")
        throw new Error("Codoxear composer module failed to load");
      const codoxearAttachments = window.CodoxearAttachments;
      if (!codoxearAttachments || typeof codoxearAttachments.createAttachmentsController !== "function")
        throw new Error("Codoxear attachments module failed to load");
      const codoxearMessageFlow = window.CodoxearMessageFlow;
      if (!codoxearMessageFlow || typeof codoxearMessageFlow.createMessageFlowController !== "function")
        throw new Error("Codoxear message flow module failed to load");
      const codoxearSecondaryPoll = window.CodoxearSecondaryPoll;
      if (!codoxearSecondaryPoll || typeof codoxearSecondaryPoll.createSecondaryPollController !== "function")
        throw new Error("Codoxear secondary poll controller failed to load");
      const codoxearInterrupt = window.CodoxearInterrupt;
      if (!codoxearInterrupt || typeof codoxearInterrupt.createInterruptController !== "function")
        throw new Error("Codoxear interrupt controller failed to load");
      const codoxearDialogMenus = window.CodoxearDialogMenus;
      if (!codoxearDialogMenus || typeof codoxearDialogMenus.createDialogMenusController !== "function")
        throw new Error("Codoxear dialog menus controller failed to load");
      const codoxearFileEditMode = window.CodoxearFileEditMode;
      if (!codoxearFileEditMode || typeof codoxearFileEditMode.createFileEditModeController !== "function")
        throw new Error("Codoxear file edit mode controller failed to load");
      const codoxearPendingUser = window.CodoxearPendingUser;
      if (!codoxearPendingUser || typeof codoxearPendingUser.createPendingUserController !== "function")
        throw new Error("Codoxear pending user controller failed to load");
      const codoxearNavigationPulse = window.CodoxearNavigationPulse;
      if (!codoxearNavigationPulse || typeof codoxearNavigationPulse.createNavigationPulseController !== "function")
        throw new Error("Codoxear navigation pulse controller failed to load");
      const codoxearFileTouch = window.CodoxearFileTouch;
      if (!codoxearFileTouch || typeof codoxearFileTouch.createFileTouchController !== "function")
        throw new Error("Codoxear file touch controller failed to load");

      const codoxearPerfHelpers = window.CodoxearPerf;
      if (!codoxearPerfHelpers || typeof codoxearPerfHelpers.pushSample !== "function" || typeof codoxearPerfHelpers.summarize !== "function") throw new Error("Codoxear performance helpers failed to load");
      function pushPerfSample(name, valueMs) {
        return codoxearPerfHelpers.pushSample(name, valueMs);
      }
      function summarizePerf() {
        return codoxearPerfHelpers.summarize();
      }

      window.codoxearPerf = summarizePerf;

      const codoxearUrls = window.CodoxearUrls;
      if (
        !codoxearUrls ||
        typeof codoxearUrls.resolveAppUrl !== "function" ||
        typeof codoxearUrls.sessionIdFromHash !== "function" ||
        typeof codoxearUrls.setSessionHash !== "function"
      )
        throw new Error("Codoxear URL helpers failed to load");
      function resolveAppUrl(path) {
        return codoxearUrls.resolveAppUrl(path);
      }
      function versionedShellAssetPath(path) {
        const version = String(window.CODOXEAR_ASSET_VERSION || "").trim();
        if (!version) return path;
        return `${path}?v=${encodeURIComponent(version)}`;
      }

      const codoxearStorage = window.CodoxearStorage;
      if (!codoxearStorage || typeof codoxearStorage.getItem !== "function" || typeof codoxearStorage.setItem !== "function" || typeof codoxearStorage.removeItem !== "function") throw new Error("Codoxear storage helpers failed to load");
      function optionalLocalStorage() {
        return typeof codoxearStorage.optionalLocalStorage === "function" ? codoxearStorage.optionalLocalStorage() : null;
      }
      function storageGetItem(key) {
        return codoxearStorage.getItem(key);
      }
      function storageSetItem(key, value) {
        return codoxearStorage.setItem(key, value);
      }
      function storageRemoveItem(key) {
        return codoxearStorage.removeItem(key);
      }

      let newSessionDefaults = {
        default_backend: "pi",
        backends: {
          codex: null,
          pi: null,
          cc: null,
        },
      };
      let latestSessions = [];
      const codoxearLaunch = window.CodoxearLaunch;
      if (
        !codoxearLaunch ||
        typeof codoxearLaunch.lastProviderKey !== "function" ||
        typeof codoxearLaunch.lastProviderModelKey !== "function" ||
        typeof codoxearLaunch.loadRememberedBackendChoice !== "function" ||
        typeof codoxearLaunch.rememberBackendChoice !== "function" ||
        typeof codoxearLaunch.loadRememberedProviderChoice !== "function" ||
        typeof codoxearLaunch.rememberProviderChoice !== "function" ||
        typeof codoxearLaunch.loadRememberedProviderModelChoice !== "function" ||
        typeof codoxearLaunch.rememberedProviderModelAbsentChoice !== "function" ||
        typeof codoxearLaunch.rememberProviderModelChoice !== "function" ||
        typeof codoxearLaunch.normalizeAgentBackendName !== "function" ||
        typeof codoxearLaunch.agentBackendDisplayName !== "function" ||
        typeof codoxearLaunch.agentBackendLogoPath !== "function" ||
        typeof codoxearLaunch.sessionAgentBackend !== "function" ||
        typeof codoxearLaunch.legacyCodexLaunchDefaults !== "function" ||
        typeof codoxearLaunch.emptyPiLaunchDefaults !== "function" ||
        typeof codoxearLaunch.emptyCcLaunchDefaults !== "function" ||
        typeof codoxearLaunch.defaultsForAgentBackend !== "function" ||
        typeof codoxearLaunch.providerChoicesForBackend !== "function" ||
        typeof codoxearLaunch.reasoningChoicesForBackend !== "function" ||
        typeof codoxearLaunch.backendSupportsFast !== "function" ||
        typeof codoxearLaunch.providerChoiceToSettings !== "function" ||
        typeof codoxearLaunch.sessionProviderChoice !== "function" ||
        typeof codoxearLaunch.modelOptionMatches !== "function" ||
        typeof codoxearLaunch.providerModelDisplay !== "function" ||
        typeof codoxearLaunch.redactedLaunchErrorText !== "function"
      )
        throw new Error("Codoxear launch helpers failed to load");
      const codoxearNewSession = window.CodoxearNewSession;
      if (
        !codoxearNewSession ||
        typeof codoxearNewSession.createNewSessionController !== "function" ||
        typeof codoxearNewSession.createNewSessionDialogController !== "function"
      )
        throw new Error("Codoxear new session controller failed to load");
      function lastProviderKey(backend) {
        return codoxearLaunch.lastProviderKey(backend);
      }
      function lastProviderModelKey(backend) {
        return codoxearLaunch.lastProviderModelKey(backend);
      }
      function loadRememberedBackendChoice() {
        return codoxearLaunch.loadRememberedBackendChoice();
      }
      function rememberBackendChoice(backend) {
        return codoxearLaunch.rememberBackendChoice(backend);
      }
      function loadRememberedProviderChoice(backend) {
        return codoxearLaunch.loadRememberedProviderChoice(backend);
      }
      function rememberProviderChoice(backend, provider) {
        return codoxearLaunch.rememberProviderChoice(backend, provider);
      }
      function loadRememberedProviderModelChoice(backend) {
        return codoxearLaunch.loadRememberedProviderModelChoice(backend);
      }
      function rememberedProviderModelAbsentChoice(value) {
        return codoxearLaunch.rememberedProviderModelAbsentChoice(value);
      }
      function rememberProviderModelChoice(backend, provider, model, options = {}) {
        return codoxearLaunch.rememberProviderModelChoice(backend, provider, model, options);
      }

      const codoxearApi = window.CodoxearApi;
      if (!codoxearApi || typeof codoxearApi.api !== "function" || typeof codoxearApi.apiResponseNotModified !== "function" || typeof codoxearApi.clearApiCache !== "function") throw new Error("Codoxear API helpers failed to load");
      function apiResponseNotModified(obj) {
        return codoxearApi.apiResponseNotModified(obj);
      }
      function clearApiCache() {
        return codoxearApi.clearApiCache();
      }
      async function api(path, options = {}) {
        return codoxearApi.api(path, options);
      }

      function fmtTs(ts) {
        return codoxearDisplay.fmtTs(ts);
      }

      function fmtBytes(n) {
        return codoxearDisplay.fmtBytes(n);
      }

      const codoxearFileHelpers = window.CodoxearFileHelpers;
      if (
        !codoxearFileHelpers ||
        typeof codoxearFileHelpers.listFromFilesField !== "function" ||
        typeof codoxearFileHelpers.stripPathLocationSuffix !== "function" ||
        typeof codoxearFileHelpers.isTextFileKind !== "function" ||
        typeof codoxearFileHelpers.isDiffableFileKind !== "function" ||
        typeof codoxearFileHelpers.blockedFileMessage !== "function" ||
        typeof codoxearFileHelpers.formatPriorityOffset !== "function" ||
        typeof codoxearFileHelpers.fileVideoPreviewErrorText !== "function" ||
        typeof codoxearFileHelpers.fileSearchScore !== "function" ||
        typeof codoxearFileHelpers.normalizeDraftFilePath !== "function" ||
        typeof codoxearFileHelpers.filePickerFoldedSearchText !== "function" ||
        typeof codoxearFileHelpers.filePickerOriginalRangeForFolded !== "function" ||
        typeof codoxearFileHelpers.filePickerMatchRanges !== "function" ||
        typeof codoxearFileHelpers.filePickerMatchRangesForQuery !== "function" ||
        typeof codoxearFileHelpers.filePickerCandidateScore !== "function" ||
        typeof codoxearFileHelpers.compareFilePickerEntries !== "function" ||
        typeof codoxearFileHelpers.normalizeFileCandidateSource !== "function" ||
        typeof codoxearFileHelpers.filePickerSectionLabel !== "function" ||
        typeof codoxearFileHelpers.duplicateFilePickerPaths !== "function" ||
        typeof codoxearFileHelpers.rawByteDuplicatePaths !== "function" ||
        typeof codoxearFileHelpers.filePickerIdentityHint !== "function" ||
        typeof codoxearFileHelpers.filePickerTitle !== "function" ||
        typeof codoxearFileHelpers.positionAfterInsertedText !== "function" ||
        typeof codoxearFileHelpers.fileEditorDeleteCommandForKey !== "function" ||
        typeof codoxearFileHelpers.dataTransferHasFiles !== "function" ||
        typeof codoxearFileHelpers.extractFilesFromClipboardData !== "function" ||
        typeof codoxearFileHelpers.extractFilesFromDropData !== "function" ||
        typeof codoxearFileHelpers.attachmentSafeStem !== "function" ||
        typeof codoxearFileHelpers.attachmentExtensionLower !== "function" ||
        typeof codoxearFileHelpers.attachmentIsLikelyHeic !== "function" ||
        typeof codoxearFileHelpers.attachmentLooksLikeImage !== "function" ||
        typeof codoxearFileHelpers.bytesToBase64 !== "function"
      )
        throw new Error("Codoxear file helpers failed to load");
      function listFromFilesField(val) {
        return codoxearFileHelpers.listFromFilesField(val);
      }

      function listFromFileRecords(val) {
        return codoxearFileHelpers.listFromFileRecords(val);
      }

      function baseName(p) {
        return codoxearDisplay.baseName(p);
      }

      function fuzzyRecentCwdScore(candidate, query) {
        return codoxearDisplay.fuzzyRecentCwdScore(candidate, query);
      }

      function shortSessionId(sid) {
        return codoxearDisplay.shortSessionId(sid);
      }

      function sessionDisplayName(s) {
        return codoxearDisplay.sessionDisplayName(s);
      }

      const SIDEBAR_REASONING_EFFORT_CODES = Object.freeze({
        off: "off",
        minimal: "min",
        low: "low",
        medium: "med",
        high: "hi",
        xhigh: "xh",
        max: "max",
      });

      function sidebarEffortCode(effort, agentBackend) {
        const normalized = typeof effort === "string" ? effort.trim().toLowerCase() : "";
        const code = SIDEBAR_REASONING_EFFORT_CODES[normalized] || "";
        // Claude Code logs carry no effort evidence: the value is launch-time
        // metadata, not verified current state, so mark it instead of
        // presenting it as live.
        if (agentBackend === "cc" && code) return `${code}*`;
        return code;
      }

      function sidebarModelText(s) {
        const model = s && typeof s.model === "string" ? s.model.trim() : "";
        if (!model || model.toLowerCase() === "default") return "";
        if (model.length <= 16) return model;
        const slash = model.indexOf("/");
        if (slash > 0 && slash < model.length - 1) {
          const provider = model.slice(0, slash);
          const modelName = model.slice(slash + 1);
          const suffixBudget = 8;
          if (modelName.length <= suffixBudget) {
            const providerBudget = Math.max(1, 16 - modelName.length - 2);
            return `${provider.slice(0, providerBudget)}…/${modelName}`;
          }
        }
        return `${model.slice(0, 6)}…${model.slice(-8)}`;
      }

      function sessionIdFromHash() {
        return codoxearUrls.sessionIdFromHash();
      }

      function setSessionHash(sessionId) {
        codoxearUrls.setSessionHash(sessionId);
      }

      const codoxearSessionHelpers = window.CodoxearSessionHelpers;
      if (
        !codoxearSessionHelpers ||
        !Array.isArray(codoxearSessionHelpers.SESSION_SIDEBAR_GROUPS) ||
        typeof codoxearSessionHelpers.sessionLaunchFailed !== "function" ||
        typeof codoxearSessionHelpers.sessionLaunchPending !== "function" ||
        typeof codoxearSessionHelpers.sessionLaunchKind !== "function" ||
        typeof codoxearSessionHelpers.sessionLaunchIcon !== "function" ||
        typeof codoxearSessionHelpers.sessionHasUnknownSend !== "function" ||
        typeof codoxearSessionHelpers.sessionIsOrphanRecovery !== "function" ||
        typeof codoxearSessionHelpers.sessionHasOrphanQueueRecovery !== "function" ||
        typeof codoxearSessionHelpers.sessionSidebarGroupKey !== "function" ||
        typeof codoxearSessionHelpers.sidebarSessionEntries !== "function" ||
        typeof codoxearSessionHelpers.sidebarRenderSignature !== "function" ||
        typeof codoxearSessionHelpers.sessionSelectable !== "function" ||
        typeof codoxearSessionHelpers.sessionIsFast !== "function" ||
        typeof codoxearSessionHelpers.diagnosticsProviderDisplay !== "function" ||
        typeof codoxearSessionHelpers.diagnosticsCopyText !== "function" ||
        typeof codoxearSessionHelpers.normalizeQueueItems !== "function"
      )
        throw new Error("Codoxear session helpers failed to load");
      const SESSION_SIDEBAR_GROUPS = codoxearSessionHelpers.SESSION_SIDEBAR_GROUPS;

      function sessionLaunchKind(s) {
        return codoxearSessionHelpers.sessionLaunchKind(s);
      }

      function sessionLaunchIcon(s) {
        return codoxearSessionHelpers.sessionLaunchIcon(s);
      }

      function sessionLaunchFailed(s) {
        return codoxearSessionHelpers.sessionLaunchFailed(s);
      }

      function sessionLaunchPending(s) {
        return codoxearSessionHelpers.sessionLaunchPending(s);
      }

      function sessionHasUnknownSend(s) {
        return codoxearSessionHelpers.sessionHasUnknownSend(s);
      }

      function sessionIsOrphanRecovery(s) {
        return codoxearSessionHelpers.sessionIsOrphanRecovery(s);
      }

      function sessionHasOrphanQueueRecovery(s) {
        return codoxearSessionHelpers.sessionHasOrphanQueueRecovery(s);
      }

      function sessionSidebarGroupKey(s) {
        return codoxearSessionHelpers.sessionSidebarGroupKey(s);
      }

      function sidebarSessionEntries(sessions) {
        return codoxearSessionHelpers.sidebarSessionEntries(sessions);
      }

      function sidebarRenderSignature(entries, { selectedId = "", swipeActions = false } = {}) {
        return codoxearSessionHelpers.sidebarRenderSignature(entries, { selectedId, swipeActions });
      }

      function sessionSelectable(s) {
        return codoxearSessionHelpers.sessionSelectable(s);
      }

      function diagnosticsProviderDisplay(d) {
        return codoxearSessionHelpers.diagnosticsProviderDisplay(d, sessionAgentBackend(d));
      }

      function diagnosticsCopyText(sessionId, rows) {
        return codoxearSessionHelpers.diagnosticsCopyText(sessionId, rows);
      }

      function normalizeQueueItems(data) {
        return codoxearSessionHelpers.normalizeQueueItems(data);
      }

      const codoxearPolling = window.CodoxearPolling;
      if (
        !codoxearPolling ||
        !codoxearPolling.POLLING_INTERVALS ||
        typeof codoxearPolling.sessionsPollDelayMs !== "function" ||
        typeof codoxearPolling.secondaryPollDelayMs !== "function" ||
        typeof codoxearPolling.browserOffline !== "function" ||
        typeof codoxearPolling.messagePollErrorDelayMs !== "function" ||
        typeof codoxearPolling.networkRetryDelayMs !== "function" ||
        typeof codoxearPolling.messagePollDelayMs !== "function" ||
        typeof codoxearPolling.normalizeMessagePollKickDelay !== "function"
      )
        throw new Error("Codoxear polling helpers failed to load");

      const codoxearNetwork = window.CodoxearNetwork;
      if (!codoxearNetwork || typeof codoxearNetwork.createNetworkStatusController !== "function")
        throw new Error("Codoxear network status helpers failed to load");

      const codoxearConversationCopy = window.CodoxearConversationCopy;
      if (
        !codoxearConversationCopy ||
        typeof codoxearConversationCopy.formatConversationForCopy !== "function" ||
        typeof codoxearConversationCopy.formatConversationForCopyResult !== "function" ||
        typeof codoxearConversationCopy.transcriptExportTooLargeCopyMessage !== "function"
      )
        throw new Error("Codoxear conversation-copy helpers failed to load");

      function transcriptExportTooLargeCopyMessage(err) {
        return codoxearConversationCopy.transcriptExportTooLargeCopyMessage(err);
      }

      function copyConversationFailureToast(err) {
        return transcriptExportTooLargeCopyMessage(err) || `copy failed: ${err && err.message ? err.message : "unknown error"}`;
      }

      function normalizeAgentBackendName(value) {
        return codoxearLaunch.normalizeAgentBackendName(value);
      }
      function agentBackendDisplayName(value) {
        return codoxearLaunch.agentBackendDisplayName(value);
      }
      function agentBackendLogoPath(value) {
        return codoxearLaunch.agentBackendLogoPath(value);
      }
      function sessionAgentBackend(session) {
        return codoxearLaunch.sessionAgentBackend(session);
      }
      function legacyCodexLaunchDefaults(seed = {}) {
        return codoxearLaunch.legacyCodexLaunchDefaults(seed);
      }
      function emptyPiLaunchDefaults(seed = {}) {
        return codoxearLaunch.emptyPiLaunchDefaults(seed);
      }
      function emptyCcLaunchDefaults(seed = {}) {
        return codoxearLaunch.emptyCcLaunchDefaults(seed);
      }
      function defaultsForAgentBackend(backend) {
        return codoxearLaunch.defaultsForAgentBackend(backend, newSessionDefaults);
      }
      function providerChoicesForBackend(backend) {
        return codoxearLaunch.providerChoicesForBackend(backend, newSessionDefaults);
      }
      function reasoningChoicesForBackend(backend, options = {}) {
        return codoxearLaunch.reasoningChoicesForBackend(backend, newSessionDefaults, options);
      }
      function backendSupportsFast(backend) {
        return codoxearLaunch.backendSupportsFast(backend, newSessionDefaults);
      }

      function redactedLaunchErrorText(value) {
        return codoxearLaunch.redactedLaunchErrorText(value);
      }

      function sessionLaunchLabel(s) {
        const kind = sessionLaunchKind(s);
        if (kind === "failed") return redactedLaunchErrorText(s && s.launch_error) || "session launch failed";
        if (kind === "web_tmux") return "web-owned tmux session";
        return kind === "web" ? "web-owned session" : "terminal-owned session";
      }

      function sessionIsFast(s) {
        return codoxearSessionHelpers.sessionIsFast(s);
      }

      function providerChoiceToSettings(choice, agentBackend = "codex") {
        return codoxearLaunch.providerChoiceToSettings(choice, agentBackend);
      }
      function sessionProviderChoice(session) {
        return codoxearLaunch.sessionProviderChoice(session);
      }
      function modelOptionMatches(option, query) {
        return codoxearLaunch.modelOptionMatches(option, query);
      }
      function providerModelDisplay(model, providerChoice = "", options = {}) {
        return codoxearLaunch.providerModelDisplay(model, providerChoice, options);
      }

	      function fmtIdleAge(seconds) {
        return codoxearDisplay.fmtIdleAge(seconds);
      }

	      function fmtRelativeAge(seconds) {
        return codoxearDisplay.fmtRelativeAge(seconds);
      }

      function sessionTitleWithId(s) {
        return codoxearDisplay.sessionTitleWithId(s);
      }

      function stripPathLocationSuffix(rawPath) {
        return codoxearFileHelpers.stripPathLocationSuffix(rawPath);
      }

      function isTextFileKind(kind) {
        return codoxearFileHelpers.isTextFileKind(kind);
      }

      function isDiffableFileKind(kind) {
        return codoxearFileHelpers.isDiffableFileKind(kind);
      }

      function blockedFileMessage(rel, reason, viewerMaxBytes, size) {
        return codoxearFileHelpers.blockedFileMessage(rel, reason, viewerMaxBytes, size);
      }

      function formatPriorityOffset(value) {
        return codoxearFileHelpers.formatPriorityOffset(value);
      }

      function fileSearchScore(candidate, query) {
        return codoxearFileHelpers.fileSearchScore(candidate, query);
      }

      function normalizeDraftFilePath(raw) {
        return codoxearFileHelpers.normalizeDraftFilePath(raw);
      }

      function filePickerFoldedSearchText(text) {
        return codoxearFileHelpers.filePickerFoldedSearchText(text);
      }

      function filePickerOriginalRangeForFolded(mapped, start, end) {
        return codoxearFileHelpers.filePickerOriginalRangeForFolded(mapped, start, end);
      }

      function filePickerMatchRanges(text, query) {
        return codoxearFileHelpers.filePickerMatchRanges(text, query);
      }

      function filePickerMatchRangesForQuery(text, query) {
        return codoxearFileHelpers.filePickerMatchRangesForQuery(text, query);
      }

      function filePickerCandidateScore(path, query) {
        return codoxearFileHelpers.filePickerCandidateScore(path, query);
      }

      function compareFilePickerEntries(a, b) {
        return codoxearFileHelpers.compareFilePickerEntries(a, b);
      }

      function normalizeFileCandidateSource(source) {
        return codoxearFileHelpers.normalizeFileCandidateSource(source);
      }

      function filePickerSectionLabel(source) {
        return codoxearFileHelpers.filePickerSectionLabel(source);
      }

      function duplicateFilePickerPaths(entries) {
        return codoxearFileHelpers.duplicateFilePickerPaths(entries);
      }

      function rawByteDuplicatePaths(entries) {
        return codoxearFileHelpers.rawByteDuplicatePaths(entries);
      }

      function filePickerIdentityHint(entry, duplicatePaths, options) {
        return codoxearFileHelpers.filePickerIdentityHint(entry, duplicatePaths, options);
      }

      function filePickerTitle(entry, hint = "") {
        return codoxearFileHelpers.filePickerTitle(entry, hint);
      }

      function dataTransferHasFiles(data) {
        return codoxearFileHelpers.dataTransferHasFiles(data);
      }

      function extractFilesFromClipboardData(data) {
        return codoxearFileHelpers.extractFilesFromClipboardData(data);
      }

      function extractFilesFromDropData(data) {
        return codoxearFileHelpers.extractFilesFromDropData(data);
      }

      function safeAttachmentStem(name) {
        return codoxearFileHelpers.attachmentSafeStem(name);
      }

      function isLikelyHeic(file) {
        return codoxearFileHelpers.attachmentIsLikelyHeic(file);
      }

      function looksLikeImage(file) {
        return codoxearFileHelpers.attachmentLooksLikeImage(file);
      }

      function b64FromBytes(bytes) {
        return codoxearFileHelpers.bytesToBase64(bytes, btoa);
      }

      const codoxearFilePicker = window.CodoxearFilePicker;
      if (
        !codoxearFilePicker ||
        typeof codoxearFilePicker.appendDraftFileMenuItem !== "function" ||
        typeof codoxearFilePicker.appendFilePickerEntryItem !== "function" ||
        typeof codoxearFilePicker.appendFilePickerSection !== "function" ||
        typeof codoxearFilePicker.appendFilePickerStatusRow !== "function" ||
        typeof codoxearFilePicker.appendHighlightedFileMenuPath !== "function" ||
        typeof codoxearFilePicker.createEntryRuntime !== "function" ||
        typeof codoxearFilePicker.createInputRuntime !== "function" ||
        typeof codoxearFilePicker.createMenuDomRuntime !== "function" ||
        typeof codoxearFilePicker.createMenuRenderRuntime !== "function" ||
        typeof codoxearFilePicker.createMenuState !== "function" ||
        typeof codoxearFilePicker.createSearchState !== "function" ||
        typeof codoxearFilePicker.localFilePickerSearchEntries !== "function" ||
        typeof codoxearFilePicker.visibleFilePickerEntries !== "function"
      )
        throw new Error("Codoxear file picker helpers failed to load");

      const codoxearFileViewer = window.CodoxearFileViewer;
      if (
        !codoxearFileViewer ||
        typeof codoxearFileViewer.bindFileTouchClick !== "function" ||
        typeof codoxearFileViewer.bindFileTouchPress !== "function" ||
        typeof codoxearFileViewer.createFileDownloadRuntime !== "function" ||
        typeof codoxearFileViewer.createFileFallbackRuntime !== "function" ||
        typeof codoxearFileViewer.createFileInspectRuntime !== "function" ||
        typeof codoxearFileViewer.createFileLoadResultRuntime !== "function" ||
        typeof codoxearFileViewer.createFileCandidateRefreshRuntime !== "function" ||
        typeof codoxearFileViewer.createFileViewerPanelRuntime !== "function" ||
        typeof codoxearFileViewer.createFileViewerLifecycleRuntime !== "function" ||
        typeof codoxearFileViewer.createFileModeControlsRuntime !== "function" ||
        typeof codoxearFileViewer.createFilePasteDialogRuntime !== "function" ||
        typeof codoxearFileViewer.createFilePdfRenderRuntime !== "function" ||
        typeof codoxearFileViewer.createFileReferenceRuntime !== "function" ||
        typeof codoxearFileViewer.createFileRenderSurfaceRuntime !== "function" ||
        typeof codoxearFileViewer.createOpenedFileRuntime !== "function" ||
        typeof codoxearFileViewer.createFileTouchToolbarRuntime !== "function" ||
        typeof codoxearFileViewer.createFileUnsavedDialogRuntime !== "function" ||
        typeof codoxearFileViewer.createFileViewerModalRuntime !== "function" ||
        typeof codoxearFileViewer.createFileViewerController !== "function" ||
        typeof codoxearFileViewer.createFileVideoPreviewRuntime !== "function" ||
        typeof codoxearFileViewer.createPdfLoader !== "function"
      )
        throw new Error("Codoxear file viewer controller failed to load");

      const codoxearFileEditor = window.CodoxearFileEditor;
      if (
        !codoxearFileEditor ||
        typeof codoxearFileEditor.createFileEditorRuntime !== "function" ||
        typeof codoxearFileEditor.createFileEditorRenderer !== "function" ||
        typeof codoxearFileEditor.createMonacoLoader !== "function"
      )
        throw new Error("Codoxear file editor runtime failed to load");

      const codoxearMarkdown = window.CodoxearMarkdown;
      if (
        !codoxearMarkdown ||
        typeof codoxearMarkdown.normalizeLineNumber !== "function" ||
        typeof codoxearMarkdown.parseLocalFileRef !== "function" ||
        typeof codoxearMarkdown.isMarkdownPreviewable !== "function" ||
        typeof codoxearMarkdown.markdownPreviewHtml !== "function" ||
        typeof codoxearMarkdown.chatMarkdownHtmlCached !== "function"
      )
        throw new Error("Codoxear markdown helpers failed to load");
      function normalizeLineNumber(value) {
        return codoxearMarkdown.normalizeLineNumber(value);
      }
      function parseLocalFileRef(rawValue) {
        return codoxearMarkdown.parseLocalFileRef(rawValue);
      }
      function isMarkdownPreviewable(path) {
        return codoxearMarkdown.isMarkdownPreviewable(path);
      }
      function markdownPreviewHtml(src, options = {}) {
        return codoxearMarkdown.markdownPreviewHtml(src, options);
      }
      function chatMarkdownHtmlCached(src, sessionId) {
        return codoxearMarkdown.chatMarkdownHtmlCached(src, sessionId);
      }

      function iconSvg(name) {
        return codoxearDisplay.iconSvg(name);
      }

      let activeAppCleanup = null;
      function cleanupActiveApp() {
        if (typeof activeAppCleanup !== "function") return;
        const cleanup = activeAppCleanup;
        activeAppCleanup = null;
        cleanup();
      }

      function renderLogin(onAuthed) {
        cleanupActiveApp();
        const root = $("#root");
        root.innerHTML = "";
        const err = el("div", { class: "err", id: "loginError", role: "alert" });
        const pwInput = el("input", {
          type: "password",
          id: "pw",
          name: "password",
          placeholder: "Password",
          "aria-label": "Password",
          autocomplete: "current-password",
          "aria-describedby": "loginError",
        });
        const loginBtn = el("button", { class: "primary", id: "loginBtn", type: "submit", text: "Login" });
        const wrap = el("div", { class: "loginWrap" });
        const form = el("form", { class: "login", id: "loginForm" }, [
          el("h1", { text: "Codoxear login" }),
          el("label", { class: "sr-only", for: "pw", text: "Password" }),
          el("div", { class: "row2" }, [
            pwInput,
            loginBtn,
            err,
          ]),
        ]);
        wrap.appendChild(form);
        root.appendChild(wrap);
        form.onsubmit = async (e) => {
          e.preventDefault();
          err.textContent = "";
          const pw = pwInput.value;
          try {
            await api("/api/login", { method: "POST", body: { password: pw } });
          onAuthed();
          } catch (e2) {
          console.error("login app initialization failed", e2);
          err.textContent = e2.obj?.error || e2.message;
          }
        };
        pwInput.focus();
        if (typeof window.__codoxearMarkBootstrapped === "function") window.__codoxearMarkBootstrapped();
      }

	      const codoxearApplicationComposition = window.CodoxearApplicationComposition;
      if (!codoxearApplicationComposition || typeof codoxearApplicationComposition.createApplicationComposition !== "function")
        throw new Error("Codoxear application composition failed to load");
      const applicationComposition = codoxearApplicationComposition.createApplicationComposition({
        window, document, navigator, HTMLElement, EventSource, AbortController, getComputedStyle,
        requestAnimationFrame, setTimeout, clearTimeout, $, UI_VERSION, ATTACH_UPLOAD_MAX_BYTES,
        codoxearViewport, codoxearDisplay, defaultButtonTooltip, codoxearVoiceHelpers, codoxearVoice,
        codoxearDom, el, codoxearShell, codoxearSessions, codoxearComposer, codoxearAttachments,
        codoxearMessageFlow, codoxearSecondaryPoll, codoxearInterrupt, codoxearDialogMenus,
        codoxearFileEditMode, codoxearPendingUser, codoxearNavigationPulse,
        codoxearFileTouch, codoxearPerfHelpers, pushPerfSample, summarizePerf, codoxearUrls,
        resolveAppUrl, versionedShellAssetPath, codoxearStorage, optionalLocalStorage, storageGetItem,
        storageSetItem, storageRemoveItem, codoxearLaunch, codoxearNewSession, lastProviderKey,
        lastProviderModelKey, loadRememberedBackendChoice, rememberBackendChoice,
        loadRememberedProviderChoice, rememberProviderChoice, loadRememberedProviderModelChoice,
        rememberedProviderModelAbsentChoice, rememberProviderModelChoice, codoxearApi,
        apiResponseNotModified, clearApiCache, api, fmtTs, fmtBytes, codoxearFileHelpers,
        listFromFilesField, listFromFileRecords, baseName, fuzzyRecentCwdScore, shortSessionId,
        sessionDisplayName, sidebarEffortCode, sidebarModelText, sessionIdFromHash, setSessionHash,
        codoxearSessionHelpers, sessionLaunchKind, sessionLaunchIcon, sessionLaunchFailed,
        sessionLaunchPending, sessionHasUnknownSend, sessionIsOrphanRecovery,
        sessionHasOrphanQueueRecovery, sessionSidebarGroupKey, sidebarSessionEntries,
        sidebarRenderSignature, sessionSelectable, diagnosticsProviderDisplay, diagnosticsCopyText,
        normalizeQueueItems, codoxearPolling, codoxearNetwork, codoxearConversationCopy,
        transcriptExportTooLargeCopyMessage, copyConversationFailureToast, normalizeAgentBackendName,
        agentBackendDisplayName, agentBackendLogoPath, sessionAgentBackend, legacyCodexLaunchDefaults,
        emptyPiLaunchDefaults, emptyCcLaunchDefaults, defaultsForAgentBackend, providerChoicesForBackend,
        reasoningChoicesForBackend, backendSupportsFast, redactedLaunchErrorText, sessionLaunchLabel,
        sessionIsFast, providerChoiceToSettings, sessionProviderChoice, modelOptionMatches,
        providerModelDisplay, fmtIdleAge, fmtRelativeAge, sessionTitleWithId, stripPathLocationSuffix,
        isTextFileKind, isDiffableFileKind, blockedFileMessage, formatPriorityOffset, fileSearchScore,
        normalizeDraftFilePath, filePickerFoldedSearchText, filePickerOriginalRangeForFolded,
        filePickerMatchRanges, filePickerMatchRangesForQuery, filePickerCandidateScore,
        compareFilePickerEntries, normalizeFileCandidateSource, filePickerSectionLabel,
        duplicateFilePickerPaths, rawByteDuplicatePaths, filePickerIdentityHint, filePickerTitle,
        dataTransferHasFiles, extractFilesFromClipboardData, extractFilesFromDropData, safeAttachmentStem,
        isLikelyHeic, looksLikeImage, b64FromBytes, codoxearFilePicker, codoxearFileViewer,
        codoxearFileEditor, codoxearMarkdown, normalizeLineNumber, parseLocalFileRef,
        isMarkdownPreviewable, markdownPreviewHtml, chatMarkdownHtmlCached, iconSvg,
        cleanupActiveApp, renderLogin,
        setActiveAppCleanup: (cleanup) => { activeAppCleanup = cleanup; },
        clearActiveAppCleanup: (cleanup) => { if (activeAppCleanup === cleanup) activeAppCleanup = null; },
      });
      function renderApp() {
        return applicationComposition.renderApp();
      }


    return Object.freeze({ api, cleanupActiveApp, renderApp, renderLogin });
  }

  global.CodoxearApplication = Object.freeze({ createApplicationController });
})(window);
