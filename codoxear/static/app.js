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
      if (!codoxearVoice || typeof codoxearVoice.createVoiceController !== "function")
        throw new Error("Codoxear voice controller failed to load");

      const codoxearDom = window.CodoxearDom;
      if (!codoxearDom || typeof codoxearDom.createElement !== "function") throw new Error("Codoxear DOM helpers failed to load");
      const el = (tag, attrs = {}, children = []) => codoxearDom.createElement(tag, attrs, children, defaultButtonTooltip);
      const codoxearShell = window.CodoxearShell;
      if (
        !codoxearShell ||
        typeof codoxearShell.createShellDOM !== "function" ||
        typeof codoxearShell.createSidebarController !== "function"
      )
        throw new Error("Codoxear shell module failed to load");
      const codoxearComposer = window.CodoxearComposer;
      if (!codoxearComposer || typeof codoxearComposer.createComposerController !== "function")
        throw new Error("Codoxear composer module failed to load");

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

      let newSessionBackend = "pi";
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
        typeof codoxearNewSession.createNewSessionController !== "function"
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

      const REASONING_EFFORT_MARKERS = Object.freeze({
        xhigh: "X",
        high: "H",
        medium: "M",
        low: "L",
        max: "M+",
        minimal: "m",
        off: "–",
      });

      function reasoningEffortMarker(effortTxt) {
        return REASONING_EFFORT_MARKERS[effortTxt] || "";
      }

      function sidebarModelText(s) {
        const model = s && typeof s.model === "string" ? s.model.trim() : "";
        return model && model.toLowerCase() !== "default" ? model : "";
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
        typeof codoxearPolling.messagePollDelayMs !== "function" ||
        typeof codoxearPolling.normalizeMessagePollKickDelay !== "function"
      )
        throw new Error("Codoxear polling helpers failed to load");

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
            err.textContent = e2.obj?.error || e2.message;
          }
        };
        pwInput.focus();
        if (typeof window.__codoxearMarkBootstrapped === "function") window.__codoxearMarkBootstrapped();
      }

	      function renderApp() {
            cleanupActiveApp();
	        const root = $("#root");
        const shellDOM = codoxearShell.createShellDOM({
          root,
          el,
          iconSvg,
          resolveAppUrl,
          versionedShellAssetPath,
        });
        const {
          app,
          backdrop,
          sessionsWrap,
          sidebarEmptyHint,
          chatWrap,
          chatEmptyState,
          chat,
          chatInner,
          olderWrap,
          olderBtn,
          olderRetryBtn,
          olderError,
          olderErrorText,
          bottomSentinel,
          jumpBtn,
          chatTimeChip,
          chatSearchInput,
          chatSearchPrevBtn,
          chatSearchNextBtn,
          chatSearchCloseBtn,
          chatSearchStatus,
          chatSearchAllHintEl,
          chatSearchBar,
          chatNavRail,
          titleLabel,
          statusChip,
          ctxChip,
          interruptBtn,
          toast,
          toggleSidebarBtn,
          unattendedBtn,
          announceBtn,
          notificationBtn,
          diagBtn,
          prevUserBtn,
          nextUserBtn,
          chatSearchBtn,
          fileBtn,
          unattendedMenu,
          liveAudio,
          composer,
          form,
          textarea,
          msgPh,
          modelPicker,
          imgInput,
          attachBtn,
          queueBtn,
          composerStopBtn,
          sendBtn,
        } = shellDOM.elements;

        let selected = null;
        let pendingHashSessionId = "";
        let pendingHashSessionSelectInFlight = false;
        const INIT_PAGE_LIMIT_DESKTOP = 60;
        const INIT_PAGE_LIMIT_MOBILE = 24;
        const OLDER_PAGE_LIMIT = 60;
        const CHAT_DOM_WINDOW = 260;
        const CHAT_DOM_WINDOW_WITH_HISTORY_SLACK = CHAT_DOM_WINDOW + OLDER_PAGE_LIMIT;
        const OLDER_TOP_TRIGGER_PX = 1;
        const OLDER_CANCEL_PX = 48;
        let openSessionTailAbortController = null;
        let messagePollAbortController = null;
        let messageEventSource = null;
        let messageSseRetryTimer = null;
        let messageSseOpen = false;
        let messageSseFallbackUntil = 0;
        const OLDER_AUTO_COOLDOWN_MS = 450;
        let pollTimer = null;
        let pollGen = 0;
        let pollLoopBusy = false;
        let pollKickPending = false;
        let pollKickDelayMs = null;
        let messagePollErrorStreak = 0;
	        let pollFastUntilMs = 0;
	         let turnOpen = false;
	         let sessionsTimer = null;
         let secondaryPollTimer = null;
         let sessionsPollingEnabled = true;
         let secondaryPollingEnabled = true;
         let currentRunning = false;
         let sessionsRefreshInFlight = null;
         let sessionsRefreshQueued = false;
	        let sessionIndex = new Map(); // session_id -> session info
        let recentCwds = [];
	        let sending = false;
	        let attachedFiles = 0;
        let stagedAttachments = [];
        let composerController = null;
        function resizeComposer() {
          if (composerController) composerController.autoGrow();
        }
        function clearComposerInput() {
          if (composerController) composerController.clearComposer();
        }
        function saveSelectedComposerDraft(sessionId) {
          if (composerController) composerController.saveSessionDraft(sessionId);
        }
        function loadSelectedComposerDraft(sessionId) {
          if (composerController) composerController.loadSessionDraft(sessionId);
        }
        function syncComposerSendButton() {
          if (composerController) composerController.syncSendButtonState();
        }
        function closeSendChoiceDialog(options) {
          if (composerController) composerController.hideSendChoice(options);
        }
				    let lastToken = null;
        let attachBadgeEl = null;
        let queueBadgeEl = null;
        let editDependencyMenuOpen = false;
        let newSessionCwdMenuOpen = false;
        let newSessionCwdMenuFocus = -1;
        let newSessionModelMenuOpen = false;
        let newSessionModelMenuFocus = -1;
        let newSessionReasoningMenuOpen = false;
        let newSessionResumeMenuOpen = false;
        let newSessionStartBusy = false;
        let newSessionLiteralModelInputValue = "";
        let newSessionLaunchPresetProviderAbsent = false;
        newSessionBackend = "pi";
        let newSessionProvider = "chatgpt";
        let newSessionFast = false;
        newSessionDefaults = {
          default_backend: "pi",
          backends: {
            codex: legacyCodexLaunchDefaults(),
            pi: emptyPiLaunchDefaults(),
            cc: emptyCcLaunchDefaults(),
          },
        };
        latestSessions = [];
        let tmuxAvailable = false;
                 let clickLoadT0 = 0;
                 let clickMetricPending = false;
              // Unattended menu state, cfg cache, number-input drafts, and the
              // per-session save timers/in-flight/pending maps live in the
              // CodoxearUnattended controller (codoxear/static/app_unattended.js).
              let editSessionId = null;
        let appDisposed = false;
        const appEventCleanups = [];
        function addAppEvent(target, type, handler, options) {
          if (!target || typeof target.addEventListener !== "function") return handler;
          target.addEventListener(type, handler, options);
          appEventCleanups.push(() => target.removeEventListener(type, handler, options));
          return handler;
        }
        function closeMessageEventSource() {
          if (messageSseRetryTimer) clearTimeout(messageSseRetryTimer);
          messageSseRetryTimer = null;
          const source = messageEventSource;
          messageEventSource = null;
          messageSseOpen = false;
          if (source && typeof source.close === "function") {
            try { source.close(); } catch (_error) {}
          }
        }
        function scheduleMessageEventSourceRetry(sessionId, gen) {
          if (appDisposed || selected !== sessionId || pollGen !== gen || messageSseRetryTimer) return;
          const delay = Math.max(1000, messageSseFallbackUntil - Date.now());
          messageSseRetryTimer = setTimeout(() => {
            messageSseRetryTimer = null;
            if (!appDisposed && selected === sessionId && pollGen === gen) openMessageEventSource(sessionId, gen);
          }, delay);
        }
        function openMessageEventSource(sessionId = selected, gen = pollGen) {
          if (appDisposed || !sessionId || selected !== sessionId || pollGen !== gen) return;
          if (typeof EventSource !== "function") return;
          const snapshot = transcriptSlotRuntime.activeSnapshot();
          if (snapshot.state !== "bound" || !snapshot.liveCursor) return;
          closeMessageEventSource();
          const url = resolveAppUrl(`/api/sessions/${sessionId}/live?cursor=${encodeURIComponent(snapshot.liveCursor)}`);
          const source = new EventSource(url);
          messageEventSource = source;
          source.onopen = () => {
            if (messageEventSource !== source || selected !== sessionId || pollGen !== gen) return;
            messageSseOpen = true;
            messageSseFallbackUntil = 0;
            messagePollErrorStreak = 0;
            abortMessagePollRequest();
            if (pollTimer) clearTimeout(pollTimer);
            pollTimer = null;
          };
          source.addEventListener("message", (event) => {
            if (messageEventSource !== source || selected !== sessionId || pollGen !== gen) return;
            let data;
            try { data = JSON.parse(event.data); } catch (error) {
              console.warn("message SSE payload was invalid", error);
              return;
            }
            Promise.resolve(applyLiveMessageData(sessionId, gen, data)).catch((error) => {
              console.warn("message SSE update failed", error);
              source.close();
              if (messageEventSource === source) {
                messageEventSource = null;
                messageSseOpen = false;
                messageSseFallbackUntil = Date.now() + 6000;
                kickPoll(0);
                scheduleMessageEventSourceRetry(sessionId, gen);
              }
            });
          });
          source.addEventListener("error", () => {
            if (messageEventSource !== source || selected !== sessionId || pollGen !== gen) return;
            source.close();
            messageEventSource = null;
            messageSseOpen = false;
            messageSseFallbackUntil = Date.now() + 6000;
            markMessagePollFailure();
            kickPoll(0);
            scheduleMessageEventSourceRetry(sessionId, gen);
          });
        }
        function stopMessagePolling() {
          selected = null;
          pollGen += 1;
          closeMessageEventSource();
          abortOpenSessionTailRequest();
          abortMessagePollRequest();
          if (pollTimer) clearTimeout(pollTimer);
          pollTimer = null;
          pollKickPending = false;
          pollKickDelayMs = null;
          messagePollErrorStreak = 0;
          pollFastUntilMs = 0;
          turnOpen = false;
        }
        function abortController(controller) {
          if (!controller || typeof controller.abort !== "function") return;
          try {
            controller.abort();
          } catch (_error) {}
        }
        function abortOpenSessionTailRequest() {
          const controller = openSessionTailAbortController;
          openSessionTailAbortController = null;
          abortController(controller);
        }
        function beginOpenSessionTailRequest(sessionId, gen) {
          abortOpenSessionTailRequest();
          const controller = typeof AbortController === "function" ? new AbortController() : null;
          openSessionTailAbortController = controller;
          return Object.freeze({ sessionId, gen, controller, signal: controller ? controller.signal : undefined });
        }
        function isCurrentOpenSessionTailRequest(request) {
          return Boolean(request && selected === request.sessionId && pollGen === request.gen);
        }
        function isOpenSessionTailAbortError(request, error) {
          return Boolean(error && error.name === "AbortError" && request && request.signal && request.signal.aborted);
        }
        function finishOpenSessionTailRequest(request) {
          if (request && openSessionTailAbortController === request.controller) openSessionTailAbortController = null;
        }
        function abortMessagePollRequest() {
          const controller = messagePollAbortController;
          messagePollAbortController = null;
          abortController(controller);
        }
        function beginMessagePollRequest(sessionId, gen) {
          abortMessagePollRequest();
          const controller = typeof AbortController === "function" ? new AbortController() : null;
          messagePollAbortController = controller;
          return Object.freeze({ sessionId, gen, controller, signal: controller ? controller.signal : undefined });
        }
        function isMessagePollAbortError(request, error) {
          return Boolean(error && error.name === "AbortError" && request && request.signal && request.signal.aborted);
        }
        function finishMessagePollRequest(request) {
          if (request && messagePollAbortController === request.controller) messagePollAbortController = null;
        }
        function cleanupApp() {
          if (appDisposed) return;
          appDisposed = true;
          sessionsPollingEnabled = false;
          secondaryPollingEnabled = false;
          stopMessagePolling();
          stopAllPolling();
          if (newSessionController) newSessionController.disposeResumeLoadTimer();
          if (voiceController) voiceController.dispose();
          if (unattendedController) unattendedController.dispose();
          filePickerSearchState.dispose();
          if (iosViewportGuardTimer) clearTimeout(iosViewportGuardTimer);
          iosViewportGuardTimer = null;
          if (chatSearchController) chatSearchController.dispose();
          if (queueController) queueController.dispose();
          if (diagController) diagController.dispose();
          if (chatNavigationController) chatNavigationController.dispose();
          if (hintModeController) hintModeController.dispose();
          olderLoadRuntime.invalidate();
          fileViewerController.abortPendingFileOpenTransport();
          hideUnattendedMenu();
          hideFilePasteDialog();
          hideFileUnsavedDialog("cancel");
          closeSendChoiceDialog();
          if (composerController) composerController.dispose();
          sidebarController.dispose();
          while (appEventCleanups.length) {
            const cleanup = appEventCleanups.pop();
            try {
              cleanup();
            } catch (_error) {}
          }
          clearApiCache();
          shellDOM.cleanup();
          if (activeAppCleanup === cleanupApp) activeAppCleanup = null;
        }
        function handleAppAuthLoss() {
          if (appDisposed) return;
          cleanupApp();
          renderLogin(renderApp);
        }
        function sessionsPollDelayMs() {
          return codoxearPolling.sessionsPollDelayMs(document.visibilityState);
        }
        function secondaryPollDelayMs() {
          return codoxearPolling.secondaryPollDelayMs(document.visibilityState);
        }
        function browserOffline() {
          return codoxearPolling.browserOffline(typeof navigator === "undefined" ? undefined : navigator);
        }
        function messagePollErrorDelayMs() {
          return codoxearPolling.messagePollErrorDelayMs(messagePollErrorStreak);
        }
        function messagePollDelayMs(now = Date.now()) {
          return codoxearPolling.messagePollDelayMs({
            now,
            visibilityState: document.visibilityState,
            offline: browserOffline(),
            errorStreak: messagePollErrorStreak,
            pollFastUntilMs,
            turnOpen,
          });
        }
        function markMessagePollSuccess() {
          messagePollErrorStreak = 0;
        }
        function markMessagePollFailure() {
          messagePollErrorStreak = Math.min(messagePollErrorStreak + 1, 20);
        }
        function normalizeMessagePollKickDelay(ms = 0) {
          return codoxearPolling.normalizeMessagePollKickDelay({
            requested: ms,
            visibilityState: document.visibilityState,
            offline: browserOffline(),
            errorStreak: messagePollErrorStreak,
            pollFastUntilMs,
            turnOpen,
          });
        }
        function stopSessionsPolling() {
          if (sessionsTimer) clearTimeout(sessionsTimer);
          sessionsTimer = null;
        }
        function stopSecondaryPolling() {
          if (secondaryPollTimer) clearTimeout(secondaryPollTimer);
          secondaryPollTimer = null;
        }
        function stopAllPolling() {
          stopSessionsPolling();
          stopSecondaryPolling();
        }
        async function runSessionsPollTick() {
          if (appDisposed || !sessionsPollingEnabled) return;
          try {
            await refreshSessions();
          } catch (e2) {
            if (e2 && e2.status === 401) {
              handleAppAuthLoss();
              return;
            }
            console.error("refreshSessions timer failed", e2);
          }
          scheduleSessionsPoll();
        }
        async function runSecondaryPollTick() {
          if (appDisposed || !secondaryPollingEnabled) return;
          try {
            await loadVoiceSettings();
            await syncNotificationState();
            if (notificationsEnabledLocally()) await pollNotificationFeed();
          } catch (e2) {
            if (e2 && e2.status === 401) {
              handleAppAuthLoss();
              return;
            }
            console.error("secondary poll failed", e2);
          }
          scheduleSecondaryPoll();
        }
        function scheduleSessionsPoll(delayMs = sessionsPollDelayMs()) {
          if (appDisposed || !sessionsPollingEnabled) return;
          stopSessionsPolling();
          sessionsTimer = setTimeout(() => {
            sessionsTimer = null;
            void runSessionsPollTick();
          }, Math.max(0, Number(delayMs) || 0));
        }
        function scheduleSecondaryPoll(delayMs = secondaryPollDelayMs()) {
          if (appDisposed || !secondaryPollingEnabled) return;
          stopSecondaryPolling();
          secondaryPollTimer = setTimeout(() => {
            secondaryPollTimer = null;
            void runSecondaryPollTick();
          }, Math.max(0, Number(delayMs) || 0));
        }

            titleLabel.style.cursor = "default";
            titleLabel.title = "No session selected";
            titleLabel.onclick = () => {
              if (!selected) return;
              openEditSession(selected);
            };
            titleLabel.onkeydown = (e) => {
              if (!selected) return;
              if (e.key !== "Enter" && e.key !== " ") return;
              e.preventDefault();
              openEditSession(selected);
            };
            function syncTitleEditState() {
              const interactive = Boolean(selected);
              titleLabel.style.cursor = interactive ? "pointer" : "default";
              titleLabel.title = interactive ? "Edit conversation" : "No session selected";
              titleLabel.tabIndex = interactive ? 0 : -1;
              if (interactive) {
                titleLabel.setAttribute("role", "button");
                titleLabel.setAttribute("aria-label", "Edit conversation");
                titleLabel.removeAttribute("aria-disabled");
              } else {
                titleLabel.removeAttribute("role");
                titleLabel.removeAttribute("aria-label");
                titleLabel.setAttribute("aria-disabled", "true");
              }
            }
            syncTitleEditState();

        const fileBackdrop = el("div", { class: "modalBackdrop", id: "fileBackdrop" });
        const fileCloseBtn = el("button", {
          id: "fileCloseBtn",
          class: "icon-btn",
          title: "Close",
          "aria-label": "Close",
          type: "button",
          html: iconSvg("x"),
        });
        const fileStatus = el("div", { class: "muted fileStatus", id: "fileStatus", role: "status", "aria-live": "polite", text: "" });
        const filePickerInput = el("input", {
          id: "filePickerInput",
          class: "filePickerInput",
          type: "text",
          placeholder: "Choose or search files",
          autocomplete: "off",
          spellcheck: "false",
          role: "combobox",
          "aria-autocomplete": "list",
          "aria-controls": "filePickerMenu",
          "aria-expanded": "false",
        });
        const filePickerMenu = el("div", { id: "filePickerMenu", class: "filePickerMenu", role: "listbox" });
        const filePickerField = el("div", { class: "pickerField filePickerField", id: "filePickerField" }, [
          el("span", { class: "filePickerIcon", html: iconSvg("chevronDown"), "aria-hidden": "true" }),
          filePickerInput,
          filePickerMenu,
        ]);
        const fileModeDiffBtn = el("button", {
          id: "fileModeDiffBtn",
          class: "icon-btn",
          type: "button",
          title: "Toggle diff",
          "aria-label": "Toggle diff",
          html: iconSvg("diff"),
        });
        const fileModePreviewBtn = el("button", {
          id: "fileModePreviewBtn",
          class: "icon-btn",
          type: "button",
          title: "Toggle markdown preview",
          "aria-label": "Toggle markdown preview",
          html: iconSvg("preview"),
        });
        const fileEditBtn = el("button", {
          id: "fileEditBtn",
          class: "icon-btn",
          type: "button",
          title: "Edit file",
          "aria-label": "Edit file",
          html: iconSvg("edit"),
        });
        const fileVideoPreviewBtn = el("button", {
          id: "fileVideoPreviewBtn",
          class: "icon-btn",
          type: "button",
          title: "Use compatible MP4 preview",
          "aria-label": "Use compatible MP4 preview",
          html: iconSvg("play"),
        });
        fileVideoPreviewBtn.style.display = "none";
        const fileDownloadBtn = el("button", {
          id: "fileDownloadBtn",
          class: "icon-btn",
          type: "button",
          title: "Download file",
          "aria-label": "Download file",
          html: iconSvg("download"),
        });
        const fileTouchSelectBtn = el("button", {
          id: "fileTouchSelectBtn",
          class: "icon-btn fileTouchBtn",
          type: "button",
          title: "Select",
          "aria-label": "Select",
          html: iconSvg("select"),
        });
        const fileTouchCopyBtn = el("button", {
          id: "fileTouchCopyBtn",
          class: "icon-btn fileTouchBtn",
          type: "button",
          title: "Copy selection",
          "aria-label": "Copy selection",
          html: iconSvg("copy"),
        });
        const fileTouchPasteBtn = el("button", {
          id: "fileTouchPasteBtn",
          class: "icon-btn fileTouchBtn",
          type: "button",
          title: "Paste",
          "aria-label": "Paste",
          html: iconSvg("paste"),
        });
        const fileTouchUpBtn = el("button", {
          id: "fileTouchUpBtn",
          class: "icon-btn fileTouchBtn",
          type: "button",
          title: "Select up",
          "aria-label": "Select up",
          html: iconSvg("up"),
        });
        const fileTouchLeftBtn = el("button", {
          id: "fileTouchLeftBtn",
          class: "icon-btn fileTouchBtn",
          type: "button",
          title: "Select left",
          "aria-label": "Select left",
          html: iconSvg("left"),
        });
        const fileTouchDownBtn = el("button", {
          id: "fileTouchDownBtn",
          class: "icon-btn fileTouchBtn",
          type: "button",
          title: "Select down",
          "aria-label": "Select down",
          html: iconSvg("down"),
        });
        const fileTouchRightBtn = el("button", {
          id: "fileTouchRightBtn",
          class: "icon-btn fileTouchBtn",
          type: "button",
          title: "Select right",
          "aria-label": "Select right",
          html: iconSvg("right"),
        });
        const fileTouchDpad = el("div", { id: "fileTouchDpad", class: "fileTouchDpad" }, [
          el("span", { class: "fileTouchSpacer", "aria-hidden": "true" }),
          fileTouchUpBtn,
          el("span", { class: "fileTouchSpacer", "aria-hidden": "true" }),
          fileTouchLeftBtn,
          fileTouchDownBtn,
          fileTouchRightBtn,
        ]);
        const fileTouchActions = el("div", { id: "fileTouchActions", class: "fileTouchActions" }, [
          fileTouchSelectBtn,
          fileTouchCopyBtn,
          fileTouchPasteBtn,
        ]);
        const fileTouchToolbar = el("div", { id: "fileTouchToolbar", class: "fileTouchToolbar" }, [
          fileTouchDpad,
          fileTouchActions,
        ]);
        const fileDiff = el("div", { class: "fileDiff", id: "fileDiff" });
        const fileImage = el("img", { id: "fileImage", class: "fileImage", alt: "" });
        const fileVideo = el("video", { id: "fileVideo", class: "fileVideo", controls: true, preload: "metadata" });
        const fileViewer = el("div", { class: "fileViewer", id: "fileViewer", role: "dialog", "aria-modal": "true", "aria-label": "File viewer" }, [
          el("div", { class: "fileViewerHeader" }, [
            el("div", { class: "title", text: "View file" }),
            el("div", { class: "actions" }, [fileModeDiffBtn, fileModePreviewBtn, fileEditBtn, fileVideoPreviewBtn, fileDownloadBtn, fileCloseBtn]),
          ]),
          el("div", { class: "fileCandRow", id: "fileCandRow" }, [filePickerField]),
          fileStatus,
          fileDiff,
          fileImage,
          fileVideo,
          fileTouchToolbar,
        ]);
        root.appendChild(fileBackdrop);
        root.appendChild(fileViewer);

        const fileUnsavedBackdrop = el("div", { class: "modalBackdrop", id: "fileUnsavedBackdrop" });
        const fileUnsavedDialog = el("div", { class: "sendChoice fileUnsavedDialog", id: "fileUnsavedDialog", role: "dialog", "aria-modal": "true", "aria-label": "Unsaved file changes" }, [
          el("div", { class: "title", text: "Unsaved changes" }),
          el("div", { class: "muted", text: "Save this file before leaving the editor?" }),
          el("div", { class: "sendChoiceActions" }, [
            el("button", { class: "primary", id: "fileUnsavedSaveBtn", type: "button", text: "Save" }),
            el("button", { id: "fileUnsavedDiscardBtn", type: "button", text: "Discard" }),
            el("button", { id: "fileUnsavedCancelBtn", type: "button", text: "Cancel" }),
          ]),
        ]);
        root.appendChild(fileUnsavedBackdrop);
        root.appendChild(fileUnsavedDialog);
        const filePasteBackdrop = el("div", { class: "modalBackdrop", id: "filePasteBackdrop" });
        const filePasteInput = el("textarea", {
          id: "filePasteInput",
          class: "filePasteInput",
          placeholder: "Paste text here",
          spellcheck: "false",
          autocapitalize: "off",
          autocomplete: "off",
          autocorrect: "off",
        });
        const filePasteDialog = el("div", { class: "sendChoice filePasteDialog", id: "filePasteDialog", role: "dialog", "aria-modal": "true", "aria-label": "Paste into file" }, [
          el("div", { class: "title", text: "Paste into file" }),
          el("div", { class: "muted", text: "Long-press in this box to use the browser paste menu, then insert into the editor." }),
          filePasteInput,
          el("div", { class: "sendChoiceActions" }, [
            el("button", { class: "primary", id: "filePasteInsertBtn", type: "button", text: "Insert" }),
            el("button", { id: "filePasteCancelBtn", type: "button", text: "Cancel" }),
          ]),
        ]);
        root.appendChild(filePasteBackdrop);
        root.appendChild(filePasteDialog);

        const sendChoiceBackdrop = el("div", { class: "modalBackdrop", id: "sendChoiceBackdrop" });
        const sendChoice = el("div", { class: "sendChoice", id: "sendChoice", role: "dialog", "aria-modal": "true", "aria-label": "Send options" }, [
          el("div", { class: "title", text: "Current response is running" }),
          el("div", { class: "muted", text: "Choose how to handle your next message." }),
          el("div", { class: "sendChoiceActions" }, [
            el("button", { class: "primary", id: "sendChoiceNow", type: "button", text: "Send now" }),
            el("button", { id: "sendChoiceLater", type: "button", text: "Send after current" }),
            el("button", { id: "sendChoiceCancel", type: "button", text: "Cancel" }),
          ]),
        ]);
        root.appendChild(sendChoiceBackdrop);
        root.appendChild(sendChoice);

        const appConfirmBackdrop = el("div", { class: "modalBackdrop appConfirmBackdrop", id: "appConfirmBackdrop" });
        const appConfirmTitle = el("div", { class: "title", id: "appConfirmTitle", text: "Confirm action" });
        const appConfirmMessage = el("div", { class: "muted appConfirmMessage", id: "appConfirmMessage", text: "" });
        const appConfirmConfirmBtn = el("button", { class: "primary", id: "appConfirmConfirmBtn", type: "button", text: "Confirm" });
        const appConfirmCancelBtn = el("button", { id: "appConfirmCancelBtn", type: "button", text: "Cancel" });
        const appConfirm = el("div", {
          class: "sendChoice appConfirm",
          id: "appConfirm",
          role: "dialog",
          "aria-modal": "true",
          "aria-labelledby": "appConfirmTitle",
          "aria-describedby": "appConfirmMessage",
        }, [
          appConfirmTitle,
          appConfirmMessage,
          el("div", { class: "sendChoiceActions appConfirmActions" }, [appConfirmConfirmBtn, appConfirmCancelBtn]),
        ]);
        root.appendChild(appConfirmBackdrop);
        root.appendChild(appConfirm);

        const queueBackdrop = el("div", { class: "modalBackdrop", id: "queueBackdrop" });
        const queueCloseBtn = el("button", {
          id: "queueCloseBtn",
          class: "icon-btn",
          title: "Close",
          "aria-label": "Close",
          type: "button",
          html: iconSvg("x"),
        });
        const queueList = el("div", { class: "queueList", id: "queueList" });
        const queueEmpty = el("div", { class: "muted", id: "queueEmpty", text: "No queued messages." });
        const queueViewer = el("div", { class: "queueViewer", id: "queueViewer", role: "dialog", "aria-modal": "true", "aria-label": "Queued messages" }, [
          el("div", { class: "queueHeader" }, [
            el("div", { class: "title", text: "Queued messages" }),
            el("div", { class: "actions" }, [queueCloseBtn]),
          ]),
          queueEmpty,
          queueList,
        ]);
        root.appendChild(queueBackdrop);
        root.appendChild(queueViewer);

        const helpBackdrop = el("div", { class: "modalBackdrop", id: "helpBackdrop" });
        const helpCloseBtn = el("button", {
          id: "helpCloseBtn",
          class: "icon-btn",
          title: "Close",
          "aria-label": "Close",
          type: "button",
          html: iconSvg("x"),
        });
        let helpReturnFocusEl = null;
        const helpViewer = el("div", { class: "helpViewer", id: "helpViewer", role: "dialog", "aria-modal": "true", "aria-label": "Help" }, [
          el("div", { class: "queueHeader" }, [
            el("div", { class: "title", text: "Help" }),
            el("div", { class: "actions" }, [helpCloseBtn]),
          ]),
          el("div", {
            class: "helpBody",
            html: `<div class="muted">Sessions</div>
<ul class="md">
  <li>Choose a conversation from the sidebar. On desktop, hover a row to reveal <b>Edit</b>, <b>Duplicate</b>, and <b>Delete</b>. On touch, swipe left for <b>Edit</b>/<b>Duplicate</b> and right for <b>Delete</b>.</li>
  <li>The dot on the title row shows state: <b>filled + pulsing</b> = busy, <b>hollow</b> = idle, <b>filled orange</b> = snoozed or blocked, <b>filled amber</b> = starting.</li>
  <li>The metadata line shows the agent-backend icon first, then the session-type icon, then the reasoning marker (<b>X/H/M/L</b>) when available, followed by recency, folder, and branch.</li>
  <li>Click the conversation title to rename or reprioritize it. <b>Details</b> in the session utilities bar shows the exact backend, provider, model, reasoning level, queue state, and token usage.</li>
</ul>
<div class="muted">New session</div>
<ul class="md">
  <li><b>New session</b> can start fresh or resume a matching conversation for the currently selected backend in the current working directory.</li>
  <li>The backend tabs choose between the supported agent backends. Right now that is <b>Codex</b>, <b>Pi</b>, and <b>Claude</b>.</li>
  <li>You can choose working directory, a combined provider/model pair, reasoning level, and whether the session should start in tmux. If the directory is a Git repo, you can also start in a new worktree branch.</li>
  <li>For Pi, the reasoning level is set when the session launches. To change it later, use <b>Shift+Tab</b> in Pi's terminal; Codoxear does not guess or inject thinking-level cycles.</li>
  <li>Codoxear remembers the last backend you used and the last provider/model pair for each backend.</li>
</ul>
<div class="muted">Messages and queue</div>
<ul class="md">
  <li>If the selected session is idle, <b>Send</b> submits immediately. If it is busy, choose <b>Send after current</b> to queue the prompt.</li>
  <li>The queue is stored per session and drains automatically when that session becomes idle. Use <b>Queued messages</b> to review or edit queued prompts.</li>
  <li><b>Load older messages</b> fetches more scrollback. <b>Jump to latest</b> returns to the newest turn when you are reading history.</li>
  <li>The <b>Search</b> button and <b>Previous</b>/<b>Next</b> message controls live in the navigation bar at the top of the conversation (not a floating rail). Use <b>/</b> to start a search of the loaded chat; when the match count shows more results, Previous/Next can load an older matching window.</li>
  <li>On a <b>Pi</b> session, type <b>/model</b> in the composer to switch models live. Start typing a provider or model name to filter the list, then choose an entry.</li>
  <li>Press <b>f</b> to show keyboard hints for visible controls: <b>1</b>–<b>9</b> switch sessions; <b>s</b> sidebar; <b>b</b> files; <b>d</b> details; <b>u</b> unattended; <b>i</b> interrupt; <b>r</b> search; <b>p</b>/<b>n</b> previous/next user message; <b>o</b> older messages; <b>g</b> latest; <b>a</b> attach; <b>q</b> queued messages; <b>x</b> stop; <b>e</b> send; <b>m</b> message box; <b>c</b> new session. Clickable file references in the conversation also get dynamically-assigned letter hints. Press <b>Escape</b> or <b>Backspace</b> to cancel.</li>
  <li>Direct shortcuts (no leader): <b>i</b> focus message box; <b>j</b>/<b>k</b> scroll down/up; <b>d</b>/<b>u</b> scroll half-page down/up; <b>G</b> go to bottom; <b>D</b> delete current session (confirm); <b>/</b> search; <b>Esc</b> exit message box or close dialog.</li>
</ul>
<div class="muted">Unattended mode</div>
<ul class="md">
  <li>Unattended mode is a per-session idle nudge. Open the Unattended button in the session utilities bar, turn it on, and optionally add an extra request to append to the built-in unattended-work prompt.</li>
  <li><b>Cooldown time</b> is how many idle minutes must pass after the assistant finishes before the next unattended prompt is injected.</li>
  <li><b>Number of injections</b> is the remaining auto-injection budget for that session. Each unattended prompt decrements it, and unattended mode turns itself off when it reaches zero.</li>
  <li>Unattended mode runs in the server process, so it keeps working even if you close the browser tab. Enabled sessions show an <b>unattended</b> badge in the sidebar.</li>
</ul>
<div class="muted">Files</div>
<ul class="md">
  <li><b>View file</b> opens recent or changed files from the selected session, with diff, file, and preview modes where available.</li>
  <li>File paths mentioned in assistant messages become clickable when the server can resolve them.</li>
  <li><b>Attach file</b> adds local files or images to the current prompt.</li>
</ul>
<div class="muted">Announcements and notifications</div>
<ul class="md">
  <li><b>Announcement</b> is a per-browser toggle. It plays the shared server audio stream and announces every end-of-turn response. Narration announcements are optional in Settings.</li>
  <li><b>Notification</b> is a per-browser toggle. On desktop it enables live browser notifications for final responses. On iPhone/iPad it can also enable Web Push when you use the installed Home Screen app over HTTPS.</li>
  <li>If Announcement cannot be enabled yet, open <b>Settings</b> and fill in the OpenAI-compatible API base URL and API key used for summarization and speech.</li>
</ul>`,
          }),
        ]);
        root.appendChild(helpBackdrop);
        root.appendChild(helpViewer);

        const diagBackdrop = el("div", { class: "modalBackdrop", id: "diagBackdrop" });
        const diagNewLikeBtn = el("button", {
          id: "diagNewLikeBtn",
          class: "icon-btn text-btn",
          title: "Start a new session with these visible launch settings",
          "aria-label": "New like this",
          type: "button",
          text: "New like this",
        });
        const diagCopyConversationBtn = el("button", {
          id: "diagCopyConversationBtn",
          class: "icon-btn text-btn",
          title: "Copy conversation",
          "aria-label": "Copy conversation",
          type: "button",
          text: "Copy conversation",
        });
        const diagCopyBtn = el("button", {
          id: "diagCopyBtn",
          class: "icon-btn text-btn",
          title: "Copy details",
          "aria-label": "Copy details",
          type: "button",
          text: "Copy details",
        });
        const diagCloseBtn = el("button", {
          id: "diagCloseBtn",
          class: "icon-btn",
          title: "Close",
          "aria-label": "Close",
          type: "button",
          html: iconSvg("x"),
        });
        // Detail actions start disabled until the controller loads the selected
        // session's details and enables their corresponding payloads.
        diagNewLikeBtn.disabled = true;
        diagCopyConversationBtn.disabled = true;
        diagCopyBtn.disabled = true;
        const diagStatus = el("div", { class: "muted", id: "diagStatus", text: "" });
        const diagContent = el("div", { class: "detailsGrid", id: "diagContent" });
        const diagViewer = el("div", { class: "diagViewer", id: "diagViewer", role: "dialog", "aria-modal": "true", "aria-label": "Details" }, [
          el("div", { class: "queueHeader" }, [
            el("div", { class: "title", text: "Details" }),
            el("div", { class: "actions" }, [diagNewLikeBtn, diagCopyConversationBtn, diagCopyBtn, diagCloseBtn]),
          ]),
          diagStatus,
          diagContent,
        ]);
        root.appendChild(diagBackdrop);
        root.appendChild(diagViewer);

        const editCloseBtn = el("button", {
          id: "editCloseBtn",
          class: "icon-btn",
          title: "Close",
          "aria-label": "Close",
          type: "button",
          html: iconSvg("x"),
        });
        const editStatus = el("div", { class: "muted", id: "editStatus", text: "" });
        const editNameInput = el("input", {
          id: "editNameInput",
          type: "text",
          placeholder: "Conversation title",
          maxlength: "80",
          autocomplete: "off",
        });
        const editPriorityRange = el("input", {
          id: "editPriorityRange",
          type: "range",
          min: "-1",
          max: "1",
          step: "0.05",
          value: "0",
        });
        const editPriorityValue = el("span", { class: "rangeValue", id: "editPriorityValue", text: "+0.00" });
        const editPriorityResetBtn = el("button", {
          id: "editPriorityResetBtn",
          class: "icon-btn text-btn subtleBtn",
          type: "button",
          text: "Reset",
        });
        const editSnoozeModeButtons = new Map();
        let editSnoozeMode = "none";
        const editSnoozeButtons = el("div", { class: "choiceChips", id: "editSnoozeButtons" });
        for (const [value, label] of [
          ["none", "No snooze"],
          ["4h", "4 hours"],
          ["tomorrow", "Tomorrow"],
          ["custom", "Custom"],
        ]) {
          const btn = el("button", {
            type: "button",
            class: "choiceChip",
            "data-snooze-mode": value,
            text: label,
          });
          editSnoozeModeButtons.set(value, btn);
          editSnoozeButtons.appendChild(btn);
        }
        const editSnoozeCustomDate = el("input", { id: "editSnoozeCustomDate", type: "date" });
        const editSnoozeCustomTime = el("input", { id: "editSnoozeCustomTime", type: "time", step: "60" });
        const editSnoozeCustomRow = el("div", { class: "customSnoozeRow", id: "editSnoozeCustomRow" }, [
          editSnoozeCustomDate,
          editSnoozeCustomTime,
        ]);
        const editDependencyBtn = el("button", {
          id: "editDependencyBtn",
          class: "filePickerBtn dialogPickerBtn",
          type: "button",
          "aria-label": "Choose dependency",
        });
        const editDependencyMenu = el("div", { id: "editDependencyMenu", class: "filePickerMenu dialogPickerMenu" });
        const editDependencyField = el("div", { class: "pickerField" }, [editDependencyBtn]);
        const editSaveBtn = el("button", { class: "primary", id: "editSaveBtn", type: "button", text: "Save" });
        const editViewer = el("dialog", { class: "formViewer formDialog", id: "editViewer", "aria-label": "Edit conversation" }, [
          el("div", { class: "queueHeader" }, [
            el("div", { class: "title", text: "Edit conversation" }),
            el("div", { class: "actions" }, [editCloseBtn]),
          ]),
          editStatus,
          el("div", { class: "formBody" }, [
            el("label", { class: "field" }, [
              el("span", { class: "fieldLabel", text: "Conversation name" }),
              editNameInput,
            ]),
            el("label", { class: "field editPriorityField" }, [
              el("span", { class: "fieldLabel", text: "Priority offset" }),
              el("div", { class: "sliderRow" }, [editPriorityRange, editPriorityValue, editPriorityResetBtn]),
            ]),
            el("label", { class: "field" }, [
              el("span", { class: "fieldLabel", text: "Snooze" }),
              editSnoozeButtons,
              editSnoozeCustomRow,
            ]),
            el("label", { class: "field" }, [
              el("span", { class: "fieldLabel", text: "Depends on" }),
              editDependencyField,
            ]),
          ]),
          el("div", { class: "formActions" }, [
            el("button", { id: "editCancelBtn", type: "button", text: "Cancel" }),
            editSaveBtn,
          ]),
        ]);
        root.appendChild(editViewer);
        editViewer.appendChild(editDependencyMenu);
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
              el("label", { class: "voiceToggleRow" }, [
                voiceClearApiKeyToggle,
                el("span", { text: "Clear saved API key" }),
              ]),
            ]),
            el("div", { class: "field" }, [
              el("label", { class: "voiceToggleRow" }, [
                narrationSettingToggle,
                el("span", { text: "Announce narration messages" }),
              ]),
            ]),
            el("div", { class: "field" }, [
              el("span", { class: "fieldLabel", text: "Unattended mode prompt" }),
              unattendedPromptInput,
              el("span", { class: "fieldHint", id: "unattendedPromptHint", text: "Sent when unattended mode resumes an idle session. Reset then Save to restore the built-in constitution." }),
              unattendedPromptResetBtn,
            ]),
          ]),
          el("div", { class: "formActions" }, [
            el("button", { id: "voiceSettingsCancelBtn", type: "button", text: "Cancel" }),
            el("button", { id: "voiceSettingsSaveBtn", class: "primary", type: "button", text: "Save" }),
          ]),
        ]);
        root.appendChild(voiceSettingsBackdrop);
        root.appendChild(voiceSettingsViewer);

        const newSessionBackdrop = el("div", { class: "modalBackdrop", id: "newSessionBackdrop" });
        const newSessionCloseBtn = el("button", {
          id: "newSessionCloseBtn",
          class: "icon-btn",
          title: "Close",
          "aria-label": "Close",
          type: "button",
          html: iconSvg("x"),
        });
        const newSessionStatus = el("div", { class: "muted", id: "newSessionStatus", text: "" });
        let newSessionReturnFocusEl = null;
        const newSessionCwdInput = el("input", {
          id: "newSessionCwdInput",
          type: "text",
          placeholder: "/path/to/project",
          autocomplete: "off",
          spellcheck: "false",
          role: "combobox",
          "aria-autocomplete": "list",
          "aria-controls": "newSessionCwdMenu",
          "aria-expanded": "false",
        });
        const newSessionCwdMenu = el("div", {
          id: "newSessionCwdMenu",
          class: "filePickerMenu dialogPickerMenu cwdSuggestionMenu",
          role: "listbox",
        });
        const newSessionCwdField = el("div", { class: "pickerField cwdPickerField cwdComboboxField", id: "newSessionCwdField" }, [
          el("span", { class: "cwdComboboxIcon", html: iconSvg("chevronDown"), "aria-hidden": "true" }),
          newSessionCwdInput,
          newSessionCwdMenu,
        ]);
        const newSessionCwdHint = el("div", { class: "fieldHint", id: "newSessionCwdHint", text: "" });
        const newSessionNameInput = el("input", {
          id: "newSessionNameInput",
          type: "text",
          placeholder: "session-name",
          autocomplete: "off",
          spellcheck: "false",
        });
        const newSessionModelInput = el("input", {
          id: "newSessionModelInput",
          type: "text",
          placeholder: "Provider/model",
          autocomplete: "off",
          spellcheck: "false",
          role: "combobox",
          "aria-autocomplete": "list",
          "aria-controls": "newSessionModelMenu",
          "aria-expanded": "false",
        });
        const newSessionModelMenu = el("div", {
          id: "newSessionModelMenu",
          class: "filePickerMenu dialogPickerMenu cwdSuggestionMenu",
          role: "listbox",
        });
        const newSessionModelField = el("div", { class: "pickerField comboboxField cwdComboboxField", id: "newSessionModelField" }, [
          el("span", { class: "cwdComboboxIcon", html: iconSvg("chevronDown"), "aria-hidden": "true" }),
          newSessionModelInput,
          newSessionModelMenu,
        ]);
        const newSessionModelLabel = el("span", { class: "fieldLabel", text: "Provider / model" });
        const newSessionBackendTabs = el("div", { class: "agentBackendTabs", id: "newSessionBackendTabs" });
        const newSessionBackendName = el("span", { class: "agentBackendTabName", id: "newSessionBackendName" });
        let newSessionReasoningEffort = "high";
        const newSessionReasoningBtn = el("button", {
          id: "newSessionReasoningBtn",
          class: "filePickerBtn dialogPickerBtn sidePickerBtn",
          type: "button",
          "aria-label": "Choose reasoning effort",
          "aria-haspopup": "menu",
          "aria-expanded": "false",
        });
        const newSessionReasoningMenu = el("div", { id: "newSessionReasoningMenu", class: "filePickerMenu dialogPickerMenu" });
        const newSessionReasoningField = el("div", { class: "pickerField comboboxField pickerButtonField", id: "newSessionReasoningField" }, [
          el("span", { class: "cwdComboboxIcon", html: iconSvg("chevronDown"), "aria-hidden": "true" }),
          newSessionReasoningBtn,
        ]);
        const newSessionResumeBtn = el("button", {
          id: "newSessionResumeBtn",
          class: "filePickerBtn dialogPickerBtn sidePickerBtn",
          type: "button",
          "aria-label": "Choose a conversation to resume",
          "aria-haspopup": "menu",
          "aria-expanded": "false",
        });
        const newSessionResumeMenu = el("div", { id: "newSessionResumeMenu", class: "filePickerMenu dialogPickerMenu" });
        const newSessionTmuxToggle = el("input", {
          id: "newSessionTmuxToggle",
          type: "checkbox",
        });
        const newSessionTmuxField = el("div", { class: "field", id: "newSessionTmuxField" }, [
          el("span", { class: "fieldLabel", text: "Launch mode" }),
          el("label", { class: "checkField" }, [
            newSessionTmuxToggle,
            el("span", { text: "Create in tmux" }),
          ]),
        ]);
        const newSessionLaunchRow = el("div", { class: "formGrid newSessionOptionsRow" }, [
          newSessionTmuxField,
        ]);
        const newSessionFastToggle = el("input", {
          id: "newSessionFastToggle",
          type: "checkbox",
        });
        const newSessionFastField = el("div", { class: "field compactToggleField", id: "newSessionFastField" }, [
          el("span", { class: "fieldLabel", text: "Speed" }),
          el("label", { class: "checkField" }, [
            newSessionFastToggle,
            el("span", { text: "Fast" }),
          ]),
        ]);
        const newSessionWorktreeToggle = el("input", {
          id: "newSessionWorktreeToggle",
          type: "checkbox",
        });
        const newSessionWorktreeInput = el("input", {
          id: "newSessionWorktreeBranchInput",
          type: "text",
          placeholder: "feature/my-branch",
          autocomplete: "off",
          spellcheck: "false",
          disabled: true,
        });
        const newSessionWorktreeField = el("div", { class: "field", id: "newSessionWorktreeField" }, [
          el("span", { class: "fieldLabel", text: "Git worktree branch" }),
          el("label", { class: "checkField" }, [
            newSessionWorktreeToggle,
            el("span", { text: "Create a new worktree for this session" }),
          ]),
          newSessionWorktreeInput,
        ]);
        const newSessionStartBtn = el("button", { class: "primary", id: "newSessionStartBtn", type: "button", text: "Start session" });
        const newSessionViewer = el("div", { class: "formViewer newSessionViewer", id: "newSessionViewer", role: "dialog", "aria-modal": "true", "aria-label": "New session" }, [
          el("div", { class: "queueHeader" }, [
            el("div", { class: "newSessionHeaderLead" }, [
              el("div", { class: "title", text: "New session" }),
              newSessionBackendTabs,
              newSessionBackendName,
            ]),
            el("div", { class: "actions" }, [newSessionCloseBtn]),
          ]),
          newSessionStatus,
          el("div", { class: "formBody" }, [
            el("label", { class: "field" }, [
              el("span", { class: "fieldLabel", text: "Working directory" }),
              newSessionCwdField,
              newSessionCwdHint,
            ]),
            el("label", { class: "field" }, [
              el("span", { class: "fieldLabel", text: "Session name" }),
              newSessionNameInput,
            ]),
            el("div", { class: "formGrid newSessionRunConfigRow" }, [
              el("label", { class: "field" }, [
                newSessionModelLabel,
                newSessionModelField,
              ]),
              el("label", { class: "field" }, [
                el("span", { class: "fieldLabel", text: "Reasoning effort" }),
                newSessionReasoningField,
              ]),
              newSessionFastField,
            ]),
            el("label", { class: "field" }, [
              el("span", { class: "fieldLabel", text: "Resume conversation" }),
              newSessionResumeBtn,
            ]),
            newSessionLaunchRow,
            newSessionWorktreeField,
          ]),
          el("div", { class: "formActions" }, [
            el("button", { id: "newSessionCancelBtn", type: "button", text: "Cancel" }),
            newSessionStartBtn,
          ]),
        ]);
        root.appendChild(newSessionBackdrop);
        root.appendChild(newSessionViewer);
        newSessionViewer.appendChild(newSessionModelMenu);
        newSessionViewer.appendChild(newSessionReasoningMenu);
        newSessionViewer.appendChild(newSessionResumeMenu);

        const codoxearModal = window.CodoxearModal;
        if (
          !codoxearModal ||
          typeof codoxearModal.isModalTargetOpen !== "function" ||
          typeof codoxearModal.syncModalIsolation !== "function" ||
          typeof codoxearModal.restoreModalFocus !== "function" ||
          typeof codoxearModal.focusModalCloseButton !== "function"
        )
          throw new Error("Codoxear modal helpers failed to load");

        const modalIsolationTargets = [
          fileViewer,
          fileUnsavedDialog,
          filePasteDialog,
          sendChoice,
          appConfirm,
          queueViewer,
          helpViewer,
          diagViewer,
          editViewer,
          voiceSettingsViewer,
          newSessionViewer,
        ];

        function isModalTargetOpen(node) {
          return codoxearModal.isModalTargetOpen(node);
        }

        function syncModalIsolation() {
          return codoxearModal.syncModalIsolation(app, modalIsolationTargets);
        }

        function closeTransientOverlays({ closeSearch = false } = {}) {
          if (unattendedController.isOpen()) hideUnattendedMenu();
          if (closeSearch && chatSearchController.isOpen()) closeChatSearch();
          if (document.body.classList.contains("sidebar-open")) setSidebarOpen(false);
          filePickerMenuState.close();
          filePickerMenu.classList.remove("open");
          filePickerInput.setAttribute("aria-expanded", "false");
          newSessionCwdMenuOpen = false;
          newSessionCwdMenuFocus = -1;
          newSessionModelMenuOpen = false;
          newSessionModelMenuFocus = -1;
          newSessionReasoningMenuOpen = false;
          newSessionResumeMenuOpen = false;
          editDependencyMenuOpen = false;
          applyDialogMenus();
        }

        function prepareModalOpen(options = {}) {
          closeTransientOverlays(options);
        }

        function afterModalVisibilityChanged() {
          syncModalIsolation();
        }

        function restoreModalFocus(target, isStillOpen) {
          return codoxearModal.restoreModalFocus(target, isStillOpen);
        }

        function focusModalCloseButton(viewer, closeBtn) {
          return codoxearModal.focusModalCloseButton(viewer, closeBtn);
        }

        let appConfirmPending = null;
        let appConfirmReturnFocusEl = null;

        function normalizeAppConfirmOptions(options = {}) {
          if (typeof options === "string") return { title: "Confirm action", message: options, confirmText: "Confirm", cancelText: "Cancel", destructive: false };
          const raw = options && typeof options === "object" ? options : {};
          return {
            title: String(raw.title || "Confirm action"),
            message: String(raw.message || ""),
            confirmText: String(raw.confirmText || "Confirm"),
            cancelText: String(raw.cancelText || "Cancel"),
            destructive: Boolean(raw.destructive),
          };
        }

        function appConfirmFocusableControls() {
          return [appConfirmCancelBtn, appConfirmConfirmBtn].filter((control) => control && !control.disabled && typeof control.focus === "function");
        }

        function focusAppConfirmInitial({ destructive = false } = {}) {
          requestAnimationFrame(() => {
            if (appConfirm.style.display !== "flex") return;
            const preferred = destructive ? appConfirmCancelBtn : appConfirmConfirmBtn;
            const fallback = destructive ? appConfirmConfirmBtn : appConfirmCancelBtn;
            const target = preferred && !preferred.disabled ? preferred : fallback && !fallback.disabled ? fallback : null;
            if (!target || typeof target.focus !== "function") return;
            try {
              target.focus({ preventScroll: true });
            } catch {}
          });
        }

        function resolveAppConfirm(result, { restoreFocus = true } = {}) {
          const pending = appConfirmPending;
          const target = appConfirmReturnFocusEl;
          appConfirmPending = null;
          appConfirmReturnFocusEl = null;
          appConfirmBackdrop.style.display = "none";
          appConfirm.style.display = "none";
          afterModalVisibilityChanged();
          if (restoreFocus) restoreModalFocus(target, () => appConfirm.style.display === "flex");
          if (pending && !pending.settled) {
            pending.settled = true;
            pending.resolve(Boolean(result));
          }
        }

        function confirmApp(options = {}) {
          if (appConfirmPending) resolveAppConfirm(false, { restoreFocus: false });
          const normalized = normalizeAppConfirmOptions(options);
          prepareModalOpen();
          appConfirmTitle.textContent = normalized.title;
          appConfirmMessage.textContent = normalized.message;
          appConfirmConfirmBtn.textContent = normalized.confirmText;
          appConfirmCancelBtn.textContent = normalized.cancelText;
          appConfirmReturnFocusEl = document.activeElement instanceof HTMLElement ? document.activeElement : null;
          appConfirmBackdrop.style.display = "block";
          appConfirm.style.display = "flex";
          afterModalVisibilityChanged();
          focusAppConfirmInitial(normalized);
          return new Promise((resolve) => {
            appConfirmPending = { resolve, settled: false };
          });
        }

        appConfirmConfirmBtn.onclick = () => resolveAppConfirm(true);
        appConfirmCancelBtn.onclick = () => resolveAppConfirm(false);
        appConfirmBackdrop.onclick = () => resolveAppConfirm(false);

        const codoxearClipboard = window.CodoxearClipboard;
        if (!codoxearClipboard || typeof codoxearClipboard.copyToClipboard !== "function")
          throw new Error("Codoxear clipboard helpers failed to load");
        const codoxearCodeCopy = window.CodoxearCodeCopy;
        if (!codoxearCodeCopy || typeof codoxearCodeCopy.createCodeBlockCopyRuntime !== "function")
          throw new Error("Codoxear code copy helpers failed to load");

        function setToast(text) {
          toast.textContent = text || "";
          if (!text) return;
          setTimeout(() => {
            if (toast.textContent === text) toast.textContent = "";
          }, 2200);
        }

        async function copyToClipboard(text) {
          return codoxearClipboard.copyToClipboard(text);
        }

        const codeBlockCopyRuntime = codoxearCodeCopy.createCodeBlockCopyRuntime({
          copyToClipboard,
          setToast,
          setTimeout,
          clearTimeout,
        });

        function formatConversationForCopy(events) {
          return codoxearConversationCopy.formatConversationForCopy(events);
        }

        function formatConversationForCopyResult(events) {
          return codoxearConversationCopy.formatConversationForCopyResult(events);
        }

        function copiedConversationToast(messageCount) {
          return messageCount === 1 ? "Copied 1 message" : `Copied ${messageCount} messages`;
        }

        async function copyConversation() {
          if (!selected) return;
          const sid = selected;
          try {
            const data = await api(`/api/sessions/${sid}/messages/export`);
            if (selected !== sid) return;
            const events = Array.isArray(data && data.events) ? data.events : [];
            const formatted = formatConversationForCopyResult(events);
            if (!formatted.text) {
              setToast("No conversation to copy");
              return;
            }
            await copyToClipboard(formatted.text);
            setToast(copiedConversationToast(formatted.messageCount));
          } catch (err) {
            setToast(copyConversationFailureToast(err));
          }
        }

        let currentQueueLen = 0;
        function setStatus({ running, queueLen }) {
          const q = Math.max(0, Number(queueLen) || 0);
          const mobile = isMobile();
          const wasRunning = currentRunning;
          currentRunning = Boolean(running);
          currentQueueLen = q;
          if (running) {
            statusChip.style.display = "none";
            statusChip.classList.remove("running");
          } else {
            statusChip.style.display = "inline-flex";
               if (q) statusChip.textContent = mobile ? `Q ${q}` : `Queue ${q}`;
               else statusChip.textContent = "Idle";
          }
          const canInterrupt = Boolean(running && selected);
          interruptBtn.style.display = canInterrupt ? "inline-flex" : "none";
          interruptBtn.disabled = !canInterrupt;
          const composerStopControl = $("#composerStopBtn");
          if (composerStopControl) {
            composerStopControl.classList.toggle("is-visible", canInterrupt);
            composerStopControl.disabled = !canInterrupt;
          }
          if (wasRunning && !currentRunning) {
            // no-op placeholder; keep transition boundary for future UI behavior
          }
          syncAttachButtonState();
          updateQueueBadge();
        }

	        function setContext(tok) {
	          if (!tok || typeof tok !== "object") {
	            lastToken = null;
	            ctxChip.style.display = "none";
	            ctxChip.disabled = true;
	            ctxChip.textContent = "";
	            ctxChip.title = "";
	            return;
	          }
	          const ctx = Number(tok.context_window);
	          const used = Number(tok.tokens_in_context);
	          const pct = Number(tok.percent_remaining);
	          if (!Number.isFinite(ctx) || !Number.isFinite(used) || ctx <= 0 || used < 0) {
	            lastToken = null;
	            ctxChip.style.display = "none";
	            ctxChip.disabled = true;
	            return;
	          }
	          const p = Number.isFinite(pct) ? Math.max(0, Math.min(100, Math.round(pct))) : null;
	          const maxInput = Number(tok.max_input_tokens);
	          const reserved = Number(tok.reserved_tokens);
	          const effectiveMaxInput = Number.isFinite(maxInput) && maxInput >= 0 ? maxInput : ctx;
	          const effectiveReserved = Number.isFinite(reserved) && reserved >= 0 ? reserved : Math.max(ctx - effectiveMaxInput, 0);
	          lastToken = { ctx, used, pct: p, remaining: Math.max(effectiveMaxInput - used, 0), maxInput: effectiveMaxInput, reserved: effectiveReserved, asOf: tok.as_of || "" };
	          ctxChip.style.display = "inline-flex";
	          ctxChip.disabled = false;
	          ctxChip.textContent = p === null ? "Ctx" : `Ctx ${p}%`;
	          ctxChip.title = `Context input: ${used}/${lastToken.maxInput} tokens (${lastToken.reserved} reserved; window ${ctx}).`;
	        }
        ctxChip.onclick = () => {
          if (!lastToken) return;
          setToast(`ctx ${lastToken.used}/${lastToken.ctx} (${lastToken.pct ?? "?"}% left)`);
        };

        function invalidateOlderLoad() {
          olderLoadRuntime.invalidate();
        }

        function resetChatRenderState() {
          invalidateOlderLoad();
          transcriptScrollRuntime.enableAutoScroll();
          sending = false;
          transcriptEventRuntime.resetRecentEvents();
          transcriptSlotRuntime.clearLiveCursor();
          transcriptScrollRuntime.markLiveTail();
          olderLoadRuntime.resetAutoTrigger();
              clickMetricPending = false;
          clearTranscriptDom();
          messageCopyNavigationRuntime.reset();
              setOlderState({ hasMore: false, isLoading: false });
          typingRowRuntime.reset();
          jumpBtn.style.display = "none";
              updateChatNavButtons();
              if (chatSearchController.isOpen()) closeChatSearch();
          transcriptScrollRuntime.reset({ scrollTop: 0 });
              transcriptScrollRuntime.syncVisibleTimeIndicator();
	        }

        function clearTranscriptDom() {
          transcriptDomRuntime.clear();
        }

        function clearOlderLoadError() {
          olderLoadRuntime.clearError();
        }

        function showOlderLoadError() {
          olderLoadRuntime.showError();
        }

        function setOlderState({ hasMore, isLoading }) {
          olderLoadRuntime.setState({ hasMore, isLoading });
        }

        const codoxearMessageRows = window.CodoxearMessageRows;
        if (
          !codoxearMessageRows ||
          typeof codoxearMessageRows.makeRow !== "function" ||
          typeof codoxearMessageRows.safeMakeRow !== "function" ||
          typeof codoxearMessageRows.messageCopyButtonForRow !== "function" ||
          typeof codoxearMessageRows.renderedMessageRows !== "function" ||
          typeof codoxearMessageRows.loadedUserMessageRows !== "function" ||
          typeof codoxearMessageRows.loadedCopyMessageRows !== "function" ||
          typeof codoxearMessageRows.activeElementIsMessageCopyButton !== "function" ||
          typeof codoxearMessageRows.createMessageCopyNavigationRuntime !== "function" ||
          typeof codoxearMessageRows.rowSearchText !== "function" ||
          typeof codoxearMessageRows.compareRowsInDomOrder !== "function" ||
          typeof codoxearMessageRows.loadedUserJumpTarget !== "function" ||
          typeof codoxearMessageRows.loadedCopyJumpTarget !== "function" ||
          typeof codoxearMessageRows.clearChatSearchMarks !== "function" ||
          typeof codoxearMessageRows.applyChatSearchMarks !== "function" ||
          typeof codoxearMessageRows.oldestRenderedHistoryCursor !== "function" ||
          typeof codoxearMessageRows.firstVisibleMessageRow !== "function" ||
          typeof codoxearMessageRows.trimRenderedRowTargets !== "function" ||
          typeof codoxearMessageRows.trimRowsBeforeViewportTargets !== "function"
        )
          throw new Error("Codoxear message row helpers failed to load");

        const messageCopyNavigationRuntime = codoxearMessageRows.createMessageCopyNavigationRuntime({ root: chatInner });

        function renderedMessageRows() {
          return codoxearMessageRows.renderedMessageRows(chatInner);
        }

        function loadedUserMessageRows() {
          return codoxearMessageRows.loadedUserMessageRows(chatInner);
        }

        function loadedCopyMessageRows() {
          return codoxearMessageRows.loadedCopyMessageRows(chatInner);
        }

        function messageCopyButtonForRow(row) {
          return codoxearMessageRows.messageCopyButtonForRow(row);
        }

        function activeElementIsMessageCopyButton() {
          return codoxearMessageRows.activeElementIsMessageCopyButton(document);
        }

        function rowSearchText(row) {
          return codoxearMessageRows.rowSearchText(row);
        }

        function compareRowsInDomOrder(a, b) {
          return codoxearMessageRows.compareRowsInDomOrder(a, b, Node);
        }

        function loadedUserJumpTarget(rows, direction, threshold) {
          return codoxearMessageRows.loadedUserJumpTarget(rows, direction, threshold);
        }

        function loadedCopyJumpTarget(rows, direction, threshold) {
          return messageCopyNavigationRuntime.jumpTarget(rows, direction, threshold);
        }

        function applyChatSearchMarks(matches, currentRow) {
          codoxearMessageRows.applyChatSearchMarks(matches, currentRow);
        }

        function firstVisibleMessageRow() {
          return codoxearMessageRows.firstVisibleMessageRow(renderedMessageRows(), chat.scrollTop + 1);
        }

        function syncMessageCopyTabStops() {
          messageCopyNavigationRuntime.syncTabStops(renderedMessageRows());
        }

        function setActiveMessageCopyRow(row, { focusCopy = false } = {}) {
          messageCopyNavigationRuntime.setActiveRow(row, { focusCopy });
        }

        addAppEvent(chatInner, "pointerover", (e) => {
          if (activeElementIsMessageCopyButton()) return;
          const row = e.target && typeof e.target.closest === "function" ? e.target.closest(".msg-row") : null;
          if (row && chatInner.contains(row)) setActiveMessageCopyRow(row);
        });

        addAppEvent(chatInner, "focusin", (e) => {
          const row = e.target && typeof e.target.closest === "function" ? e.target.closest(".msg-row") : null;
          if (row && chatInner.contains(row)) setActiveMessageCopyRow(row);
        });

        function prefersReducedMotion() {
          return codoxearViewport.prefersReducedMotion();
        }

        function pulseNavigatedRow(row) {
          if (!row) return;
          setActiveMessageCopyRow(row, { focusCopy: activeElementIsMessageCopyButton() });
          row.classList.remove("nav-pulse");
          void row.offsetWidth;
          row.classList.add("nav-pulse");
          setTimeout(() => row.classList.remove("nav-pulse"), 1400);
        }

        const hintModeController = (function instantiateHintModeController() {
          const codoxearHintMode = window.CodoxearHintMode;
          if (!codoxearHintMode || typeof codoxearHintMode.createHintModeController !== "function")
            throw new Error("Codoxear hint mode controller failed to load");
          return codoxearHintMode.createHintModeController({
            documentTarget: document,
            isTextEntryElement,
            isMobile,
            modalIsolationTargets,
            isModalTargetOpen,
            addAppEvent,
            shellHints: [
              { label: "s", element: toggleSidebarBtn },
              { label: "b", element: fileBtn },
              { label: "d", element: diagBtn },
              { label: "u", element: unattendedBtn },
              { label: "i", element: interruptBtn },
              { label: "r", element: chatSearchBtn },
              { label: "p", element: prevUserBtn },
              { label: "n", element: nextUserBtn },
              { label: "o", element: olderBtn },
              { label: "g", element: jumpBtn },
              { label: "a", element: attachBtn },
              { label: "q", element: queueBtn },
              { label: "x", element: composerStopBtn },
              { label: "e", element: sendBtn },
              { label: "m", element: textarea },
              { label: "c", element: $("#newBtn") },
            ],
          });
        })();

        // --- Direct (no-leader) Vimium-style shortcuts ---
        // These fire when not in a text-entry element, no modal is open, and
        // hint mode is not active. They don't conflict with hint-mode letters
        // (which require `f` leader first) — the context disambiguates.
        addAppEvent(document, "keydown", (e) => {
          if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey) return;
          if (e.isComposing) return;
          if (hintModeController && hintModeController.isActive()) return;
          if (isTextEntryElement(document.activeElement)) return;
          if (isModalTargetOpen(appConfirm) || isModalTargetOpen(sendChoice) || isModalTargetOpen(queueViewer) || isModalTargetOpen(helpViewer) || isModalTargetOpen(diagViewer) || isModalTargetOpen(editViewer) || isModalTargetOpen(newSessionViewer) || isFileViewerOpen()) return;
          // Capital keys (Shift) are distinct actions: G = go to bottom, D = delete.
          if (e.shiftKey) {
            const shifted = String(e.key || "");
            if (shifted === "G") {
              e.preventDefault();
              void jumpToLatest();
              return;
            }
            if (shifted === "D") {
              e.preventDefault();
              void (async () => {
                if (!selected) return;
                const sid = selected;
                const confirmed = await confirmApp({
                  title: "Delete session?",
                  message: "Delete the current session? This cannot be undone.",
                  confirmText: "Delete",
                  cancelText: "Cancel",
                  destructive: true,
                });
                if (!confirmed) return;
                if (selected !== sid) return;
                try {
                  await api(`/api/sessions/${sid}/delete`, { method: "POST", body: {} });
                  clearDeletedSessionClientState(sid);
                  await refreshSessions();
                  setToast("session deleted");
                } catch (err) {
                  setToast(`delete error: ${err && err.message ? err.message : "unknown error"}`);
                }
              })();
              return;
            }
            return;
          }
          const key = String(e.key || "").toLowerCase();
          if (key === "i") {
            e.preventDefault();
            textarea.focus({ preventScroll: true });
            return;
          }
          const step = Math.round(chat.clientHeight * 0.15);
          const halfPage = Math.round(chat.clientHeight * 0.45);
          const scrollBehavior = prefersReducedMotion() ? "auto" : "smooth";
          if (key === "j") {
            e.preventDefault();
            chat.scrollBy({ top: step, behavior: scrollBehavior });
          } else if (key === "k") {
            e.preventDefault();
            chat.scrollBy({ top: -step, behavior: scrollBehavior });
          } else if (key === "d") {
            e.preventDefault();
            chat.scrollBy({ top: halfPage, behavior: scrollBehavior });
          } else if (key === "u") {
            e.preventDefault();
            chat.scrollBy({ top: -halfPage, behavior: scrollBehavior });
          }
        });

        // Loaded-chat navigation rail + direct-to-search shortcut orchestration
        // lives in the CodoxearChatNavigation controller
        // (codoxear/static/app_chat_navigation.js). app.js keeps DOM
        // construction for prevUserBtn/nextUserBtn and the thin wrappers below.
        // The controller is instantiated after the DOM nodes and message-row
        // helpers exist; it wires the prev/next button handlers and the
        // document keydown listener itself. Chat search internals stay here.
        const chatNavigationController = (function instantiateChatNavigationController() {
          const codoxearChatNavigation = window.CodoxearChatNavigation;
          if (!codoxearChatNavigation || typeof codoxearChatNavigation.createChatNavigationController !== "function")
            throw new Error("Codoxear chat navigation controller failed to load");
          return codoxearChatNavigation.createChatNavigationController({
            prevUserBtn,
            nextUserBtn,
            getSelected: () => selected,
            loadedUserMessageRows,
            loadedCopyMessageRows,
            loadedUserJumpTarget,
            loadedCopyJumpTarget,
            getScrollTop: () => chat.scrollTop,
            prefersReducedMotion,
            pulseNavigatedRow,
            setToast,
            openChatSearch,
            isTextEntryElement,
            modalIsolationTargets,
            isModalTargetOpen,
            addAppEvent,
            documentTarget: document,
          });
        })();

        function updateChatNavButtons() {
          chatNavigationController.syncButtons();
        }

        function jumpToLoadedUserMessage(direction) {
          chatNavigationController.jumpToLoadedUserMessage(direction);
        }

        function jumpToLoadedMessage(direction) {
          chatNavigationController.jumpToLoadedMessage(direction);
        }

        function clearChatSearchMarks() {
          codoxearMessageRows.clearChatSearchMarks(renderedMessageRows());
        }

        function compactChatSearchSnippet(text, query, limit = 96) {
          return codoxearDisplay.compactChatSearchSnippet(text, query, limit);
        }

        function chatSearchTranscriptHint(match, query) {
          return codoxearDisplay.chatSearchTranscriptHint(match, query);
        }

        // --- Loaded-chat search + older-history search orchestration now
        // lives in the CodoxearChatSearch controller
        // (codoxear/static/app_chat_search.js). app.js keeps DOM construction
        // for the search bar/controls, the row/text/mark helpers, transcript
        // rendering + older-load authority, and the thin wrappers below that
        // other app.js call sites and the chat navigation controller use.
        let chatSearchController;

        function openChatSearch() {
          chatSearchController.open();
        }

        function closeChatSearch() {
          chatSearchController.close();
        }

        function refreshLoadedChatSearch(options) {
          chatSearchController.refreshLoaded(options);
        }

        function stepChatSearch(delta) {
          return chatSearchController.step(delta);
        }

        // The direct-to-search shortcut (`/`) lives in the
        // CodoxearChatNavigation controller (codoxear/static/app_chat_navigation.js),
        // wired via chatNavigationController above.

        let activeTailHistoryCursor = null;

        function usableOlderHistoryCursor(data) {
          return codoxearTranscript.hasUsableOlderHistory(data) ? codoxearTranscript.historyCursorFromPayload(data) : null;
        }

        function oldestRenderedHistoryCursor() {
          return codoxearMessageRows.oldestRenderedHistoryCursor(renderedMessageRows()) || activeTailHistoryCursor;
        }

        function clearRenderedTranscriptRange() {
          activeTailHistoryCursor = null;
          setOlderState({ hasMore: false, isLoading: false });
          transcriptScrollRuntime.markLiveTail();
        }

        function initPageLimit() {
          return isMobile() ? INIT_PAGE_LIMIT_MOBILE : INIT_PAGE_LIMIT_DESKTOP;
        }

        function olderPageLimit() {
          return OLDER_PAGE_LIMIT;
        }

        const codoxearTranscript = window.CodoxearTranscript;
        if (
          !codoxearTranscript ||
          typeof codoxearTranscript.normalizeTailEvent !== "function" ||
          typeof codoxearTranscript.normalizeTranscriptState !== "function" ||
          typeof codoxearTranscript.normalizedTranscriptEvents !== "function" ||
          typeof codoxearTranscript.transcriptKey !== "function" ||
          typeof codoxearTranscript.historyCursorFromPayload !== "function" ||
          typeof codoxearTranscript.hasUsableOlderHistory !== "function" ||
          typeof codoxearTranscript.transcriptSnapshotFromData !== "function" ||
          typeof codoxearTranscript.transcriptIdentityFromData !== "function" ||
          typeof codoxearTranscript.tailCacheMatchesSession !== "function" ||
          typeof codoxearTranscript.rememberTailSnapshot !== "function" ||
          typeof codoxearTranscript.appendTailSnapshotEvents !== "function" ||
          typeof codoxearTranscript.createTranscriptSlotRuntime !== "function" ||
          typeof codoxearTranscript.createTypingRowRuntime !== "function" ||
          typeof codoxearTranscript.createTranscriptRenderRuntime !== "function" ||
          typeof codoxearTranscript.createTranscriptDomRuntime !== "function" ||
          typeof codoxearTranscript.createTranscriptScrollRuntime !== "function" ||
          typeof codoxearTranscript.createTranscriptEventRuntime !== "function" ||
          typeof codoxearTranscript.createOlderLoadRuntime !== "function" ||
          typeof codoxearTranscript.createLoadedChatSearchRuntime !== "function" ||
          typeof codoxearTranscript.createChatSearchAllRuntime !== "function"
        )
          throw new Error("Codoxear transcript helpers failed to load");

        const olderLoadRuntime = codoxearTranscript.createOlderLoadRuntime({
          olderWrap,
          olderButton: olderBtn,
          olderError,
          olderErrorText,
          AbortControllerCtor: AbortController,
          nowMs: () => performance.now(),
          autoCooldownMs: OLDER_AUTO_COOLDOWN_MS,
        });

        chatSearchController = (function instantiateChatSearchController() {
          const codoxearChatSearch = window.CodoxearChatSearch;
          if (!codoxearChatSearch || typeof codoxearChatSearch.createChatSearchController !== "function")
            throw new Error("Codoxear chat search controller failed to load");
          return codoxearChatSearch.createChatSearchController({
            chatSearchBtn,
            chatSearchInput,
            chatSearchPrevBtn,
            chatSearchNextBtn,
            chatSearchCloseBtn,
            chatSearchStatus,
            chatSearchAllHintEl,
            chatSearchBar,
            createLoadedChatSearchRuntime: codoxearTranscript.createLoadedChatSearchRuntime,
            createChatSearchAllRuntime: codoxearTranscript.createChatSearchAllRuntime,
            getSelected: () => selected,
            getPollGen: () => pollGen,
            api,
            setToast,
            openSession,
            handleAppAuthLoss,
            chatSearchTranscriptHint,
            syncVisibleTimeIndicator: () => transcriptScrollRuntime.syncVisibleTimeIndicator(),
            renderedMessageRows,
            rowSearchText,
            compareRowsInDomOrder,
            clearChatSearchMarks,
            applyChatSearchMarks,
            pulseNavigatedRow,
            prefersReducedMotion,
            oldestRenderedHistoryCursor,
            renderDetachedTranscriptWindow,
            invalidateOlderLoad,
            setOlderState,
            showOlderLoadError,
            hasOlderMessages,
            isLoadingOlderMessages,
            olderPageLimit,
            loadOlderMessages,
            olderLoadRuntime,
          });
        })();

        const transcriptSlotRuntime = codoxearTranscript.createTranscriptSlotRuntime({
          getSession: (sessionId) => sessionIndex.get(sessionId) || null,
          maxTailEvents: Math.max(INIT_PAGE_LIMIT_DESKTOP, INIT_PAGE_LIMIT_MOBILE),
        });

        function activeTranscriptSnapshot() {
          return transcriptSlotRuntime.activeSnapshot();
        }

        const typingRowRuntime = codoxearTranscript.createTypingRowRuntime({
          root: chatInner,
          bottomSentinel,
          el,
          shouldAutoScroll: () => transcriptScrollRuntime.snapshot().autoScroll,
          scheduleScrollToBottom: () => transcriptScrollRuntime.scheduleScrollToBottom(),
        });

        const transcriptScrollRuntime = codoxearTranscript.createTranscriptScrollRuntime({
          chat,
          jumpButton: jumpBtn,
          timeChip: chatTimeChip,
          requestAnimationFrame: (callback) => requestAnimationFrame(callback),
          hasSelection: () => Boolean(selected),
          isSearchOpen: () => chatSearchController.isOpen(),
          firstVisibleMessageRow,
          dayLabel,
          time24,
          shouldCancelOlderLoad: () => olderLoadRuntime.shouldCancelOnScroll(),
          cancelOlderLoad: invalidateOlderLoad,
          autoLoadOlder: () => { void loadOlderMessages({ auto: true }); },
          bottomThresholdPx: 80,
          olderTopTriggerPx: OLDER_TOP_TRIGGER_PX,
          olderCancelPx: OLDER_CANCEL_PX,
        });

        const transcriptDomRuntime = codoxearTranscript.createTranscriptDomRuntime({
          root: chatInner,
          olderWrap,
          bottomSentinel,
          el,
          ymd,
          dayLabel,
          getRenderedRows: renderedMessageRows,
          trimRenderedRowTargets: codoxearMessageRows.trimRenderedRowTargets,
          trimRowsBeforeViewportTargets: codoxearMessageRows.trimRowsBeforeViewportTargets,
          scrollRuntime: transcriptScrollRuntime,
          defaultWindowRows: CHAT_DOM_WINDOW,
          afterDecorate: () => {
            updateChatNavButtons();
            syncMessageCopyTabStops();
            if (chatSearchController.isOpen()) chatSearchController.refreshLoaded({ jump: false, preserveCurrent: true });
          },
        });

        function olderLoadSnapshot() {
          return olderLoadRuntime.snapshot();
        }

        function hasOlderMessages() {
          return olderLoadSnapshot().hasMore;
        }

        function isLoadingOlderMessages() {
          return olderLoadSnapshot().isLoading;
        }

        function normalizeTailEvent(ev) {
          return codoxearTranscript.normalizeTailEvent(ev);
        }

        function normalizeTranscriptState(data) {
          return codoxearTranscript.normalizeTranscriptState(data);
        }

        function transcriptKey(threadId, logPath) {
          return codoxearTranscript.transcriptKey(threadId, logPath);
        }

        function transcriptSnapshotFromData(data) {
          return codoxearTranscript.transcriptSnapshotFromData(data);
        }

        function transcriptIdentityFromData(data, fallback = null) {
          return codoxearTranscript.transcriptIdentityFromData(data, fallback);
        }

        function getSessionTranscriptSlot(sessionId) {
          return transcriptSlotRuntime.getSlot(sessionId);
        }

        function syncActiveTranscriptSlot(sessionId) {
          return transcriptSlotRuntime.syncActiveSlot(sessionId);
        }

        function dropPendingUserRows(sessionId, predicate = null) {
          if (!sessionId) return;
          const dropped = transcriptEventRuntime.dropPendingUsers(sessionId, predicate);
          if (selected !== sessionId) return;
          for (const item of dropped) {
            if (!item || !item.id) continue;
            const pendingEl = chatInner.querySelector(`.msg.user[data-local-id="${item.id}"]`);
            const row = pendingEl ? pendingEl.closest(".msg-row") : null;
            if (row) row.remove();
          }
        }

        function updateSessionTranscriptSlot(sessionId, data) {
          const change = transcriptSlotRuntime.updateSlot(sessionId, data);
          if (change.resetPending) dropPendingUserRows(sessionId, () => true);
          if (selected === sessionId) syncActiveTranscriptSlot(sessionId);
          return change;
        }

        function beginTranscriptRenewal(sessionId) {
          const change = transcriptSlotRuntime.beginRenewal(sessionId);
          if (!change) return;
          dropPendingUserRows(sessionId, () => true);
          if (selected === sessionId) syncActiveTranscriptSlot(sessionId);
        }

        function tailCacheMatchesSession(cache, session) {
          return transcriptSlotRuntime.tailCacheMatchesSession(cache, session);
        }

        function rememberTailSnapshot(sessionId, session, data) {
          return transcriptSlotRuntime.rememberTail(sessionId, session, data);
        }

        function appendTailSnapshotEvents(sessionId, events, { session = null, identityData = null, liveCursor: nextLiveCursor, busy, queueLen, token } = {}) {
          return transcriptSlotRuntime.appendTailEvents(sessionId, events, {
            session,
            identityData,
            liveCursor: nextLiveCursor,
            busy,
            queueLen,
            token,
          });
        }

        function restorePendingUserRowsForSession(sessionId) {
          if (!sessionId) return;
          const slot = getSessionTranscriptSlot(sessionId);
          const items = transcriptEventRuntime.pendingUsersForSession(sessionId, Number(slot.epoch || 0));
          for (const item of items) {
            if (!item || !item.id) continue;
            if (chatInner.querySelector(`.msg.user[data-local-id="${item.id}"]`)) continue;
            appendEvent({ role: "user", text: item.text, pending: true, localId: item.id, ts: item.t0 });
          }
        }

        function applySessionListTranscriptIdentity(sessionId, sessionMeta) {
          if (!sessionId || selected !== sessionId || !sessionMeta) return;
          const slotChange = updateSessionTranscriptSlot(sessionId, sessionMeta);
          if (!slotChange.resetPending) return;

          transcriptSlotRuntime.deleteTailCache(sessionId);
          transcriptSlotRuntime.clearLiveCursor();
          clearRenderedTranscriptRange();
          setAttachCount(0);
          invalidateOlderLoad();
          transcriptEventRuntime.resetRecentEvents();
          transcriptScrollRuntime.enableAutoScroll();
          clearTranscriptDom();
          if (slotChange.current.state === "pending_bind") {
            renderPendingTranscriptSlot(sessionId);
          } else {
            setOlderState({ hasMore: false, isLoading: false });
            transcriptScrollRuntime.syncJumpButton();
            kickPoll(0);
          }

          const running = Boolean(sessionMeta.busy);
          const queueLen = Number.isFinite(Number(sessionMeta.queue_len)) ? Number(sessionMeta.queue_len) : 0;
          turnOpen = running;
          setStatus({ running, queueLen });
          setContext(sessionMeta.token || null);
          setTyping(running);
        }

        function updateQueueBadge() {
          if (!queueBadgeEl) return;
          if (!selected) {
            queueBadgeEl.textContent = "";
            queueBadgeEl.style.display = "none";
            return;
          }
          const n = Math.max(0, Number(currentQueueLen) || 0);
          if (n > 0) {
            queueBadgeEl.textContent = String(n);
            queueBadgeEl.style.display = "inline-flex";
          } else {
            queueBadgeEl.textContent = "";
            queueBadgeEl.style.display = "none";
          }
          if (queueViewer.style.display === "flex") {
            void refreshQueueViewer();
          }
        }

          function markClickFirstPaint() {
            if (!clickMetricPending) return;
            clickMetricPending = false;
            const dt = performance.now() - clickLoadT0;
            pushPerfSample("click_to_first_message_ms", dt);
          }

        function updateTypingStatsFromSession(session) {
          if (!session) return;
          const current = typingRowRuntime.snapshot().stats || { thinking: 0, tools: 0 };
          const stats = {
            thinking: session.thinking,
            tools: session.tools,
          };
          // Live deltas are authoritative once observed; session-list counts
          // fill the initial projection and recover counts while idle polling
          // catches up.
          if (!turnOpen || (!current.thinking && !current.tools)) typingRowRuntime.updateTypingStats(stats);
        }

        function applyTypingMetaDelta(data) {
          const delta = data && data.meta_delta;
          if (!delta || typeof delta !== "object") return;
          typingRowRuntime.updateTypingStats(
            { thinking: delta.thinking, tools: delta.tool },
            { delta: true },
          );
        }

	        function setTyping(show) {
	          typingRowRuntime.setVisible(show);
	        }

        function ymd(d) {
          return codoxearDisplay.ymd(d);
        }

        function dayLabel(d) {
          return codoxearDisplay.dayLabel(d);
        }

        function time24(d) {
          return codoxearDisplay.time24(d);
        }

        function rebuildDecorations({ preserveScroll }) {
          transcriptDomRuntime.rebuildDecorations({ preserveScroll });
        }

        function trimRenderedRows({ fromTop, maxRows = CHAT_DOM_WINDOW }) {
          transcriptDomRuntime.trimRenderedRows({ fromTop, maxRows });
        }

        function trimRenderedRowsBeforeViewport({ maxRows = CHAT_DOM_WINDOW } = {}) {
          transcriptDomRuntime.trimRowsBeforeViewport({ maxRows, viewportTop: chat.scrollTop + 1 });
        }

        function messageRowDeps() {
          return {
            el,
            chatMarkdownHtmlCached,
            selectedSessionId: selected,
            upgradeCandidateFileRefs,
            time24,
            iconSvg,
            copyToClipboard,
            setToast,
            chatAssistantDedupeKey,
            setTimeout: window.setTimeout.bind(window),
            consoleError: console.error.bind(console),
          };
        }

        function makeRow(ev, { ts, pending }) {
          return codoxearMessageRows.makeRow(ev, { ts, pending }, messageRowDeps());
        }

        function safeMakeRow(ev, opts) {
          return codoxearMessageRows.safeMakeRow(ev, opts, messageRowDeps());
        }

      const codoxearMessageIdentity = window.CodoxearMessageIdentity;
      if (
        !codoxearMessageIdentity ||
        typeof codoxearMessageIdentity.normalizeTextForPendingMatch !== "function" ||
        typeof codoxearMessageIdentity.pendingMatchKey !== "function" ||
        typeof codoxearMessageIdentity.eventKey !== "function" ||
        typeof codoxearMessageIdentity.chatAssistantDedupeKey !== "function"
      )
        throw new Error("Codoxear message identity helpers failed to load");

      function normalizeTextForPendingMatch(s) {
        return codoxearMessageIdentity.normalizeTextForPendingMatch(s);
      }

      const transcriptEventRuntime = codoxearTranscript.createTranscriptEventRuntime({
        eventKey: codoxearMessageIdentity.eventKey,
        pendingMatchKey: codoxearMessageIdentity.pendingMatchKey,
        normalizePendingText: codoxearMessageIdentity.normalizeTextForPendingMatch,
        assistantDedupeKey: codoxearMessageIdentity.chatAssistantDedupeKey,
        maxRecentEventKeys: 320,
      });

      function eventKey(ev) {
        return codoxearMessageIdentity.eventKey(ev);
      }

        function markEventSeen(ev) {
          transcriptEventRuntime.markEventSeen(ev);
        }

        function isDuplicateEvent(ev) {
          return transcriptEventRuntime.isDuplicateEvent(ev);
        }

        function chatAssistantDedupeKey(ev) {
          return codoxearMessageIdentity.chatAssistantDedupeKey(ev);
        }

        function isAdjacentAssistantDuplicateEvent(ev) {
          return transcriptEventRuntime.isAdjacentAssistantDuplicateEvent(ev, {
            renderedAtLiveTail: transcriptScrollRuntime.snapshot().renderedAtLiveTail,
            rows: renderedMessageRows(),
          });
        }

        function pendingMatchKey(s) {
          return codoxearMessageIdentity.pendingMatchKey(s);
        }

      function isTranscriptRenewalCommand(raw, sessionId = selected) {
        const session = sessionId ? sessionIndex.get(sessionId) : null;
        if (!session || sessionAgentBackend(session) !== "codex") return false;
        return String(raw || "").trim() === "/new";
      }

      function takePendingUserMatch(ev, sessionId = selected, { allowUntimedCommit = true } = {}) {
        const slot = getSessionTranscriptSlot(sessionId);
        return transcriptEventRuntime.takePendingUserMatch(ev, sessionId, Number(slot.epoch || 0), { allowUntimedCommit });
      }

      function consumePendingUserIfMatches(ev, sessionId = selected) {
        const match = takePendingUserMatch(ev, sessionId);
        if (!match) return false;
        const { id } = match;
        const pendingEl = chatInner.querySelector(`.msg.user[data-local-id="${id}"]`);
        if (!pendingEl) return false;

          pendingEl.style.opacity = "1";
          pendingEl.removeAttribute("data-local-id");
          pendingEl.removeAttribute("data-pending");

          const mdEl = pendingEl.querySelector(".md");
          if (mdEl && typeof ev.text === "string") mdEl.innerHTML = chatMarkdownHtmlCached(ev.text, sessionId);

          const row = pendingEl.closest(".msg-row");
          if (row && typeof ev.ts === "number" && Number.isFinite(ev.ts)) row.dataset.ts = String(ev.ts);
          const tsEl = pendingEl.querySelector(".ts");
          if (tsEl && typeof ev.ts === "number" && Number.isFinite(ev.ts)) tsEl.textContent = time24(new Date(ev.ts * 1000));
          rebuildDecorations({ preserveScroll: true });
          markEventSeen(ev);
          return true;
        }

        const transcriptRenderRuntime = codoxearTranscript.createTranscriptRenderRuntime({
          root: chatInner,
          bottomSentinel,
          document,
          safeMakeRow,
          normalizeEvents: normalizedTranscriptEvents,
          consumePendingUserIfMatches,
          isDuplicateEvent,
          isAdjacentAssistantDuplicateEvent,
          markEventSeen,
          markFirstPaint: markClickFirstPaint,
          restorePendingRows: restorePendingUserRowsForSession,
          resetRecentEvents: () => transcriptEventRuntime.resetRecentEvents(),
          setOlderState,
          firstVisibleMessageRow,
          getScrollTop: () => chat.scrollTop,
          getSelectedSessionId: () => selected,
          domRuntime: transcriptDomRuntime,
          scrollRuntime: transcriptScrollRuntime,
          typingRowRuntime,
          historySlackRows: CHAT_DOM_WINDOW_WITH_HISTORY_SLACK,
        });

        function isMobile() {
          return codoxearViewport.isMobile();
        }

        function useDesktopSessionActions() {
          return codoxearViewport.useDesktopSessionActions();
        }

        function useTouchFileEditorControls() {
          return codoxearViewport.useTouchFileEditorControls();
        }

        function setSidebarOpen(open) {
          if (open) {
            document.body.classList.add("sidebar-open");
            storageSetItem("codexweb.sidebarOpen", "1");
          } else {
            document.body.classList.remove("sidebar-open");
            storageRemoveItem("codexweb.sidebarOpen");
          }
        }

        function setSidebarCollapsed(collapsed) {
          if (collapsed) {
            document.body.classList.add("sidebar-collapsed");
            storageSetItem("codexweb.sidebarCollapsed", "1");
          } else {
            document.body.classList.remove("sidebar-collapsed");
            storageRemoveItem("codexweb.sidebarCollapsed");
          }
        }

        async function clearCommitUnknownSend(sid, previewText = "") {
          const sessionId = String(sid || "").trim();
          if (!sessionId) return false;
          const preview = String(previewText || "").trim();
          const suffix = preview ? `\n\nPrompt: ${preview.slice(0, 240)}${preview.length > 240 ? "..." : ""}` : "";
          const confirmed = await confirmApp({
            title: "Clear unknown-send marker?",
            message: `Clear the unknown-send marker only after checking the transcript or terminal. This does not undo a prompt that may already have been sent.${suffix}`,
            confirmText: "Clear marker",
            cancelText: "Cancel",
            destructive: true,
          });
          if (!confirmed) return false;
          try {
            await api(`/api/sessions/${sessionId}/commit_unknown_send/clear`, { method: "POST", body: {} });
            setToast("unknown send marker cleared");
            await refreshSessions();
            updateQueueBadge();
            if (selected === sessionId) syncRecoveryUiForSession(sessionId);
            return true;
          } catch (e) {
            if (e && e.status === 401) {
              handleAppAuthLoss();
              return false;
            }
            setToast(`clear unknown send error: ${e && e.message ? e.message : "unknown error"}`);
            return false;
          }
        }

         const sidebarController = codoxearShell.createSidebarController({
           sessionsWrap,
           sidebarEmptyHint,
           el,
           iconSvg,
           sidebarRenderSignature,
           sessionDisplayName,
           sessionLaunchFailed,
           sessionLaunchPending,
           redactedLaunchErrorText,
           fmtRelativeAge,
           reasoningEffortMarker,
           sidebarModelText,
           baseName,
           sessionIsFast,
           agentBackendLogoPath,
           agentBackendDisplayName,
           sessionAgentBackend,
           sessionLaunchIcon,
           sessionLaunchLabel,
           confirmAction: (options) => confirmApp(options),
           api,
           clearDeletedSessionClientState,
           refreshSessions,
           setToast,
           openEditSession,
           duplicateSession: async (session) => {
             const cwd = session && session.cwd && session.cwd !== "?" ? session.cwd : "";
             if (!cwd) {
               setToast("cwd unavailable");
               return;
             }
             await spawnSessionWithCwd(
               cwd,
               null,
               null,
               "",
               sessionProviderChoice(session),
               session && session.model ? session.model : "default",
               session && session.reasoning_effort ? session.reasoning_effort : "high",
               sessionIsFast(session),
               !!(session && session.transport === "tmux"),
               null,
               sessionAgentBackend(session)
             );
           },
           selectSession,
           setSidebarOpen,
           now: () => Date.now(),
           performanceNow: () => performance.now(),
           consoleError: (...args) => console.error(...args),
         });

         async function refreshSessions() {
           if (sessionsRefreshInFlight) {
             sessionsRefreshQueued = true;
             return sessionsRefreshInFlight;
           }
           sessionsRefreshInFlight = (async () => {
             let result = latestSessions;
             try {
               do {
                 sessionsRefreshQueued = false;
                 result = await refreshSessionsOnce();
               } while (sessionsRefreshQueued && !appDisposed);
               return result;
             } finally {
               sessionsRefreshInFlight = null;
             }
           })();
           return sessionsRefreshInFlight;
         }

	         async function refreshSessionsOnce() {
	           const data = await api("/api/sessions");
          if (appDisposed) return latestSessions;
          const notModified = apiResponseNotModified(data);
          if (notModified && !sidebarController.hasDeferredRefresh()) return latestSessions;
          if (!notModified) {
            latestSessions = Array.isArray(data.sessions) ? data.sessions.slice() : [];
            newSessionDefaults =
              data && typeof data.new_session_defaults === "object" && data.new_session_defaults
                ? data.new_session_defaults
                : {
                    default_backend: "pi",
                    backends: {
                      codex: legacyCodexLaunchDefaults(),
                      pi: emptyPiLaunchDefaults(),
                      cc: emptyCcLaunchDefaults(),
                    },
                  };
            tmuxAvailable = !!data.tmux_available;
            recentCwds = Array.isArray(data.recent_cwds)
              ? data.recent_cwds.filter((cwd, idx, arr) => typeof cwd === "string" && cwd.trim() && arr.indexOf(cwd) === idx)
              : [];
            if (newSessionViewer.style.display === "flex") {
              const statusText = String(newSessionStatus.textContent || "").trim();
              syncNewSessionTmuxUi();
              renderNewSessionModelMenu();
              renderNewSessionReasoningMenu();
              syncNewSessionRunConfigUi();
              if (!statusText || statusText.startsWith("Launch defaults degraded for ")) newSessionStatus.textContent = newSessionDefaultsWarningText();
            }
            fileReferenceRuntime.clearDiscoveryCaches();
          }
          const swipeActions = !useDesktopSessionActions();
          const sessions = latestSessions
            .slice()
            .sort((a, b) => {
              const p = Number(b.final_priority || 0) - Number(a.final_priority || 0);
              if (p) return p;
              const u = Number(b.updated_ts || b.start_ts || 0) - Number(a.updated_ts || a.start_ts || 0);
              if (u) return u;
              const s0 = Number(b.start_ts || 0) - Number(a.start_ts || 0);
              if (s0) return s0;
              return String(a.session_id || "").localeCompare(String(b.session_id || ""));
            });
          sessionIndex = new Map();
          for (const session of sessions) sessionIndex.set(session.session_id, session);
          if (selected && !sessionIndex.has(selected)) clearSelectedSessionAfterRemoval(selected);
          if (selected) {
            applySessionListTranscriptIdentity(selected, sessionIndex.get(selected));
            syncRecoveryUiForSession(selected);
          }
          if (selected) syncStagedAttachmentsFromSelectedSession();
          else setStagedAttachments([]);
          const renderedSidebar = sidebarController.render(sidebarSessionEntries(sessions), {
            selectedId: selected,
            swipeActions,
          });
          if (!renderedSidebar) return sessions;
          if (selected) {
            const session = sessionIndex.get(selected);
            if (session) {
              titleLabel.textContent = sessionTitleWithId(session);
              updateTypingStatsFromSession(session);
            }
          }
          updateUnattendedBtnState();
          updateQueueBadge();
          syncComposerSendButton();
          syncQueueSubmitState();
          maybeSelectPendingHashSession();
          return sessions;
        }

        function appendEvent(ev) {
          transcriptRenderRuntime.appendEvent(ev);
        }

        function normalizedTranscriptEvents(events, { consumePending = false } = {}) {
          return codoxearTranscript.normalizedTranscriptEvents(events, {
            consumePending,
            selectedSessionId: selected,
            eventKey,
            takePendingMatch: takePendingUserMatch,
          });
        }

        function renderTranscript(events, { preserveScroll = false } = {}) {
          return transcriptRenderRuntime.renderTranscript(events, { preserveScroll });
        }

        function renderDetachedTranscriptWindow(events, { hasMore = false } = {}) {
          return transcriptRenderRuntime.renderDetachedTranscriptWindow(events, { hasMore });
        }

        function prependOlderEvents(allEvents, { preserveViewport = false } = {}) {
          return transcriptRenderRuntime.prependOlderEvents(allEvents, { preserveViewport });
        }

        async function loadOlderMessages({ auto = false, cancelOnScroll = true } = {}) {
          const state = olderLoadSnapshot();
          if (!selected || !state.hasMore || state.isLoading) return false;
          if (auto && !olderLoadRuntime.markAutoTrigger()) return false;
          const sid = selected;
          const gen = pollGen;
          const load = olderLoadRuntime.beginLoad({ cancelOnScroll });
          try {
            const reqCursor = oldestRenderedHistoryCursor();
            if (!reqCursor) throw new Error("history cursor missing");
            const data = await api(`/api/sessions/${sid}/messages/history?cursor=${encodeURIComponent(reqCursor)}&limit=${olderPageLimit()}`, {
              signal: load.signal,
            });
            if (selected !== sid || pollGen !== gen || !olderLoadRuntime.isCurrent(load)) return false;
            const evs = Array.isArray(data.events) ? data.events : [];
            activeTailHistoryCursor = usableOlderHistoryCursor(data);
            const nextHasOlder = Boolean(activeTailHistoryCursor);
            clearOlderLoadError();
            setOlderState({ hasMore: nextHasOlder, isLoading: false });
            if (evs.length) {
              prependOlderEvents(evs, { preserveViewport: auto });
              return true;
            }
            return false;
          } catch (e) {
            if (e && e.status === 401) {
              handleAppAuthLoss();
              return false;
            }
            if (selected !== sid || pollGen !== gen || !olderLoadRuntime.isCurrent(load)) return false;
            if (e && e.status === 409) {
              await openSession(sid, { useCache: false });
              return false;
            }
            setOlderState({ hasMore: hasOlderMessages(), isLoading: false });
            showOlderLoadError();
            return false;
          } finally {
            olderLoadRuntime.finishLoad(load);
          }
        }

        // Older-history search window loading (loadNearestOlderChatSearchWindow /
        // loadChatSearchCursorWindow) now lives in the CodoxearChatSearch
        // controller (codoxear/static/app_chat_search.js). app.js keeps the
        // transcript/older-load authority those paths invoke through injected
        // deps (olderLoadRuntime, loadOlderMessages, renderDetachedTranscript
        // Window, openSession, handleAppAuthLoss, invalidateOlderLoad,
        // setOlderState, showOlderLoadError).

        function maybeAutoLoadOlder() {
          transcriptScrollRuntime.maybeAutoLoadOlder();
        }

        function applySessionRuntimeFromTail(sessionId, data) {
          const slot = syncActiveTranscriptSlot(sessionId);
          transcriptSlotRuntime.setLiveCursor(slot.state === "bound" && typeof data.live_cursor === "string" && data.live_cursor ? data.live_cursor : null);
          activeTailHistoryCursor = usableOlderHistoryCursor(data);
          setOlderState({ hasMore: Boolean(activeTailHistoryCursor), isLoading: false });
          const nowBusy = Boolean(data && data.busy);
          turnOpen = nowBusy;
          const queueLen = data && Number.isFinite(Number(data.queue_len)) ? Number(data.queue_len) : 0;
          const session = sessionIndex.get(sessionId);
          updateTypingStatsFromSession(session);
          setStatus({ running: nowBusy, queueLen });
          setContext(data ? data.token : null);
          setTyping(nowBusy);
          if (slot.state === "bound") {
            const s = sessionIndex.get(sessionId);
            if (s) rememberTailSnapshot(sessionId, s, data);
          } else {
            transcriptSlotRuntime.deleteTailCache(sessionId);
          }
        }

        function renderSessionTail(events) {
          renderTranscript(events, { preserveScroll: false });
          markClickFirstPaint();
          transcriptScrollRuntime.scheduleScrollToBottom({ double: true });
        }


        function recoveryPromptPreview(text, maxLen = 320) {
          return codoxearDisplay.recoveryPromptPreview(text, maxLen);
        }

        function launchPresetFromSessionInfo(s) {
          return s && typeof s === "object" ? {
            session_id: s.session_id,
            cwd: s.cwd,
            agent_backend: s.agent_backend,
            provider_choice: s.provider_choice,
            model_provider: s.model_provider,
            preferred_auth_method: s.preferred_auth_method,
            model: s.model,
            reasoning_effort: s.reasoning_effort,
            service_tier: s.service_tier,
            transport: s.transport,
            tmux_session: s.tmux_session,
            tmux_window: s.tmux_window,
          } : null;
        }

        function recoveryDetailsText(sessionId, s) {
          const lines = [
            "Codoxear recovery details",
            `Session: ${sessionId}`,
          ];
          if (s && s.cwd) lines.push(`cwd: ${s.cwd}`);
          if (s && s.agent_backend) lines.push(`backend: ${s.agent_backend}`);
          if (s && sessionLaunchFailed(s)) {
            lines.push("state: launch failed");
            if (s.launch_stage) lines.push(`launch stage: ${s.launch_stage}`);
            const safeLaunchError = redactedLaunchErrorText(s.launch_error);
            if (safeLaunchError) lines.push(`launch error: ${safeLaunchError}`);
            if (s.model_provider) lines.push(`model provider: ${s.model_provider}`);
            if (s.model) lines.push(`model: ${s.model}`);
            if (s.reasoning_effort) lines.push(`reasoning: ${s.reasoning_effort}`);
            if (s.service_tier) lines.push(`service tier: ${s.service_tier}`);
            if (s.tmux_session || s.tmux_window) lines.push(`tmux: ${s.tmux_session || "-"}${s.tmux_window ? ":" + s.tmux_window : ""}`);
            const submitted = Number.isFinite(Number(s.submitted_user_message_count)) ? Number(s.submitted_user_message_count) : 0;
            if (submitted > 0) lines.push(`submitted prompts: ${submitted}`);
          }
          if (s && s.orphan_recovery) lines.push("state: missing session/orphan recovery");
          if (s && s.queue_recovery) lines.push("state: queued recovery items present");
          if (s && s.commit_unknown_send) lines.push("state: direct send commit unknown");
          const qn = s && Number.isFinite(Number(s.queue_len)) ? Number(s.queue_len) : 0;
          if (qn > 0) lines.push(`queued recovery items: ${qn}`);
          const preview = recoveryPromptPreview(s && s.commit_unknown_send_text ? s.commit_unknown_send_text : "", 2000);
          if (preview) lines.push("", "Unknown-send prompt:", preview);
          return lines.join("\n");
        }

        function clearSelectedSessionAfterRemoval(sessionId, { incrementPollGen = false, clearPollState = false } = {}) {
          if (selected !== sessionId) return false;
          handleFileViewerSessionUnavailable(sessionId);
          selected = null;
          abortMessagePollRequest();
          if (incrementPollGen) pollGen += 1;
          if (clearPollState) {
            if (pollTimer) clearTimeout(pollTimer);
            pollTimer = null;
            pollKickPending = false;
            pollKickDelayMs = null;
          }
          transcriptSlotRuntime.setActivePending();
          clearRenderedTranscriptRange();
          turnOpen = false;
          storageRemoveItem("codexweb.selected");
          setSessionHash("");
          titleLabel.textContent = "No session selected";
          setStatus({ running: false, queueLen: 0 });
          setContext(null);
          setTyping(false);
          if (typeof setStagedAttachments === "function") setStagedAttachments([]);
          else setAttachCount(0);
          resetChatRenderState();
          updateQueueBadge();
          if (unattendedController.isOpen()) hideUnattendedMenu();
          updateUnattendedBtnState();
          syncComposerSendButton();
          syncQueueSubmitState();
          syncAttachButtonState();
          return true;
        }

        function clearDeletedSessionClientState(sessionId) {
          const selectedCleared = clearSelectedSessionAfterRemoval(sessionId);
          transcriptSlotRuntime.deleteSession(sessionId);
          dropPendingUserRows(sessionId, () => true);
          return selectedCleared;
        }

        async function dismissFailedLaunchRecord(sessionId) {
          const s = sessionIndex.get(sessionId);
          if (!sessionLaunchFailed(s)) {
            setToast("launch record is not failed");
            return;
          }
          const confirmed = await confirmApp({
            title: "Dismiss launch record?",
            message: "Dismiss this launch record?",
            confirmText: "Dismiss",
            cancelText: "Cancel",
            destructive: true,
          });
          if (!confirmed) return;
          try {
            await api(`/api/sessions/${sessionId}/delete`, { method: "POST", body: {} });
            clearDeletedSessionClientState(sessionId);
            await refreshSessions();
            setToast("Dismissed launch record");
          } catch (err) {
            setToast(`dismiss error: ${err && err.message ? err.message : "unknown error"}`);
          }
        }

        function syncRecoveryUiForSession(sessionId) {
          if (selected !== sessionId) return;
          const s = sessionIndex.get(sessionId) || null;
          if (s) {
            const queueLen = Number.isFinite(Number(s.queue_len)) ? Number(s.queue_len) : 0;
            setStatus({ running: currentRunning, queueLen });
          }
          syncAttachButtonState();
          syncQueueSubmitState();
          syncComposerSendButton();
          updateUnattendedBtnState();
          updateQueueBadge();
        }

        function renderPendingTranscriptSlot(sessionId) {
          clearTranscriptDom();
          activeTailHistoryCursor = null;
          setOlderState({ hasMore: false, isLoading: false });
          transcriptScrollRuntime.markLiveTail();
          restorePendingUserRowsForSession(sessionId);
          markClickFirstPaint();
          transcriptScrollRuntime.syncJumpButton();
        }

        function renderTranscriptLoading(sessionId) {
          clearTranscriptDom();
          activeTailHistoryCursor = null;
          setOlderState({ hasMore: false, isLoading: false });
          transcriptScrollRuntime.markLiveTail();
          restorePendingUserRowsForSession(sessionId);
          const row = el("div", { class: "msg-row assistant typing-row transcript-loading-row" });
          row.dataset.role = "assistant";
          row.appendChild(el("div", { class: "msg assistant loading", role: "status", "aria-live": "polite", text: "Loading transcript…" }));
          chatInner.insertBefore(row, bottomSentinel);
          transcriptScrollRuntime.syncJumpButton();
        }

        function renderTranscriptLoadError(sessionId, err, { preserveTranscript = false } = {}) {
          for (const row of Array.from(chatInner.querySelectorAll(".transcript-error-row"))) row.remove();
          if (!preserveTranscript) {
            clearTranscriptDom();
            activeTailHistoryCursor = null;
            setOlderState({ hasMore: false, isLoading: false });
            transcriptScrollRuntime.markLiveTail();
            restorePendingUserRowsForSession(sessionId);
          }
          const reason = err && err.message ? ` ${err.message}` : "";
          const row = el("div", { class: "msg-row assistant typing-row transcript-error-row" });
          row.dataset.role = "assistant";
          const bubble = el("div", { class: "msg assistant error transcript-error", role: "alert" });
          bubble.appendChild(el("span", { class: "transcriptErrorText", text: `Could not load transcript.${reason}` }));
          const retryBtn = el("button", {
            class: "icon-btn text-btn transcriptRetryBtn",
            type: "button",
            text: "Retry",
            title: "Retry loading this transcript",
            "aria-label": "Retry loading this transcript",
          });
          retryBtn.onclick = (e) => {
            e.preventDefault();
            e.stopPropagation();
            if (selected !== sessionId) return;
            void openSession(sessionId, { useCache: true });
          };
          bubble.appendChild(retryBtn);
          row.appendChild(bubble);
          chatInner.insertBefore(row, bottomSentinel);
          turnOpen = false;
          setTyping(false);
          markClickFirstPaint();
          transcriptScrollRuntime.syncJumpButton();
        }

        function applyCachedTail(sessionId, cache, sessionMeta) {
          updateSessionTranscriptSlot(sessionId, {
            transcript_state: "bound",
            thread_id: cache.threadId || (sessionMeta ? sessionMeta.thread_id : null),
            log_path: cache.logPath || (sessionMeta ? sessionMeta.log_path : null),
          });
          syncActiveTranscriptSlot(sessionId);
          transcriptSlotRuntime.setLiveCursor(cache.liveCursor || null);
          activeTailHistoryCursor = typeof cache.historyCursor === "string" && cache.historyCursor ? cache.historyCursor : null;
          setOlderState({ hasMore: Boolean(cache.hasOlder && activeTailHistoryCursor), isLoading: false });
          renderSessionTail(cache.events);
          const metaBusy = Boolean(sessionMeta && sessionMeta.busy);
          const cachedBusy = Boolean(cache.busy) || metaBusy;
          const queueLen =
            sessionMeta && Number.isFinite(Number(sessionMeta.queue_len))
              ? Number(sessionMeta.queue_len)
              : Number.isFinite(Number(cache.queueLen))
                ? Number(cache.queueLen)
                : 0;
          turnOpen = cachedBusy;
          setStatus({ running: cachedBusy, queueLen });
          setContext(cache.token || (sessionMeta ? sessionMeta.token : null));
          updateTypingStatsFromSession(sessionMeta);
          setTyping(cachedBusy);
        }

        async function openSession(sessionId, { useCache = true, fallbackToCacheOnFailure = false } = {}) {
          pollGen += 1;
          const myGen = pollGen;
          if (typeof closeMessageEventSource === "function") closeMessageEventSource();
          abortOpenSessionTailRequest();
          abortMessagePollRequest();
          if (pollTimer) {
            clearTimeout(pollTimer);
            pollTimer = null;
          }
          pollKickPending = false;
          pollKickDelayMs = null;

          const oldSelected = selected;
          selected = sessionId;
          // Optimistically update sidebar active state for immediate visual feedback.
          // The next poll cycle re-renders the full sidebar, but this avoids the
          // perceived lag where the transcript switches before the highlight moves.
          sessionsWrap.querySelectorAll(".session.active").forEach((el) => el.classList.remove("active"));
          const optimisticActive = sessionsWrap.querySelector(`.session[data-session-id="${sessionId}"]`);
          if (optimisticActive) optimisticActive.classList.add("active");
          // Save the outgoing session's draft, load the incoming session's draft.
          if (oldSelected && oldSelected !== sessionId) saveSelectedComposerDraft(oldSelected);
          loadSelectedComposerDraft(sessionId);
          if (unattendedController.isOpen() && unattendedController.menuSessionId() !== sessionId) hideUnattendedMenu();
          storageSetItem("codexweb.selected", sessionId);
          setSessionHash(sessionId);
          transcriptSlotRuntime.setActivePending();
          clearRenderedTranscriptRange();
          turnOpen = false;
          if (typeof syncStagedAttachmentsFromSelectedSession === "function") syncStagedAttachmentsFromSelectedSession();
          else setAttachCount(0);
          updateQueueBadge();
          setStatus({ running: false, queueLen: 0 });
          setContext(null);
          setTyping(false);
          resetChatRenderState();

          const s = sessionIndex.get(sessionId);
          titleLabel.textContent = s ? sessionTitleWithId(s) : sessionId ? String(sessionId) : "No session selected";
          clickLoadT0 = performance.now();
          clickMetricPending = true;
          const optimisticBusy = Boolean(s && s.busy);
          const optimisticQueueLen = s && Number.isFinite(Number(s.queue_len)) ? Number(s.queue_len) : 0;
          turnOpen = optimisticBusy;
          setStatus({ running: optimisticBusy, queueLen: optimisticQueueLen });
          setContext(s ? s.token || null : null);
          updateTypingStatsFromSession(s);
          setTyping(optimisticBusy);
          const fileViewerSyncStarted = Boolean(isFileViewerOpen() && !currentFileDirty());
          if (fileViewerSyncStarted) {
            void ensureCurrentFileViewerSession().catch((e) => console.error("file viewer session sync failed after selection", e));
          }

          const cachedTail = s ? transcriptSlotRuntime.getTailCache(sessionId) : null;
          let displayedCachedTail = false;
          if (useCache && s && cachedTail && tailCacheMatchesSession(cachedTail, s) && Array.isArray(cachedTail.events) && cachedTail.events.length) {
            applyCachedTail(sessionId, cachedTail, s);
            displayedCachedTail = true;
          }
          if (!displayedCachedTail) renderTranscriptLoading(sessionId);

          let data;
          const tailRequest = beginOpenSessionTailRequest(sessionId, myGen);
          try {
            data = await api(`/api/sessions/${sessionId}/messages/tail?limit=${initPageLimit()}`, {
              signal: tailRequest.signal,
            });
          } catch (e) {
            if (e && e.status === 401) {
              handleAppAuthLoss();
              return null;
            }
            if (isOpenSessionTailAbortError(tailRequest, e)) return null;
            if (!isCurrentOpenSessionTailRequest(tailRequest)) return null;
            markMessagePollFailure();
            if (e && e.status === 404) {
              clearSelectedSessionAfterRemoval(sessionId, { clearPollState: true });
              void refreshSessions().catch((e2) => {
                if (e2 && e2.status === 401) handleAppAuthLoss();
                else console.error("refreshSessions failed after session disappeared", e2);
              });
              return null;
            }
            if (fallbackToCacheOnFailure && !displayedCachedTail && !useCache && s && cachedTail && tailCacheMatchesSession(cachedTail, s) && Array.isArray(cachedTail.events) && cachedTail.events.length) {
              applyCachedTail(sessionId, cachedTail, s);
              displayedCachedTail = true;
            }
            renderTranscriptLoadError(sessionId, e, { preserveTranscript: displayedCachedTail });
            if (!appDisposed && selected === sessionId && pollGen === myGen) kickPoll(messagePollDelayMs());
            return null;
          } finally {
            finishOpenSessionTailRequest(tailRequest);
          }
          if (!isCurrentOpenSessionTailRequest(tailRequest)) return null;
          markMessagePollSuccess();
          const slotChange = updateSessionTranscriptSlot(sessionId, data);
          if (slotChange.ignoredStaleBound) {
            renderPendingTranscriptSlot(sessionId);
            applySessionRuntimeFromTail(sessionId, { transcript_state: "pending_bind", busy: data.busy, queue_len: data.queue_len, token: data.token });
            if (slotChange.current.state !== "failed") kickPoll(900);
            return data;
          }
          if (slotChange.current.state === "bound" || slotChange.current.state === "failed") renderSessionTail(Array.isArray(data.events) ? data.events : []);
          else renderPendingTranscriptSlot(sessionId);
          applySessionRuntimeFromTail(sessionId, data);
          if (slotChange.current.state !== "failed") {
            if (typeof openMessageEventSource === "function") openMessageEventSource(sessionId, myGen);
            kickPoll(900);
          }
          if (isMobile()) setSidebarOpen(false);
          updateUnattendedBtnState();
          if (isFileViewerOpen() && !currentFileDirty() && !fileViewerSyncStarted) {
            void ensureCurrentFileViewerSession();
          } else if (isFileViewerOpen() && !currentFileDirty() && currentFileViewerSessionId() === sessionId) {
            void refreshFileCandidates({ sessionId }).catch((e) => console.error("file candidates refresh failed after transcript load", e));
          }
          return data;
        }

			        async function applyLiveMessageData(sid, gen, data) {
          if (gen !== pollGen || sid !== selected) return;
          markMessagePollSuccess();
          const slotInfo = transcriptSnapshotFromData(data);
          const nowBusy = Boolean(data.busy);
          const wasTurnOpen = turnOpen;
          if (activeTranscriptSnapshot().state === "bound" && slotInfo.state === "pending_bind") {
            updateSessionTranscriptSlot(sid, data);
            resetChatRenderState();
            renderPendingTranscriptSlot(sid);
            setAttachCount(0);
            applySessionRuntimeFromTail(sid, data);
            return;
          }
          if (activeTranscriptSnapshot().state === "bound" && slotInfo.state === "bound" && slotInfo.logPath !== activeTranscriptSnapshot().logPath) {
            await openSession(sid, { useCache: false });
            return;
          }
          transcriptSlotRuntime.setLiveCursor(typeof data.live_cursor === "string" && data.live_cursor ? data.live_cursor : null);
          const evs = Array.isArray(data.events) ? data.events : [];
          for (const ev of evs) appendEvent(ev);
          const turnStart = Boolean(data.turn_start);
          const turnEnd = Boolean(data.turn_end);
          const turnAborted = Boolean(data.turn_aborted);
          const newTurn = turnStart || (!wasTurnOpen && nowBusy);
          if (newTurn) typingRowRuntime.resetTypingStats();
          if (turnStart) turnOpen = true;
          if (!turnOpen && nowBusy) turnOpen = true;
          if ((turnEnd || turnAborted) && turnOpen) turnOpen = false;
          if (turnOpen && !nowBusy) turnOpen = false;
          applyTypingMetaDelta(data);
          setStatus({ running: Boolean(turnOpen || nowBusy), queueLen: data.queue_len });
          setContext(data.token);
          setTyping(Boolean(turnOpen || nowBusy));
          const s2 = sessionIndex.get(sid);
          if (evs.length) {
            appendTailSnapshotEvents(sid, evs, {
              session: s2,
              liveCursor: transcriptSlotRuntime.activeSnapshot().liveCursor,
              busy: Boolean(turnOpen || nowBusy),
              queueLen: data.queue_len,
              token: data.token,
              identityData: data,
            });
          }
          if (s2) titleLabel.textContent = sessionTitleWithId(s2);
        }

        async function pollMessages(sid = selected, gen = pollGen) {
          if (appDisposed || !sid) return;
          let pollRequest = null;
          try {
            if (!transcriptSlotRuntime.activeSnapshot().liveCursor) {
              if (activeTranscriptSnapshot().state === "pending_bind") {
                pollRequest = beginMessagePollRequest(sid, gen);
                const data = await api(`/api/sessions/${sid}/messages/tail?limit=${initPageLimit()}`, { signal: pollRequest.signal });
                if (gen !== pollGen || sid !== selected) return;
                markMessagePollSuccess();
                const slotChange = updateSessionTranscriptSlot(sid, data);
                if (slotChange.ignoredStaleBound) {
                  renderPendingTranscriptSlot(sid);
                  applySessionRuntimeFromTail(sid, { transcript_state: "pending_bind", busy: data.busy, queue_len: data.queue_len, token: data.token });
                  return;
                }
                if (slotChange.current.state === "bound" || slotChange.current.state === "failed") renderSessionTail(Array.isArray(data.events) ? data.events : []);
                applySessionRuntimeFromTail(sid, data);
                return;
              }
              if (activeTranscriptSnapshot().state === "failed") return;
              await openSession(sid, { useCache: false });
              return;
            }
            const reqCursor = transcriptSlotRuntime.activeSnapshot().liveCursor;
            pollRequest = beginMessagePollRequest(sid, gen);
            const data = await api(`/api/sessions/${sid}/messages/live?cursor=${encodeURIComponent(reqCursor)}`, { signal: pollRequest.signal });
            await applyLiveMessageData(sid, gen, data);
          } catch (e) {
            if (e && e.status === 401) {
              handleAppAuthLoss();
              return;
            }
            if (isMessagePollAbortError(pollRequest, e)) return;
            if (gen !== pollGen || sid !== selected) return;
            if (e && e.status === 409) {
              await openSession(sid, { useCache: false });
              return;
            }
            if (e && e.status === 404) {
              clearSelectedSessionAfterRemoval(sid, { incrementPollGen: true, clearPollState: true });
              try {
                await refreshSessions();
              } catch (e2) {
                console.error("refreshSessions failed after session disappeared", e2);
                toast.textContent = `refresh error: ${e2 && e2.message ? e2.message : "unknown error"}`;
              }
              return;
            }
            markMessagePollFailure();
            // Transient network errors (fetch failed entirely, no HTTP status)
            // are self-recovering via poll backoff — don't toast them. Only
            // surface errors where the server actually responded with a status.
            if (e && typeof e.status === "number") {
              toast.textContent = `error: ${e.message}`;
            } else {
              console.warn("message poll network error", e && e.message);
            }
          } finally {
            finishMessagePollRequest(pollRequest);
          }
        }

        async function pollLoop() {
          if (appDisposed || !selected || messageSseOpen) return;
          if (pollLoopBusy) {
            pollKickPending = true;
            return;
          }
          pollLoopBusy = true;
          const mySid = selected;
          const myGen = pollGen;
          try {
            await pollMessages(mySid, myGen);
          } finally {
            pollLoopBusy = false;
          }
          if (pollKickPending) {
            const delay = pollKickDelayMs == null ? 0 : pollKickDelayMs;
            pollKickPending = false;
            pollKickDelayMs = null;
            kickPoll(delay);
            return;
          }
          if (appDisposed || selected !== mySid || pollGen !== myGen) return;
          pollTimer = setTimeout(pollLoop, messagePollDelayMs());
        }

        function kickPoll(ms = 0) {
          if (appDisposed || messageSseOpen) return;
          const delay = normalizeMessagePollKickDelay(ms);
          if (pollTimer) {
            clearTimeout(pollTimer);
            pollTimer = null;
          }
          if (pollLoopBusy) {
            pollKickPending = true;
            pollKickDelayMs = delay;
            return;
          }
          pollTimer = setTimeout(pollLoop, delay);
        }

        async function jumpToLatest() {
          if (!selected) return;
          const sid = selected;
          invalidateOlderLoad();
          transcriptScrollRuntime.enableAutoScroll();
          try {
            await openSession(sid, { useCache: false, fallbackToCacheOnFailure: true });
          } catch (e) {
            if (selected !== sid) return;
            setToast(`jump error: ${e && e.message ? e.message : "unknown error"}`);
          }
          if (selected !== sid) return;
          transcriptScrollRuntime.scheduleScrollToBottom({ syncJump: true });
          kickPoll(0);
        }

        async function selectSession(id) {
          await openSession(id, { useCache: true });
        }

        function rememberPendingHashSession(sid) {
          pendingHashSessionId = String(sid || "").trim();
        }

        function maybeSelectPendingHashSession() {
          const sid = pendingHashSessionId;
          if (!sid || pendingHashSessionSelectInFlight) return;
          if (sessionIdFromHash() !== sid) {
            rememberPendingHashSession("");
            return;
          }
          if (sid === selected) {
            rememberPendingHashSession("");
            return;
          }
          const session = sessionIndex.get(sid);
          if (!sessionSelectable(session)) return;
          rememberPendingHashSession("");
          pendingHashSessionSelectInFlight = true;
          void selectSession(sid)
            .catch((e) => {
              if (e && e.status === 401) handleAppAuthLoss();
              else console.error("pending hash session select failed", e);
            })
            .finally(() => {
              pendingHashSessionSelectInFlight = false;
            });
        }

        async function selectSessionFromHash({ refreshIfMissing = false, deferIfMissing = false } = {}) {
          const sid = sessionIdFromHash();
          if (!sid) {
            rememberPendingHashSession("");
            return;
          }
          if (sid === selected) {
            rememberPendingHashSession("");
            return;
          }
          let session = sessionIndex.get(sid);
          if (!session && refreshIfMissing) {
            try {
              await refreshSessions();
            } catch (e) {
              if (e && e.status === 401) handleAppAuthLoss();
              else console.error("hash session refresh failed", e);
              return;
            }
            session = sessionIndex.get(sid);
          }
          if (!sessionSelectable(session)) {
            if (deferIfMissing) rememberPendingHashSession(sid);
            return;
          }
          rememberPendingHashSession("");
          await selectSession(sid);
        }

        // Unattended menu state, async load/save orchestration, input draft
        // handling, menu focus/visibility, and control event handling live in
        // the CodoxearUnattended controller (codoxear/static/app_unattended.js).
        // app.js owns DOM construction for the unattended button/menu/controls,
        // the updateUnattendedBtnState shell projection (which delegates the
        // unattended-specific projection to the controller), and the thin
        // delegating wrappers below. The controller is instantiated after the
        // DOM nodes exist; it wires the button/menu/input handlers and the
        // document Escape/click + window resize listeners itself.
        const unattendedController = (function instantiateUnattendedController() {
          const codoxearUnattended = window.CodoxearUnattended;
          if (!codoxearUnattended || typeof codoxearUnattended.createUnattendedController !== "function")
            throw new Error("Codoxear unattended controller failed to load");
          return codoxearUnattended.createUnattendedController({
            unattendedBtn,
            unattendedMenu,
            enabledEl: $("#unattendedEnabled"),
            cooldownEl: $("#unattendedCooldownMinutes"),
            remainingEl: $("#unattendedRemainingInjections"),
            requestEl: $("#unattendedRequest"),
            getSelected: () => selected,
            getSessionInfo: (sid) => sessionIndex.get(sid),
            isAppDisposed: () => appDisposed,
            api,
            refreshSessions,
            handleAppAuthLoss,
            setToast,
            addAppEvent,
            documentTarget: document,
            windowTarget: window,
            requestFrame: requestAnimationFrame,
            setTimeout,
            clearTimeout,
            requestShellProjection: updateUnattendedBtnState,
          });
        })();

        // App-shell button projection. The unattended-specific projection
        // (button disabled/title/active, cfg cache sync from session fields,
        // number-input draft sync, menu enabled-checkbox sync, and the
        // close-menu-when-selected-changes guard) is delegated to the
        // controller. Everything else (title edit, attach/file/send/queue/diag
        // buttons, context bar, chat nav, chat-search close) stays here.
        function updateUnattendedBtnState() {
          syncTitleEditState();
          unattendedController.syncButtonState();
          syncAttachButtonState();
          const fileViewerBlocked = Boolean(selected && selectedSessionLaunchFailed());
          const fileViewerLabel = !selected ? "Select a session to view files" : fileViewerBlocked ? "Failed launch has no file browser" : "View file";
          fileBtn.disabled = !selected || fileViewerBlocked;
          fileBtn.title = fileViewerLabel;
          fileBtn.setAttribute("aria-label", fileViewerLabel);
          chatSearchBtn.disabled = !selected;
          chatNavRail.style.display = selected ? "flex" : "none";
          chatEmptyState.style.display = selected ? "none" : "flex";
          if (!selected && chatSearchController.isOpen()) closeChatSearch();
          updateChatNavButtons();
          syncQueueSubmitState();
          syncComposerSendButton();
          diagBtn.disabled = !selected;
        }

        function hideUnattendedMenu(opts) {
          return unattendedController.hide(opts);
        }

        function showUnattendedMenu(opts) {
          return unattendedController.show(opts);
        }

        function toggleUnattendedMenu(opts) {
          return unattendedController.toggle(opts);
        }
        // --- Voice / Settings / Notifications / Announcement orchestration
        // now lives in the CodoxearVoice controller
        // (codoxear/static/app_voice.js). app.js keeps DOM construction for
        // announceBtn, notificationBtn, liveAudio, and the voice settings
        // dialog nodes, and delegates every voice/settings/notification/
        // announcement call site through the thin wrappers below. The
        // controller owns its state, handlers, timers, and HLS lifecycle and
        // exposes dispose() for cleanupApp.
        let voiceController;
        function instantiateVoiceController() {
          return codoxearVoice.createVoiceController({
            announceBtn,
            notificationBtn,
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
            voiceSettingsCancelBtn: $("#voiceSettingsCancelBtn"),
            voiceSettingsSaveBtn: $("#voiceSettingsSaveBtn"),
            isAppDisposed: () => appDisposed,
            api,
            setToast,
            handleAppAuthLoss,
            prepareModalOpen,
            afterModalVisibilityChanged,
            resolveAppUrl,
            versionedShellAssetPath,
            storageGetItem,
            storageSetItem,
            storageRemoveItem,
            focusSessionFromNotification: (sid) => {
              if (sessionIdFromHash() !== sid) setSessionHash(sid);
              void selectSessionFromHash({ refreshIfMissing: true, deferIfMissing: true }).catch((e) => {
                if (e && e.status === 401) handleAppAuthLoss();
                else console.error("desktop notification session select failed", e);
              });
            },
          });
        }
        voiceController = instantiateVoiceController();
        function voiceAnnouncementsEnabled() {
          return voiceController.voiceAnnouncementsEnabled();
        }
        function notificationsEnabledLocally() {
          return voiceController.notificationsEnabledLocally();
        }
        function loadVoiceSettings() {
          return voiceController.loadVoiceSettings();
        }
        function syncNotificationState(serverSnapshot) {
          return voiceController.syncNotificationState(serverSnapshot);
        }
        function pollNotificationFeed(opts) {
          return voiceController.pollNotificationFeed(opts);
        }
        function resumeAnnouncementRuntime(opts) {
          return voiceController.resumeAnnouncementRuntime(opts);
        }
        function showVoiceSettingsDialog() {
          return voiceController.showVoiceSettingsDialog();
        }
        function hideVoiceSettingsDialog() {
          return voiceController.hideVoiceSettingsDialog();
        }
        function renderRecentCwdOptions() {
          return newSessionController.renderRecentCwdOptions();
        }

        function filteredRecentCwdOptions() {
          return newSessionController.filteredRecentCwdOptions();
        }

        function hideEditSession() {
          editSessionId = null;
          editStatus.textContent = "";
          editSaveBtn.disabled = false;
          editDependencyMenuOpen = false;
          applyDialogMenus();
          if (editViewer.open) editViewer.close();
          afterModalVisibilityChanged();
        }

        function syncEditPriorityLabel() {
          editPriorityValue.textContent = formatPriorityOffset(editPriorityRange.value);
        }

        function setEditSnoozeMode(mode) {
          editSnoozeMode = ["none", "4h", "tomorrow", "custom"].includes(mode) ? mode : "none";
          for (const [value, btn] of editSnoozeModeButtons.entries()) {
            btn.classList.toggle("active", value === editSnoozeMode);
          }
          editSnoozeCustomRow.style.display = editSnoozeMode === "custom" ? "grid" : "none";
        }

        function tomorrowSnoozeSeconds() {
          const d = new Date();
          d.setDate(d.getDate() + 1);
          d.setHours(9, 0, 0, 0);
          return Math.floor(d.getTime() / 1000);
        }

        function fillCustomSnoozeInputs(tsSeconds) {
          const ts = Number(tsSeconds);
          const d = Number.isFinite(ts) && ts > 0 ? new Date(ts * 1000) : new Date(Date.now() + 24 * 3600 * 1000);
          const yyyy = String(d.getFullYear()).padStart(4, "0");
          const mm = String(d.getMonth() + 1).padStart(2, "0");
          const dd = String(d.getDate()).padStart(2, "0");
          const hh = String(d.getHours()).padStart(2, "0");
          const mi = String(d.getMinutes()).padStart(2, "0");
          editSnoozeCustomDate.value = `${yyyy}-${mm}-${dd}`;
          editSnoozeCustomTime.value = `${hh}:${mi}`;
        }

        function fillDependencyOptions(currentSid, currentDependencySid) {
          editDependencyMenu.innerHTML = "";
          const addItem = (value, label, active) => {
            const btn = el("button", {
              class: "fileMenuItem" + (active ? " active" : ""),
              type: "button",
              title: label,
            });
            btn.appendChild(el("span", { class: "fileMenuPath", text: label }));
            btn.onclick = () => {
              editDependencyBtn.dataset.value = value || "";
              setDependencyButtonContent();
              editDependencyMenuOpen = false;
              applyDialogMenus();
            };
            editDependencyMenu.appendChild(btn);
          };
          addItem("", "No dependency", !currentDependencySid);
          for (const s of sessionIndex.values()) {
            if (!s || s.session_id === currentSid) continue;
            const label = `${sessionDisplayName(s)}${s.cwd ? ` | ${baseName(s.cwd)}` : ""}`;
            addItem(s.session_id, label, currentDependencySid === s.session_id);
          }
          editDependencyBtn.dataset.value = currentDependencySid || "";
          setDependencyButtonContent();
        }

        function setDependencyButtonContent() {
          const value = String(editDependencyBtn.dataset.value || "");
          let label = "No dependency";
          if (value) {
            const s = sessionIndex.get(value);
            if (s) label = `${sessionDisplayName(s)}${s.cwd ? ` | ${baseName(s.cwd)}` : ""}`;
          }
          setPickerButtonContent(editDependencyBtn, label);
        }

        function applyNewSessionCwdSuggestion(cwd) {
          return newSessionController.applyNewSessionCwdSuggestion(cwd);
        }

        function renderRecentCwdMenu() {
          return newSessionController.renderRecentCwdMenu();
        }

        function syncNewSessionNamePlaceholder() {
          return newSessionController.syncNewSessionNamePlaceholder();
        }

        function newSessionResumeLabel(item) {
          return newSessionController.newSessionResumeLabel(item);
        }

        function setPickerButtonContent(button, primaryText, secondaryText = "", placeholder = false) {
          if (!button) return;
          button.innerHTML = "";
          const textWrap = el("span", { class: `pickerButtonText${placeholder ? " placeholder" : ""}` });
          textWrap.appendChild(el("span", { class: "pickerButtonPrimary", text: String(primaryText || "") }));
          if (secondaryText) textWrap.appendChild(el("span", { class: "pickerButtonSecondary", text: String(secondaryText) }));
          button.appendChild(textWrap);
          button.appendChild(el("span", { class: "pickerButtonChevron", html: iconSvg("chevronDown") }));
        }

        function setNewSessionResumeSelection(item) {
          return newSessionController.setNewSessionResumeSelection(item);
        }

        function renderNewSessionBackendTabs() {
          newSessionBackendTabs.innerHTML = "";
          for (const backend of ["pi", "codex", "cc"]) {
            const active = newSessionBackend === backend;
            const btn = el("button", {
              class: `agentBackendTab${active ? " active" : ""}`,
              type: "button",
              title: agentBackendDisplayName(backend),
              "aria-label": agentBackendDisplayName(backend),
            }, [
              el("img", {
                class: "agentBackendTabLogo",
                src: agentBackendLogoPath(backend),
                alt: `${agentBackendDisplayName(backend)} logo`,
                width: "20",
                height: "20",
              }),
            ]);
            btn.onclick = () => setNewSessionBackend(backend, { resetSelections: true });
            newSessionBackendTabs.appendChild(btn);
          }
          newSessionBackendName.textContent = agentBackendDisplayName(newSessionBackend);
        }

        const newSessionController = codoxearNewSession.createNewSessionController({
          backend: () => newSessionBackend,
          provider: () => newSessionProvider,
          reasoningEffort: () => newSessionReasoningEffort,
          literalModelInputValue: () => newSessionLiteralModelInputValue,
          launchPresetProviderAbsent: () => newSessionLaunchPresetProviderAbsent,
          defaultsSource: () => newSessionDefaults,
          latestSessions: () => latestSessions,
          tmuxAvailable: () => tmuxAvailable,
          assignProvider: (value) => {
            newSessionProvider = value;
          },
          assignReasoningEffort: (value) => {
            newSessionReasoningEffort = value;
          },
          assignLiteralModelInputValue: (value) => {
            newSessionLiteralModelInputValue = value;
          },
          assignLaunchPresetProviderAbsent: (value) => {
            newSessionLaunchPresetProviderAbsent = Boolean(value);
          },
          modelInput: newSessionModelInput,
          modelField: newSessionModelField,
          status: newSessionStatus,
          reasoningBtn: newSessionReasoningBtn,
          setPickerButtonContent: (button, primaryText, secondaryText, placeholder) => setPickerButtonContent(button, primaryText, secondaryText, placeholder),
          renderReasoningMenu: () => renderNewSessionReasoningMenu(),
          renderModelMenu: () => renderNewSessionModelMenu(),
          setFast: (value) => setNewSessionFast(value),
          setBackend: (value, opts) => setNewSessionBackend(value, opts),
          setTmuxChecked: (value) => {
            newSessionTmuxToggle.checked = value;
          },
          applyDialogMenus: () => applyDialogMenus(),
          closeModelMenu: () => {
            newSessionModelMenuOpen = false;
            newSessionModelMenuFocus = -1;
          },
          cwdInput: newSessionCwdInput,
          cwdMenu: newSessionCwdMenu,
          cwdField: newSessionCwdField,
          cwdHint: newSessionCwdHint,
          nameInput: newSessionNameInput,
          recentCwds: () => recentCwds,
          cwdMenuFocus: () => newSessionCwdMenuFocus,
          assignCwdMenuFocus: (value) => {
            newSessionCwdMenuFocus = value;
          },
          closeCwdMenu: () => {
            newSessionCwdMenuOpen = false;
            newSessionCwdMenuFocus = -1;
          },
          el,
          resumeMenu: newSessionResumeMenu,
          resumeBtn: newSessionResumeBtn,
          closeResumeMenu: () => {
            newSessionResumeMenuOpen = false;
          },
          fetchResumeCandidates: (cwd, backend) => api(`/api/session_resume_candidates?cwd=${encodeURIComponent(cwd)}&agent_backend=${encodeURIComponent(backend)}`),
          tmuxToggle: newSessionTmuxToggle,
          tmuxField: newSessionTmuxField,
          worktreeToggle: newSessionWorktreeToggle,
          worktreeInput: newSessionWorktreeInput,
          worktreeField: newSessionWorktreeField,
          startBtn: newSessionStartBtn,
        });
        function newSessionProviderChoices() {
          return newSessionController.newSessionProviderChoices();
        }

        function newSessionHasProviderChoices() {
          return newSessionController.newSessionHasProviderChoices();
        }

        function defaultNewSessionProviderChoice() {
          return newSessionController.defaultNewSessionProviderChoice();
        }

        function newSessionProviderModelDisplay(model, providerChoice = "") {
          return newSessionController.newSessionProviderModelDisplay(model, providerChoice);
        }

        function newSessionAllowsCustomProvider() {
          return newSessionController.newSessionAllowsCustomProvider();
        }

        function parseNewSessionProviderModelInput(value = newSessionModelInput.value) {
          return newSessionController.parseNewSessionProviderModelInput(value);
        }

        function rememberedNewSessionProviderModelChoice() {
          return newSessionController.rememberedNewSessionProviderModelChoice();
        }

        function newSessionDefaultsWarningText() {
          return newSessionController.newSessionDefaultsWarningText();
        }

        function clearNewSessionProviderModelError() {
          return newSessionController.clearNewSessionProviderModelError();
        }

        function syncNewSessionProviderFromModelInput() {
          return newSessionController.syncNewSessionProviderFromModelInput();
        }

        function currentNewSessionModelForCapabilities() {
          return newSessionController.currentNewSessionModelForCapabilities();
        }

        function currentReasoningChoices() {
          return newSessionController.currentReasoningChoices();
        }

        function renderNewSessionReasoningMenu() {
          newSessionReasoningMenu.innerHTML = "";
          const items = currentReasoningChoices();
          for (const value of items) {
            const label = value;
            const btn = el("button", {
              class: "fileMenuItem" + (newSessionReasoningEffort === value ? " active" : ""),
              type: "button",
              title: label,
            });
            btn.appendChild(el("span", { class: "fileMenuPath", text: label }));
            btn.onclick = () => {
              setNewSessionReasoningEffort(value);
              newSessionReasoningMenuOpen = false;
              applyDialogMenus();
            };
            newSessionReasoningMenu.appendChild(btn);
          }
        }

        function syncNewSessionRunConfigUi() {
          const defaults = defaultsForAgentBackend(newSessionBackend);
          const supportsFast = !!defaults.supports_fast;
          const hasProviders = newSessionHasProviderChoices() || newSessionAllowsCustomProvider();
          newSessionModelLabel.textContent = hasProviders ? "Provider / model" : "Model";
          newSessionModelInput.placeholder = hasProviders ? "provider/model or model" : "Model";
          newSessionFastField.style.display = supportsFast ? "" : "none";
          if (!supportsFast) setNewSessionFast(false);
        }

        function setNewSessionBackend(value, { resetSelections = false } = {}) {
          const next = normalizeAgentBackendName(value);
          const previous = newSessionBackend;
          newSessionBackend = next;
          rememberBackendChoice(next);
          const defaults = defaultsForAgentBackend(next);
          const providerChoices = providerChoicesForBackend(next);
          const defaultProvider = typeof defaults.provider_choice === "string" ? defaults.provider_choice.trim() : "";
          const rememberedProvider = loadRememberedProviderChoice(next);
          if (resetSelections || previous !== next || !providerChoices.includes(newSessionProvider)) {
            setNewSessionProvider((rememberedProvider && providerChoices.includes(rememberedProvider) ? rememberedProvider : "") || defaultProvider || providerChoices[0] || "");
          } else {
            setNewSessionProvider(newSessionProvider);
          }
          const modelDefault = typeof defaults.model === "string" ? defaults.model.trim() : "";
          if (resetSelections || previous !== next) {
            const rememberedPair = rememberedNewSessionProviderModelChoice();
            const selectedPair = rememberedPair || parseNewSessionProviderModelInput(newSessionProviderModelDisplay(modelDefault || "default", newSessionProvider));
            if (selectedPair.providerChoice && (providerChoices.includes(selectedPair.providerChoice) || newSessionAllowsCustomProvider())) {
              setNewSessionProvider(selectedPair.providerChoice);
            }
            newSessionModelInput.value = newSessionProviderModelDisplay(selectedPair.model || modelDefault || "default", selectedPair.providerAbsent ? "" : selectedPair.providerChoice || newSessionProvider);
            newSessionLiteralModelInputValue = selectedPair.providerAbsent ? newSessionModelInput.value : "";
            newSessionLaunchPresetProviderAbsent = Boolean(selectedPair.providerAbsent);
            clearNewSessionProviderModelError();
          }
          const reasoningChoices = currentReasoningChoices();
          const defaultEffort = typeof defaults.reasoning_effort === "string" ? defaults.reasoning_effort.trim().toLowerCase() : "";
          if (resetSelections || previous !== next || !reasoningChoices.includes(newSessionReasoningEffort)) {
            setNewSessionReasoningEffort(defaultEffort || reasoningChoices[0] || "high");
          } else {
            setNewSessionReasoningEffort(newSessionReasoningEffort);
          }
          if (resetSelections || previous !== next) {
            setNewSessionFast(String(defaults.service_tier || "").trim().toLowerCase() === "fast");
          }
          syncNewSessionRunConfigUi();
          renderNewSessionBackendTabs();
          renderNewSessionReasoningMenu();
          renderNewSessionModelMenu();
          scheduleNewSessionResumeLoad();
        }

        function setNewSessionProvider(value) {
          return newSessionController.setNewSessionProvider(value);
        }

        function newSessionModelOption(model, { providerChoice = "", recent = false, configured = false, providerAbsent = false } = {}) {
          return newSessionController.newSessionModelOption(model, { providerChoice, recent, configured, providerAbsent });
        }

        function sessionModelOptions() {
          return newSessionController.sessionModelOptions();
        }

        function filteredNewSessionModelOptions() {
          return newSessionController.filteredNewSessionModelOptions();
        }

        function setNewSessionReasoningEffort(value) {
          return newSessionController.setNewSessionReasoningEffort(value);
        }

        function setNewSessionFast(value) {
          newSessionFast = !!value;
          newSessionFastToggle.checked = newSessionFast;
        }

        function syncNewSessionCwdHint() {
          return newSessionController.syncNewSessionCwdHint();
        }

        function setNewSessionCwdError(message) {
          return newSessionController.setNewSessionCwdError(message);
        }

        function clearNewSessionCwdInfo() {
          return newSessionController.clearNewSessionCwdInfo();
        }

        function syncNewSessionTmuxUi() {
          return newSessionController.syncNewSessionTmuxUi();
        }

        function syncNewSessionWorktreeUi() {
          return newSessionController.syncNewSessionWorktreeUi();
        }

        function renderNewSessionResumeMenu() {
          return newSessionController.renderNewSessionResumeMenu();
        }

        function selectNewSessionModel(option) {
          return newSessionController.selectNewSessionModel(option);
        }

        function renderNewSessionModelMenu() {
          newSessionModelMenu.innerHTML = "";
          const items = filteredNewSessionModelOptions();
          const raw = String(newSessionModelInput.value || "").trim();
          const defaults = defaultsForAgentBackend(newSessionBackend);
          const configured = typeof defaults.model === "string" ? defaults.model.trim() : "";
          if (newSessionModelMenuFocus < 0) {
            const selected = raw || configured;
            if (selected) {
              const selectedIdx = items.findIndex((item) => item.displayText === selected || item.model === selected);
              if (selectedIdx >= 0) newSessionModelMenuFocus = selectedIdx;
            }
          }
          if (newSessionModelMenuFocus >= items.length) newSessionModelMenuFocus = items.length ? items.length - 1 : -1;
          if (!items.length) {
            newSessionModelMenu.appendChild(el("div", { class: "pickerEmpty", text: "No matching models" }));
            newSessionModelInput.removeAttribute("aria-activedescendant");
            return items;
          }
          for (const [idx, item] of items.entries()) {
            const model = item.model;
            const title = item.displayText || newSessionProviderModelDisplay(model, item.providerChoice);
            const active = newSessionModelMenuFocus === idx || (newSessionModelMenuFocus < 0 && (raw === title || raw === model));
            const btn = el("button", {
              id: `newSessionModelOption-${idx}`,
              class: "fileMenuItem" + (active ? " active" : ""),
              type: "button",
              role: "option",
              "aria-selected": active ? "true" : "false",
              "aria-label": title,
              title,
            });
            btn.appendChild(el("span", { class: "fileMenuPath", text: model }));
            if (item.providerChoice) btn.appendChild(el("span", { class: "fileMenuHint", text: item.providerChoice }));
            btn.onmousedown = (e) => e.preventDefault();
            btn.onclick = () => selectNewSessionModel(item);
            newSessionModelMenu.appendChild(btn);
          }
          if (newSessionModelMenuFocus >= 0) newSessionModelInput.setAttribute("aria-activedescendant", `newSessionModelOption-${newSessionModelMenuFocus}`);
          else newSessionModelInput.removeAttribute("aria-activedescendant");
          return items;
        }

        function scheduleNewSessionResumeLoad() {
          return newSessionController.scheduleNewSessionResumeLoad();
        }

        function applyDialogMenus() {
          editDependencyMenu.classList.toggle("open", editDependencyMenuOpen);
          newSessionCwdMenu.classList.toggle("open", newSessionCwdMenuOpen);
          newSessionModelMenu.classList.toggle("open", newSessionModelMenuOpen);
          newSessionReasoningMenu.classList.toggle("open", newSessionReasoningMenuOpen);
          newSessionResumeMenu.classList.toggle("open", newSessionResumeMenuOpen);
          editDependencyBtn.setAttribute("aria-expanded", editDependencyMenuOpen ? "true" : "false");
          newSessionCwdInput.setAttribute("aria-expanded", newSessionCwdMenuOpen ? "true" : "false");
          if (!newSessionCwdMenuOpen && newSessionCwdMenuFocus < 0) newSessionCwdInput.removeAttribute("aria-activedescendant");
          newSessionModelInput.setAttribute("aria-expanded", newSessionModelMenuOpen ? "true" : "false");
          if (!newSessionModelMenuOpen && newSessionModelMenuFocus < 0) newSessionModelInput.removeAttribute("aria-activedescendant");
          newSessionReasoningBtn.setAttribute("aria-expanded", newSessionReasoningMenuOpen ? "true" : "false");
          newSessionResumeBtn.setAttribute("aria-expanded", newSessionResumeMenuOpen ? "true" : "false");
          if (editDependencyMenuOpen) positionDialogMenu(editDependencyMenu, editDependencyBtn);
          if (newSessionCwdMenuOpen) positionDialogMenu(newSessionCwdMenu, newSessionCwdInput);
          if (newSessionModelMenuOpen) positionDialogMenu(newSessionModelMenu, newSessionModelInput);
          if (newSessionReasoningMenuOpen) positionDialogMenu(newSessionReasoningMenu, newSessionReasoningBtn);
          if (newSessionResumeMenuOpen) positionDialogMenu(newSessionResumeMenu, newSessionResumeBtn);
        }

        function positionDialogMenu(menu, anchorBtn) {
          if (!menu || !anchorBtn) return;
          const host = menu.parentElement;
          if (!host) return;
          const vv = window.visualViewport;
          const rect = anchorBtn.getBoundingClientRect();
          const hostRect = host.getBoundingClientRect();
          const viewportW = hostRect.width;
          const viewportTop = vv ? vv.offsetTop : 0;
          const viewportBottom = viewportTop + (vv ? vv.height : window.innerHeight);
          const margin = 12;
          const desiredWidth = Math.min(Math.max(rect.width, 280), viewportW - margin * 2);
          menu.style.position = "absolute";
          const left = Math.max(margin, Math.min(viewportW - margin - desiredWidth, rect.left - hostRect.left));
          menu.style.left = `${left}px`;
          menu.style.width = `${desiredWidth}px`;
          menu.style.right = "auto";
          menu.style.bottom = "auto";
          menu.style.maxHeight = "";
          const menuHeight = Math.min(menu.scrollHeight || 260, Math.floor((viewportBottom - viewportTop) * 0.5));
          const spaceBelow = viewportBottom - rect.bottom - margin;
          const spaceAbove = rect.top - viewportTop - margin;
          const openAbove = spaceBelow < Math.min(220, menuHeight) && spaceAbove > spaceBelow;
          if (openAbove) {
            const maxHeight = Math.max(120, spaceAbove - 8);
            menu.style.maxHeight = `${maxHeight}px`;
            const top = Math.max(viewportTop + margin - hostRect.top, rect.top - hostRect.top - Math.min(menuHeight, maxHeight) - 8);
            menu.style.top = `${top}px`;
          } else {
            const maxHeight = Math.max(120, spaceBelow - 8);
            menu.style.maxHeight = `${maxHeight}px`;
            const top = Math.min(viewportBottom - margin - hostRect.top - Math.min(menuHeight, maxHeight), rect.bottom - hostRect.top + 8);
            menu.style.top = `${top}px`;
          }
        }

        function openEditSession(sid) {
          if (!sid) return;
          const s = sessionIndex.get(sid);
          if (!s) return;
          editSessionId = sid;
          editStatus.textContent = "";
          editSaveBtn.disabled = false;
          editNameInput.value = typeof s.alias === "string" ? s.alias : "";
          editNameInput.placeholder = sessionDisplayName(s) || "Conversation title";
          editPriorityRange.value = String(Number(s.priority_offset || 0));
          syncEditPriorityLabel();
          const snoozeUntil = Number(s.snooze_until || 0);
          if (snoozeUntil > Date.now() / 1000) {
            setEditSnoozeMode("custom");
            fillCustomSnoozeInputs(snoozeUntil);
          } else {
            setEditSnoozeMode("none");
            fillCustomSnoozeInputs(tomorrowSnoozeSeconds());
          }
          fillDependencyOptions(sid, s.dependency_session_id || "");
          prepareModalOpen();
          if (!editViewer.open) editViewer.showModal();
          afterModalVisibilityChanged();
        }

        function restoreNewSessionFocus() {
          const target = newSessionReturnFocusEl;
          newSessionReturnFocusEl = null;
          if (!target || !target.isConnected || typeof target.focus !== "function") return;
          if (typeof target.disabled === "boolean" && target.disabled) return;
          requestAnimationFrame(() => {
            if (isModalTargetOpen(newSessionViewer)) return;
            try {
              target.focus({ preventScroll: true });
            } catch {}
          });
        }

        function focusNewSessionInitialControl() {
          const target = isMobile() ? newSessionCloseBtn : newSessionCwdInput;
          requestAnimationFrame(() => {
            if (!isModalTargetOpen(newSessionViewer)) return;
            target.focus({ preventScroll: true });
            if (target !== newSessionCwdInput) return;
            const end = newSessionCwdInput.value.length;
            try {
              newSessionCwdInput.setSelectionRange(end, end);
            } catch {}
          });
        }

        function hideNewSessionDialog() {
          const wasOpen = isModalTargetOpen(newSessionViewer);
          if (newSessionController) newSessionController.disposeResumeLoadTimer();
          newSessionStatus.textContent = "";
          newSessionCwdMenuOpen = false;
          newSessionCwdMenuFocus = -1;
          newSessionModelMenuOpen = false;
          newSessionModelMenuFocus = -1;
          newSessionReasoningMenuOpen = false;
          newSessionResumeMenuOpen = false;
          applyDialogMenus();
          newSessionBackdrop.style.display = "none";
          newSessionViewer.style.display = "none";
          afterModalVisibilityChanged();
          if (wasOpen) restoreNewSessionFocus();
          else newSessionReturnFocusEl = null;
        }

        function launchPresetProviderChoice(s) {
          return newSessionController.launchPresetProviderChoice(s);
        }

        function applyNewSessionLaunchPreset(sessionInfo) {
          return newSessionController.applyNewSessionLaunchPreset(sessionInfo);
        }

        function openNewSessionDialog({ cwd = null, statusText = "", likeSession = null, returnFocusEl = null } = {}) {
          newSessionReturnFocusEl = returnFocusEl instanceof HTMLElement ? returnFocusEl : document.activeElement instanceof HTMLElement ? document.activeElement : null;
          prepareModalOpen();
          const cur = selected ? sessionIndex.get(selected) : null;
          const like = likeSession && typeof likeSession === "object" ? likeSession : null;
          const initialCwd = typeof cwd === "string" && cwd.trim() ? cwd.trim() : like && like.cwd && like.cwd !== "?" ? like.cwd : cur && cur.cwd && cur.cwd !== "?" ? cur.cwd : "";
          const rememberedBackend = loadRememberedBackendChoice();
          const currentBackend = like ? sessionAgentBackend(like) : cur ? sessionAgentBackend(cur) : "";
          const defaultBackend = normalizeAgentBackendName(newSessionDefaults && newSessionDefaults.default_backend);
          const initialBackend = currentBackend || rememberedBackend || defaultBackend;
          newSessionStatus.textContent = String(statusText || newSessionDefaultsWarningText() || "");
          newSessionCwdInput.value = initialCwd;
          newSessionNameInput.value = "";
          newSessionModelInput.value = "";
          newSessionLiteralModelInputValue = "";
          newSessionLaunchPresetProviderAbsent = false;
          syncNewSessionNamePlaceholder();
          newSessionController.clearNewSessionResumeCandidates();
          setNewSessionResumeSelection(null);
          setNewSessionCwdError("");
          clearNewSessionCwdInfo();
          newSessionTmuxToggle.checked = tmuxAvailable;
          newSessionWorktreeToggle.checked = false;
          newSessionWorktreeInput.value = "";
          newSessionWorktreeInput.disabled = true;
          newSessionWorktreeInput.style.display = "none";
          newSessionWorktreeField.style.display = "none";
          newSessionCwdMenuOpen = false;
          newSessionCwdMenuFocus = -1;
          newSessionModelMenuOpen = false;
          newSessionModelMenuFocus = -1;
          newSessionReasoningMenuOpen = false;
          renderRecentCwdMenu();
          setNewSessionBackend(initialBackend, { resetSelections: true });
          if (like) applyNewSessionLaunchPreset(like);
          renderNewSessionResumeMenu();
          newSessionBackdrop.style.display = "block";
          newSessionViewer.style.display = "flex";
          afterModalVisibilityChanged();
          scheduleNewSessionResumeLoad();
          syncNewSessionTmuxUi();
          syncNewSessionWorktreeUi();
          focusNewSessionInitialControl();
        }

        editPriorityRange.oninput = syncEditPriorityLabel;
        editPriorityResetBtn.onclick = () => {
          editPriorityRange.value = "0";
          syncEditPriorityLabel();
        };
        for (const [mode, btn] of editSnoozeModeButtons.entries()) {
          btn.onclick = () => {
            setEditSnoozeMode(mode);
            if (mode === "tomorrow") fillCustomSnoozeInputs(tomorrowSnoozeSeconds());
            else if (mode === "4h") fillCustomSnoozeInputs(Math.floor(Date.now() / 1000) + 4 * 3600);
          };
        }
        editDependencyBtn.onclick = (e) => {
          e.preventDefault();
          e.stopPropagation();
          editDependencyMenuOpen = !editDependencyMenuOpen;
          newSessionCwdMenuOpen = false;
          newSessionCwdMenuFocus = -1;
          newSessionResumeMenuOpen = false;
          applyDialogMenus();
        };
        newSessionResumeBtn.onclick = (e) => {
          e.preventDefault();
          e.stopPropagation();
          renderNewSessionResumeMenu();
          newSessionResumeMenuOpen = !newSessionResumeMenuOpen;
          editDependencyMenuOpen = false;
          newSessionCwdMenuOpen = false;
          newSessionCwdMenuFocus = -1;
          newSessionModelMenuOpen = false;
          newSessionModelMenuFocus = -1;
          newSessionReasoningMenuOpen = false;
          applyDialogMenus();
        };
        editCloseBtn.onclick = () => hideEditSession();
        $("#editCancelBtn").onclick = () => hideEditSession();
        editViewer.addEventListener("cancel", (e) => {
          e.preventDefault();
          hideEditSession();
        });
        editViewer.onclick = (e) => {
          if (e.target === editViewer) hideEditSession();
        };
        editSaveBtn.onclick = async () => {
          const sid = editSessionId;
          if (!sid || editSaveBtn.disabled) return;
          let snoozeUntil = null;
          const snoozeMode = editSnoozeMode;
          if (snoozeMode === "4h") {
            snoozeUntil = Math.floor(Date.now() / 1000) + 4 * 3600;
          } else if (snoozeMode === "tomorrow") {
            snoozeUntil = tomorrowSnoozeSeconds();
          } else if (snoozeMode === "custom") {
            const dateRaw = String(editSnoozeCustomDate.value || "").trim();
            const timeRaw = String(editSnoozeCustomTime.value || "").trim();
            if (!dateRaw || !timeRaw) {
              editStatus.textContent = "Choose both a custom date and time.";
              return;
            }
            const parsed = Date.parse(`${dateRaw}T${timeRaw}`);
            if (!Number.isFinite(parsed)) {
              editStatus.textContent = "Invalid snooze time.";
              return;
            }
            snoozeUntil = Math.floor(parsed / 1000);
          }
          try {
            editSaveBtn.disabled = true;
            editStatus.textContent = "Saving...";
            await api(`/api/sessions/${sid}/edit`, {
              method: "POST",
              body: {
                name: String(editNameInput.value || ""),
                priority_offset: Number(editPriorityRange.value || 0),
                snooze_until: snoozeUntil,
                dependency_session_id: String(editDependencyBtn.dataset.value || "") || null,
              },
            });
            await refreshSessions();
            if (editSessionId !== sid) return;
            hideEditSession();
            if (selected === sid) {
              const s2 = sessionIndex.get(sid);
              if (s2) titleLabel.textContent = sessionTitleWithId(s2);
            }
            setToast("conversation updated");
          } catch (e) {
            if (editSessionId !== sid) return;
            editStatus.textContent = e && e.message ? e.message : "Save failed";
          } finally {
            if (editSessionId === sid) editSaveBtn.disabled = false;
          }
        };

        newSessionCloseBtn.onclick = () => hideNewSessionDialog();
        $("#newSessionCancelBtn").onclick = () => hideNewSessionDialog();
        newSessionBackdrop.onclick = () => hideNewSessionDialog();
        newSessionViewer.onclick = (e) => e.stopPropagation();
        newSessionCwdInput.onclick = () => {
          newSessionCwdMenuFocus = -1;
          renderRecentCwdMenu();
          newSessionCwdMenuOpen = true;
          editDependencyMenuOpen = false;
          newSessionModelMenuOpen = false;
          newSessionModelMenuFocus = -1;
          newSessionResumeMenuOpen = false;
          applyDialogMenus();
        };
        newSessionCwdInput.oninput = () => {
          newSessionCwdMenuFocus = -1;
          setNewSessionCwdError("");
          syncNewSessionNamePlaceholder();
          renderRecentCwdMenu();
          newSessionCwdMenuOpen = true;
          newSessionModelMenuOpen = false;
          newSessionModelMenuFocus = -1;
          scheduleNewSessionResumeLoad();
          applyDialogMenus();
        };
        newSessionCwdInput.onblur = () => {
          requestAnimationFrame(() => {
            if (newSessionCwdField.contains(document.activeElement)) return;
            newSessionCwdMenuOpen = false;
            newSessionCwdMenuFocus = -1;
            applyDialogMenus();
          });
        };
        newSessionCwdInput.onkeydown = (e) => {
          const items = renderRecentCwdMenu();
          if (e.key === "ArrowDown" || e.key === "ArrowUp") {
            if (!items.length) return;
            e.preventDefault();
            newSessionCwdMenuOpen = true;
            editDependencyMenuOpen = false;
            newSessionModelMenuOpen = false;
            newSessionModelMenuFocus = -1;
            newSessionResumeMenuOpen = false;
            const delta = e.key === "ArrowDown" ? 1 : -1;
            if (newSessionCwdMenuFocus < 0) newSessionCwdMenuFocus = delta > 0 ? 0 : items.length - 1;
            else newSessionCwdMenuFocus = (newSessionCwdMenuFocus + delta + items.length) % items.length;
            renderRecentCwdMenu();
            applyDialogMenus();
            const active = document.getElementById(`newSessionCwdOption-${newSessionCwdMenuFocus}`);
            if (active && typeof active.scrollIntoView === "function") active.scrollIntoView({ block: "nearest" });
            return;
          }
          if (e.key === "Enter" && newSessionCwdMenuOpen && newSessionCwdMenuFocus >= 0) {
            const active = items[newSessionCwdMenuFocus];
            if (!active) return;
            e.preventDefault();
            applyNewSessionCwdSuggestion(active.cwd);
            return;
          }
          if (e.key === "Escape" && newSessionCwdMenuOpen) {
            e.preventDefault();
            e.stopPropagation();
            newSessionCwdMenuOpen = false;
            newSessionCwdMenuFocus = -1;
            applyDialogMenus();
            return;
          }
          if (e.key === "Tab" && newSessionCwdMenuOpen) {
            newSessionCwdMenuOpen = false;
            newSessionCwdMenuFocus = -1;
            applyDialogMenus();
          }
        };
        newSessionModelInput.onclick = () => {
          newSessionModelMenuFocus = -1;
          renderNewSessionModelMenu();
          newSessionModelMenuOpen = true;
          editDependencyMenuOpen = false;
          newSessionCwdMenuOpen = false;
          newSessionCwdMenuFocus = -1;
          newSessionReasoningMenuOpen = false;
          newSessionResumeMenuOpen = false;
          applyDialogMenus();
        };
        newSessionModelInput.oninput = () => {
          newSessionLiteralModelInputValue = "";
          newSessionLaunchPresetProviderAbsent = false;
          newSessionModelMenuFocus = -1;
          syncNewSessionProviderFromModelInput();
          renderNewSessionModelMenu();
          setNewSessionReasoningEffort(newSessionReasoningEffort);
          renderNewSessionReasoningMenu();
          newSessionModelMenuOpen = true;
          newSessionReasoningMenuOpen = false;
          applyDialogMenus();
        };
        newSessionModelInput.onblur = () => {
          requestAnimationFrame(() => {
            if (newSessionModelField.contains(document.activeElement)) return;
            newSessionModelMenuOpen = false;
            newSessionModelMenuFocus = -1;
            applyDialogMenus();
          });
        };
        newSessionModelInput.onkeydown = (e) => {
          const items = renderNewSessionModelMenu();
          if (e.key === "ArrowDown" || e.key === "ArrowUp") {
            if (!items.length) return;
            e.preventDefault();
            newSessionModelMenuOpen = true;
            editDependencyMenuOpen = false;
            newSessionCwdMenuOpen = false;
            newSessionCwdMenuFocus = -1;
            newSessionReasoningMenuOpen = false;
            newSessionResumeMenuOpen = false;
            const delta = e.key === "ArrowDown" ? 1 : -1;
            if (newSessionModelMenuFocus < 0) newSessionModelMenuFocus = delta > 0 ? 0 : items.length - 1;
            else newSessionModelMenuFocus = (newSessionModelMenuFocus + delta + items.length) % items.length;
            renderNewSessionModelMenu();
            applyDialogMenus();
            const active = document.getElementById(`newSessionModelOption-${newSessionModelMenuFocus}`);
            if (active && typeof active.scrollIntoView === "function") active.scrollIntoView({ block: "nearest" });
            return;
          }
          if (e.key === "Enter" && newSessionModelMenuOpen && newSessionModelMenuFocus >= 0) {
            const active = items[newSessionModelMenuFocus];
            if (!active) return;
            e.preventDefault();
            selectNewSessionModel(active);
            return;
          }
          if (e.key === "Escape" && newSessionModelMenuOpen) {
            e.preventDefault();
            e.stopPropagation();
            newSessionModelMenuOpen = false;
            newSessionModelMenuFocus = -1;
            applyDialogMenus();
            return;
          }
          if (e.key === "Tab" && newSessionModelMenuOpen) {
            newSessionModelMenuOpen = false;
            newSessionModelMenuFocus = -1;
            applyDialogMenus();
          }
        };
        newSessionWorktreeToggle.onchange = () => {
          syncNewSessionWorktreeUi();
          if (newSessionWorktreeToggle.checked) newSessionWorktreeInput.focus();
        };
        newSessionWorktreeInput.oninput = () => syncNewSessionWorktreeUi();
        newSessionFastToggle.onchange = () => setNewSessionFast(newSessionFastToggle.checked);
        newSessionReasoningBtn.onclick = (e) => {
          e.preventDefault();
          e.stopPropagation();
          renderNewSessionReasoningMenu();
          newSessionReasoningMenuOpen = !newSessionReasoningMenuOpen;
          editDependencyMenuOpen = false;
          newSessionCwdMenuOpen = false;
          newSessionCwdMenuFocus = -1;
          newSessionModelMenuOpen = false;
          newSessionModelMenuFocus = -1;
          newSessionResumeMenuOpen = false;
          applyDialogMenus();
        };
        newSessionStartBtn.onclick = async () => {
          if (newSessionStartBusy) return;
          const cwd = String(newSessionCwdInput.value || "").trim();
          const agentBackend = newSessionBackend;
          setNewSessionCwdError("");
          if (!cwd) {
            newSessionStatus.textContent = "";
            setNewSessionCwdError("Working directory is required.");
            return;
          }
          const sessionName = String(newSessionNameInput.value || "").trim();
          const parsedProviderModel = syncNewSessionProviderFromModelInput();
          if (parsedProviderModel.providerError) {
            newSessionStatus.textContent = parsedProviderModel.providerError;
            return;
          }
          const providerChoice = String(parsedProviderModel.providerAbsent ? "" : parsedProviderModel.providerChoice || newSessionProvider || "").trim();
          const model = String(parsedProviderModel.model || "default").trim() || "default";
          rememberProviderModelChoice(agentBackend, providerChoice, model, { providerAbsent: Boolean(parsedProviderModel.providerAbsent) });
          const resumeSessionId = (newSessionController.currentResumeSelection() || {}).session_id || null;
          const createInTmux = !!newSessionTmuxToggle.checked;
          const worktreeBranch = !resumeSessionId && newSessionWorktreeToggle.checked ? String(newSessionWorktreeInput.value || "").trim() : null;
          if (newSessionWorktreeToggle.checked && !worktreeBranch) {
            newSessionStatus.textContent = "Branch name is required.";
            return;
          }
          newSessionStatus.textContent = resumeSessionId ? "Resuming..." : worktreeBranch ? "Creating worktree..." : createInTmux ? "Starting in tmux..." : "Starting...";
          let cwdStartError = false;
          let startErrorText = "";
          newSessionStartBusy = true;
          newSessionStartBtn.disabled = true;
          try {
            const brokerPid = await spawnSessionWithCwd(cwd, resumeSessionId, worktreeBranch, sessionName, providerChoice, model, newSessionReasoningEffort, newSessionFast, createInTmux, (e) => {
              if (e && e.obj && e.obj.field === "cwd") {
                cwdStartError = true;
                newSessionStatus.textContent = "";
                setNewSessionCwdError(e.message);
                return;
              }
              const launchId = e && e.obj && e.obj.launch_id ? String(e.obj.launch_id) : "";
              startErrorText = launchId ? `${e.message} (${launchId})` : e && e.message ? e.message : "Start failed.";
            }, agentBackend);
            if (brokerPid) hideNewSessionDialog();
            else if (!cwdStartError) newSessionStatus.textContent = startErrorText || "Start failed.";
          } finally {
            newSessionStartBusy = false;
            newSessionStartBtn.disabled = false;
          }
        };
        const FILE_CANDIDATE_CACHE_TTL_MS = 15000;
        const filePickerMenuState = codoxearFilePicker.createMenuState({
          normalizeLineNumber,
        });
        const filePickerDomRuntime = codoxearFilePicker.createMenuDomRuntime({
          field: filePickerField,
          menu: filePickerMenu,
          input: filePickerInput,
          menuState: filePickerMenuState,
        });
        const filePickerSearchState = codoxearFilePicker.createSearchState({
          blocked: () => blockUnavailableFileAction(),
          currentSessionId: () => currentFileViewerSessionId() || selected || "",
          api,
          inputValue: () => filePickerInput.value,
          isMenuOpen: () => filePickerMenuState.isOpen(),
          renderMenu: () => renderFilePickerMenu(),
          applyMenuState: () => applyFileMenuState(),
          normalizeFileApiPath: (value) => normalizeFileApiPath(value),
        });
        const filePickerEntryRuntime = codoxearFilePicker.createEntryRuntime({
          menuState: filePickerMenuState,
          inputValue: () => filePickerInput.value,
          candidateKeys: () => fileViewerController.currentFileCandidateKeys(),
          entryForKey: (key) => fileViewerController.fileEntryForKey(key),
          pickerEntryForKey: (key, options) => fileViewerController.pickerEntryForKey(key, options),
          pickerEntryForPath: (path, options) => fileViewerController.pickerEntryForPath(path, options),
          keyForPath: (path, gitPath, apiPath) => fileCandidateKey(path, gitPath, apiPath),
          activeFileDraft: () => currentActiveFileDraft(),
          activeFilePath: () => activeFilePathValue(),
          searchSnapshot: () => filePickerSearchSnapshot(),
          normalizeFileApiPath: (value) => normalizeFileApiPath(value),
        });
        const filePickerRenderRuntime = codoxearFilePicker.createMenuRenderRuntime({
          menu: filePickerMenu,
          menuState: filePickerMenuState,
          inputValue: () => filePickerInput.value,
          visibleEntries: () => filePickerEntryRuntime.visibleEntries(),
          searchSnapshot: () => filePickerSearchSnapshot(),
          normalizeDraftFilePath: (query) => normalizeDraftFilePath(query),
          draftSuppressed: () => filePickerDraftSuppressed(),
          draftEntry: (path) => filePickerEntryRuntime.draftEntry(path),
          syncActiveDescendant: (focusIndex) => filePickerDomRuntime.syncActiveDescendant(focusIndex),
          sectionLabel: (source) => filePickerSectionLabel(source),
          duplicatePaths: (entries) => duplicateFilePickerPaths(entries),
          rawByteDuplicatePaths: (entries) => rawByteDuplicatePaths(entries),
          identityHint: (entry, duplicatePaths, options) => filePickerIdentityHint(entry, duplicatePaths, options),
          titleForEntry: (entry, hint) => filePickerTitle(entry, hint),
          normalizeFileApiPath: (value) => normalizeFileApiPath(value),
          activeIdentity: () => currentActiveFileIdentity(),
          gitStatusMessage: () => fileViewerController.currentFileCandidateGitStateMessage(),
          openDraftFilePath: (draftPath) => openDraftFilePathWithGuard(draftPath),
          openEntry: async (selectedEntry) => {
            try {
              await openFilePathWithResolvedMode(selectedEntry.path, { line: filePickerSelectionLine(), changed: Boolean(selectedEntry.changed), gitPath: Boolean(selectedEntry.gitPath), apiPath: selectedEntry.apiPath });
            } catch (e) {
              fileStatus.textContent = `error: ${e && e.message ? e.message : "unable to inspect path"}`;
            }
          },
          el,
          createTextNode: (value) => document.createTextNode(value),
        });
        const filePickerInputRuntime = codoxearFilePicker.createInputRuntime({
          input: filePickerInput,
          menuState: filePickerMenuState,
          ensureCurrentSession: () => ensureCurrentFileViewerSession(),
          renderMenu: () => renderFilePickerMenu(),
          applyMenuState: () => applyFileMenuState(),
          resetInput: () => resetFilePickerInput(),
          closeMenu: (options) => closeFilePickerMenu(options),
          currentSessionId: () => currentFileViewerSessionId(),
          selectedSessionId: () => selected,
          resetSearchState: () => resetFileSearchState(),
          setSearchSessionId: (sessionId) => filePickerSearchState.setSessionId(sessionId),
          scheduleSearch: (query) => filePickerSearchState.schedule(query),
          selectionLine: () => filePickerSelectionLine(),
          openDraftFilePathWithGuard: (path) => openDraftFilePathWithGuard(path),
          openFilePathWithResolvedMode: (path, options) => openFilePathWithResolvedMode(path, options),
          setStatus: (status) => {
            fileStatus.textContent = status;
          },
          optionElementById: (id) => document.getElementById(id),
          isFocusInsideField: () => filePickerField.contains(document.activeElement),
          requestAnimationFrame: (callback) => requestAnimationFrame(callback),
        });
        const MONACO_LOADER_TIMEOUT_MS = 4000;
        const PDFJS_LOADER_TIMEOUT_MS = 6000;
        const fileEditorRuntime = codoxearFileEditor.createFileEditorRuntime();
        const fileEditorMonacoLoader = codoxearFileEditor.createMonacoLoader({
          resolveAppUrl,
          timeoutMs: MONACO_LOADER_TIMEOUT_MS,
        });
        const fileEditorRenderer = codoxearFileEditor.createFileEditorRenderer({
          runtime: fileEditorRuntime,
          monacoLoader: fileEditorMonacoLoader,
          host: fileDiff,
          normalizeLineNumber,
          requestAnimationFrame: (callback) => requestAnimationFrame(callback),
          setTimeout: (callback, delay) => setTimeout(callback, delay),
          isCurrentFileOpenRequest: (request) => isCurrentFileOpenRequest(request),
          renderPlainTextFallback: (rel, text, lineNumber, reason) => renderPlainTextFallback(rel, text, lineNumber, reason),
          disposeFileEditor: () => disposeFileEditor(),
          currentEditorKind: () => currentFileEditorKind(),
          setEditorKind: (kind) => setFileEditorKind(kind),
          currentFileEditMode: () => currentFileEditMode(),
          currentActiveFileEditable: () => currentActiveFileEditable(),
          isUnavailable: () => isFileViewerSessionUnavailable(),
          isProgrammaticChange: () => fileViewerController.isFileEditorProgrammaticChange(),
          currentTouchSelectMode: () => currentFileTouchSelectMode(),
          resetTouchSelectionState: () => resetFileTouchSelectionState(),
          currentActiveFileText: () => currentActiveFileText(),
          setDirty: (dirty) => setFileDirty(dirty),
          runProgrammaticChange: (callback) => fileViewerController.runFileEditorProgrammaticChange(callback),
          syncReadOnly: () => syncFileEditorReadOnly(),
          updateTouchToolbar: () => updateFileTouchToolbar(),
        });
        const filePdfLoader = codoxearFileViewer.createPdfLoader({
          resolveAppUrl,
          timeoutMs: PDFJS_LOADER_TIMEOUT_MS,
        });
        const fileFallbackRuntime = codoxearFileViewer.createFileFallbackRuntime({
          host: fileDiff,
          el,
          normalizeLineNumber,
          requestAnimationFrame: (callback) => requestAnimationFrame(callback),
          disposeFileEditor: () => disposeFileEditor(),
          disposePdfRender: () => disposePdfRender(),
          clearFileVideo: () => clearFileVideo(),
          setFileRenderSurface: (surface) => setFileRenderSurface(surface),
          setFileEditorKind: (kind) => setFileEditorKind(kind),
          applyPlainTextFallbackState: () => fileViewerController.applyPlainTextFallbackState(),
          updateFileTouchToolbar: () => updateFileTouchToolbar(),
          currentSessionId: () => currentFileViewerSessionId() || selected || "",
          markdownPreviewHtml: (body, context) => markdownPreviewHtml(body, context),
          upgradeCandidateFileRefs: (node) => upgradeCandidateFileRefs(node),
          blockedFileMessage: (rel, reason, viewerMaxBytes, size) => blockedFileMessage(rel, reason, viewerMaxBytes, size),
        });
        const fileDownloadRuntime = codoxearFileViewer.createFileDownloadRuntime({
          resolveAppUrl,
          document,
        });
        const filePdfRenderRuntime = codoxearFileViewer.createFilePdfRenderRuntime({
          host: fileDiff,
          el,
          ensurePdfJs: () => ensurePdfJs(),
          createCanvas: () => document.createElement("canvas"),
          devicePixelRatio: () => window.devicePixelRatio || 1,
          disposeFileEditor: () => disposeFileEditor(),
          disposePdfRender: () => disposePdfRender(),
          clearFileVideo: () => clearFileVideo(),
          setFileRenderSurface: (surface) => setFileRenderSurface(surface),
          renderDownloadFallback: (rel, url, reason) => renderDownloadFallback(rel, url, reason),
          isCurrentFileOpenRequest: (request) => isCurrentFileOpenRequest(request),
          setActivePdfRenderState: (state) => fileViewerController.setActivePdfRenderState(state),
          isActivePdfRenderState: (state) => fileViewerController.isActivePdfRenderState(state),
          updateFileTouchToolbar: () => updateFileTouchToolbar(),
          IntersectionObserverCtor: typeof IntersectionObserver === "function" ? IntersectionObserver : null,
        });
        const filePasteDialogRuntime = codoxearFileViewer.createFilePasteDialogRuntime({
          backdrop: filePasteBackdrop,
          dialog: filePasteDialog,
          input: filePasteInput,
          prepareModalOpen,
          afterModalVisibilityChanged,
          focusActiveEditor: () => fileEditorRuntime.focusActiveCodeEditor(currentFileEditorKind()),
          requestAnimationFrame: (callback) => requestAnimationFrame(callback),
        });
        const fileRenderSurfaceRuntime = codoxearFileViewer.createFileRenderSurfaceRuntime({
          diff: fileDiff,
          image: fileImage,
          video: fileVideo,
          videoPreviewButton: fileVideoPreviewBtn,
          clearActiveVideoFallback: () => fileViewerController.clearActiveVideoFallback(),
        });
        const fileModeControlsRuntime = codoxearFileViewer.createFileModeControlsRuntime({
          diffButton: fileModeDiffBtn,
          previewButton: fileModePreviewBtn,
          downloadButton: fileDownloadBtn,
          videoPreviewButton: fileVideoPreviewBtn,
          hideFilePasteDialog: () => hideFilePasteDialog(),
          setFileEditMode: (mode) => setFileEditMode(mode),
          syncFileEditorReadOnly: () => syncFileEditorReadOnly(),
          updateFileEditButton: () => updateFileEditButton(),
        });
        const fileTouchToolbarRuntime = codoxearFileViewer.createFileTouchToolbarRuntime({
          toolbar: fileTouchToolbar,
          actions: fileTouchActions,
          dpad: fileTouchDpad,
          copyButton: fileTouchCopyBtn,
          pasteButton: fileTouchPasteBtn,
          selectButton: fileTouchSelectBtn,
        });
        const fileViewerModalRuntime = codoxearFileViewer.createFileViewerModalRuntime({
          backdrop: fileBackdrop,
          viewer: fileViewer,
          pickerInput: filePickerInput,
          closeButton: fileCloseBtn,
          prepareModalOpen,
          afterModalVisibilityChanged,
          focusModalCloseButton,
          restoreModalFocus,
          isModalTargetOpen,
          setReturnFocusElement: (element, ElementCtor) => fileViewerController.setFileViewerReturnFocusElement(element, ElementCtor),
          takeReturnFocusElement: () => fileViewerController.takeFileViewerReturnFocusElement(),
        });
        const fileUnsavedDialogRuntime = codoxearFileViewer.createFileUnsavedDialogRuntime({
          backdrop: fileUnsavedBackdrop,
          dialog: fileUnsavedDialog,
          viewer: fileViewer,
          title: fileUnsavedDialog.querySelector(".title"),
          message: fileUnsavedDialog.querySelector(".muted"),
          saveButton: $("#fileUnsavedSaveBtn"),
          discardButton: $("#fileUnsavedDiscardBtn"),
          cancelButton: $("#fileUnsavedCancelBtn"),
          prepareModalOpen,
          afterModalVisibilityChanged,
          restoreModalFocus,
          isModalTargetOpen,
          requestAnimationFrame: (callback) => requestAnimationFrame(callback),
          promptPlan: () => fileViewerController.fileUnsavedPromptPlan(),
          beginPrompt: () => fileViewerController.beginFileUnsavedPrompt(),
          resolvePrompt: (choice) => fileViewerController.resolveFileUnsavedPrompt(choice),
          setReturnFocusElement: (element, ElementCtor) => fileViewerController.setFileUnsavedReturnFocusElement(element, ElementCtor),
          takeReturnFocusElement: () => fileViewerController.takeFileUnsavedReturnFocusElement(),
          isUnavailable: () => isFileViewerSessionUnavailable(),
        });

        function currentFileViewerSessionId() {
          return fileViewerController.currentFileViewerSessionId();
        }

        function currentFileSessionId() {
          return String(currentFileViewerSessionId() || selected || "").trim();
        }

        function isFileViewerSessionUnavailable() {
          return fileViewerController.isFileViewerSessionUnavailable();
        }

        function blockUnavailableFileAction() {
          return fileViewerController.blockUnavailableFileAction();
        }

        function currentActiveFileIdentity() {
          return fileViewerController.currentActiveFileIdentity();
        }

        function activeFilePathValue() {
          return currentActiveFileIdentity().path;
        }

        function currentFileEditorKind() {
          return fileViewerController.currentFileEditorKind();
        }

        function setFileEditorKind(kind) {
          return fileViewerController.setFileEditorKind(kind);
        }

        function isCurrentFileOpenRequest(request) {
          return fileViewerController.isCurrentFileOpenRequest(request);
        }

        function clearFileVideo() {
          return fileRenderSurfaceRuntime.clearVideo();
        }

        function setFileRenderSurface(surface) {
          return fileRenderSurfaceRuntime.setSurface(surface);
        }

        function resetFileViewerPanel() {
          return fileViewerPanelRuntime.resetPanel();
        }

        function renderEmptyFileViewerTarget({ updateTouchToolbar = false } = {}) {
          return fileViewerPanelRuntime.renderEmptyTarget({ updateTouchToolbar });
        }

        async function ensureCurrentFileViewerSession() {
          return await fileViewerLifecycleRuntime.ensureCurrentSession();
        }

        function disposeFileEditor() {
          return fileEditorRuntime.disposeCurrentFile({
            finishProgrammaticChange: () => fileViewerController.finishFileEditorProgrammaticChange(),
            clearHost: () => {
              fileDiff.innerHTML = "";
            },
            setFileEditorKind: (kind) => setFileEditorKind(kind),
            clearFileTouchSelectionState: () => clearFileTouchSelectionState(),
          });
        }

        function disposePdfRender() {
          return fileViewerController.disposeActivePdfRender();
        }

        function isFileViewerOpen() {
          return fileViewerModalRuntime.isOpen();
        }

        function syncFileEditorReadOnly() {
          return fileViewerController.syncFileEditorReadOnly();
        }

        function updateFileTouchToolbar() {
          return fileTouchToolbarRuntime.update(fileViewerController.currentFileTouchToolbarState());
        }

        function clearFileTouchSelectionState() {
          return fileViewerController.clearFileTouchSelectionState();
        }

        function currentFileTouchSelectMode() {
          return fileViewerController.currentFileTouchSelectMode();
        }

        function resetFileTouchSelectionState(options) {
          return fileViewerController.resetFileTouchSelectionState(options);
        }

        function toggleFileTouchSelectionMode() {
          return fileViewerController.toggleFileTouchSelectionMode();
        }

        function handleFileTouchMoveButtonPress(direction) {
          return fileViewerController.handleFileTouchMoveButtonPress(direction);
        }

        function handleFileEditorSaveShortcut(e) {
          return fileViewerController.handleFileEditorSaveShortcut(e);
        }

        function handleFileEditorDeleteKeydown(e) {
          return fileViewerController.handleFileEditorDeleteKeydown(e);
        }

        function suppressFileEditorNativeDelete(e) {
          return fileViewerController.suppressFileEditorNativeDelete(e);
        }

        async function copyActiveFileSelection() {
          return await fileViewerController.copyActiveFileSelection();
        }

        function hideFilePasteDialog({ restoreFocus = false } = {}) {
          return filePasteDialogRuntime.hide({ restoreFocus });
        }

        function showFilePasteDialog() {
          return filePasteDialogRuntime.show();
        }

        async function pasteFromClipboardIntoActiveFile() {
          return await fileViewerController.pasteFromClipboardIntoActiveFile();
        }

        function handleFilePasteInsert(text) {
          return fileViewerController.handleFilePasteInsert(text);
        }

        function updateFileEditButton() {
          return fileViewerController.updateFileEditButton();
        }

        function currentFileDirty() {
          return fileViewerController.currentFileDirty();
        }

        function setFileDirty(nextDirty) {
          return fileViewerController.setFileDirty(nextDirty);
        }

        function resetActiveFileBufferState() {
          fileViewerController.resetActiveFileBufferState();
        }

        function currentActiveFileText() {
          return fileViewerController.currentActiveFileText();
        }

        function currentActiveFileEditable() {
          return fileViewerController.currentActiveFileEditable();
        }

        function currentActiveFileDraft() {
          return fileViewerController.currentActiveFileDraft();
        }

        function getFileEditorText() {
          return fileEditorRuntime.currentFileText(currentFileEditorKind(), currentActiveFileText());
        }

        function restoreFileEditorText(text) {
          return fileEditorRuntime.restoreCurrentFileText(text, {
            prepareFileEditorTextRestore: (value) => fileViewerController.prepareFileEditorTextRestore(value),
            currentFileEditorKind: () => currentFileEditorKind(),
            runFileEditorProgrammaticChange: (callback) => fileViewerController.runFileEditorProgrammaticChange(callback),
            finishFileEditorTextRestore: () => fileViewerController.finishFileEditorTextRestore(),
          });
        }

        function renderPlainTextFallback(rel, text, lineNumber = null, reason = "Rich file viewer unavailable") {
          return fileFallbackRuntime.applyPlainText(rel, text, lineNumber, reason);
        }

        function renderDownloadFallback(rel, url, reason = "Preview unavailable") {
          return fileFallbackRuntime.applyDownload(rel, url, reason);
        }

        async function ensurePdfJs() {
          return await filePdfLoader.ensure();
        }

        async function renderMonacoFile(rel, text, lineNumber = null, langOverride = "", request = null) {
          return await fileEditorRenderer.renderFile(rel, text, lineNumber, langOverride, request);
        }

        async function renderMonacoDiff(rel, originalText, modifiedText, lineNumber = null, request = null) {
          return await fileEditorRenderer.renderDiff(rel, originalText, modifiedText, lineNumber, request);
        }

        function renderMarkdownPreview(rel, text) {
          return fileFallbackRuntime.applyMarkdown(rel, text);
        }

        function renderBlockedFileNotice(rel, reason, viewerMaxBytes, size) {
          return fileFallbackRuntime.applyBlocked(rel, reason, viewerMaxBytes, size);
        }

        async function renderPdfFile(rel, url, request) {
          return await filePdfRenderRuntime.render(rel, url, request);
        }

        function currentFileEditMode() {
          return fileViewerController.currentFileEditMode();
        }

        function setFileEditMode(nextMode) {
          return fileViewerController.setFileEditMode(nextMode);
        }

        function hideFileUnsavedDialog(choice = "cancel") {
          return fileUnsavedDialogRuntime.hide(choice);
        }

        function promptFileUnsavedChoice() {
          return fileUnsavedDialogRuntime.promptChoice(document.activeElement, HTMLElement);
        }

        const fileInspectRuntime = codoxearFileViewer.createFileInspectRuntime({
          currentSessionId: () => currentFileViewerSessionId(),
          selectedSessionId: () => selected,
          normalizeFileApiPath: (value) => normalizeFileApiPath(value),
          api: (url, options) => api(url, options),
        });

        const fileViewerController = codoxearFileViewer.createFileViewerController({
          el,
          fileStatus,
          fileEditButton: fileEditBtn,
          iconSvg,
          currentSessionId: () => currentFileViewerSessionId(),
          currentFileSessionId: () => currentFileSessionId(),
          normalizeLineNumber,
          normalizeFileApiPath,
          isFileViewerOpen: () => isFileViewerOpen(),
          hideFileUnsavedDialog: (choice) => hideFileUnsavedDialog(choice),
          resetFileSearchState: () => resetFileSearchState(),
          closeFilePickerMenu: (options) => closeFilePickerMenu(options),
          isTextFileKind: (kind) => isTextFileKind(kind),
          isDiffableFileKind: (kind) => isDiffableFileKind(kind),
          confirmReload: (message) => confirmApp({ title: "Reload file from disk?", message, confirmText: "Reload", cancelText: "Cancel", destructive: true }),
          promptUnsavedFileChoice: () => promptFileUnsavedChoice(),
          restoreFileEditorText: (text) => restoreFileEditorText(text),
          hideFileViewer: () => hideFileViewer(),
          setFilePath: (path, options) => setFilePath(path, options),
          resetFileViewerPanel: () => resetFileViewerPanel(),
          applyFileLoadResult: (rel, result, request, options) => applyFileLoadResult(rel, result, request, options),
          normalizeDraftFilePath: (path) => normalizeDraftFilePath(path),
          inspectSessionFilePath: (path, options) => fileInspectRuntime.inspectSessionFilePath(path, options),
          api: (url, options) => api(url, options),
          focusEditor: () => fileEditorRuntime.focusActiveCodeEditor(currentFileEditorKind()),
          disposeOpenRender: () => disposePdfRender(),
          initialFileViewMode: storageGetItem("codexweb.fileViewMode") || "diff",
          initialFileNonDiffMode: storageGetItem("codexweb.fileNonDiffMode") === "preview" ? "preview" : "file",
          persistFileViewMode: (mode) => storageSetItem("codexweb.fileViewMode", mode),
          persistFileNonDiffMode: (mode) => storageSetItem("codexweb.fileNonDiffMode", mode),
          isMarkdownPreviewable,
          resetActiveFileBufferState: () => resetActiveFileBufferState(),
          updateFileTouchToolbar: () => updateFileTouchToolbar(),
          useTouchFileEditorControls: () => useTouchFileEditorControls(),
          hasActiveFileCodeEditor: () => Boolean(fileEditorRuntime.activeCodeEditor(currentFileEditorKind())),
          hasBlockingFileEditorModal: () => modalIsolationTargets.some((node) => node !== fileViewer && isModalTargetOpen(node)),
          isTextEntryTarget: (target) => isTextEntryElement(target),
          eventTargetElement: (value) => value instanceof HTMLElement ? value : null,
          normalizeFileEditorPosition: (editor, position) => fileEditorRuntime.normalizePosition(editor, position),
          applyFileEditorSelection: (editor, cursor, anchor) => fileEditorRuntime.applySelection(editor, cursor, anchor, fileEditorMonacoLoader.selectionCtor()),
          isCollapsedFileSelection: (selection) => fileEditorRuntime.isCollapsedSelection(selection),
          fileEditorEditSupportAvailable: () => fileEditorMonacoLoader.editSupportAvailable(),
          updateFileDiffEditorOptions: (options) => fileEditorRuntime.updateEditorOptions(currentFileEditorKind(), options),
          showFilePasteDialog: () => showFilePasteDialog(),
          hideFilePasteDialog: (options) => hideFilePasteDialog(options),
          clipboardReadAvailable: () => Boolean(window.isSecureContext && navigator.clipboard && typeof navigator.clipboard.readText === "function"),
          readClipboardText: () => navigator.clipboard.readText(),
          isActiveFileEditorInput: (target) => fileEditorRuntime.isActiveInput(currentFileEditorKind(), target, HTMLElement),
          getActiveFileSelectionText: () => fileEditorRuntime.activeSelectionText(currentFileEditorKind()),
          copyToClipboard: (text) => copyToClipboard(text),
          focusActiveFileCodeEditor: () => fileEditorRuntime.focusActiveCodeEditor(currentFileEditorKind()),
          nowMs: () => Date.now(),
          setToast: (message) => setToast(message),
          setFileViewMode: (mode) => setFileViewMode(mode),
          renderMonacoFile: (rel, text, lineNumber, langOverride, request) => renderMonacoFile(rel, text, lineNumber, langOverride, request),
          getFileEditorText: () => getFileEditorText(),
          fmtBytes: (value) => fmtBytes(value),
          applyFileMode: () => applyFileMode(),
          rememberOpenedFile: (rel, absPath) => rememberOpenedFile(rel, absPath),
          historyFileSelectionForSession: (sessionId) => openedFileRuntime.historySelection(sessionId),
          renderFilePickerMenu: () => renderFilePickerMenu(),
        });
        const fileViewerPanelRuntime = codoxearFileViewer.createFileViewerPanelRuntime({
          controller: fileViewerController,
          disposeFileEditor: () => disposeFileEditor(),
          resetRenderSurface: () => fileRenderSurfaceRuntime.reset(),
          resetFilePickerInput: () => resetFilePickerInput(),
          renderFilePickerMenu: () => renderFilePickerMenu(),
          closeFilePickerMenu: () => closeFilePickerMenu(),
          applyFileMode: () => applyFileMode(),
          updateFileTouchToolbar: () => updateFileTouchToolbar(),
          setStatus: (status) => {
            fileStatus.textContent = status;
          },
        });
        const fileViewerLifecycleRuntime = codoxearFileViewer.createFileViewerLifecycleRuntime({
          controller: fileViewerController,
          beginHide: () => fileViewerModalRuntime.beginHide(),
          hideDisplay: () => fileViewerModalRuntime.hideDisplay(),
          finishHide: (state) => fileViewerModalRuntime.finishHide(state),
          hideFileUnsavedDialog: () => hideFileUnsavedDialog(),
          hideFilePasteDialog: () => hideFilePasteDialog(),
          resetFileViewerPanel: () => resetFileViewerPanel(),
          closeFilePickerMenu: (options) => closeFilePickerMenu(options),
          resetFileSearchState: () => resetFileSearchState(),
          setFileSearchSessionId: (sessionId) => filePickerSearchState.setSessionId(sessionId),
          updateFileTouchToolbar: () => updateFileTouchToolbar(),
          isFileViewerOpen: () => isFileViewerOpen(),
          selectedSessionId: () => selected,
          maybeHandleUnsavedFileChanges: () => maybeHandleUnsavedFileChanges(),
          filePickerSearchSessionId: () => filePickerSearchSnapshot().sessionId,
          refreshFileCandidates: (options) => refreshFileCandidates(options),
          setFilePath: (path, options) => setFilePath(path, options),
          openFilePathWithResolvedMode: (path, options) => openFilePathWithResolvedMode(path, options),
          renderEmptyFileViewerTarget: (options) => renderEmptyFileViewerTarget(options),
          setStatus: (status) => {
            fileStatus.textContent = status;
          },
          showModal: (options) => fileViewerModalRuntime.show({ ...options, activeElement: document.activeElement, ElementCtor: HTMLElement }),
          setFileViewMode: (nextMode) => setFileViewMode(nextMode),
          applyFileMode: () => applyFileMode(),
          openFilePickerSearchQuery: (query, options) => openFilePickerSearchQuery(query, options),
          setPreserveSearchOnFocus: (value) => filePickerMenuState.setPreserveSearchOnFocus(value),
          focusFilePickerInput: () => {
            try {
              filePickerInput.focus({ preventScroll: true });
            } catch (_) {
              filePickerInput.focus();
            }
          },
        });
        const fileVideoPreviewRuntime = codoxearFileViewer.createFileVideoPreviewRuntime({
          controller: fileViewerController,
          fetchPreview: (url, options) => fetch(url, options),
          resolveAppUrl: (url) => resolveAppUrl(url),
          handleAuthLoss: () => handleAppAuthLoss(),
          errorText: (error) => codoxearFileHelpers.fileVideoPreviewErrorText(error),
          video: fileVideo,
        });
        const fileLoadResultRuntime = codoxearFileViewer.createFileLoadResultRuntime({
          controller: fileViewerController,
          resolveAppUrl,
          setStatus: (status) => {
            fileStatus.textContent = status;
          },
          disposeFileEditor: () => disposeFileEditor(),
          renderMonacoDiff: (rel, originalText, modifiedText, lineNumber, request, options) => renderMonacoDiff(rel, originalText, modifiedText, lineNumber, request, options),
          renderMonacoFile: (rel, text, lineNumber, langOverride, request) => renderMonacoFile(rel, text, lineNumber, langOverride, request),
          renderMarkdownPreview: (rel, text) => renderMarkdownPreview(rel, text),
          renderBlockedFileNotice: (rel, reason, viewerMaxBytes, size) => renderBlockedFileNotice(rel, reason, viewerMaxBytes, size),
          renderPdfFile: (rel, url, request) => renderPdfFile(rel, url, request),
          showImage: (src, alt) => fileRenderSurfaceRuntime.showImage(src, alt),
          showVideo: (loadPlan, options) => fileRenderSurfaceRuntime.showVideo(loadPlan, options),
          loadCompatibleVideoPreview: (token, options) => fileVideoPreviewRuntime.loadCompatibleVideoPreview(token, options),
        });
        const fileCandidateRefreshRuntime = codoxearFileViewer.createFileCandidateRefreshRuntime({
          controller: fileViewerController,
          currentSessionId: () => currentFileViewerSessionId(),
          selectedSessionId: () => selected,
          blockUnavailableFileAction: () => blockUnavailableFileAction(),
          isSessionCurrent: (sessionId, syncToken) => fileViewerLifecycleRuntime.isSessionCurrent(sessionId, syncToken),
          ttlMs: FILE_CANDIDATE_CACHE_TTL_MS,
          nowMs: () => Date.now(),
          collectMessageFileRefs: () => collectMessageFileRefs(),
          sessionFiles: (sessionId) => {
            const s = sessionId ? sessionIndex.get(sessionId) : null;
            return listFromFilesField(s && s.files);
          },
          sessionFileRecords: (sessionId) => {
            const s = sessionId ? sessionIndex.get(sessionId) : null;
            return listFromFileRecords(s && s.files);
          },
          sessionRelativePath: (rawPath, sessionId) => sessionRelativePath(rawPath, sessionId),
          api: (url) => api(url),
          normalizeFileApiPath: (value) => normalizeFileApiPath(value),
          renderMenu: () => renderFilePickerMenu(),
        });
        const openedFileRuntime = codoxearFileViewer.createOpenedFileRuntime({
          currentSessionId: () => currentFileViewerSessionId(),
          selectedSessionId: () => selected,
          sessionRelativePath: (rawPath, sessionId) => sessionRelativePath(rawPath, sessionId),
          activeIdentity: () => currentActiveFileIdentity(),
          fileEntryForPath: (rel, gitPath, apiPath) => fileEntryForPath(rel, gitPath, apiPath),
          upsertFileEntry: (entry) => upsertFileEntry(entry),
          sessionById: (sessionId) => sessionIndex.get(sessionId) || null,
          listFromFilesField: (files) => listFromFilesField(files),
          listFromFileRecords: (files) => listFromFileRecords(files),
          deleteCandidateCache: (sessionId) => fileViewerController.deleteFileCandidateCache(sessionId),
        });
        const fileReferenceRuntime = codoxearFileViewer.createFileReferenceRuntime({
          selectedSessionId: () => selected,
          sessionById: (sessionId) => sessionIndex.get(sessionId) || null,
          sessions: () => Array.from(sessionIndex.values()),
          chatRoot: chatInner,
          ElementCtor: Element,
          sessionRelativePath: (rawPath, sessionId) => sessionRelativePath(rawPath, sessionId),
          listFromFilesField: (files) => listFromFilesField(files),
          listFromFileRecords: (files) => listFromFileRecords(files),
          normalizeFileApiPath: (value) => normalizeFileApiPath(value),
          normalizeLineNumber: (value) => normalizeLineNumber(value),
          parseLocalFileRef,
          showFileViewer: (options) => showFileViewer(options),
          selectSession: (sessionId) => selectSession(sessionId),
          openDirectorySession: (options) => openNewSessionDialog(options),
          setToast: (message) => setToast(message),
          api: (url, options) => api(url, options),
          el,
        });

        async function maybeHandleUnsavedFileChanges() {
          return await fileViewerController.maybeHandleUnsavedFileChanges();
        }

        function handleFileUnsavedSaveChoice() {
          return fileViewerController.handleFileUnsavedSaveChoice();
        }

        function handleFileUnsavedDiscardChoice() {
          return fileViewerController.handleFileUnsavedDiscardChoice();
        }

        function handleFileUnsavedCancelChoice() {
          return fileViewerController.handleFileUnsavedCancelChoice();
        }

        async function openDraftFilePathWithGuard(path) {
          return await fileViewerController.openDraftFilePathWithGuard(path);
        }

        async function requestHideFileViewer() {
          return await fileViewerController.requestHideFileViewer();
        }

        async function handleFileDiffModeButtonPress() {
          return await fileViewerController.handleFileDiffModeButtonPress();
        }

        async function handleFilePreviewModeButtonPress() {
          return await fileViewerController.handleFilePreviewModeButtonPress();
        }

        async function handleFileEditButtonPress() {
          return await fileViewerController.handleFileEditButtonPress();
        }

        function activeFileDownloadApiPath() {
          return fileViewerController.activeFileDownloadApiPath();
        }

        function setFileViewMode(mode) {
          return fileViewerController.setFileViewMode(mode);
        }

        function applyFileMode() {
          return fileModeControlsRuntime.apply(fileViewerController.currentFileModeControlState());
        }

        function applyFileMenuState() {
          return filePickerDomRuntime.apply();
        }

        function resetFilePickerInput() {
          return filePickerDomRuntime.resetInput(activeFilePathValue() || "");
        }

        function closeFilePickerMenu({ restoreInput = false } = {}) {
          return filePickerDomRuntime.close({ restoreInput, inputValue: activeFilePathValue() || "" });
        }

        function filePickerSelectionLine() {
          return filePickerMenuState.selectionLine(filePickerInput.value);
        }

        function openFilePickerSearchQuery(query, { line = null, suppressDraft = false } = {}) {
          return filePickerInputRuntime.openSearchQuery(query, { line, suppressDraft });
        }

        function normalizeFileApiPath(value) {
          return typeof value === "string" && value !== "" ? value : "";
        }

        function setFilePath(rel, { line = null, gitPath = undefined, apiPath = undefined } = {}) {
          return fileViewerPanelRuntime.setFilePath(rel, { line, gitPath, apiPath });
        }

        function fileCandidateKey(path, gitPath = false, apiPath = "") {
          return fileViewerController.fileCandidateKey(path, gitPath, apiPath);
        }

        function fileEntryForPath(path, gitPath = false, apiPath = "") {
          return fileViewerController.fileEntryForPath(path, gitPath, apiPath);
        }

        async function openFilePathWithResolvedMode(path, { line = null, changed = null, isCurrent = null, gitPath = null, apiPath = "" } = {}) {
          return await fileViewerController.openFilePathWithResolvedMode(path, { line, changed, isCurrent, gitPath, apiPath });
        }

        function upsertFileEntry(entry) {
          return fileViewerController.upsertFileEntry(entry);
        }

        function rememberOpenedFile(relPath, absPath = null) {
          return openedFileRuntime.remember(relPath, absPath);
        }

        function collectMessageFileRefs() {
          return fileReferenceRuntime.collectMessageFileRefs();
        }

        function resetFileSearchState() {
          filePickerSearchState.reset();
        }

        function filePickerSearchSnapshot() {
          return filePickerSearchState.snapshot();
        }

        function renderFilePickerMenu() {
          return filePickerRenderRuntime.render();
        }

        async function upgradeCandidateFileRefs(root) {
          return await fileReferenceRuntime.upgradeCandidateRefs(root);
        }

        function sessionRelativePath(rawPath, sidOverride = null) {
          const sid = typeof sidOverride === "string" && sidOverride ? sidOverride : selected;
          const s = sid ? sessionIndex.get(sid) : null;
          if (!s || !s.cwd) return null;
          const abs = stripPathLocationSuffix(rawPath);
          const cwd = String(s.cwd || "").replace(/\/+$/, "");
          if (!abs) return null;
          if (abs === cwd) return ".";
          if (abs.startsWith(cwd + "/")) return abs.slice(cwd.length + 1);
          return null;
        }

        async function refreshFileCandidates({ force = false, sessionId = null, syncToken = null } = {}) {
          return await fileCandidateRefreshRuntime.refresh({ force, sessionId, syncToken });
        }

        async function showFileViewer({ path = "", mode = "", manual = false, line = null, pickerQuery = "" } = {}) {
          void manual;
          if (selectedSessionLaunchFailed()) {
            setToast("failed launch has no file browser");
            return false;
          }
          return await fileViewerLifecycleRuntime.show({ path, mode, line, pickerQuery });
        }
        function hideFileViewer() {
          return fileViewerLifecycleRuntime.hide();
        }
        function handleFileViewerSessionUnavailable(sessionId) {
          return fileViewerController.handleFileViewerSessionUnavailable(sessionId);
        }
        async function applyFileLoadResult(rel, result, request, { viewMode = "file" } = {}) {
          return await fileLoadResultRuntime.apply(rel, result, request, { viewMode });
        }

        fileBtn.onclick = (e) => {
          e.preventDefault();
          e.stopPropagation();
          void showFileViewer();
        };
        filePickerInput.onfocus = () => filePickerInputRuntime.focus();
        filePickerInput.onclick = (e) => filePickerInputRuntime.click(e);
        filePickerInput.oninput = () => filePickerInputRuntime.input();
        filePickerInput.onblur = () => filePickerInputRuntime.blur();
        filePickerInput.onkeydown = (e) => filePickerInputRuntime.keydown(e);
        fileModeDiffBtn.onclick = (e) => {
          e.preventDefault();
          e.stopPropagation();
          void handleFileDiffModeButtonPress();
        };
        fileModePreviewBtn.onclick = (e) => {
          e.preventDefault();
          e.stopPropagation();
          void handleFilePreviewModeButtonPress();
        };
        fileEditBtn.onclick = async (e) => {
          e.preventDefault();
          e.stopPropagation();
          await handleFileEditButtonPress();
        };
        fileVideoPreviewBtn.onclick = (e) => {
          e.preventDefault();
          e.stopPropagation();
          void fileVideoPreviewRuntime.handleButtonPress();
        };

        fileDownloadBtn.onclick = (e) => {
          e.preventDefault();
          e.stopPropagation();
          fileDownloadRuntime.download(activeFileDownloadApiPath());
        };
        codoxearFileViewer.bindFileTouchPress(fileTouchSelectBtn, () => {
          toggleFileTouchSelectionMode();
        });
        codoxearFileViewer.bindFileTouchClick(fileTouchCopyBtn, () => {
          void copyActiveFileSelection();
        });
        codoxearFileViewer.bindFileTouchClick(fileTouchPasteBtn, () => {
          void pasteFromClipboardIntoActiveFile();
        });
        codoxearFileViewer.bindFileTouchPress(fileTouchUpBtn, () => {
          handleFileTouchMoveButtonPress("up");
        });
        codoxearFileViewer.bindFileTouchPress(fileTouchLeftBtn, () => {
          handleFileTouchMoveButtonPress("left");
        });
        codoxearFileViewer.bindFileTouchPress(fileTouchDownBtn, () => {
          handleFileTouchMoveButtonPress("down");
        });
        codoxearFileViewer.bindFileTouchPress(fileTouchRightBtn, () => {
          handleFileTouchMoveButtonPress("right");
        });
        fileCloseBtn.onclick = (e) => {
          e.preventDefault();
          e.stopPropagation();
          void requestHideFileViewer();
        };
        fileBackdrop.onclick = () => void requestHideFileViewer();
        $("#fileUnsavedSaveBtn").onclick = () => handleFileUnsavedSaveChoice();
        $("#fileUnsavedDiscardBtn").onclick = () => handleFileUnsavedDiscardChoice();
        $("#fileUnsavedCancelBtn").onclick = () => handleFileUnsavedCancelChoice();
        fileUnsavedBackdrop.onclick = () => handleFileUnsavedCancelChoice();
        $("#filePasteInsertBtn").onclick = () => {
          handleFilePasteInsert(filePasteInput.value);
        };
        $("#filePasteCancelBtn").onclick = () => hideFilePasteDialog({ restoreFocus: true });
        filePasteBackdrop.onclick = () => hideFilePasteDialog({ restoreFocus: true });
        chatInner.addEventListener("click", (e) => {
          if (codeBlockCopyRuntime.handleClick(e)) return;
          void fileReferenceRuntime.handleClick(e);
        });
        fileDiff.addEventListener("click", (e) => {
          void fileReferenceRuntime.handleClick(e);
        });
        addAppEvent(document, "click", (e) => {
          const t = e.target instanceof Element ? e.target : null;
          if (!t) return;
          if (isFileViewerOpen() && filePickerMenuState.isOpen() && !t.closest("#fileCandRow")) {
            closeFilePickerMenu({ restoreInput: true });
          }
          if (editDependencyMenuOpen && !t.closest("#editDependencyBtn") && !t.closest("#editDependencyMenu")) {
            editDependencyMenuOpen = false;
            applyDialogMenus();
          }
          if (newSessionCwdMenuOpen && !t.closest("#newSessionCwdField")) {
            newSessionCwdMenuOpen = false;
            newSessionCwdMenuFocus = -1;
            applyDialogMenus();
          }
          if (newSessionModelMenuOpen && !t.closest("#newSessionModelField")) {
            newSessionModelMenuOpen = false;
            newSessionModelMenuFocus = -1;
            applyDialogMenus();
          }
          if (newSessionReasoningMenuOpen && !t.closest("#newSessionReasoningBtn") && !t.closest("#newSessionReasoningMenu")) {
            newSessionReasoningMenuOpen = false;
            applyDialogMenus();
          }
          if (newSessionResumeMenuOpen && !t.closest("#newSessionResumeBtn") && !t.closest("#newSessionResumeMenu")) {
            newSessionResumeMenuOpen = false;
            applyDialogMenus();
          }
        });
        function handleFileTouchSelectionKeydown(e) {
          return fileViewerController.handleFileTouchSelectionKeydown(e);
        }
        addAppEvent(document, "keydown", handleFileTouchSelectionKeydown, true);
        addAppEvent(document, "keydown", handleFileEditorSaveShortcut, true);
        addAppEvent(document, "keydown", handleFileEditorDeleteKeydown, true);
        addAppEvent(
          document,
          "beforeinput",
          (e) => {
            suppressFileEditorNativeDelete(e);
          },
          true
        );
        addAppEvent(
          document,
          "input",
          (e) => {
            suppressFileEditorNativeDelete(e);
          },
          true
        );
        addAppEvent(document, "keydown", (e) => {
          if (e.key === "Tab" && appConfirm.style.display === "flex") {
            const focusable = appConfirmFocusableControls();
            e.preventDefault();
            e.stopPropagation();
            if (!focusable.length) return;
            const currentIndex = focusable.indexOf(document.activeElement);
            const offset = e.shiftKey ? -1 : 1;
            const nextIndex = currentIndex < 0 ? (e.shiftKey ? focusable.length - 1 : 0) : (currentIndex + offset + focusable.length) % focusable.length;
            try {
              focusable[nextIndex].focus({ preventScroll: true });
            } catch {}
            return;
          }
          if (e.key !== "Escape") return;
          if (appConfirm.style.display === "flex") {
            e.preventDefault();
            e.stopPropagation();
            resolveAppConfirm(false);
            return;
          }
          if (filePasteDialogRuntime.isOpen()) {
            hideFilePasteDialog({ restoreFocus: true });
            return;
          }
          if (fileUnsavedDialog.style.display === "flex") {
            hideFileUnsavedDialog("cancel");
            return;
          }
          if (isFileViewerOpen()) {
            e.preventDefault();
            void requestHideFileViewer();
            return;
          }
          if (sendChoice.style.display === "flex") {
            e.preventDefault();
            e.stopPropagation();
            closeSendChoiceDialog({ restoreFocus: true });
            return;
          }
          if (queueViewer.style.display === "flex") hideQueueViewer();
          if (helpViewer.style.display === "flex") hideHelpViewer();
          if (diagViewer.style.display === "flex") hideDiagViewer();
          if (voiceController.isSettingsOpen()) hideVoiceSettingsDialog();
          if (editViewer.style.display === "flex") hideEditSession();
          if (newSessionViewer.style.display === "flex") hideNewSessionDialog();
        });

        const queueController = (function instantiateQueueController() {
          const codoxearQueue = window.CodoxearQueue;
          if (!codoxearQueue || typeof codoxearQueue.createQueueController !== "function")
            throw new Error("Codoxear queue controller failed to load");
          return codoxearQueue.createQueueController({
            queueBackdrop,
            queueCloseBtn,
            queueList,
            queueEmpty,
            queueViewer,
            queueBtn: $("#queueBtn"),
            getSelected: () => selected,
            getSessionInfo: (sid) => sessionIndex.get(sid),
            isAppDisposed: () => appDisposed,
            api,
            setToast,
            clearCommitUnknownSend,
            refreshSessions,
            updateQueueBadge,
            syncRecoveryUiForSession,
            kickPoll,
            setPollFastUntilMs: (ms) => { pollFastUntilMs = ms; },
            handleAppAuthLoss,
            prepareModalOpen,
            afterModalVisibilityChanged,
            el,
            iconSvg,
            confirmAction: (options) => confirmApp(options),
            recoveryPanelFocusFallback: () => null,
          });
        })();

        function selectedSessionLaunchFailed() {
          return sessionLaunchFailed(selected ? sessionIndex.get(selected) : null);
        }

        function syncQueueSubmitState() {
          queueController.syncQueueSubmitState();
        }

        async function enqueueComposerText(raw, opts) {
          return queueController.enqueueComposerText(raw, opts);
        }

        async function refreshQueueViewer() {
          return queueController.refreshQueueViewer();
        }

        function showQueueViewer(opts) {
          return queueController.showQueueViewer(opts);
        }

        function hideQueueViewer() {
          return queueController.hideQueueViewer();
        }

        function showHelpViewer({ opener = null } = {}) {
          helpReturnFocusEl = opener instanceof HTMLElement ? opener : document.activeElement instanceof HTMLElement ? document.activeElement : null;
          prepareModalOpen();
          helpBackdrop.style.display = "block";
          helpViewer.style.display = "flex";
          afterModalVisibilityChanged();
          focusModalCloseButton(helpViewer, helpCloseBtn);
        }
        function hideHelpViewer() {
          const wasOpen = isModalTargetOpen(helpViewer);
          const focusTarget = helpReturnFocusEl;
          helpReturnFocusEl = null;
          helpBackdrop.style.display = "none";
          helpViewer.style.display = "none";
          afterModalVisibilityChanged();
          if (wasOpen) restoreModalFocus(focusTarget, () => isModalTargetOpen(helpViewer));
        }

        // Details/diagnostics modal state, rendering decisions, and the
        // New-like-this / Copy conversation / Copy details / show / hide
        // behavior live in the CodoxearDiagnostics controller
        // (codoxear/static/app_diagnostics.js).
        // app.js owns DOM construction for the diag nodes and the thin
        // delegating wrappers below; all diag rendering authority is delegated.
        const diagController = (function instantiateDiagnosticsController() {
          const codoxearDiagnostics = window.CodoxearDiagnostics;
          if (!codoxearDiagnostics || typeof codoxearDiagnostics.createDiagnosticsController !== "function")
            throw new Error("Codoxear diagnostics controller failed to load");
          return codoxearDiagnostics.createDiagnosticsController({
            diagBackdrop,
            diagViewer,
            diagContent,
            diagStatus,
            diagCloseBtn,
            diagNewLikeBtn,
            diagCopyConversationBtn,
            diagCopyBtn,
            getSelected: () => selected,
            getSessionInfo: (sid) => sessionIndex.get(sid),
            api,
            setToast,
            copyToClipboard,
            copyConversation,
            openNewSessionDialog,
            recoveryDetailsText,
            launchPresetFromSessionInfo,
            redactedLaunchErrorText,
            sessionLaunchLabel,
            agentBackendDisplayName,
            diagnosticsProviderDisplay,
            diagnosticsCopyText,
            fmtTs,
            fmtRelativeAge,
            formatPriorityOffset,
            prepareModalOpen,
            afterModalVisibilityChanged,
            el,
            uiVersion: UI_VERSION,
          });
        })();

        diagNewLikeBtn.onclick = (e) => diagController.onNewLikeClick(e);
        diagCopyConversationBtn.onclick = (e) => void diagController.onCopyConversationClick(e);
        diagCopyBtn.onclick = (e) => diagController.onCopyClick(e);

        async function showDiagViewer(opts) {
          return diagController.show(opts);
        }

        function hideDiagViewer(opts) {
          return diagController.hide(opts);
        }

        if (queueBtn) {
          queueBtn.onclick = (e) => {
            e.preventDefault();
            e.stopPropagation();
            const selectedInfo = sessionIndex.get(selected);
            if (sessionLaunchFailed(selectedInfo)) {
              setToast("failed session cannot receive messages");
              return;
            }
            const raw = $("#msg") ? $("#msg").value : "";
            if (raw && raw.trim()) {
              if (!selected) return;
              const sid = selected;
              void enqueueComposerText(raw, { sid }).then((ok) => {
                if (ok && selected === sid && $("#msg").value === raw) clearComposerInput();
              });
              return;
            }
            showQueueViewer({ opener: e.currentTarget });
          };
        }
        syncQueueSubmitState();
        queueCloseBtn.onclick = (e) => {
          e.preventDefault();
          e.stopPropagation();
          hideQueueViewer();
        };
        queueBackdrop.onclick = () => hideQueueViewer();

        $("#helpBtnSide").onclick = (e) => {
          e.preventDefault();
          e.stopPropagation();
          showHelpViewer({ opener: e.currentTarget });
        };
        $("#settingsBtnSide").onclick = (e) => {
          e.preventDefault();
          e.stopPropagation();
          showVoiceSettingsDialog();
        };
        helpCloseBtn.onclick = (e) => {
          e.preventDefault();
          e.stopPropagation();
          hideHelpViewer();
        };
        helpBackdrop.onclick = () => hideHelpViewer();

        diagBtn.onclick = (e) => {
          e.preventDefault();
          e.stopPropagation();
          void showDiagViewer({ opener: e.currentTarget });
        };
        diagCloseBtn.onclick = (e) => {
          e.preventDefault();
          e.stopPropagation();
          hideDiagViewer();
        };
        diagBackdrop.onclick = () => hideDiagViewer();
        async function spawnSessionWithCwd(cwd, resumeSessionId = null, worktreeBranch = null, sessionName = "", providerChoice = "chatgpt", model = "default", reasoningEffort = "high", fast = false, createInTmux = false, errorHandler = null, agentBackend = "codex") {
          if (!cwd || !String(cwd).trim()) {
            setToast("cwd unavailable");
            return null;
          }
          try {
            const backend = normalizeAgentBackendName(agentBackend);
            const modeLabel = resumeSessionId ? "resuming..." : worktreeBranch ? "creating worktree..." : createInTmux ? "starting in tmux..." : "starting...";
            const alias = String(sessionName || "").trim();
            const providerName = String(providerChoice || "").trim();
            const providerSettings = providerChoiceToSettings(providerName, backend);
            const modelName = String(model || "").trim();
            const effortName = String(reasoningEffort || "").trim().toLowerCase();
            setToast(modeLabel);
            const body = { cwd: String(cwd), agent_backend: backend };
            if (resumeSessionId) body.resume_session_id = String(resumeSessionId);
            if (worktreeBranch) body.worktree_branch = String(worktreeBranch);
            if (providerSettings.model_provider) body.model_provider = providerSettings.model_provider;
            if (providerSettings.preferred_auth_method) body.preferred_auth_method = providerSettings.preferred_auth_method;
            if (modelName) body.model = modelName;
            if (effortName) body.reasoning_effort = effortName;
            if (backendSupportsFast(backend) && fast) body.service_tier = "fast";
            if (createInTmux) body.create_in_tmux = true;
            const res = await api("/api/sessions", { method: "POST", body });
            if (res && res.pending && res.launch_id) {
              setToast(createInTmux ? "tmux session still starting" : "session still starting");
              await refreshSessions();
              return String(res.launch_id);
            }
            const brokerPid = res && res.broker_pid ? Number(res.broker_pid) : null;
            if (!brokerPid) {
              setToast("start failed");
              return null;
            }
            const doneLabel = resumeSessionId ? "resumed" : worktreeBranch ? "worktree started" : createInTmux ? "tmux started" : "started";
            setToast(`${doneLabel} (broker ${brokerPid})`);
            for (let i = 0; i < 60; i++) {
              const sessions = await refreshSessions();
              let found = (sessions || []).find((x) => Number(x.broker_pid || 0) === brokerPid);
              if (found) {
                if (alias && String(found.alias || "").trim() !== alias) {
                  await api(`/api/sessions/${found.session_id}/rename`, { method: "POST", body: { name: alias } });
                  const renamed = await refreshSessions();
                  found = (renamed || []).find((x) => x.session_id === found.session_id) || found;
                }
                selectSession(found.session_id);
                return brokerPid;
              }
              await new Promise((r) => setTimeout(r, 250));
            }
            setToast(`${doneLabel} session will appear once the agent writes its session log`);
            return brokerPid;
          } catch (e) {
            const errLabel = resumeSessionId ? "resume" : worktreeBranch ? "worktree start" : "start";
            if (typeof errorHandler === "function") errorHandler(e);
            setToast(`${errLabel} error: ${e.message}`);
            void refreshSessions().catch((err) => console.error("refreshSessions failed after launch error", err));
            return null;
          }
        }
        $("#newBtn").onclick = async () => {
          openNewSessionDialog();
        };
        $("#chatEmptyNewBtn").onclick = async () => {
          openNewSessionDialog();
        };
	        async function interruptSelectedSession() {
	          if (!selected) return;
	          try {
	            setToast("interrupting...");
            await api(`/api/sessions/${selected}/interrupt`, { method: "POST" });
            pollFastUntilMs = Date.now() + 2500;
            kickPoll(0);
          } catch (e) {
            setToast(`interrupt error: ${e.message}`);
          }
        }
        interruptBtn.onclick = (e) => {
          e.preventDefault();
          e.stopPropagation();
          void interruptSelectedSession();
        };
        if (composerStopBtn) {
          composerStopBtn.onclick = (e) => {
            e.preventDefault();
            e.stopPropagation();
            void interruptSelectedSession();
          };
        }

        $("#logoutBtnSide").onclick = async () => {
          try {
            await api("/api/logout", { method: "POST" });
          } catch (e) {
            console.error("logout failed", e);
          } finally {
            if (appDisposed) return;
            cleanupApp();
            renderLogin(renderApp);
          }
        };

        toggleSidebarBtn.onclick = () => {
          if (isMobile()) {
            setSidebarOpen(!document.body.classList.contains("sidebar-open"));
            return;
          }
          setSidebarCollapsed(!document.body.classList.contains("sidebar-collapsed"));
        };
	        backdrop.onclick = () => setSidebarOpen(false);

        chat.addEventListener("scroll", () => {
          transcriptScrollRuntime.handleScroll();
        });
        chat.addEventListener(
          "wheel",
          (e) => {
            transcriptScrollRuntime.handleWheel(e);
          },
          { passive: true }
        );
        chat.addEventListener(
          "touchstart",
          (e) => {
            transcriptScrollRuntime.handleTouchStart(e);
          },
          { passive: true }
        );
        chat.addEventListener(
          "touchmove",
          (e) => {
            // Finger moves down -> content scrolls up.
            transcriptScrollRuntime.handleTouchMove(e);
          },
          { passive: true }
        );
        jumpBtn.onclick = () => {
          void jumpToLatest();
        };
        olderBtn.onclick = () => {
          void loadOlderMessages({ auto: false });
        };
        olderRetryBtn.onclick = () => {
          clearOlderLoadError();
          void loadOlderMessages({ auto: false });
        };

         const isIOS =
           /iP(hone|od|ad)/.test(navigator.userAgent || "") ||
           (navigator.platform === "MacIntel" && navigator.maxTouchPoints && navigator.maxTouchPoints > 1);
	         function activeTextEntryElement() {
	           const active = document.activeElement;
	           return isTextEntryElement(active) ? active : null;
	         }
	         let iosViewportGuardTimer = null;
	         let iosViewportGuardUntil = 0;
	         function normalizePageScroll() {
	           if (!isIOS) return;
	           const activeEntry = activeTextEntryElement();
	           if (activeEntry && activeEntry !== textarea) return;
	           const y = window.scrollY || document.documentElement.scrollTop || document.body.scrollTop || 0;
	           if (y <= 0) return;
	           window.scrollTo(0, 0);
	           document.documentElement.scrollTop = 0;
	           document.body.scrollTop = 0;
	         }
	         function stopIOSViewportGuard() {
	           if (iosViewportGuardTimer) clearTimeout(iosViewportGuardTimer);
	           iosViewportGuardTimer = null;
	           iosViewportGuardUntil = 0;
	         }
	         function isIOSViewportGuardActive() {
	           return isIOS && Date.now() < iosViewportGuardUntil;
	         }
	         function runIOSViewportGuard({ preserveChatBottom, durationMs = 1400 } = {}) {
	           if (!isIOS) return;
	           stopIOSViewportGuard();
	           iosViewportGuardUntil = Date.now() + Math.max(0, Number(durationMs) || 0);
	           const tick = () => {
	             const activeEntry = activeTextEntryElement();
	             if (activeEntry && activeEntry !== textarea) {
	               stopIOSViewportGuard();
	               return;
	             }
	             updateAppHeightVar();
	             normalizePageScroll();
	             if (preserveChatBottom && transcriptScrollRuntime.shouldAutoScrollOrNearBottom()) transcriptScrollRuntime.scrollToBottom();
	             if (!isIOSViewportGuardActive()) {
	               iosViewportGuardTimer = null;
	               return;
	             }
	             iosViewportGuardTimer = setTimeout(tick, 50);
	           };
	           tick();
	         }
	         if (window.visualViewport) {
	           const onViewportShift = () => {
	             updateAppHeightVar();
	             if (!isIOS) return;
	             const activeEntry = activeTextEntryElement();
	             if (activeEntry && activeEntry !== textarea) {
	               stopIOSViewportGuard();
	               return;
	             }
	             if (document.activeElement === textarea || isIOSViewportGuardActive()) {
	               normalizePageScroll();
	               if (transcriptScrollRuntime.shouldAutoScrollOrNearBottom()) transcriptScrollRuntime.scheduleScrollToBottom();
	             }
	           };
	           addAppEvent(window.visualViewport, "resize", onViewportShift);
	           addAppEvent(window.visualViewport, "scroll", onViewportShift);
	         }
         if (!attachBadgeEl) {
           attachBadgeEl = el("span", { class: "attachBadge", id: "attachBadge" });
           attachBtn.appendChild(attachBadgeEl);
         }
        if (!queueBadgeEl && queueBtn) {
          queueBadgeEl = el("span", { class: "attachBadge queueBadge", id: "queueBadge" });
          queueBtn.appendChild(queueBadgeEl);
        }
        function normalizedStagedAttachments(list) {
          if (!Array.isArray(list)) return [];
          return list
            .filter((item) => item && typeof item === "object" && typeof item.id === "string" && item.id)
            .map((item) => ({
              id: String(item.id),
              display_name: String(item.display_name || item.filename || "file"),
              filename: String(item.filename || item.display_name || "file"),
              size: Number.isFinite(Number(item.size)) ? Number(item.size) : 0,
              created_ts: Number.isFinite(Number(item.created_ts)) ? Number(item.created_ts) : 0,
            }));
        }
        function attachmentIdentityText(item) {
          const name = item && (item.display_name || item.filename) ? String(item.display_name || item.filename) : "staged attachment";
          const id = item && item.id ? String(item.id).slice(0, 8) : "";
          const size = item && Number.isFinite(Number(item.size)) ? fmtBytes(Number(item.size)) : "0 B";
          return id ? `${name} · ${size} · attachment ${id}` : `${name} · ${size}`;
        }
        function setStagedAttachments(list) {
          stagedAttachments = normalizedStagedAttachments(list);
          attachedFiles = stagedAttachments.length;
          renderStagedAttachments();
          if (textarea) resizeComposer();
          projectSelectedAttachmentIndicator();
        }
        const setAttachCount = (n) => {
          attachedFiles = Math.max(0, Number(n) || 0);
          if (attachedFiles === 0 && stagedAttachments.length) stagedAttachments = [];
          renderStagedAttachments();
          projectSelectedAttachmentIndicator();
        };
        function setSelectedSessionStagedAttachments(list) {
          if (selected) {
            const info = sessionIndex.get(selected);
            if (info) {
              info.staged_attachments = normalizedStagedAttachments(list);
              info.pending_attachment = info.staged_attachments.length > 0;
              sessionIndex.set(selected, info);
            }
          }
          setStagedAttachments(list);
        }
        function syncStagedAttachmentsFromSelectedSession() {
          const info = selected ? sessionIndex.get(selected) : null;
          setStagedAttachments(info && Array.isArray(info.staged_attachments) ? info.staged_attachments : []);
        }
        function renderStagedAttachments() {
          const tray = $("#stagedAttachments");
          if (!tray) return;
          tray.innerHTML = "";
          if (!stagedAttachments.length) {
            tray.style.display = "none";
            return;
          }
          tray.style.display = "flex";
          for (const item of stagedAttachments) {
            const chip = el("div", { class: "stagedAttachmentChip", title: attachmentIdentityText(item) });
            chip.appendChild(el("span", { class: "stagedAttachmentName", text: item.display_name || item.filename || "file" }));
            chip.appendChild(el("span", { class: "stagedAttachmentMeta", text: fmtBytes(item.size || 0) }));
            const removeBtn = el("button", { class: "stagedAttachmentRemove", type: "button", text: "×", title: `Remove ${item.display_name || "attachment"}`, "aria-label": `Remove ${item.display_name || "attachment"}` });
            removeBtn.onclick = async () => {
              if (!selected) return;
              const sid = selected;
              try {
                const res = await api(`/api/sessions/${sid}/attachments/delete`, { method: "POST", body: { id: item.id } });
                if (selected === sid) {
                  setSelectedSessionStagedAttachments(res && Array.isArray(res.attachments) ? res.attachments : []);
                  setToast("attachment removed");
                  void refreshSessions().catch((e) => {
                    if (e && e.status === 401) handleAppAuthLoss();
                    else console.error("refreshSessions failed", e);
                  });
                }
              } catch (err) {
                if (err && err.status === 401) {
                  handleAppAuthLoss();
                  return;
                }
                if (selected === sid) setToast(`remove attachment error: ${err && err.message ? err.message : "unknown error"}`);
              }
            };
            chip.appendChild(removeBtn);
            tray.appendChild(chip);
          }
          const clearBtn = el("button", { class: "stagedAttachmentsClear", type: "button", text: "Clear", title: "Clear staged attachments", "aria-label": "Clear staged attachments" });
          clearBtn.onclick = async () => {
            if (!selected) return;
            const sid = selected;
            try {
              const res = await api(`/api/sessions/${sid}/attachments/clear`, { method: "POST", body: {} });
              if (selected === sid) {
                setSelectedSessionStagedAttachments(res && Array.isArray(res.attachments) ? res.attachments : []);
                setToast("attachments cleared");
                void refreshSessions().catch((e) => {
                  if (e && e.status === 401) handleAppAuthLoss();
                  else console.error("refreshSessions failed", e);
                });
              }
            } catch (err) {
              if (err && err.status === 401) {
                handleAppAuthLoss();
                return;
              }
              if (selected === sid) setToast(`clear attachments error: ${err && err.message ? err.message : "unknown error"}`);
            }
          };
          tray.appendChild(clearBtn);
        }
        // The visible attachment indicator is a projection of the selected
        // session's server-owned staged attachment list; the legacy
        // pending_attachment flag is only a compatibility fallback.
        function projectSelectedAttachmentIndicator() {
          if (!attachBadgeEl) return;
          const sessionInfo = selected ? sessionIndex.get(selected) : null;
          const serverListCount = sessionInfo && Array.isArray(sessionInfo.staged_attachments) ? normalizedStagedAttachments(sessionInfo.staged_attachments).length : 0;
          const serverPending = Boolean(sessionInfo && sessionInfo.pending_attachment);
          const visible = Math.max(stagedAttachments.length, serverListCount, serverPending ? 1 : 0);
          if (visible > 0) {
            attachBadgeEl.textContent = String(visible);
            attachBadgeEl.style.display = "inline-flex";
          } else {
            attachBadgeEl.textContent = "";
            attachBadgeEl.style.display = "none";
          }
        };
        // Mutate the selected session's cached pending_attachment only when the
        // frontend has direct evidence it changed (successful attach -> true;
        // successful send with allow_pending_attachment -> false; successful
        // pending_attachment/clear -> false). This keeps the cached value in
        // step with what the server now knows until the next refreshSessions()
        // returns the authoritative value, so the indicator does not re-render
        // against stale pending_attachment=true right after a send/clear.
        function setSelectedSessionPendingAttachment(value) {
          if (!selected) return false;
          const info = sessionIndex.get(selected);
          if (!info) return false;
          info.pending_attachment = Boolean(value);
          if (!value) info.staged_attachments = [];
          sessionIndex.set(selected, info);
          if (!value) setStagedAttachments([]);
          else projectSelectedAttachmentIndicator();
          return true;
        }
        function attachmentBlockerForSession(sessionId, sessionInfo = null) {
          if (!sessionId) return "Select a session to attach a file";
          const info = sessionInfo || sessionIndex.get(sessionId) || null;
          if (info && sessionLaunchFailed(info)) return "Failed launch cannot receive file attachments";
          if (info && sessionHasUnknownSend(info)) return "Resolve the unknown send before attaching a file";
          if (info && sessionIsOrphanRecovery(info)) return "Missing session can only be reviewed";
          if (info && sessionHasOrphanQueueRecovery(info)) return "Review preserved queued recovery items before attaching a file";
          if (sending) return "Wait for the current send to finish before attaching a file";
          return "";
        }
        function latestAttachmentBlockerForSession(sessionId) {
          return attachmentBlockerForSession(sessionId, sessionId ? sessionIndex.get(sessionId) || null : null);
        }
        function syncAttachButtonState() {
          const attachControl = $("#attachBtn");
          if (!attachControl) return;
          const selectedInfo = selected ? sessionIndex.get(selected) || null : null;
          const attachBlocker = attachmentBlockerForSession(selected, selectedInfo);
          const attachLabel = attachBlocker || `Attach file (max ${fmtBytes(ATTACH_UPLOAD_MAX_BYTES)})`;
          attachControl.disabled = Boolean(attachBlocker);
          attachControl.title = attachLabel;
          attachControl.setAttribute("aria-label", attachLabel);
        }
        setAttachCount(0);
        syncAttachButtonState();
        if (!selected) {
          attachBtn.disabled = true;
          attachBtn.title = "Select a session to attach a file";
          attachBtn.setAttribute("aria-label", "Select a session to attach a file");
        }
        updateQueueBadge();
        syncQueueSubmitState();
        syncComposerSendButton();
	          textarea.addEventListener(
	            "focus",
	            () => {
	              const wasNear = transcriptScrollRuntime.isNearBottom();
              if (wasNear) {
                transcriptScrollRuntime.enableAutoScroll();
                transcriptScrollRuntime.syncJumpButton();
              }
	              if (isIOS) runIOSViewportGuard({ preserveChatBottom: wasNear, durationMs: 1800 });
	              else {
	                const tick = () => {
	                  updateAppHeightVar();
	                  if (wasNear) transcriptScrollRuntime.scrollToBottom();
	                };
	                requestAnimationFrame(tick);
	                setTimeout(tick, 120);
	              }
	            },
	            { passive: true }
	          );
	          textarea.addEventListener(
	            "blur",
	            () => {
	              setTimeout(() => {
	                if (isIOS) {
	                  const activeEntry = activeTextEntryElement();
	                  if (activeEntry && activeEntry !== textarea) {
	                    stopIOSViewportGuard();
	                    updateAppHeightVar();
	                    return;
	                  }
	                  runIOSViewportGuard({ preserveChatBottom: false, durationMs: 900 });
	                  return;
	                }
	                updateAppHeightVar();
	              }, 0);
	            },
	            { passive: true }
	          );
        textarea.addEventListener("keydown", (e) => {
          if (e.key === "Escape") {
            // Exit the message box and return focus to the document so global
            // shortcuts (hint mode, etc.) are reachable again. Without this,
            // Esc does nothing in the composer and the user is stranded.
            e.preventDefault();
            e.stopPropagation();
            textarea.blur();
            return;
          }
          if (e.key !== "Enter") return;
          if (e.isComposing) return;
          if (!(e.ctrlKey || e.metaKey)) return;
          e.preventDefault();
          form.requestSubmit();
        });

        async function toJpegBlob(file, { maxDim = 2048, quality = 0.86 } = {}) {
          const url = URL.createObjectURL(file);
          try {
            const img = new Image();
            img.decoding = "async";
            img.src = url;
            if (img.decode) await img.decode();
            else
              await new Promise((resolve, reject) => {
                img.onload = resolve;
                img.onerror = () => reject(new Error("decode failed"));
              });
            const w0 = img.naturalWidth || img.width || 0;
            const h0 = img.naturalHeight || img.height || 0;
            if (!w0 || !h0) throw new Error("invalid image dimensions");
            const scale = Math.min(1, maxDim / Math.max(w0, h0));
            const w = Math.max(1, Math.round(w0 * scale));
            const h = Math.max(1, Math.round(h0 * scale));
            const canvas = document.createElement("canvas");
            canvas.width = w;
            canvas.height = h;
            const ctx = canvas.getContext("2d", { alpha: false });
            if (!ctx) throw new Error("no canvas");
            ctx.drawImage(img, 0, 0, w, h);
            const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", quality));
            if (!blob) throw new Error("jpeg encode failed");
            return blob;
          } finally {
            URL.revokeObjectURL(url);
          }
        }

        function imageExtensionFromMimeType(type, fallback = "") {
          const normalized = String(type || "").toLowerCase();
          if (normalized === "image/jpeg" || normalized === "image/jpg") return "jpg";
          if (normalized === "image/png") return "png";
          if (normalized === "image/gif") return "gif";
          if (normalized === "image/webp") return "webp";
          if (normalized === "image/heic") return "heic";
          if (normalized === "image/heif") return "heif";
          if (normalized === "image/avif") return "avif";
          return normalized.startsWith("image/") ? fallback : "";
        }

        function pastedFileName(file, index, seed) {
          const suffix = index > 0 ? `-${index + 1}` : "";
          const base = `pasted-${seed}${suffix}`;
          const ext = imageExtensionFromMimeType(file && file.type, "png");
          return ext ? `${base}.${ext}` : base;
        }

        async function stageFiles(files, { sid = selected, source = "picker" } = {}) {
          const sessionId = sid || selected;
          const uploadFiles = Array.from(files || []).filter(Boolean);
          if (!uploadFiles.length) return false;

          const producer = String(source || "picker");
          const progressVerb = producer === "paste" ? "pasting" : producer === "drop" ? "dropping" : "uploading";
          const producerNameSeed = Date.now();
          let successes = 0;
          let stoppedByBlocker = "";
          const failures = [];
          for (let fileIndex = 0; fileIndex < uploadFiles.length; fileIndex += 1) {
            const f = uploadFiles[fileIndex];
            try {
              if (selected !== sessionId) break;
              const attachBlocker = latestAttachmentBlockerForSession(sessionId);
              if (attachBlocker) {
                stoppedByBlocker = attachBlocker;
                break;
              }
              setToast(uploadFiles.length > 1 ? `${progressVerb} ${fileIndex + 1}/${uploadFiles.length}...` : "uploading file...");
              const maxBytes = ATTACH_UPLOAD_MAX_BYTES;
              let uploadBlob = f;
              let uploadName = f.name || (producer === "paste" ? pastedFileName(f, fileIndex, producerNameSeed) : "file");
              if (looksLikeImage(f) && (f.size > maxBytes || isLikelyHeic(f))) {
                setToast("compressing image...");
                const stem = safeAttachmentStem(uploadName);
                uploadName = `${stem}.jpg`;
                const tries = [
                  { maxDim: 2048, quality: 0.86 },
                  { maxDim: 1600, quality: 0.82 },
                  { maxDim: 1600, quality: 0.72 },
                  { maxDim: 1280, quality: 0.68 },
                  { maxDim: 1280, quality: 0.58 },
                ];
                let blob = null;
                for (const t of tries) {
                  blob = await toJpegBlob(f, t);
                  if (blob.size <= maxBytes) break;
                }
                if (!blob || blob.size > maxBytes) throw new Error(`image too large (max ${fmtBytes(maxBytes)})`);
                uploadBlob = blob;
              }
              const ab = await uploadBlob.arrayBuffer();
              if (ab.byteLength > maxBytes) throw new Error(`file too large (max ${fmtBytes(maxBytes)})`);
              const b64 = b64FromBytes(new Uint8Array(ab));
              const res = await api(`/api/sessions/${sessionId}/inject_file`, {
                method: "POST",
                body: { filename: uploadName, data_b64: b64 },
              });
              if (selected === sessionId && res && res.ok) {
                successes += 1;
                setSelectedSessionStagedAttachments(Array.isArray(res.attachments) ? res.attachments : []);
              }
            } catch (e) {
              if (e && e.status === 401) {
                handleAppAuthLoss();
                return false;
              }
              failures.push(`${f && f.name ? f.name : "file"}: ${e && e.message ? e.message : "unknown error"}`);
            }
          }
          if (selected === sessionId) {
            if (successes && failures.length) setToast(`attached ${successes}; ${failures.length} failed: ${failures[0]}`);
            else if (successes && stoppedByBlocker) setToast(`attached ${successes}; stopped: ${stoppedByBlocker}`);
            else if (successes) setToast(successes === 1 ? "file staged" : `${successes} files staged`);
            else if (failures.length) setToast(`attach error: ${failures[0]}`);
            else if (stoppedByBlocker) setToast(stoppedByBlocker);
            pollFastUntilMs = Date.now() + 4000;
            kickPoll(0);
            void refreshSessions().catch((refreshErr) => {
              if (refreshErr && refreshErr.status === 401) handleAppAuthLoss();
              else console.error("refreshSessions failed", refreshErr);
            });
          }
          return successes > 0;
        }

        attachBtn.onclick = () => {
          const sid = selected;
          const sessionInfo = sid ? sessionIndex.get(sid) || null : null;
          const attachBlocker = attachmentBlockerForSession(sid, sessionInfo);
          if (attachBlocker) {
            setToast(attachBlocker);
            return;
          }
          imgInput.value = "";
          imgInput.click();
        };
        imgInput.addEventListener("change", async () => {
          const sid = selected;
          if (!sid) return;
          const files = Array.from(imgInput.files || []);
          imgInput.value = "";
          await stageFiles(files, { sid, source: "picker" });
        });


        function clipboardPlainText(data) {
          if (!data || typeof data.getData !== "function") return "";
          try {
            return data.getData("text/plain") || data.getData("text") || "";
          } catch (_) {
            return "";
          }
        }

        function insertComposerPastedText(text) {
          const value = String(text || "");
          if (!value) return false;
          const start = Number.isFinite(textarea.selectionStart) ? textarea.selectionStart : textarea.value.length;
          const end = Number.isFinite(textarea.selectionEnd) ? textarea.selectionEnd : start;
          if (typeof textarea.setRangeText === "function") {
            textarea.setRangeText(value, start, end, "end");
          } else {
            textarea.value = `${textarea.value.slice(0, start)}${value}${textarea.value.slice(end)}`;
            textarea.selectionStart = start + value.length;
            textarea.selectionEnd = start + value.length;
          }
          textarea.dispatchEvent(new Event("input", { bubbles: true }));
          return true;
        }

        textarea.addEventListener("paste", (e) => {
          const files = extractFilesFromClipboardData(e.clipboardData);
          if (!files.length) return;
          const pastedText = clipboardPlainText(e.clipboardData);
          e.preventDefault();
          if (pastedText) insertComposerPastedText(pastedText);
          void stageFiles(files, { sid: selected, source: "paste" });
        });

        let composerDragDepth = 0;
        function setComposerDropActive(active) {
          composer.classList.toggle("drop-active", Boolean(active));
        }
        function clearComposerDropActive() {
          composerDragDepth = 0;
          setComposerDropActive(false);
        }
        addAppEvent(composer, "dragenter", (e) => {
          if (!dataTransferHasFiles(e.dataTransfer)) return;
          e.preventDefault();
          composerDragDepth += 1;
          setComposerDropActive(true);
        }, { passive: false });
        addAppEvent(composer, "dragover", (e) => {
          if (!dataTransferHasFiles(e.dataTransfer)) return;
          e.preventDefault();
          if (e.dataTransfer) e.dataTransfer.dropEffect = "copy";
          setComposerDropActive(true);
        }, { passive: false });
        addAppEvent(composer, "dragleave", (e) => {
          if (!dataTransferHasFiles(e.dataTransfer)) return;
          composerDragDepth = Math.max(0, composerDragDepth - 1);
          if (composerDragDepth === 0) setComposerDropActive(false);
        }, { passive: false });
        addAppEvent(composer, "drop", (e) => {
          if (!dataTransferHasFiles(e.dataTransfer)) return;
          e.preventDefault();
          clearComposerDropActive();
          const files = extractFilesFromDropData(e.dataTransfer);
          if (!files.length) return;
          void stageFiles(files, { sid: selected, source: "drop" });
        }, { passive: false });
        addAppEvent(window, "dragover", (e) => {
          if (!dataTransferHasFiles(e.dataTransfer)) return;
          e.preventDefault();
        }, { passive: false });
        addAppEvent(window, "dragleave", (e) => {
          const outsideWindow =
            e.clientX <= 0 ||
            e.clientY <= 0 ||
            e.clientX >= window.innerWidth ||
            e.clientY >= window.innerHeight ||
            (!e.relatedTarget && (e.target === document || e.target === document.documentElement || e.target === document.body));
          if (outsideWindow) clearComposerDropActive();
        }, { passive: false });
        addAppEvent(window, "dragend", () => {
          clearComposerDropActive();
        }, { passive: false });
        addAppEvent(window, "drop", (e) => {
          if (dataTransferHasFiles(e.dataTransfer)) e.preventDefault();
          clearComposerDropActive();
        }, { passive: false });

        composerController = codoxearComposer.createComposerController({
          form,
          textarea,
          msgPh,
          modelPicker,
          sendBtn,
          sendChoice,
          sendChoiceBackdrop,
          sendChoiceNowBtn: $("#sendChoiceNow"),
          sendChoiceLaterBtn: $("#sendChoiceLater"),
          sendChoiceCancelBtn: $("#sendChoiceCancel"),
          getSelected: () => selected,
          getSessionInfo: (sessionId) => sessionIndex.get(sessionId) || null,
          getNewSessionDefaults: () => newSessionDefaults,
          patchSessionInfo: (sessionId, patch) => {
            const current = sessionIndex.get(sessionId);
            if (!current) return;
            Object.assign(current, patch || {});
            sessionIndex.set(sessionId, current);
          },
          sessionLaunchFailed,
          getSending: () => sending,
          setSending: (value) => { sending = Boolean(value); },
          getCurrentRunning: () => currentRunning,
          setCurrentRunning: (value) => { currentRunning = Boolean(value); },
          setTurnOpen: (value) => { turnOpen = Boolean(value); },
          getStagedAttachments: () => stagedAttachments.slice(),
          normalizedStagedAttachments,
          setSelectedSessionPendingAttachment: (sessionId, value) => {
            if (selected === sessionId) setSelectedSessionPendingAttachment(value);
          },
          setAttachCount,
          syncAttachButtonState,
          syncQueueSubmitState,
          syncRecoveryUiForSession,
          confirmAction: (options) => confirmApp(options),
          api,
          setToast,
          handleAppAuthLoss,
          refreshSessions,
          setPollFastUntilMs: (value) => { pollFastUntilMs = value; },
          kickPoll,
          isTranscriptRenewalCommand,
          nextLocalEchoId: () => transcriptEventRuntime.nextLocalEchoId(),
          renderedAtLiveTail: () => transcriptScrollRuntime.snapshot().renderedAtLiveTail,
          clearTranscriptDom,
          clearRenderedTranscriptRange,
          setOlderState,
          getSessionTranscriptSlot,
          addPendingUser: (pending) => transcriptEventRuntime.addPendingUser(pending),
          appendEvent,
          deleteTailCache: (sessionId) => transcriptSlotRuntime.deleteTailCache(sessionId),
          beginTranscriptRenewal,
          clearLiveCursor: () => transcriptSlotRuntime.clearLiveCursor(),
          invalidateOlderLoad,
          renderPendingTranscriptSlot,
          dropPendingUser: (sessionId, localId) => transcriptEventRuntime.dropPendingUsers(sessionId, (pending) => pending && pending.id === localId),
          removePendingUserRow: (localId) => {
            const pendingEl = chatInner.querySelector(`.msg.user[data-local-id="${localId}"]`);
            if (!pendingEl) return;
            const pendingRow = pendingEl.closest(".msg-row");
            if (pendingRow) pendingRow.remove();
            else pendingEl.remove();
          },
          hasPendingForSession: (sessionId) => transcriptEventRuntime.hasPendingForSession(sessionId),
          enqueueComposerText,
          prepareModalOpen,
          afterModalVisibilityChanged,
          restoreModalFocus,
          storageGetItem,
          storageSetItem,
          storageRemoveItem,
          onAutoGrow: () => {
            if (transcriptScrollRuntime.snapshot().autoScroll) transcriptScrollRuntime.scheduleScrollToBottom();
          },
          requestFrame: (callback) => requestAnimationFrame(callback),
          getComputedStyle: (node) => getComputedStyle(node),
          activeElement: () => document.activeElement,
          isHTMLElement: (value) => value instanceof HTMLElement,
          now: () => Date.now(),
          consoleError: (...args) => console.error(...args),
          windowTarget: window,
        });

        activeAppCleanup = cleanupApp;
        if (typeof window.__codoxearMarkBootstrapped === "function") window.__codoxearMarkBootstrapped();

	        (async () => {
	          if (storageGetItem("codexweb.sidebarCollapsed") === "1") setSidebarCollapsed(true);
	          if (storageGetItem("codexweb.sidebarOpen") === "1") setSidebarOpen(true);

	          try {
	            const sessions = await refreshSessions();
              const hashed = sessionIdFromHash();
	            const remembered = storageGetItem("codexweb.selected");
	            const first = sessions && sessions.length ? (sessions.find(sessionSelectable) || {}).session_id || null : null;
	            const pick =
	              hashed && sessionSelectable(sessionIndex.get(hashed))
	                ? hashed
	                : remembered && sessionSelectable(sessionIndex.get(remembered))
	                  ? remembered
	                  : first;
	            if (pick) await selectSession(pick);
              void (async () => {
                try {
                  await Promise.all([loadVoiceSettings(), syncNotificationState()]);
                  if (appDisposed) return;
                  if (voiceAnnouncementsEnabled()) resumeAnnouncementRuntime({ resetSource: false });
                  if (notificationsEnabledLocally()) await pollNotificationFeed({ prime: true });
                } catch (e) {
                  if (e && e.status === 401) handleAppAuthLoss();
                  else console.error("initial voice and notification sync failed", e);
                }
              })();
	          } catch (e) {
	            if (e && e.status === 401) {
              handleAppAuthLoss();
	              return;
	            }
	            console.error("initial refreshSessions failed", e);
	            setToast(`sessions error: ${e && e.message ? e.message : "unknown error"}`);
	          } finally {
              if (appDisposed) return;
	            if (msgPh) msgPh.style.display = textarea.value ? "none" : "flex";
	            resizeComposer();

	            scheduleSessionsPoll();
            scheduleSecondaryPoll();
              addAppEvent(window, "hashchange", async () => {
                await selectSessionFromHash({ refreshIfMissing: true, deferIfMissing: true });
              });
              addAppEvent(window, "beforeunload", () => {
                cleanupApp();
              });
              addAppEvent(document, "visibilitychange", () => {
                if (appDisposed) return;
                if (document.visibilityState === "visible") {
                  resumeAnnouncementRuntime({ resetSource: false });
                  if (selected) kickPoll(0);
                  scheduleSessionsPoll(0);
                  scheduleSecondaryPoll(0);
                  return;
                }
                if (selected) kickPoll(messagePollDelayMs());
                scheduleSessionsPoll(sessionsPollDelayMs());
                scheduleSecondaryPoll(secondaryPollDelayMs());
              });
              addAppEvent(window, "online", () => {
                if (appDisposed) return;
                messagePollErrorStreak = 0;
                if (selected) kickPoll(0);
                scheduleSessionsPoll(0);
              });
              addAppEvent(window, "offline", () => {
                if (appDisposed) return;
                if (selected) kickPoll(messagePollDelayMs());
              });
              addAppEvent(window, "pageshow", () => {
                if (!appDisposed) resumeAnnouncementRuntime({ resetSource: false });
              });
              addAppEvent(window, "online", () => {
                if (!appDisposed) resumeAnnouncementRuntime({ resetSource: true });
              });
              addAppEvent(window, "focus", () => {
                if (!appDisposed) resumeAnnouncementRuntime({ resetSource: false });
              });
	          }
	        })();
      }

      (async function boot() {
        try {
          await api("/api/me");
          renderApp();
        } catch (e) {
          if (e && e.status === 401) {
            renderLogin(renderApp);
            return;
          }
          console.error("boot auth check failed", e);
          const err = document.createElement("pre");
          err.textContent = `error: unable to contact server (${e && e.message ? e.message : "unknown error"})`;
          document.body.innerHTML = "";
          document.body.appendChild(err);
        }
      })();
