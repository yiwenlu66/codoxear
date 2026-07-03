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

      function browserSupportsNativeLiveAudioPlayback() {
        return codoxearVoiceHelpers.browserSupportsNativeLiveAudioPlayback(liveAudio);
      }

      function browserSupportsMseLiveAudioPlayback() {
        return codoxearVoiceHelpers.browserSupportsMseLiveAudioPlayback(window);
      }

      function shouldPreferNativeLiveAudioPlayback() {
        return codoxearVoiceHelpers.shouldPreferNativeLiveAudioPlayback(liveAudio, navigator);
      }

      function browserSupportsLiveAudioPlayback() {
        return codoxearVoiceHelpers.browserSupportsLiveAudioPlayback(liveAudio, window);
      }

      function base64UrlToUint8Array(value) {
        return codoxearVoiceHelpers.base64UrlToUint8Array(value, atob);
      }

      function isMobileNotificationDevice() {
        return codoxearVoiceHelpers.isMobileNotificationDevice(navigator);
      }

      function notificationDeviceClass() {
        return codoxearVoiceHelpers.notificationDeviceClass(navigator);
      }

      const codoxearDom = window.CodoxearDom;
      if (!codoxearDom || typeof codoxearDom.createElement !== "function") throw new Error("Codoxear DOM helpers failed to load");
      const el = (tag, attrs = {}, children = []) => codoxearDom.createElement(tag, attrs, children, defaultButtonTooltip);

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

      let newSessionBackend = "codex";
      let newSessionDefaults = {
        default_backend: "codex",
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
        typeof codoxearFileHelpers.filePickerIdentityHint !== "function" ||
        typeof codoxearFileHelpers.filePickerTitle !== "function" ||
        typeof codoxearFileHelpers.positionAfterInsertedText !== "function" ||
        typeof codoxearFileHelpers.fileEditorDeleteCommandForKey !== "function" ||
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
        typeof codoxearSessionHelpers.sessionNeedsReview !== "function" ||
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

      function sessionNeedsReview(s) {
        return codoxearSessionHelpers.sessionNeedsReview(s);
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

      function renderSessionGroupHeader(entry) {
        const count = Number(entry.count) || 0;
        return el("div", {
          class: "sessionGroupHeader",
          "data-session-group": entry.key,
          role: "heading",
          "aria-level": "2",
          "aria-label": `${entry.label}: ${count} session${count === 1 ? "" : "s"}`,
        }, [
          el("span", { class: "sessionGroupLabel", text: entry.label }),
          el("span", { class: "sessionGroupCount", "aria-hidden": "true", text: String(count) }),
        ]);
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
      if (!codoxearConversationCopy || typeof codoxearConversationCopy.formatConversationForCopy !== "function") throw new Error("Codoxear conversation-copy helpers failed to load");

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

      function filePickerIdentityHint(entry, duplicatePaths, options) {
        return codoxearFileHelpers.filePickerIdentityHint(entry, duplicatePaths, options);
      }

      function filePickerTitle(entry, hint = "") {
        return codoxearFileHelpers.filePickerTitle(entry, hint);
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
      }

	      function renderApp() {
            cleanupActiveApp();
	        const root = $("#root");
	        root.innerHTML = "";

	        const backdrop = el("div", { class: "backdrop", id: "backdrop" });
	        const app = el("div", { class: "app" });
        const sidebar = el("div", { class: "sidebar" });
        const sessionsWrap = el("div", { class: "sessions" });
         const sidebarFooter = el("footer", {}, [
          el("button", { id: "helpBtnSide", type: "button", title: "Help", "aria-label": "Help", html: iconSvg("help") + "Help" }),
          el("button", { id: "settingsBtnSide", type: "button", title: "Settings", "aria-label": "Settings", html: iconSvg("settings") + "Settings" }),
          el("button", { id: "logoutBtnSide", type: "button", title: "Log out", "aria-label": "Log out", html: iconSvg("logout") + "Log out" }),
        ]);
        const main = el("div", { class: "main" });
        const chatWrap = el("div", { class: "chatWrap", id: "chatWrap" });
        const chat = el("div", { class: "chat", id: "chat" });
        const chatInner = el("div", { class: "chatInner", id: "chatInner" });
        const olderWrap = el("div", { class: "olderWrap", id: "olderWrap" });
        const olderBtn = el("button", {
          class: "olderBtn",
          id: "olderBtn",
          type: "button",
          text: "Load older messages",
        });
        const olderErrorText = el("span", { class: "olderErrorText", text: "" });
        const olderRetryBtn = el("button", { class: "olderRetryBtn", type: "button", text: "Retry" });
        const olderError = el("div", { class: "olderError", id: "olderError", role: "status" }, [
          olderErrorText,
          olderRetryBtn,
        ]);
        olderWrap.appendChild(olderBtn);
        olderWrap.appendChild(olderError);
        const bottomSentinel = el("div", { id: "bottomSentinel" });
        const jumpBtn = el("button", {
          class: "jumpBtn",
          id: "jumpBtn",
          title: "Jump to latest",
          "aria-label": "Jump to latest",
          html: iconSvg("down"),
        });
        const chatTimeChip = el("div", { id: "chatTimeChip", class: "chatTimeChip", "aria-hidden": "true" });
        const chatSearchInput = el("input", {
          id: "chatSearchInput",
          class: "chatSearchInput",
          type: "search",
          placeholder: "Search loaded chat",
          "aria-label": "Search loaded chat messages",
          autocomplete: "off",
        });
        const chatSearchPrevBtn = el("button", { id: "chatSearchPrevBtn", class: "icon-btn", type: "button", title: "Previous match", "aria-label": "Previous match", html: iconSvg("up") });
        const chatSearchNextBtn = el("button", { id: "chatSearchNextBtn", class: "icon-btn", type: "button", title: "Next match", "aria-label": "Next match", html: iconSvg("down") });
        const chatSearchCloseBtn = el("button", { id: "chatSearchCloseBtn", class: "icon-btn", type: "button", title: "Close search", "aria-label": "Close search", html: iconSvg("x") });
        const chatSearchStatus = el("span", { id: "chatSearchStatus", class: "chatSearchStatus", text: "Loaded" });
        const chatSearchAllHintEl = el("span", { id: "chatSearchAllHint", class: "chatSearchAllHint", text: "" });
        const chatSearchBar = el("div", { id: "chatSearchBar", class: "chatSearchBar", role: "search", "aria-label": "Search loaded chat messages" }, [
          chatSearchInput,
          chatSearchStatus,
          chatSearchAllHintEl,
          chatSearchPrevBtn,
          chatSearchNextBtn,
          chatSearchCloseBtn,
        ]);
        chatSearchBar.style.display = "none";
        chatInner.appendChild(olderWrap);
        chatInner.appendChild(bottomSentinel);
        chat.appendChild(chatInner);
        chatWrap.appendChild(chat);
        chatWrap.appendChild(jumpBtn);
        chatWrap.appendChild(chatTimeChip);
        chatWrap.appendChild(chatSearchBar);
        const composer = el("div", { class: "composer" });

        let selected = null;
        let pendingHashSessionId = "";
        let pendingHashSessionSelectInFlight = false;
        let liveCursor = null;
        const INIT_PAGE_LIMIT_DESKTOP = 60;
        const INIT_PAGE_LIMIT_MOBILE = 24;
        const OLDER_PAGE_LIMIT = 60;
        const CHAT_DOM_WINDOW = 260;
        const CHAT_DOM_WINDOW_WITH_HISTORY_SLACK = CHAT_DOM_WINDOW + OLDER_PAGE_LIMIT;
        const OLDER_TOP_TRIGGER_PX = 1;
        const OLDER_CANCEL_PX = 48;
        let activeTranscriptState = "pending_bind";
        let activeLogPath = null;
        let activeThreadId = null;
        const CHAT_SEARCH_ALL_DEBOUNCE_MS = 300;
        const CHAT_SEARCH_ALL_COUNT_MAX = 1000;
        let activeMessageCopyRow = null;
        let pendingRecoveryFocusDescriptor = null;
        let renderedAtLiveTail = true;
        let openSessionTailAbortController = null;
        let messagePollAbortController = null;
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
         let openSwipeContent = null;
         let openSwipeSessionId = null;
         let openSwipeTargetX = 0;
         let swipeRefreshDeferred = false;
         let lastSidebarRenderSignature = "";
         let sessionsRefreshInFlight = null;
         let sessionsRefreshQueued = false;
	        let sessionIndex = new Map(); // session_id -> session info
        const sessionTranscriptSlots = new Map();
        const sessionTailCache = new Map();
        let recentCwds = [];
	        let sending = false;
	        let localEchoSeq = 0;
	        const pendingUser = [];
	        let attachedFiles = 0;
		        let autoScroll = true;
			        let backfillToken = 0;
        let backfillState = null;
			    let lastScrollTop = 0;
				    let lastToken = null;
				    let typingRow = null;
        let attachBadgeEl = null;
        let queueBadgeEl = null;
        let editDependencyMenuOpen = false;
        let newSessionCwdMenuOpen = false;
        let newSessionCwdMenuFocus = -1;
        let newSessionModelMenuOpen = false;
        let newSessionModelMenuFocus = -1;
        let newSessionReasoningMenuOpen = false;
        let newSessionResumeMenuOpen = false;
        let newSessionResumeCandidates = [];
        let newSessionResumeSelection = null;
        let newSessionResumeLoadSeq = 0;
        let newSessionResumeLoadTimer = null;
        let newSessionStartBusy = false;
        let newSessionLiteralModelInputValue = "";
        let newSessionLaunchPresetProviderAbsent = false;
        let newSessionCwdInfo = { exists: false, will_create: false, git_repo: false, git_root: "", git_branch: "" };
        let newSessionCwdError = "";
        newSessionBackend = "codex";
        let newSessionProvider = "chatgpt";
        let newSessionFast = false;
        newSessionDefaults = {
          default_backend: "codex",
          backends: {
            codex: legacyCodexLaunchDefaults(),
            pi: emptyPiLaunchDefaults(),
            cc: emptyCcLaunchDefaults(),
          },
        };
        latestSessions = [];
        let tmuxAvailable = false;
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
        let notificationFeedSinceTs = Date.now() / 1000;
        const announcementClientId = (() => {
          const key = "codoxear.announcementClientId";
          const current = storageGetItem(key);
          if (current) return current;
          const next = (window.crypto && crypto.randomUUID ? crypto.randomUUID() : `ann-${Date.now()}-${Math.random().toString(16).slice(2)}`);
          storageSetItem(key, next);
          return next;
        })();
        let announcementHeartbeatTimer = null;
        let liveAudioRetryTimer = null;
        let liveAudioWatchdogTimer = null;
        let notificationState = {
          desktop_supported: false,
          push_supported: false,
          permission: typeof Notification === "undefined" ? "unsupported" : Notification.permission,
          desktop_enabled: false,
          endpoint: "",
          notifications_enabled: false,
          subscriptions: [],
        };
        let liveAudioStarted = false;
        let liveAudioSourceUrl = "";
        let liveAudioHls = null;
        let liveAudioLastProgressTs = 0;
        let liveAudioLastCurrentTime = 0;
        let liveAudioSuspectSinceTs = 0;
        let liveAudioLastRestartTs = 0;
        const LIVE_AUDIO_WATCHDOG_MS = 2500;
        const LIVE_AUDIO_STALL_GRACE_MS = 12000;
        const LIVE_AUDIO_RESTART_THROTTLE_MS = 4000;
        let swRegistration = null;
         const recentEventKeys = [];
         const recentEventKeySet = new Set();
         const RECENT_EVENT_KEYS_MAX = 320;
                 let clickLoadT0 = 0;
                 let clickMetricPending = false;
              let unattendedMenuOpen = false;
              let unattendedMenuToken = 0;
              let unattendedMenuSessionId = null;
              let unattendedReturnFocusEl = null;
              let unattendedCfg = { enabled: false, request: "", cooldown_minutes: 5, remaining_injections: 10 };
              let unattendedNumberDraft = { cooldown_minutes: "5", remaining_injections: "10" };
              let unattendedNumberDirty = { cooldown_minutes: false, remaining_injections: false };
              const unattendedSaveTimers = new Map();
              const unattendedSaveInFlight = new Map();
              const unattendedSavePending = new Map();
              let editSessionId = null;
        let appDisposed = false;
        const appEventCleanups = [];
        function addAppEvent(target, type, handler, options) {
          if (!target || typeof target.addEventListener !== "function") return handler;
          target.addEventListener(type, handler, options);
          appEventCleanups.push(() => target.removeEventListener(type, handler, options));
          return handler;
        }
        function stopMessagePolling() {
          selected = null;
          pollGen += 1;
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
          if (newSessionResumeLoadTimer) clearTimeout(newSessionResumeLoadTimer);
          newSessionResumeLoadTimer = null;
          if (voiceSaveTimer) clearTimeout(voiceSaveTimer);
          voiceSaveTimer = null;
          unattendedSaveTimers.forEach((timer) => clearTimeout(timer));
          unattendedSaveTimers.clear();
          unattendedSavePending.clear();
          unattendedSaveInFlight.clear();
          if (liveAudioRetryTimer) clearTimeout(liveAudioRetryTimer);
          liveAudioRetryTimer = null;
          filePickerSearchState.dispose();
          if (iosViewportGuardTimer) clearTimeout(iosViewportGuardTimer);
          iosViewportGuardTimer = null;
          desktopNotificationTimers.forEach((timer) => clearTimeout(timer));
          desktopNotificationTimers.clear();
          chatSearchAllRuntime.dispose();
          queueUpdateTimers.forEach((timer) => clearTimeout(timer));
          queueUpdateTimers.clear();
          queueMutationLocks.clear();
          queuePendingDeletes.clear();
          olderLoadRuntime.invalidate();
          fileViewerController.abortPendingFileOpenTransport();
          stopAnnouncementHeartbeat();
          stopLiveAudioWatchdog();
          resetLiveAudioState();
          hideUnattendedMenu();
          hideFilePasteDialog();
          hideFileUnsavedDialog("cancel");
          hideSendChoice();
          while (appEventCleanups.length) {
            const cleanup = appEventCleanups.pop();
            try {
              cleanup();
            } catch (_error) {}
          }
          clearApiCache();
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

            const titleLabel = el("div", { id: "threadTitle", text: "No session selected" });
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
				        const statusChip = el("span", { class: "status-chip", id: "statusChip", text: "Idle" });
				        const ctxChip = el("span", { class: "status-chip", id: "ctxChip", text: "" });
		        ctxChip.style.display = "none";
        const interruptBtn = el("button", {
          id: "interruptBtn",
          class: "icon-btn",
          title: "Interrupt (Esc)",
          "aria-label": "Interrupt (Esc)",
          type: "button",
          html: iconSvg("stop"),
        });
        interruptBtn.style.display = "none";
        const toast = el("div", { class: "muted toast", id: "toast", role: "status", "aria-live": "polite" });
			        const toggleSidebarBtn = el("button", {
	          id: "toggleSidebarBtn",
	          class: "icon-btn",
	          title: "Toggle sidebar",
	          "aria-label": "Toggle sidebar",
	          html: iconSvg("menu"),
	        });
        const unattendedBtn = el("button", {
          id: "unattendedBtn",
          class: "icon-btn",
          title: "Unattended mode",
          "aria-label": "Unattended mode",
          "aria-controls": "unattendedMenu",
          "aria-expanded": "false",
          "aria-haspopup": "dialog",
          type: "button",
          html: iconSvg("unattended"),
        });
        unattendedBtn.disabled = true;
        unattendedBtn.classList.toggle("active", false);
        const announceBtn = el("button", {
          id: "announceBtn",
          class: "icon-btn",
          title: "Voice announcements",
          "aria-label": "Voice announcements",
          type: "button",
          html: iconSvg("volume"),
        });
        const notificationBtn = el("button", {
          id: "notificationBtn",
          class: "icon-btn",
          title: "Notifications",
          "aria-label": "Notifications",
          type: "button",
          html: iconSvg("bell"),
        });
        const diagBtn = el("button", {
          id: "diagBtn",
          class: "icon-btn",
          title: "Details",
          "aria-label": "Details",
          type: "button",
          html: iconSvg("info"),
        });
        diagBtn.disabled = true;
        const copyConversationBtn = el("button", {
          id: "copyConversationBtn",
          class: "icon-btn",
          title: "Copy conversation",
          "aria-label": "Copy conversation",
          type: "button",
          html: iconSvg("copy"),
        });
        copyConversationBtn.disabled = true;
        const prevUserBtn = el("button", {
          id: "prevUserBtn",
          class: "icon-btn",
          title: "Previous user message",
          "aria-label": "Previous user message",
          type: "button",
          html: iconSvg("up"),
        });
        prevUserBtn.disabled = true;
        const nextUserBtn = el("button", {
          id: "nextUserBtn",
          class: "icon-btn",
          title: "Next user message",
          "aria-label": "Next user message",
          type: "button",
          html: iconSvg("down"),
        });
        nextUserBtn.disabled = true;
        const chatSearchBtn = el("button", {
          id: "chatSearchBtn",
          class: "icon-btn",
          title: "Search loaded messages",
          "aria-label": "Search loaded messages",
          type: "button",
          html: iconSvg("search"),
        });
        chatSearchBtn.disabled = true;
        const fileBtn = el("button", {
          id: "fileBtn",
          class: "icon-btn",
          title: "View file",
          "aria-label": "View file",
          type: "button",
          html: iconSvg("file"),
        });
        fileBtn.disabled = true;
        const unattendedMenu = el("div", { id: "unattendedMenu", class: "unattendedMenu", role: "dialog", "aria-label": "Unattended mode settings" }, [
          el("div", { class: "row" }, [
            el("label", {}, [
              el("input", { type: "checkbox", id: "unattendedEnabled" }),
              el("span", { text: "Unattended mode" }),
			            ]),
			          ]),
			          el("div", { class: "unattendedGrid" }, [
			            el("div", {}, [
			              el("div", { class: "label", text: "Cooldown time (minutes)" }),
			              el("input", { id: "unattendedCooldownMinutes", type: "number", min: "1", step: "1", inputmode: "numeric", "aria-label": "Unattended cooldown time in minutes" }),
			            ]),
			            el("div", {}, [
			              el("div", { class: "label", text: "Number of injections" }),
			              el("input", { id: "unattendedRemainingInjections", type: "number", min: "0", step: "1", inputmode: "numeric", "aria-label": "Unattended remaining injections" }),
			            ]),
			          ]),
			          el("div", { class: "label", text: "Additional request to append (optional; per session)" }),
			          el("textarea", { id: "unattendedRequest", "aria-label": "Additional request for unattended prompt" }),
			        ]);
        const liveAudio = el("audio", { id: "liveAudio", preload: "none", playsinline: "true" });
        liveAudio.style.display = "none";

        const topMeta = el("div", { class: "topMeta" }, [ctxChip]);
        const titleRow = el("div", { class: "titleRow" }, [titleLabel, topMeta]);
        const titleWrap = el("div", { class: "titleWrap" }, [titleRow]);
        const sessionContextBar = el("div", { class: "sessionContextBar", id: "sessionContextBar", "aria-label": "Session utilities" }, [
          fileBtn,
          copyConversationBtn,
          diagBtn,
          unattendedBtn,
        ]);
        const chatNavRail = el("div", { class: "chatNavRail", id: "chatNavRail", "aria-label": "Loaded chat navigation" }, [
          chatSearchBtn,
          prevUserBtn,
          nextUserBtn,
        ]);
        chatWrap.appendChild(chatNavRail);
        const topbar = el("div", { class: "topbar" }, [
          el("div", { class: "pill" }, [toggleSidebarBtn, titleWrap]),
          el("div", { class: "actions topActions" }, [
            interruptBtn,
          ]),
        ]);

        const form = el("form", {}, [
          el("button", {
            class: "icon-btn",
            id: "attachBtn",
            type: "button",
            title: "Attach file",
            "aria-label": "Attach file",
            html: iconSvg("paperclip"),
          }),
          el("div", { class: "inputWrap" }, [
            el("textarea", { id: "msg", placeholder: "", "aria-label": "Enter your instructions here" }),
            el("div", { class: "ph", id: "msgPh", text: "Enter your instructions here" }),
          ]),
          el("input", { id: "imgInput", type: "file", style: "display:none" }),
          el("button", { class: "icon-btn", id: "queueBtn", type: "button", title: "Queued messages", "aria-label": "Queued messages", html: iconSvg("queue") }),
          el("button", { class: "icon-btn composerStopBtn", id: "composerStopBtn", type: "button", title: "Stop current response", "aria-label": "Stop current response", html: iconSvg("stop") }),
          el("button", { class: "icon-btn primary", id: "sendBtn", type: "submit", title: "Send", "aria-label": "Send", html: iconSvg("send") }),
        ]);
        composer.appendChild(form);

        sidebar.appendChild(
          el("header", {}, [
            el("div", { class: "title", html: `<img class="sidebarLogo" src="${resolveAppUrl(versionedShellAssetPath("/static/codoxear-icon.png"))}" alt="" />Codoxear` }),
            el("div", { class: "actions" }, [
              el("button", { id: "newBtn", class: "icon-btn", title: "New session", "aria-label": "New session", html: iconSvg("plus") }),
              notificationBtn,
              announceBtn,
            ]),
          ])
        );
        sidebar.appendChild(sessionsWrap);
        sidebar.appendChild(sidebarFooter);
        main.appendChild(topbar);
        main.appendChild(sessionContextBar);
        main.appendChild(toast);
        main.appendChild(chatWrap);
        main.appendChild(composer);
        app.appendChild(sidebar);
        app.appendChild(main);
        app.appendChild(backdrop);
        root.appendChild(app);
        root.appendChild(unattendedMenu);
        root.appendChild(liveAudio);

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

        const queueBackdrop = el("div", { class: "modalBackdrop", id: "queueBackdrop" });
        const queueCloseBtn = el("button", {
          id: "queueCloseBtn",
          class: "icon-btn",
          title: "Close",
          "aria-label": "Close",
          type: "button",
          html: iconSvg("x"),
        });
        let queueReturnFocusEl = null;
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
  <li>The dot on the title row shows state: <b>blue</b> = busy, <b>gray</b> = idle, <b>orange</b> = snoozed or blocked.</li>
  <li>The metadata line shows the agent-backend icon first, then the session-type icon, then the reasoning marker (<b>X/H/M/L</b>) when available, followed by recency, folder, and branch.</li>
  <li>Click the conversation title to rename or reprioritize it. <b>Details</b> in the session utilities bar shows the exact backend, provider, model, reasoning level, queue state, and token usage.</li>
</ul>
<div class="muted">New session</div>
<ul class="md">
  <li><b>New session</b> can start fresh or resume a matching conversation for the currently selected backend in the current working directory.</li>
  <li>The backend tabs choose between the supported agent backends. Right now that is <b>Codex</b>, <b>Pi</b>, and <b>Claude</b>.</li>
  <li>You can choose working directory, a combined provider/model pair, reasoning level, and whether the session should start in tmux. If the directory is a Git repo, you can also start in a new worktree branch.</li>
  <li>Codoxear remembers the last backend you used and the last provider/model pair for each backend.</li>
</ul>
<div class="muted">Messages and queue</div>
<ul class="md">
  <li>If the selected session is idle, <b>Send</b> submits immediately. If it is busy, choose <b>Send after current</b> to queue the prompt.</li>
  <li>The queue is stored per session and drains automatically when that session becomes idle. Use <b>Queued messages</b> to review or edit queued prompts.</li>
  <li><b>Load older messages</b> fetches more scrollback. <b>Jump to latest</b> returns to the newest turn when you are reading history.</li>
  <li>Use <b>/</b> to search the loaded chat; Previous/Next can load an older matching window when the transcript count shows more matches.</li>
  <li>Use <b>Alt+↑</b>/<b>Alt+↓</b> to jump between loaded user messages. Use <b>Alt+Shift+↑</b>/<b>Alt+Shift+↓</b> to move the active per-message copy control across all loaded messages.</li>
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
        let diagReturnFocusEl = null;
        let diagCopyText = "";
        let diagNewLikeSession = null;
        diagNewLikeBtn.disabled = true;
        diagCopyBtn.disabled = true;
        const diagStatus = el("div", { class: "muted", id: "diagStatus", text: "" });
        const diagContent = el("div", { class: "detailsGrid", id: "diagContent" });
        const diagViewer = el("div", { class: "diagViewer", id: "diagViewer", role: "dialog", "aria-modal": "true", "aria-label": "Details" }, [
          el("div", { class: "queueHeader" }, [
            el("div", { class: "title", text: "Details" }),
            el("div", { class: "actions" }, [diagNewLikeBtn, diagCopyBtn, diagCloseBtn]),
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
        let voiceSettingsReturnFocusEl = null;
        const voiceBaseUrlInput = el("input", { id: "voiceBaseUrlInput", type: "text", autocomplete: "off", spellcheck: "false" });
        const voiceApiKeyInput = el("input", { id: "voiceApiKeyInput", type: "password", autocomplete: "off", spellcheck: "false" });
        const voiceClearApiKeyToggle = el("input", { id: "voiceClearApiKeyToggle", type: "checkbox" });
        const narrationSettingToggle = el("input", { id: "narrationSettingToggle", type: "checkbox" });
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
          if (unattendedMenuOpen) hideUnattendedMenu();
          if (closeSearch && loadedChatSearchSnapshot().open) closeChatSearch();
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

        const codoxearClipboard = window.CodoxearClipboard;
        if (!codoxearClipboard || typeof codoxearClipboard.copyTextViaSelection !== "function" || typeof codoxearClipboard.copyToClipboard !== "function")
          throw new Error("Codoxear clipboard helpers failed to load");

        function setToast(text) {
          toast.textContent = text || "";
          if (!text) return;
          setTimeout(() => {
            if (toast.textContent === text) toast.textContent = "";
          }, 2200);
        }

        function copyTextViaSelection(text) {
          return codoxearClipboard.copyTextViaSelection(text);
        }

        async function copyToClipboard(text) {
          return codoxearClipboard.copyToClipboard(text);
        }

        function formatConversationForCopy(events) {
          return codoxearConversationCopy.formatConversationForCopy(events);
        }

        async function copyConversation() {
          if (!selected) return;
          const sid = selected;
          copyConversationBtn.disabled = true;
          try {
            const data = await api(`/api/sessions/${sid}/messages/export`);
            if (selected !== sid) return;
            const events = Array.isArray(data && data.events) ? data.events : [];
            const text = formatConversationForCopy(events);
            if (!text) {
              setToast("No conversation to copy");
              return;
            }
            await copyToClipboard(text);
            setToast(`Copied ${events.length} messages`);
          } catch (err) {
            setToast(`copy failed: ${err && err.message ? err.message : "unknown error"}`);
          } finally {
            copyConversationBtn.disabled = !selected;
          }
        }

        copyConversationBtn.onclick = (e) => {
          e.preventDefault();
          e.stopPropagation();
          void copyConversation();
        };

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
	            return;
	          }
	          const p = Number.isFinite(pct) ? Math.max(0, Math.min(100, Math.round(pct))) : null;
	          const maxInput = Number(tok.max_input_tokens);
	          const reserved = Number(tok.reserved_tokens);
	          const effectiveMaxInput = Number.isFinite(maxInput) && maxInput >= 0 ? maxInput : ctx;
	          const effectiveReserved = Number.isFinite(reserved) && reserved >= 0 ? reserved : Math.max(ctx - effectiveMaxInput, 0);
	          lastToken = { ctx, used, pct: p, remaining: Math.max(effectiveMaxInput - used, 0), maxInput: effectiveMaxInput, reserved: effectiveReserved, asOf: tok.as_of || "" };
	          ctxChip.style.display = "inline-flex";
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
          autoScroll = true;
          sending = false;
          recentEventKeys.length = 0;
          recentEventKeySet.clear();
          liveCursor = null;
          renderedAtLiveTail = true;
          olderLoadRuntime.resetAutoTrigger();
              clickMetricPending = false;
          clearTranscriptDom();
              setOlderState({ hasMore: false, isLoading: false });
	          typingRow = null;
	          jumpBtn.style.display = "none";
              updateChatNavButtons();
              if (loadedChatSearchSnapshot().open) closeChatSearch();
	          backfillState = null;
	          backfillToken += 1;
	          lastScrollTop = 0;
	          chat.scrollTop = 0;
              syncVisibleTimeIndicator();
	        }

        function clearTranscriptDom() {
          chatInner.innerHTML = "";
          chatInner.appendChild(olderWrap);
          chatInner.appendChild(bottomSentinel);
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

        function loadedCopyJumpTarget(rows, activeRow, direction, threshold) {
          return codoxearMessageRows.loadedCopyJumpTarget(rows, activeRow, direction, threshold);
        }

        function applyChatSearchMarks(matches, currentRow) {
          codoxearMessageRows.applyChatSearchMarks(matches, currentRow);
        }

        function firstVisibleMessageRow() {
          return codoxearMessageRows.firstVisibleMessageRow(renderedMessageRows(), chat.scrollTop + 1);
        }

        function trimRenderedRowTargets(rows, fromTop, maxRows) {
          return codoxearMessageRows.trimRenderedRowTargets(rows, fromTop, maxRows, CHAT_DOM_WINDOW);
        }

        function trimRowsBeforeViewportTargets(rows, maxRows, viewportTop) {
          return codoxearMessageRows.trimRowsBeforeViewportTargets(rows, maxRows, CHAT_DOM_WINDOW, viewportTop);
        }

        function syncMessageCopyTabStops() {
          const buttons = Array.from(chatInner.querySelectorAll(".msg-copy-btn"));
          let activeBtn = messageCopyButtonForRow(activeMessageCopyRow);
          if (!activeBtn || !activeBtn.isConnected) {
            activeBtn = null;
            const rows = renderedMessageRows();
            for (let i = rows.length - 1; i >= 0; i -= 1) {
              const candidate = messageCopyButtonForRow(rows[i]);
              if (candidate) {
                activeBtn = candidate;
                break;
              }
            }
          }
          activeMessageCopyRow = activeBtn ? activeBtn.closest(".msg-row") : null;
          for (const btn of buttons) {
            const active = btn === activeBtn;
            btn.tabIndex = active ? 0 : -1;
            btn.disabled = !active;
            if (active) btn.removeAttribute("aria-hidden");
            else btn.setAttribute("aria-hidden", "true");
          }
        }

        function setActiveMessageCopyRow(row, { focusCopy = false } = {}) {
          activeMessageCopyRow = row && row.isConnected && messageCopyButtonForRow(row) ? row : null;
          syncMessageCopyTabStops();
          if (focusCopy && activeMessageCopyRow) {
            const btn = messageCopyButtonForRow(activeMessageCopyRow);
            if (btn && btn.tabIndex >= 0) btn.focus({ preventScroll: true });
          }
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

        function updateChatNavButtons() {
          const enabled = Boolean(selected && loadedUserMessageRows().length);
          prevUserBtn.disabled = !enabled;
          nextUserBtn.disabled = !enabled;
        }

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

        function jumpToLoadedUserMessage(direction) {
          const rows = loadedUserMessageRows();
          updateChatNavButtons();
          if (!rows.length) {
            setToast("No loaded user messages");
            return;
          }
          const result = loadedUserJumpTarget(rows, direction, chat.scrollTop + 24);
          if (!result.target) {
            setToast(result.reason === "first" ? "At first loaded user message" : "At last loaded user message");
            return;
          }
          const target = result.target;
          target.scrollIntoView({ block: "start", behavior: prefersReducedMotion() ? "auto" : "smooth" });
          pulseNavigatedRow(target);
        }

        function jumpToLoadedMessage(direction) {
          const rows = loadedCopyMessageRows();
          if (!rows.length) {
            setToast("No loaded messages");
            return;
          }
          const result = loadedCopyJumpTarget(rows, activeMessageCopyRow, direction, chat.scrollTop + 24);
          if (!result.target) {
            setToast(result.reason === "first" ? "At first loaded message" : "At last loaded message");
            return;
          }
          const target = result.target;
          target.scrollIntoView({ block: "start", behavior: prefersReducedMotion() ? "auto" : "smooth" });
          pulseNavigatedRow(target);
        }

        prevUserBtn.onclick = (e) => {
          e.preventDefault();
          e.stopPropagation();
          jumpToLoadedUserMessage(-1);
        };
        nextUserBtn.onclick = (e) => {
          e.preventDefault();
          e.stopPropagation();
          jumpToLoadedUserMessage(1);
        };

        function clearChatSearchMarks() {
          codoxearMessageRows.clearChatSearchMarks(renderedMessageRows());
        }

        function compactChatSearchSnippet(text, query, limit = 96) {
          return codoxearDisplay.compactChatSearchSnippet(text, query, limit);
        }

        function chatSearchTranscriptHint(match, query) {
          return codoxearDisplay.chatSearchTranscriptHint(match, query);
        }

        function currentChatSearchState() {
          return loadedChatSearchSnapshot();
        }

        function currentChatSearchQuery() {
          return currentChatSearchState().query;
        }

        function currentChatSearchMatches() {
          return currentChatSearchState().matches;
        }

        function syncChatSearchStatus() {
          const searchState = currentChatSearchState();
          const total = searchState.matches.length;
          const allState = chatSearchAllSnapshot();
          const allSuffix = searchState.query
            ? searchState.loadingOlder
              ? " · loading older"
              : Number.isFinite(allState.count)
                ? ` · ${allState.count}${allState.truncated ? "+" : ""} all`
                : ""
            : "";
          const canLoadOlderMatch = Boolean(
            searchState.query &&
              Number.isFinite(allState.count) &&
              (allState.truncated || allState.count > total) &&
              hasOlderMessages() &&
              !searchState.loadingOlder &&
              !isLoadingOlderMessages()
          );
          const showAllHint = Boolean(searchState.query && !searchState.loadingOlder && Number.isFinite(allState.count) && (allState.truncated || allState.count > total) && allState.hint);
          chatSearchStatus.textContent = searchState.query ? `${total ? searchState.index + 1 : 0}/${total} loaded${allSuffix}` : "Loaded";
          chatSearchAllHintEl.textContent = showAllHint ? `all: ${allState.hint}` : "";
          chatSearchAllHintEl.title = showAllHint ? allState.hint : "";
          chatSearchAllHintEl.style.display = showAllHint ? "" : "none";
          chatSearchPrevBtn.disabled = total <= 0;
          chatSearchNextBtn.disabled = total <= 0 && !canLoadOlderMatch;
        }

        function resetAllChatSearchCount() {
          chatSearchAllRuntime.reset();
        }

        function scheduleAllChatSearchCount(query) {
          const cleanQuery = String(query || "").trim();
          if (!selected || !cleanQuery) {
            resetAllChatSearchCount();
            syncChatSearchStatus();
            return;
          }
          chatSearchAllRuntime.schedule(cleanQuery, (scheduledQuery) => {
            void refreshAllChatSearchCount(scheduledQuery);
          });
          syncChatSearchStatus();
        }

        async function refreshAllChatSearchCount(query) {
          const cleanQuery = String(query || "").trim();
          if (!selected || !cleanQuery) {
            resetAllChatSearchCount();
            syncChatSearchStatus();
            return;
          }
          const sid = selected;
          const request = chatSearchAllRuntime.beginRequest();
          try {
            const data = await api(`/api/sessions/${sid}/messages/search?q=${encodeURIComponent(cleanQuery)}&limit=1&text_max=96&count_max=${CHAT_SEARCH_ALL_COUNT_MAX}`, { signal: request.signal });
            if (selected !== sid || !chatSearchAllRuntime.isCurrent(request) || currentChatSearchQuery() !== cleanQuery.toLowerCase()) return;
            const firstMatch = Array.isArray(data.matches) && data.matches.length ? data.matches[0] : null;
            chatSearchAllRuntime.completeRequest(request, {
              count: data.match_count,
              truncated: data.match_count_truncated,
              hint: chatSearchTranscriptHint(firstMatch, cleanQuery),
            });
            syncChatSearchStatus();
          } catch (e) {
            if (e && e.name === "AbortError") return;
            if (selected !== sid || !chatSearchAllRuntime.isCurrent(request)) return;
            chatSearchAllRuntime.failRequest(request);
            syncChatSearchStatus();
          } finally {
            chatSearchAllRuntime.finishRequest(request);
          }
        }

        function focusChatSearchMatch(index, { jump = true } = {}) {
          clearChatSearchMarks();
          const result = loadedChatSearchRuntime.focusIndex(index);
          if (!result.row) {
            syncChatSearchStatus();
            return;
          }
          applyChatSearchMarks(result.matches, result.row);
          syncChatSearchStatus();
          if (jump) {
            result.row.scrollIntoView({ block: "center", behavior: prefersReducedMotion() ? "auto" : "smooth" });
            pulseNavigatedRow(result.row);
          }
        }

        function ensureChatSearchTargetRow(historyCursor) {
          const targetCursor = String(historyCursor || "").trim();
          if (!targetCursor) return -1;
          const target = renderedMessageRows().find((row) => row.dataset.historyCursor === targetCursor);
          if (!target) return -1;
          return loadedChatSearchRuntime.ensureTargetRow(target, currentChatSearchQuery(), compareRowsInDomOrder);
        }

        function refreshLoadedChatSearch({ jump = false, preserveCurrent = true, refreshAllCount = true } = {}) {
          const query = loadedChatSearchRuntime.setQuery(chatSearchInput.value || "");
          clearChatSearchMarks();
          if (!query) {
            loadedChatSearchRuntime.clearMatches();
            resetAllChatSearchCount();
            syncChatSearchStatus();
            return;
          }
          if (refreshAllCount) scheduleAllChatSearchCount(query);
          const matches = renderedMessageRows().filter((row) => row.dataset.searchForcedQuery === query || rowSearchText(row).toLowerCase().includes(query));
          const nextState = loadedChatSearchRuntime.setMatches(matches, { preserveCurrent });
          if (!nextState.matches.length) {
            syncChatSearchStatus();
            return;
          }
          focusChatSearchMatch(nextState.index, { jump });
        }

        function openChatSearch() {
          if (!selected) return;
          loadedChatSearchRuntime.setOpen(true);
          chatSearchBar.style.display = "flex";
          syncVisibleTimeIndicator();
          refreshLoadedChatSearch({ jump: false, preserveCurrent: true });
          chatSearchInput.focus({ preventScroll: true });
          chatSearchInput.select();
        }

        function closeChatSearch() {
          loadedChatSearchRuntime.setOpen(false);
          chatSearchBar.style.display = "none";
          clearChatSearchMarks();
          resetAllChatSearchCount();
          loadedChatSearchRuntime.setLoadingOlder(false);
          syncVisibleTimeIndicator();
        }

        async function loadOlderUntilChatSearchMatch({ boundaryMatch = null, focus = "first" } = {}) {
          const startState = currentChatSearchState();
          if (!selected || !startState.query || startState.loadingOlder) return false;
          const sid = selected;
          const gen = pollGen;
          const query = startState.query;
          const maxPages = 12;
          loadedChatSearchRuntime.setLoadingOlder(true);
          syncChatSearchStatus();
          try {
            for (let i = 0; i < maxPages; i += 1) {
              if (selected !== sid || pollGen !== gen || currentChatSearchQuery() !== query || !hasOlderMessages()) return false;
              const loaded = await loadOlderMessages({ auto: false, cancelOnScroll: false });
              if (selected !== sid || pollGen !== gen || currentChatSearchQuery() !== query) return false;
              refreshLoadedChatSearch({ jump: false, preserveCurrent: false });
              const matches = currentChatSearchMatches();
              if (boundaryMatch) {
                const boundaryIndex = matches.indexOf(boundaryMatch);
                if (boundaryIndex > 0) {
                  focusChatSearchMatch(focus === "last" ? boundaryIndex - 1 : 0, { jump: true });
                  return true;
                }
              } else if (matches.length) {
                focusChatSearchMatch(0, { jump: true });
                return true;
              }
              if (!loaded || !hasOlderMessages()) return false;
            }
            return false;
          } finally {
            loadedChatSearchRuntime.setLoadingOlder(false);
            syncChatSearchStatus();
          }
        }

        async function stepChatSearch(delta) {
          if (!currentChatSearchState().open) openChatSearch();
          refreshLoadedChatSearch({ jump: false, preserveCurrent: true, refreshAllCount: false });
          let state = currentChatSearchState();
          if (!state.matches.length) {
            const allState = chatSearchAllSnapshot();
            if (state.query && Number.isFinite(allState.count) && allState.count > 0 && hasOlderMessages()) {
              const jumped = await loadNearestOlderChatSearchWindow();
              if (jumped) return;
              const found = await loadOlderUntilChatSearchMatch();
              if (found) return;
              setToast("No loaded matches after loading older messages");
              return;
            }
            setToast(state.query ? "No loaded matches" : "Enter a loaded-chat search");
            return;
          }
          const startIndex = state.index;
          const allState = chatSearchAllSnapshot();
          const unloadedTranscriptMatches = Number.isFinite(allState.count) ? (allState.truncated || allState.count > state.matches.length) : true;
          const canLoadOlderMatches = Boolean(state.query && unloadedTranscriptMatches && hasOlderMessages());
          const atForwardWrap = delta > 0 && startIndex >= state.matches.length - 1;
          const atBackwardWrap = delta < 0 && startIndex <= 0;
          if (canLoadOlderMatches && (atForwardWrap || atBackwardWrap)) {
            const jumped = await loadNearestOlderChatSearchWindow();
            if (jumped) return;
            state = currentChatSearchState();
            const found = await loadOlderUntilChatSearchMatch({
              boundaryMatch: state.matches[0],
              focus: atBackwardWrap ? "last" : "first",
            });
            if (found) return;
            focusChatSearchMatch(startIndex + delta, { jump: true });
            return;
          }
          focusChatSearchMatch(startIndex + delta, { jump: true });
        }

        chatSearchBtn.onclick = (e) => {
          e.preventDefault();
          e.stopPropagation();
          if (loadedChatSearchSnapshot().open) closeChatSearch();
          else openChatSearch();
        };
        chatSearchInput.oninput = () => refreshLoadedChatSearch({ jump: true, preserveCurrent: false });
        chatSearchInput.onkeydown = (e) => {
          if (e.key === "Escape") {
            e.preventDefault();
            closeChatSearch();
          } else if (e.key === "Enter") {
            e.preventDefault();
            void stepChatSearch(e.shiftKey ? -1 : 1);
          }
        };
        chatSearchPrevBtn.onclick = (e) => {
          e.preventDefault();
          e.stopPropagation();
          void stepChatSearch(-1);
        };
        chatSearchNextBtn.onclick = (e) => {
          e.preventDefault();
          e.stopPropagation();
          void stepChatSearch(1);
        };
        chatSearchCloseBtn.onclick = (e) => {
          e.preventDefault();
          e.stopPropagation();
          closeChatSearch();
        };

        function chatNavigationShortcutBlocked(target) {
          if (!selected) return true;
          if (isTextEntryElement(target)) return true;
          if (document.body.classList.contains("sidebar-open")) return true;
          return modalIsolationTargets.some(isModalTargetOpen);
        }

        function chatSearchShortcutBlocked(target) {
          return chatNavigationShortcutBlocked(target);
        }

        addAppEvent(document, "keydown", (e) => {
          if (e.defaultPrevented) return;
          if (e.key === "/" && !e.ctrlKey && !e.metaKey && !e.altKey) {
            if (chatSearchShortcutBlocked(e.target)) return;
            e.preventDefault();
            openChatSearch();
            return;
          }
          if (e.altKey && e.shiftKey && !e.ctrlKey && !e.metaKey && (e.key === "ArrowUp" || e.key === "ArrowDown")) {
            if (chatNavigationShortcutBlocked(e.target)) return;
            e.preventDefault();
            jumpToLoadedMessage(e.key === "ArrowUp" ? -1 : 1);
            return;
          }
          if (e.altKey && !e.shiftKey && !e.ctrlKey && !e.metaKey && (e.key === "ArrowUp" || e.key === "ArrowDown")) {
            if (chatNavigationShortcutBlocked(e.target)) return;
            e.preventDefault();
            jumpToLoadedUserMessage(e.key === "ArrowUp" ? -1 : 1);
          }
        });

        function oldestRenderedHistoryCursor() {
          return codoxearMessageRows.oldestRenderedHistoryCursor(renderedMessageRows());
        }

        function clearRenderedTranscriptRange() {
          setOlderState({ hasMore: false, isLoading: false });
          renderedAtLiveTail = true;
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
          typeof codoxearTranscript.transcriptKey !== "function" ||
          typeof codoxearTranscript.transcriptSnapshotFromData !== "function" ||
          typeof codoxearTranscript.transcriptIdentityFromData !== "function" ||
          typeof codoxearTranscript.tailCacheMatchesSession !== "function" ||
          typeof codoxearTranscript.rememberTailSnapshot !== "function" ||
          typeof codoxearTranscript.appendTailSnapshotEvents !== "function" ||
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

        const loadedChatSearchRuntime = codoxearTranscript.createLoadedChatSearchRuntime();

        function loadedChatSearchSnapshot() {
          return loadedChatSearchRuntime.snapshot();
        }

        const chatSearchAllRuntime = codoxearTranscript.createChatSearchAllRuntime({
          setTimeout: window.setTimeout.bind(window),
          clearTimeout: window.clearTimeout.bind(window),
          AbortControllerCtor: AbortController,
          debounceMs: CHAT_SEARCH_ALL_DEBOUNCE_MS,
        });

        function chatSearchAllSnapshot() {
          return chatSearchAllRuntime.snapshot();
        }

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
          if (!sessionId) return { state: "pending_bind", threadId: null, logPath: null, key: null, epoch: 0, ignoredKey: null };
          const current = sessionTranscriptSlots.get(sessionId);
          if (current) return current;
          return { state: "pending_bind", threadId: null, logPath: null, key: null, epoch: 0, ignoredKey: null };
        }

        function syncActiveTranscriptSlot(sessionId) {
          const slot = getSessionTranscriptSlot(sessionId);
          activeTranscriptState = slot.state;
          activeThreadId = slot.threadId;
          activeLogPath = slot.logPath;
          return slot;
        }

        function dropPendingUserRows(sessionId, predicate = null) {
          if (!sessionId || !pendingUser.length) return;
          const kept = [];
          for (const item of pendingUser) {
            const match = !!(item && item.sessionId === sessionId && (!predicate || predicate(item)));
            if (!match) {
              kept.push(item);
              continue;
            }
            if (selected === sessionId && item.id) {
              const pendingEl = chatInner.querySelector(`.msg.user[data-local-id="${item.id}"]`);
              const row = pendingEl ? pendingEl.closest(".msg-row") : null;
              if (row) row.remove();
            }
          }
          pendingUser.length = 0;
          pendingUser.push(...kept);
        }

        function updateSessionTranscriptSlot(sessionId, data) {
          const prev = getSessionTranscriptSlot(sessionId);
          const snap = transcriptSnapshotFromData(data);
          if (prev.state === "pending_bind" && prev.ignoredKey && snap.state === "bound" && snap.key === prev.ignoredKey) {
            const next = { ...prev };
            if (sessionId) sessionTranscriptSlots.set(sessionId, next);
            if (selected === sessionId) syncActiveTranscriptSlot(sessionId);
            return { previous: prev, current: next, resetPending: false, ignoredStaleBound: true };
          }
          let epoch = prev.epoch;
          let resetPending = false;
          let ignoredKey = null;
          if (snap.state === "pending_bind") {
            if (prev.state === "bound") {
              epoch += 1;
              resetPending = true;
              ignoredKey = prev.key;
            } else {
              ignoredKey = prev.ignoredKey || null;
            }
          } else if (prev.state === "bound" && prev.key !== snap.key) {
            epoch += 1;
            resetPending = true;
          }
          const next = { ...snap, epoch, ignoredKey };
          if (sessionId) sessionTranscriptSlots.set(sessionId, next);
          if (resetPending) dropPendingUserRows(sessionId, () => true);
          if (selected === sessionId) syncActiveTranscriptSlot(sessionId);
          return { previous: prev, current: next, resetPending, ignoredStaleBound: false };
        }

        function beginTranscriptRenewal(sessionId) {
          if (!sessionId) return;
          const prev = getSessionTranscriptSlot(sessionId);
          const next = {
            state: "pending_bind",
            threadId: null,
            logPath: null,
            key: null,
            epoch: Number(prev.epoch || 0) + 1,
            ignoredKey: prev.state === "bound" ? prev.key : prev.ignoredKey || null,
          };
          sessionTranscriptSlots.set(sessionId, next);
          dropPendingUserRows(sessionId, () => true);
          if (selected === sessionId) syncActiveTranscriptSlot(sessionId);
        }

        function tailCacheMatchesSession(cache, session) {
          return codoxearTranscript.tailCacheMatchesSession(cache, session);
        }

        function rememberTailSnapshot(sessionId, session, data) {
          return codoxearTranscript.rememberTailSnapshot(sessionTailCache, sessionId, session, data, Math.max(INIT_PAGE_LIMIT_DESKTOP, INIT_PAGE_LIMIT_MOBILE));
        }

        function appendTailSnapshotEvents(sessionId, events, { session = null, identityData = null, liveCursor: nextLiveCursor, busy, queueLen, token } = {}) {
          return codoxearTranscript.appendTailSnapshotEvents(sessionTailCache, sessionIndex, sessionId, events, {
            session,
            identityData,
            liveCursor: nextLiveCursor,
            busy,
            queueLen,
            token,
            maxEvents: Math.max(INIT_PAGE_LIMIT_DESKTOP, INIT_PAGE_LIMIT_MOBILE),
          });
        }

        function restorePendingUserRowsForSession(sessionId) {
          if (!sessionId) return;
          const slot = getSessionTranscriptSlot(sessionId);
          const items = pendingUser
            .filter((item) => item && item.sessionId === sessionId && Number(item.epoch || 0) === Number(slot.epoch || 0))
            .sort((a, b) => Number(a.t0 || 0) - Number(b.t0 || 0));
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

          sessionTailCache.delete(sessionId);
          liveCursor = null;
          clearRenderedTranscriptRange();
          setAttachCount(0);
          invalidateOlderLoad();
          recentEventKeys.length = 0;
          recentEventKeySet.clear();
          backfillToken += 1;
          backfillState = null;
          autoScroll = true;
          clearTranscriptDom();
          if (slotChange.current.state === "pending_bind") {
            renderPendingTranscriptSlot(sessionId);
          } else {
            setOlderState({ hasMore: false, isLoading: false });
            syncJumpButton();
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

	        function ensureTypingRow() {
	          if (typingRow && typingRow.isConnected) return typingRow;
	          const row = el("div", { class: "msg-row assistant typing-row" });
	          row.dataset.role = "assistant";
	          const bubble = el("div", { class: "msg assistant typing" });
	          const dots = el("div", { class: "typingDots", "aria-label": "Running", title: "Running" }, [
	            el("span", { class: "typingDot" }),
	            el("span", { class: "typingDot" }),
	            el("span", { class: "typingDot" }),
	          ]);
	          bubble.appendChild(dots);
	          row.appendChild(bubble);
	          typingRow = row;
	          return row;
	        }

	        function setTyping(show) {
	          if (!show) {
	            if (typingRow && typingRow.isConnected) typingRow.remove();
	            return;
	          }
	          const row = ensureTypingRow();
	          if (!row.isConnected) {
	            chatInner.insertBefore(row, bottomSentinel);
	          } else if (row.nextSibling !== bottomSentinel) {
	            chatInner.insertBefore(row, bottomSentinel);
	          }
	          if (autoScroll) requestAnimationFrame(() => scrollToBottom());
	        }

        function isNearBottom() {
          const thresholdPx = 80;
          return chat.scrollHeight - (chat.scrollTop + chat.clientHeight) <= thresholdPx;
        }

        function syncVisibleTimeIndicator() {
          if (!selected || loadedChatSearchSnapshot().open || (renderedAtLiveTail && (autoScroll || isNearBottom()))) {
            chatTimeChip.style.display = "none";
            chatTimeChip.textContent = "";
            return;
          }
          const row = firstVisibleMessageRow();
          const ts = row ? Number(row.dataset.ts || "0") : 0;
          if (!Number.isFinite(ts) || ts <= 0) {
            chatTimeChip.style.display = "none";
            chatTimeChip.textContent = "";
            return;
          }
          const d = new Date(ts * 1000);
          chatTimeChip.textContent = `${dayLabel(d)} · ${time24(d)}`;
          chatTimeChip.style.display = "inline-flex";
        }

        function syncJumpButton() {
          jumpBtn.style.display = renderedAtLiveTail && (autoScroll || isNearBottom()) ? "none" : "inline-flex";
          syncVisibleTimeIndicator();
        }

        function scrollToBottom() {
          // Avoid scrollIntoView() on mobile Safari, which can scroll the whole page when the
          // on-screen keyboard opens/closes.
          chat.scrollTop = chat.scrollHeight;
          lastScrollTop = chat.scrollTop;
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
          const oldTop = chat.scrollTop;
          const oldH = chat.scrollHeight;

          for (const n of Array.from(chatInner.querySelectorAll(".day-sep"))) n.remove();

          const rows = Array.from(chatInner.querySelectorAll(".msg-row")).filter((row) => !row.classList.contains("typing-row") && !row.classList.contains("recovery-panel-row"));
          let prevRole = null;
          let prevDay = null;
          let lastDay = null;

          for (const row of rows) {
            const role = row.classList.contains("user") ? "user" : "assistant";
            const ts = Number(row.dataset.ts || "0");
            const day = ts ? ymd(new Date(ts * 1000)) : null;

            row.classList.remove("grouped");
            if (prevRole === role && prevDay && day && prevDay === day) row.classList.add("grouped");
            prevRole = role;
            prevDay = day;

            if (day && day !== lastDay) {
              const d = new Date(ts * 1000);
              const sep = el("div", { class: "day-sep", text: dayLabel(d) });
              sep.dataset.day = day;
              chatInner.insertBefore(sep, row);
              lastDay = day;
            }
          }

          if (preserveScroll) {
            chat.scrollTop = oldTop + (chat.scrollHeight - oldH);
          }
          if (autoScroll) {
            requestAnimationFrame(() => scrollToBottom());
          }
          syncJumpButton();
          updateChatNavButtons();
          syncMessageCopyTabStops();
          if (loadedChatSearchSnapshot().open) refreshLoadedChatSearch({ jump: false, preserveCurrent: true });
        }

        function trimRenderedRows({ fromTop, maxRows = CHAT_DOM_WINDOW }) {
          const targets = trimRenderedRowTargets(renderedMessageRows(), fromTop, maxRows);
          if (!targets.length) return;
          for (const row of targets) row.remove();
          renderedAtLiveTail = Boolean(fromTop);
        }

        function trimRenderedRowsBeforeViewport({ maxRows = CHAT_DOM_WINDOW } = {}) {
          const targets = trimRowsBeforeViewportTargets(renderedMessageRows(), maxRows, chat.scrollTop + 1);
          if (!targets.length) return;
          for (const row of targets) row.remove();
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

      function eventKey(ev) {
        return codoxearMessageIdentity.eventKey(ev);
      }

        function markEventSeen(ev) {
          const key = eventKey(ev);
          if (!key) return;
          if (recentEventKeySet.has(key)) return;
          recentEventKeySet.add(key);
          recentEventKeys.push(key);
          if (recentEventKeys.length > RECENT_EVENT_KEYS_MAX) {
            const drop = recentEventKeys.splice(0, recentEventKeys.length - RECENT_EVENT_KEYS_MAX);
            for (const k of drop) recentEventKeySet.delete(k);
          }
        }

        function isDuplicateEvent(ev) {
          const key = eventKey(ev);
          if (!key) return false;
          return recentEventKeySet.has(key);
        }

        function chatAssistantDedupeKey(ev) {
          return codoxearMessageIdentity.chatAssistantDedupeKey(ev);
        }

        function isAdjacentAssistantDuplicateEvent(ev) {
          if (!renderedAtLiveTail || !ev || ev.pending || ev.role !== "assistant") return false;
          const key = chatAssistantDedupeKey(ev);
          if (!key) return false;
          const rows = renderedMessageRows();
          const last = rows.length ? rows[rows.length - 1] : null;
          return Boolean(last && last.dataset.role === "assistant" && last.dataset.assistantDedupeKey === key);
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
        if (ev.role !== "user" || ev.pending) return false;
        const slot = getSessionTranscriptSlot(sessionId);
        const slotEpoch = Number(slot.epoch || 0);
        const key = pendingMatchKey(ev.text);
        const loose = normalizeTextForPendingMatch(ev.text);
        const evTs = typeof ev.ts === "number" && Number.isFinite(ev.ts) ? ev.ts : null;
        const sameSlot = [];
        const exactCandidates = [];
        for (let i = 0; i < pendingUser.length; i++) {
          const x = pendingUser[i];
          if (!x || x.sessionId !== sessionId || Number(x.epoch || 0) !== slotEpoch) continue;
          const candidate = { i, x };
          sameSlot.push(candidate);
          if (x.key === key || x.loose === loose) exactCandidates.push(candidate);
        }
        const candidates = exactCandidates.length
          ? exactCandidates
          : sameSlot.filter(({ x }) => evTs !== null ? evTs >= Number(x.t0 || 0) - 5 : allowUntimedCommit);
          if (!candidates.length) return false;
          let best = candidates[0];
          if (exactCandidates.length && evTs !== null) {
            let bestD = Math.abs(evTs - (best.x.t0 || evTs));
            for (const c of candidates.slice(1)) {
              const d = Math.abs(evTs - (c.x.t0 || evTs));
              if (d < bestD) {
                best = c;
                bestD = d;
              }
          }
        }
        const idx = best.i;
        if (idx < 0) return null;
        const match = pendingUser[idx];
        pendingUser.splice(idx, 1);
        return match || null;
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
          const confirmed = window.confirm(
            `Clear the unknown-send marker only after checking the transcript or terminal. This does not undo a prompt that may already have been sent.${suffix}`
          );
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
          if (notModified && !swipeRefreshDeferred) return latestSessions;
          if (!notModified) {
            latestSessions = Array.isArray(data.sessions) ? data.sessions.slice() : [];
            newSessionDefaults =
              data && typeof data.new_session_defaults === "object" && data.new_session_defaults
                ? data.new_session_defaults
                : {
                    default_backend: "codex",
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
          const applyingDeferredSwipeRefresh = swipeRefreshDeferred && !openSwipeSessionId;
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
		          for (const s of sessions) sessionIndex.set(s.session_id, s);
              if (selected && !sessionIndex.has(selected)) clearSelectedSessionAfterRemoval(selected);
              if (selected) {
                applySessionListTranscriptIdentity(selected, sessionIndex.get(selected));
                syncRecoveryUiForSession(selected);
              }
              const sidebarEntries = sidebarSessionEntries(sessions);
		           if (swipeActions && openSwipeSessionId && sessionsWrap.childElementCount > 0) {
		             swipeRefreshDeferred = true;
		             return sessions;
		           }
              const sidebarSignature = sidebarRenderSignature(sidebarEntries, { selectedId: selected, swipeActions });
              const sidebarUnchanged = !applyingDeferredSwipeRefresh && sessionsWrap.childElementCount > 0 && sidebarSignature === lastSidebarRenderSignature;
              if (applyingDeferredSwipeRefresh) swipeRefreshDeferred = false;
              if (!sidebarUnchanged) {
		             sessionsWrap.innerHTML = "";
		             openSwipeContent = null;
                lastSidebarRenderSignature = sidebarSignature;
			          for (const entry of sidebarEntries) {
                    if (entry.type === "header") {
                      sessionsWrap.appendChild(renderSessionGroupHeader(entry));
                      continue;
                    }
                    const s = entry.session;
			            const card = el("div", { class: "session" + (selected === s.session_id ? " active" : "") });

             const title = sessionDisplayName(s);
             const badges = [];
             const launchFailed = sessionLaunchFailed(s);
             const launchPending = sessionLaunchPending(s);
             const launchRow = launchFailed || launchPending;
             if (launchFailed) badges.push(el("span", { class: "badge launchFailed", text: "failed", title: redactedLaunchErrorText(s.launch_error) || "Session launch failed" }));
             if (launchPending) badges.push(el("span", { class: "badge launchPending", text: "starting", title: "Session is still starting" }));
             if (s.unattended_enabled) badges.push(el("span", { class: "badge unattended", text: "unattended", title: "Unattended mode enabled" }));
             if (s.queue_len) badges.push(el("span", { class: "badge queue", text: `queue ${s.queue_len}` }));
             if (s.queue_recovery) badges.push(el("span", { class: "badge commitUnknown", text: "recovery", title: "Queued item is preserved for recovery; open the queue to resolve it" }));
             if (s.commit_unknown_send) {
               const unknownBadge = el("button", {
                 class: "badge commitUnknown",
                 type: "button",
                 text: "unknown",
                 title: "Previous send status is unknown; check transcript, then click to clear",
                 "aria-label": "Previous send status is unknown; check transcript, then clear marker",
               });
               unknownBadge.onclick = (e) => {
                 e.preventDefault();
                 e.stopPropagation();
                 closeOpenSwipe();
                 void clearCommitUnknownSend(s.session_id, s.commit_unknown_send_text || "");
               };
               badges.push(unknownBadge);
             }

	             const updatedTs = typeof s.updated_ts === "number" && Number.isFinite(s.updated_ts) ? s.updated_ts : s.start_ts;
	             const ageS = updatedTs ? Math.max(0, Date.now() / 1000 - updatedTs) : 0;
	             const effortTxt = String(s.reasoning_effort || "").trim().toLowerCase();
	             const effortMark = effortTxt === "xhigh" ? "X" : effortTxt === "high" ? "H" : effortTxt === "medium" ? "M" : effortTxt === "low" ? "L" : "";
	             const stateTxt = launchPending ? "starting" : fmtRelativeAge(ageS);
	             const cwdBase = baseName(s.cwd);
	             const branchTxt = typeof s.git_branch === "string" ? s.git_branch.trim() : "";

	            function closeOpenSwipe() {
	              if (!openSwipeContent) return;
	              openSwipeContent.style.transform = "translate3d(0px, 0, 0)";
	              openSwipeContent.dataset.swipeX = "0";
	              openSwipeContent = null;
	              openSwipeSessionId = null;
	              openSwipeTargetX = 0;
	              if (swipeRefreshDeferred) {
	                void refreshSessions().catch((e) => console.error("refreshSessions failed after swipe close", e));
	              }
	            }

             async function doDelete(e) {
               if (e) {
                 e.preventDefault();
                 e.stopPropagation();
               }
              closeOpenSwipe();
              if (!confirm(launchRow ? "Dismiss this launch record?" : "Delete this session?")) return;
              try {
                await api(`/api/sessions/${s.session_id}/delete`, { method: "POST", body: {} });
                clearDeletedSessionClientState(s.session_id);
                if (launchRow && card && card.parentNode) card.remove();
                await refreshSessions();
              } catch (err) {
                setToast(`delete error: ${err.message}`);
              }
             }

             const renameBtn = el("button", {
               class: "icon-btn",
               title: "Edit conversation",
               "aria-label": "Edit conversation",
               type: "button",
               html: iconSvg("edit"),
             });
             renameBtn.onclick = (e) => {
               e.preventDefault();
               e.stopPropagation();
               closeOpenSwipe();
               openEditSession(s.session_id);
             };
             const dupBtn = el("button", {
               class: "icon-btn",
               title: "Duplicate session",
               "aria-label": "Duplicate session",
               type: "button",
               html: iconSvg("duplicate"),
             });
             dupBtn.onclick = async (e) => {
               e.preventDefault();
               e.stopPropagation();
               closeOpenSwipe();
               if (launchRow) {
                 if (launchFailed) void selectSession(s.session_id);
                 setToast(launchFailed ? "review failed launch before retrying" : "session still starting");
                 return;
               }
               const cwd = s && s.cwd && s.cwd !== "?" ? s.cwd : "";
               if (!cwd) {
                 setToast("cwd unavailable");
                 return;
               }
               await spawnSessionWithCwd(
                 cwd,
                 null,
                 null,
                 "",
                 sessionProviderChoice(s),
                 s && s.model ? s.model : "default",
                 s && s.reasoning_effort ? s.reasoning_effort : "high",
                 sessionIsFast(s),
                 !!(s && s.transport === "tmux"),
                 null,
                 sessionAgentBackend(s)
               );
             };
             const delBtn = el("button", {
               class: "icon-btn danger sessionDel",
               title: launchRow ? "Dismiss launch record" : "Delete session",
               "aria-label": launchRow ? "Dismiss launch record" : "Delete session",
               type: "button",
               html: iconSvg("trash"),
             });
             delBtn.onclick = (e) => void doDelete(e);

             const stateDot = el("span", {
               class:
                 "stateDot" +
                 (launchPending ? " pending" : s.snoozed || s.blocked ? " suppressed" : s.busy ? " busy" : " idle"),
             });
             const titleRow = el("div", { class: "sessionTitleRow" }, [
               stateDot,
               el("div", { class: "titleLine", title: s.cwd || "" }, [
                 el("span", { class: "titleText", text: title }),
                 sessionIsFast(s)
                   ? el("span", { class: "sessionFastIcon", html: iconSvg("lightning"), title: "Fast session" })
                   : null,
               ].filter(Boolean)),
             ]);
	             const badgesWrap = el("div", { class: "sessionBadges" }, badges);
	             const metaItems = [
	               el("img", {
	                 class: "sessionBackendStatusIcon",
	                 src: agentBackendLogoPath(sessionAgentBackend(s)),
	                 alt: `${agentBackendDisplayName(sessionAgentBackend(s))} logo`,
	                 width: "12",
	                 height: "12",
	               }),
	               el("span", {
	                 class: `ownerBadge ownerIconBadge ${s.transport === "tmux" ? "owner-tmux" : s.owned ? "owner-web" : "owner-terminal"}`,
	                 html: iconSvg(sessionLaunchIcon(s)),
	                 title: sessionLaunchLabel(s),
	               })
	             ];
	             if (effortMark) {
	               metaItems.push(
	                 el("span", {
	                   class: `effortMark effort-${effortTxt}`,
	                   text: effortMark,
	                   title: `reasoning effort ${effortTxt}`,
	                 })
	               );
	             }
	             metaItems.push(el("span", { class: "metaText", text: `${stateTxt}${cwdBase ? ` | ${cwdBase}` : ""}${branchTxt ? ` | ${branchTxt}` : ""}` }));
	             const meta = el("div", { class: "muted subLine sessionMetaLine" }, metaItems);
             if (launchFailed) meta.title = redactedLaunchErrorText(s.launch_error) || "Session launch failed";
             if (launchPending) meta.title = "Session is still starting";

             const sessionEditActions = launchRow ? [] : [renameBtn, dupBtn];
             if (swipeActions) {
               const leftActions = el("div", { class: "sessionActions left" }, [delBtn]);
               const rightActions = el("div", { class: "sessionActions right" }, sessionEditActions);
               const top = el("div", { class: "row" }, [titleRow, badgesWrap]);
               const inner = el("div", { class: "sessionInner" }, [top, meta]);
	               const content = el("div", { class: "sessionContent" }, [inner]);
	               content.dataset.swipeX = "0";
	               const swipe = el("div", { class: "sessionSwipe" }, [leftActions, rightActions, content]);
	               card.appendChild(swipe);
	               if (openSwipeSessionId === s.session_id && openSwipeTargetX !== 0) {
	                  content.style.transform = `translate3d(${openSwipeTargetX}px, 0, 0)`;
	                  content.dataset.swipeX = String(openSwipeTargetX);
	                  openSwipeContent = content;
	               }

		               const leftMax = 72;
	               const rightMax = sessionEditActions.length ? 104 : 0;
	               let startX = null;
	               let startY = 0;
	               let startSwipe = 0;
	               let lastMoveTs = 0;
	               let lastMoveX = 0;
	               let swipeVelocity = 0;
	               let dragging = false;
	                content.addEventListener("pointerdown", (e) => {
	                  if (e.pointerType === "mouse" && e.button !== 0) return;
	                  startX = e.clientX;
	                  startY = e.clientY;
	                  startSwipe = Number(content.dataset.swipeX || 0);
	                  lastMoveTs = performance.now();
	                  lastMoveX = e.clientX;
	                  swipeVelocity = 0;
	                  dragging = false;
	                  if (openSwipeContent && openSwipeContent !== content) closeOpenSwipe();
	                  try {
	                    content.setPointerCapture(e.pointerId);
	                  } catch (_) {}
                });
	                 content.addEventListener("pointermove", (e) => {
	                   if (startX === null) return;
	                   const dx = e.clientX - startX;
	                   const dy = e.clientY - startY;
	                  const now = performance.now();
	                  const dt = Math.max(now - lastMoveTs, 1);
	                  swipeVelocity = ((e.clientX - lastMoveX) / dt) * 1000;
	                  lastMoveTs = now;
	                  lastMoveX = e.clientX;
	                  if (!dragging) {
	                    if (Math.abs(dx) < 4) return;
	                    if (Math.abs(dx) < Math.abs(dy) * 0.7) return;
	                    dragging = true;
	                    content.style.transition = "none";
	                  }
	                  if (dragging) e.preventDefault();
	                  let x = startSwipe + dx;
                  x = Math.min(leftMax, Math.max(-rightMax, x));
                   content.style.transform = `translate3d(${x}px, 0, 0)`;
                   content.dataset.swipeX = String(x);
                 });
                function finishSwipe(e) {
                  if (startX === null) return;
                  try {
                    if (e && e.pointerId != null) content.releasePointerCapture(e.pointerId);
                  } catch (_) {}
                  startX = null;
                  if (!dragging) return;
	                  dragging = false;
	                  content.style.transition = "";
	                  const x = Number(content.dataset.swipeX || 0);
	                 let target = 0;
	                  const commitLeft = leftMax > 0 && (x > leftMax * 0.28 || swipeVelocity > 420);
	                  const commitRight = rightMax > 0 && (-x > rightMax * 0.28 || swipeVelocity < -420);
	                  if (commitLeft) target = leftMax;
	                  else if (commitRight) target = -rightMax;
	                  content.style.transform = `translate3d(${target}px, 0, 0)`;
	                  content.dataset.swipeX = String(target);
	                  if (target !== 0) {
	                    openSwipeContent = content;
	                    openSwipeSessionId = s.session_id;
	                    openSwipeTargetX = target;
	                  } else if (openSwipeContent === content) {
	                    openSwipeContent = null;
	                    openSwipeSessionId = null;
	                    openSwipeTargetX = 0;
	                  }
	                }
               content.addEventListener("pointerup", finishSwipe);
               content.addEventListener("pointercancel", finishSwipe);

               card.onclick = () => {
                 const x = Number(content.dataset.swipeX || 0);
                 if (Math.abs(x) > 2) {
                   closeOpenSwipe();
                   return;
                 }
                 setSidebarOpen(false);
                 if (launchPending) {
                   setToast("session still starting");
                   return;
                 }
                 selectSession(s.session_id);
               };
	             } else {
	               card.classList.add("desktop");
	               const actions = el("div", { class: "sessionActionsInline" }, [...sessionEditActions, delBtn]);
	               const titleWithBadges = el("div", { class: "sessionTitleWithBadges" }, [titleRow, badgesWrap]);
	               const main = el("div", { class: "sessionMain" }, [titleWithBadges, meta]);
	               const inner = el("div", { class: "sessionInner sessionDesktopLayout" }, [main, actions]);
	               card.appendChild(inner);
	               card.onclick = () => {
	                 if (launchPending) {
	                   setToast("session still starting");
	                   return;
	                 }
	                 selectSession(s.session_id);
	               };
	             }

	             sessionsWrap.appendChild(card);
	            }
              }
	          if (openSwipeSessionId && !sessionIndex.has(openSwipeSessionId)) {
	            openSwipeSessionId = null;
	            openSwipeTargetX = 0;
	            openSwipeContent = null;
	          }
          if (selected) {
            const s = sessionIndex.get(selected);
            if (s) titleLabel.textContent = sessionTitleWithId(s);
          }
          updateUnattendedBtnState();
          updateQueueBadge();
          syncSendButtonState();
          syncQueueSubmitState();
          maybeSelectPendingHashSession();
          return sessions;
        }

        function appendEvent(ev) {
          if (!ev || (ev.role !== "user" && ev.role !== "assistant")) return;
          if (consumePendingUserIfMatches(ev)) return;
          if (isDuplicateEvent(ev)) return;
          if (isAdjacentAssistantDuplicateEvent(ev)) {
            markEventSeen(ev);
            return;
          }

          const pending = Boolean(ev.pending);
          const stick = pending || (renderedAtLiveTail && (autoScroll || isNearBottom()));
          if (!pending && !renderedAtLiveTail) {
            markEventSeen(ev);
            syncJumpButton();
            return;
          }
          const ts = typeof ev.ts === "number" && Number.isFinite(ev.ts) ? ev.ts : ev.pending ? Date.now() / 1000 : null;
          const { row } = safeMakeRow(ev, { ts, pending });
	          const anchor = typingRow && typingRow.isConnected ? typingRow : bottomSentinel;
	          chatInner.insertBefore(row, anchor);
            trimRenderedRows({ fromTop: stick });
          rebuildDecorations({ preserveScroll: false });
          if (typeof renderRecoveryPanelIfNeeded === "function") renderRecoveryPanelIfNeeded(typeof selected === "undefined" ? null : selected);
            if (!ev.pending) markClickFirstPaint();
          markEventSeen(ev);

          if (stick) {
            requestAnimationFrame(() => scrollToBottom());
          }
          syncJumpButton();
        }

        function normalizedTranscriptEvents(events, { consumePending = false } = {}) {
          const msgs = [];
          const seen = new Set();
          for (const ev of events || []) {
            if (!ev || (ev.role !== "user" && ev.role !== "assistant")) continue;
            if (consumePending) takePendingUserMatch(ev, selected, { allowUntimedCommit: false });
            const k = eventKey(ev);
            if (k && seen.has(k)) continue;
            if (k) seen.add(k);
            msgs.push(ev);
          }
          return msgs;
        }

        function renderTranscript(events, { preserveScroll = false } = {}) {
          const msgs = normalizedTranscriptEvents(events, { consumePending: true });
          renderedAtLiveTail = true;
          clearTranscriptDom();
          if (!msgs.length) {
            restorePendingUserRowsForSession(selected);
            return;
          }
          recentEventKeys.length = 0;
          recentEventKeySet.clear();
          const frag = document.createDocumentFragment();
          for (const ev of msgs) {
            const ts = typeof ev.ts === "number" && Number.isFinite(ev.ts) ? ev.ts : null;
            markEventSeen(ev);
            frag.appendChild(safeMakeRow(ev, { ts, pending: false }).row);
          }
          chatInner.insertBefore(frag, bottomSentinel);
          rebuildDecorations({ preserveScroll });
          restorePendingUserRowsForSession(selected);
        }

        function renderDetachedTranscriptWindow(events, { hasMore = false } = {}) {
          const msgs = normalizedTranscriptEvents(events, { consumePending: false });
          autoScroll = false;
          renderedAtLiveTail = false;
          clearTranscriptDom();
          setOlderState({ hasMore: Boolean(hasMore), isLoading: false });
          if (!msgs.length) {
            syncJumpButton();
            return false;
          }
          recentEventKeys.length = 0;
          recentEventKeySet.clear();
          const frag = document.createDocumentFragment();
          for (const ev of msgs) {
            const ts = typeof ev.ts === "number" && Number.isFinite(ev.ts) ? ev.ts : null;
            markEventSeen(ev);
            frag.appendChild(safeMakeRow(ev, { ts, pending: false }).row);
          }
          chatInner.insertBefore(frag, bottomSentinel);
          rebuildDecorations({ preserveScroll: false });
          chat.scrollTop = 1;
          lastScrollTop = chat.scrollTop;
          syncJumpButton();
          return true;
        }

        function prependOlderEvents(allEvents, { preserveViewport = false } = {}) {
          const msgs = [];
          for (const ev of allEvents) {
            if (!ev || (ev.role !== "user" && ev.role !== "assistant")) continue;
            msgs.push(ev);
          }
          if (!msgs.length) return;
          autoScroll = false;
          const frag = document.createDocumentFragment();
          for (const ev of msgs) {
            const ts = typeof ev.ts === "number" && Number.isFinite(ev.ts) ? ev.ts : null;
            frag.appendChild(safeMakeRow(ev, { ts, pending: false }).row);
          }
          const anchorRow = preserveViewport ? firstVisibleMessageRow() : null;
          const anchorOffset = anchorRow ? anchorRow.offsetTop - chat.scrollTop : 0;
          const firstMsg = chatInner.querySelector(".msg-row:not(.typing-row)");
          const anchor = firstMsg || (typingRow && typingRow.isConnected ? typingRow : bottomSentinel);
          chatInner.insertBefore(frag, anchor);
          const wasAtLiveTail = renderedAtLiveTail;
          if (!preserveViewport) chat.scrollTop = 1;
          trimRenderedRows({ fromTop: false, maxRows: CHAT_DOM_WINDOW_WITH_HISTORY_SLACK });
          if (wasAtLiveTail && renderedAtLiveTail === false) {
            autoScroll = false;
          }
          rebuildDecorations({ preserveScroll: false });
          if (preserveViewport && anchorRow && anchorRow.isConnected) {
            chat.scrollTop = Math.max(0, anchorRow.offsetTop - anchorOffset);
          } else {
            chat.scrollTop = 1;
          }
          lastScrollTop = chat.scrollTop;
          syncJumpButton();
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
            const nextHasOlder = Boolean(data.has_older);
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

        async function loadNearestOlderChatSearchWindow() {
          if (!selected || !currentChatSearchQuery()) return false;
          const boundaryCursor = oldestRenderedHistoryCursor();
          if (!boundaryCursor) return false;
          const sid = selected;
          const gen = pollGen;
          const query = currentChatSearchQuery();
          try {
            const data = await api(
              `/api/sessions/${sid}/messages/search?q=${encodeURIComponent(query)}&limit=1&text_max=96&order=latest&before=${encodeURIComponent(boundaryCursor)}`
            );
            if (selected !== sid || pollGen !== gen || currentChatSearchQuery() !== query) return false;
            const match = Array.isArray(data.matches) && data.matches.length ? data.matches[0] : null;
            const cursor = match && typeof match.load_cursor === "string" ? match.load_cursor : "";
            const targetHistoryCursor = match && typeof match.history_cursor === "string" ? match.history_cursor : "";
            if (!cursor) return false;
            return await loadChatSearchCursorWindow(cursor, { targetHistoryCursor });
          } catch (e) {
            if (e && e.status === 401) {
              handleAppAuthLoss();
              return false;
            }
            if (selected !== sid || pollGen !== gen || currentChatSearchQuery() !== query) return false;
            if (e && e.status === 409) {
              await openSession(sid, { useCache: false });
              return false;
            }
            return false;
          }
        }

        async function loadChatSearchCursorWindow(cursor, { targetHistoryCursor = "" } = {}) {
          const cleanCursor = String(cursor || "").trim();
          if (!selected || !cleanCursor || currentChatSearchState().loadingOlder) return false;
          const sid = selected;
          const gen = pollGen;
          const query = currentChatSearchQuery();
          invalidateOlderLoad();
          const load = olderLoadRuntime.beginLoad({ cancelOnScroll: false });
          loadedChatSearchRuntime.setLoadingOlder(true);
          syncChatSearchStatus();
          try {
            const data = await api(`/api/sessions/${sid}/messages/history?cursor=${encodeURIComponent(cleanCursor)}&limit=${olderPageLimit()}`, {
              signal: load.signal,
            });
            if (selected !== sid || pollGen !== gen || !olderLoadRuntime.isCurrent(load) || currentChatSearchQuery() !== query || String(currentChatSearchQuery() || "") === "") return false;
            const evs = Array.isArray(data.events) ? data.events : [];
            if (!evs.length) return false;
            const rendered = renderDetachedTranscriptWindow(evs, { hasMore: Boolean(data.has_older) });
            if (!rendered) return false;
            refreshLoadedChatSearch({ jump: false, preserveCurrent: false });
            const targetIndex = ensureChatSearchTargetRow(targetHistoryCursor);
            if (targetIndex >= 0) focusChatSearchMatch(targetIndex, { jump: true });
            else if (currentChatSearchMatches().length) focusChatSearchMatch(currentChatSearchMatches().length - 1, { jump: true });
            setToast("Loaded transcript match");
            return Boolean(currentChatSearchMatches().length || targetIndex >= 0);
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
            showOlderLoadError();
            return false;
          } finally {
            olderLoadRuntime.finishLoad(load);
            loadedChatSearchRuntime.setLoadingOlder(false);
            if (isLoadingOlderMessages()) setOlderState({ hasMore: hasOlderMessages(), isLoading: false });
            syncChatSearchStatus();
          }
        }

        function maybeAutoLoadOlder() {
          if (chat.scrollTop > OLDER_TOP_TRIGGER_PX) return;
          void loadOlderMessages({ auto: true });
        }

        function applySessionRuntimeFromTail(sessionId, data) {
          const slot = syncActiveTranscriptSlot(sessionId);
          liveCursor = slot.state === "bound" && typeof data.live_cursor === "string" && data.live_cursor ? data.live_cursor : null;
          setOlderState({ hasMore: slot.state === "bound" && Boolean(data && data.has_older), isLoading: false });
          const nowBusy = Boolean(data && data.busy);
          turnOpen = nowBusy;
          const queueLen = data && Number.isFinite(Number(data.queue_len)) ? Number(data.queue_len) : 0;
          setStatus({ running: nowBusy, queueLen });
          setContext(data ? data.token : null);
          setTyping(nowBusy);
          if (slot.state === "bound") {
            const s = sessionIndex.get(sessionId);
            if (s) rememberTailSnapshot(sessionId, s, data);
          } else {
            sessionTailCache.delete(sessionId);
          }
        }

        function renderSessionTail(events) {
          renderTranscript(events, { preserveScroll: false });
          renderRecoveryPanelIfNeeded(selected);
          markClickFirstPaint();
          requestAnimationFrame(() => {
            scrollToBottom();
            requestAnimationFrame(() => scrollToBottom());
          });
        }

        function recoverySessionInfo(sessionId) {
          const s = sessionIndex.get(sessionId);
          if (!s || (!sessionLaunchFailed(s) && !s.orphan_recovery && !s.queue_recovery && !s.commit_unknown_send)) return null;
          return s;
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
          activeTranscriptState = "pending_bind";
          activeLogPath = null;
          activeThreadId = null;
          liveCursor = null;
          clearRenderedTranscriptRange();
          turnOpen = false;
          storageRemoveItem("codexweb.selected");
          setSessionHash("");
          titleLabel.textContent = "No session selected";
          setStatus({ running: false, queueLen: 0 });
          setContext(null);
          setTyping(false);
          setAttachCount(0);
          resetChatRenderState();
          updateQueueBadge();
          if (unattendedMenuOpen) hideUnattendedMenu();
          updateUnattendedBtnState();
          syncSendButtonState();
          syncQueueSubmitState();
          syncAttachButtonState();
          return true;
        }

        function clearDeletedSessionClientState(sessionId) {
          const selectedCleared = clearSelectedSessionAfterRemoval(sessionId);
          sessionTranscriptSlots.delete(sessionId);
          sessionTailCache.delete(sessionId);
          dropPendingUserRows(sessionId, () => true);
          return selectedCleared;
        }

        async function dismissFailedLaunchRecord(sessionId) {
          const s = sessionIndex.get(sessionId);
          if (!sessionLaunchFailed(s)) {
            setToast("launch record is not failed");
            return;
          }
          if (!confirm("Dismiss this launch record?")) return;
          try {
            await api(`/api/sessions/${sessionId}/delete`, { method: "POST", body: {} });
            clearDeletedSessionClientState(sessionId);
            await refreshSessions();
            setToast("Dismissed launch record");
          } catch (err) {
            setToast(`dismiss error: ${err && err.message ? err.message : "unknown error"}`);
          }
        }

        function focusedRecoveryActionDescriptor(sessionId) {
          const active = document.activeElement;
          const button = active && typeof active.closest === "function" ? active.closest(".recovery-panel-row button") : null;
          if (!button) return pendingRecoveryFocusDescriptor && pendingRecoveryFocusDescriptor.sessionId === sessionId ? pendingRecoveryFocusDescriptor : null;
          return {
            sessionId,
            text: String(button.textContent || "").trim(),
            title: String(button.getAttribute("title") || ""),
          };
        }

        function focusRecoveryAction(row, descriptor) {
          if (!row || !descriptor) return false;
          const buttons = Array.from(row.querySelectorAll("button"));
          const target = buttons.find((btn) => String(btn.textContent || "").trim() === descriptor.text && String(btn.getAttribute("title") || "") === descriptor.title) || null;
          if (!target || target.disabled) return false;
          pendingRecoveryFocusDescriptor = descriptor;
          requestAnimationFrame(() => {
            try {
              if (target.isConnected && !target.disabled) {
                target.focus({ preventScroll: true });
                pendingRecoveryFocusDescriptor = null;
              }
            } catch {}
          });
          return true;
        }

        function focusRecoveryFallback(descriptor) {
          if (!descriptor) return;
          const fallback = document.querySelector(".recovery-panel .icon-btn") || queueBtn || null;
          if (!fallback || typeof fallback.focus !== "function" || fallback.disabled) {
            pendingRecoveryFocusDescriptor = null;
            return;
          }
          pendingRecoveryFocusDescriptor = descriptor;
          requestAnimationFrame(() => {
            try {
              if (fallback.isConnected && !fallback.disabled) {
                fallback.focus({ preventScroll: true });
                pendingRecoveryFocusDescriptor = null;
              }
            } catch {}
          });
        }

        function renderRecoveryPanelIfNeeded(sessionId) {
          const focusDescriptor = focusedRecoveryActionDescriptor(sessionId);
          for (const row of Array.from(chatInner.querySelectorAll(".recovery-panel-row"))) row.remove();
          const s = recoverySessionInfo(sessionId);
          if (!s) {
            focusRecoveryFallback(focusDescriptor);
            return false;
          }
          const queueLen = Number.isFinite(Number(s.queue_len)) ? Number(s.queue_len) : 0;
          const launchFailed = sessionLaunchFailed(s);
          const row = el("div", { class: "msg-row assistant recovery-panel-row" });
          row.dataset.role = "assistant";
          const panelLabel = launchFailed ? "Launch failed" : "Recovery needed";
          const bubble = el("div", { class: "msg assistant recovery-panel", role: "group", "aria-label": panelLabel });
          bubble.appendChild(el("div", { class: "recoveryPanelTitle", text: panelLabel }));
          const list = el("ul", { class: "recoveryPanelList" });
          if (launchFailed) {
            list.appendChild(el("li", { text: "This web-owned session failed before a usable session log was bound." }));
            const launchStage = String(s.launch_stage || "").trim();
            if (launchStage) list.appendChild(el("li", { text: `Stage: ${launchStage}` }));
            const launchModel = [s.model_provider, s.model].map((v) => String(v || "").trim()).filter(Boolean).join("/");
            if (launchModel) list.appendChild(el("li", { text: `Launch settings: ${launchModel}${s.reasoning_effort ? " · " + s.reasoning_effort : ""}` }));
          }
          if (s.orphan_recovery) list.appendChild(el("li", { text: "The original session is missing; preserved prompts can be reviewed here before you decide what to discard." }));
          if (s.commit_unknown_send) list.appendChild(el("li", { text: "A direct send may or may not have reached the terminal. Check the transcript or terminal before clearing the marker." }));
          if (s.queue_recovery || queueLen > 0) list.appendChild(el("li", { text: `${queueLen || "Some"} queued recovery item${queueLen === 1 ? "" : "s"} preserved for review.` }));
          bubble.appendChild(list);
          const launchError = launchFailed ? recoveryPromptPreview(redactedLaunchErrorText(s.launch_error), 1200) : "";
          if (launchError) bubble.appendChild(el("pre", { class: "recoveryPanelPreview", text: launchError }));
          const preview = recoveryPromptPreview(s.commit_unknown_send_text || "");
          if (preview) bubble.appendChild(el("pre", { class: "recoveryPanelPreview", text: preview }));
          const actions = el("div", { class: "recoveryPanelActions" });
          if (queueLen > 0) {
            const queueAction = el("button", { class: "icon-btn text-btn", type: "button", text: "Review queue", title: "Review preserved queued recovery items" });
            queueAction.onclick = (e) => {
              e.preventDefault();
              e.stopPropagation();
              showQueueViewer({ opener: e.currentTarget });
            };
            actions.appendChild(queueAction);
          }
          if (s.commit_unknown_send) {
            const clearAction = el("button", { class: "icon-btn text-btn danger", type: "button", text: "Clear unknown marker", title: "Clear only after checking transcript or terminal" });
            clearAction.onclick = async (e) => {
              e.preventDefault();
              e.stopPropagation();
              await clearCommitUnknownSend(sessionId, s.commit_unknown_send_text || "");
            };
            actions.appendChild(clearAction);
          }
          if (launchFailed) {
            const newLikeAction = el("button", { class: "icon-btn text-btn", type: "button", text: "New like this", title: "Review copied launch settings before starting" });
            newLikeAction.onclick = (e) => {
              e.preventDefault();
              e.stopPropagation();
              const preset = launchPresetFromSessionInfo(s);
              if (!preset) {
                setToast("launch details not available");
                return;
              }
              openNewSessionDialog({ likeSession: preset, statusText: "Review copied launch settings before starting.", returnFocusEl: e.currentTarget });
            };
            actions.appendChild(newLikeAction);
            const dismissAction = el("button", { class: "icon-btn text-btn danger", type: "button", text: "Dismiss launch", title: "Dismiss failed launch record" });
            dismissAction.onclick = async (e) => {
              e.preventDefault();
              e.stopPropagation();
              await dismissFailedLaunchRecord(sessionId);
            };
            actions.appendChild(dismissAction);
          }
          const copyAction = el("button", { class: "icon-btn text-btn", type: "button", text: "Copy details", title: "Copy recovery details" });
          copyAction.onclick = async (e) => {
            e.preventDefault();
            e.stopPropagation();
            try {
              await copyToClipboard(recoveryDetailsText(sessionId, s));
              setToast("Copied recovery details");
            } catch (err) {
              setToast(`copy failed: ${err && err.message ? err.message : "unknown error"}`);
            }
          };
          actions.appendChild(copyAction);
          bubble.appendChild(actions);
          row.appendChild(bubble);
          const anchor = typingRow && typingRow.isConnected ? typingRow : bottomSentinel;
          chatInner.insertBefore(row, anchor);
          if (focusDescriptor && !focusRecoveryAction(row, focusDescriptor)) focusRecoveryFallback(focusDescriptor);
          return true;
        }

        function syncRecoveryUiForSession(sessionId) {
          if (selected !== sessionId) return;
          const s = sessionIndex.get(sessionId) || null;
          if (s) {
            const queueLen = Number.isFinite(Number(s.queue_len)) ? Number(s.queue_len) : 0;
            setStatus({ running: currentRunning, queueLen });
          }
          renderRecoveryPanelIfNeeded(sessionId);
          syncAttachButtonState();
          syncQueueSubmitState();
          syncSendButtonState();
          updateUnattendedBtnState();
          updateQueueBadge();
        }

        function renderPendingTranscriptSlot(sessionId) {
          clearTranscriptDom();
          setOlderState({ hasMore: false, isLoading: false });
          renderedAtLiveTail = true;
          restorePendingUserRowsForSession(sessionId);
          renderRecoveryPanelIfNeeded(sessionId);
          markClickFirstPaint();
          syncJumpButton();
        }

        function renderTranscriptLoading(sessionId) {
          clearTranscriptDom();
          setOlderState({ hasMore: false, isLoading: false });
          renderedAtLiveTail = true;
          restorePendingUserRowsForSession(sessionId);
          const row = el("div", { class: "msg-row assistant typing-row transcript-loading-row" });
          row.dataset.role = "assistant";
          row.appendChild(el("div", { class: "msg assistant loading", role: "status", "aria-live": "polite", text: "Loading transcript…" }));
          chatInner.insertBefore(row, bottomSentinel);
          syncJumpButton();
        }

        function renderTranscriptLoadError(sessionId, err, { preserveTranscript = false } = {}) {
          for (const row of Array.from(chatInner.querySelectorAll(".transcript-error-row"))) row.remove();
          if (!preserveTranscript) {
            clearTranscriptDom();
            setOlderState({ hasMore: false, isLoading: false });
            renderedAtLiveTail = true;
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
          renderRecoveryPanelIfNeeded(sessionId);
          markClickFirstPaint();
          syncJumpButton();
        }

        function applyCachedTail(sessionId, cache, sessionMeta) {
          updateSessionTranscriptSlot(sessionId, {
            transcript_state: "bound",
            thread_id: cache.threadId || (sessionMeta ? sessionMeta.thread_id : null),
            log_path: cache.logPath || (sessionMeta ? sessionMeta.log_path : null),
          });
          syncActiveTranscriptSlot(sessionId);
          liveCursor = cache.liveCursor || null;
          setOlderState({ hasMore: Boolean(cache.hasOlder), isLoading: false });
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
          setTyping(cachedBusy);
        }

        async function openSession(sessionId, { useCache = true, fallbackToCacheOnFailure = false } = {}) {
          pollGen += 1;
          const myGen = pollGen;
          abortOpenSessionTailRequest();
          abortMessagePollRequest();
          if (pollTimer) {
            clearTimeout(pollTimer);
            pollTimer = null;
          }
          pollKickPending = false;
          pollKickDelayMs = null;

          selected = sessionId;
          if (unattendedMenuOpen && unattendedMenuSessionId !== sessionId) hideUnattendedMenu();
          storageSetItem("codexweb.selected", sessionId);
          setSessionHash(sessionId);
          activeTranscriptState = "pending_bind";
          activeLogPath = null;
          activeThreadId = null;
          liveCursor = null;
          clearRenderedTranscriptRange();
          turnOpen = false;
          setAttachCount(0);
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
          setTyping(optimisticBusy);
          const fileViewerSyncStarted = Boolean(isFileViewerOpen() && !currentFileDirty());
          if (fileViewerSyncStarted) {
            void ensureCurrentFileViewerSession().catch((e) => console.error("file viewer session sync failed after selection", e));
          }

          if (s && s.orphan_recovery) {
            renderPendingTranscriptSlot(sessionId);
            activeTranscriptState = "failed";
            setStatus({ running: false, queueLen: optimisticQueueLen });
            setContext(null);
            setTyping(false);
            syncAttachButtonState();
            syncQueueSubmitState();
            syncSendButtonState();
            updateUnattendedBtnState();
            if (isMobile()) setSidebarOpen(false);
            return { events: [], busy: false, queue_len: optimisticQueueLen, token: null, transcript_state: "failed" };
          }

          const cachedTail = s ? sessionTailCache.get(sessionId) : null;
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

          if (slotChange.current.state !== "failed") kickPoll(900);
          if (isMobile()) setSidebarOpen(false);
          updateUnattendedBtnState();
          if (isFileViewerOpen() && !currentFileDirty() && !fileViewerSyncStarted) {
            void ensureCurrentFileViewerSession();
          } else if (isFileViewerOpen() && !currentFileDirty() && currentFileViewerSessionId() === sessionId) {
            void refreshFileCandidates({ sessionId }).catch((e) => console.error("file candidates refresh failed after transcript load", e));
          }
          return data;
        }

			        async function pollMessages(sid = selected, gen = pollGen) {
          if (appDisposed || !sid) return;
          let pollRequest = null;
          try {
            if (!liveCursor) {
              if (activeTranscriptState === "pending_bind") {
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
              if (activeTranscriptState === "failed") return;
              await openSession(sid, { useCache: false });
              return;
            }
            const reqCursor = liveCursor;
            pollRequest = beginMessagePollRequest(sid, gen);
            const data = await api(`/api/sessions/${sid}/messages/live?cursor=${encodeURIComponent(reqCursor)}`, { signal: pollRequest.signal });
            if (gen !== pollGen || sid !== selected) return;
            markMessagePollSuccess();
            const slotInfo = transcriptSnapshotFromData(data);
            const nowBusy = Boolean(data.busy);
            if (activeTranscriptState === "bound" && slotInfo.state === "pending_bind") {
              updateSessionTranscriptSlot(sid, data);
              resetChatRenderState();
              renderPendingTranscriptSlot(sid);
              setAttachCount(0);
              applySessionRuntimeFromTail(sid, data);
              return;
            }
            if (activeTranscriptState === "bound" && slotInfo.state === "bound" && slotInfo.logPath !== activeLogPath) {
              await openSession(sid, { useCache: false });
              return;
            }

            liveCursor = typeof data.live_cursor === "string" && data.live_cursor ? data.live_cursor : null;
            const evs = Array.isArray(data.events) ? data.events : [];
            for (const ev of evs) appendEvent(ev);

            const turnStart = Boolean(data.turn_start);
            const turnEnd = Boolean(data.turn_end);
            const turnAborted = Boolean(data.turn_aborted);
            if (turnStart) turnOpen = true;
            if (!turnOpen && nowBusy) turnOpen = true;
            if ((turnEnd || turnAborted) && turnOpen) turnOpen = false;
            if (turnOpen && !nowBusy) turnOpen = false;

            setStatus({ running: Boolean(turnOpen || nowBusy), queueLen: data.queue_len });
            setContext(data.token);
            setTyping(Boolean(turnOpen || nowBusy));
            const s2 = sessionIndex.get(sid);
            if (evs.length) {
              appendTailSnapshotEvents(sid, evs, {
                session: s2,
                liveCursor,
                busy: Boolean(turnOpen || nowBusy),
                queueLen: data.queue_len,
                token: data.token,
                identityData: data,
              });
            }
            if (s2) titleLabel.textContent = sessionTitleWithId(s2);
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
            toast.textContent = `error: ${e.message}`;
          } finally {
            finishMessagePollRequest(pollRequest);
          }
        }

        async function pollLoop() {
          if (appDisposed || !selected) return;
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
          if (appDisposed) return;
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
          autoScroll = true;
          try {
            await openSession(sid, { useCache: false, fallbackToCacheOnFailure: true });
          } catch (e) {
            if (selected !== sid) return;
            setToast(`jump error: ${e && e.message ? e.message : "unknown error"}`);
          }
          if (selected !== sid) return;
          requestAnimationFrame(() => {
            scrollToBottom();
            syncJumpButton();
          });
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

        function parseUnattendedDraftInt(name) {
          const raw = String(unattendedNumberDraft[name] ?? "").trim();
          if (!raw) return null;
          const minValue = name === "cooldown_minutes" ? 1 : 0;
          const value = Number.parseInt(raw, 10);
          if (!Number.isInteger(value) || value < minValue) return null;
          return value;
        }

        function syncUnattendedNumberDraftsFromCfg() {
          if (!unattendedNumberDirty.cooldown_minutes) unattendedNumberDraft.cooldown_minutes = String(unattendedCfg.cooldown_minutes);
          if (!unattendedNumberDirty.remaining_injections) unattendedNumberDraft.remaining_injections = String(unattendedCfg.remaining_injections);
        }

        function syncUnattendedNumberInputs() {
          const cooldownEl = $("#unattendedCooldownMinutes");
          const remainingEl = $("#unattendedRemainingInjections");
          if (cooldownEl) {
            cooldownEl.value = unattendedNumberDirty.cooldown_minutes
              ? unattendedNumberDraft.cooldown_minutes
              : String(unattendedCfg.cooldown_minutes);
          }
          if (remainingEl) {
            remainingEl.value = unattendedNumberDirty.remaining_injections
              ? unattendedNumberDraft.remaining_injections
              : String(unattendedCfg.remaining_injections);
          }
        }

        function setUnattendedControlsDisabled(disabled) {
          const value = Boolean(disabled);
          ["unattendedEnabled", "unattendedCooldownMinutes", "unattendedRemainingInjections", "unattendedRequest"].forEach((id) => {
            const node = $(`#${id}`);
            if (node) node.disabled = value;
          });
        }

        function restoreUnattendedNumberDraft(name) {
          unattendedNumberDirty[name] = false;
          unattendedNumberDraft[name] = String(unattendedCfg[name]);
          syncUnattendedNumberInputs();
        }

        function finalizeUnattendedNumberDraft(name) {
          const value = parseUnattendedDraftInt(name);
          if (value === null || value !== unattendedCfg[name]) return;
          unattendedNumberDirty[name] = false;
          unattendedNumberDraft[name] = String(unattendedCfg[name]);
        }

        function updateUnattendedBtnState() {
          const s = selected ? sessionIndex.get(selected) : null;
          syncTitleEditState();
          const on = Boolean(s && s.unattended_enabled);
          unattendedBtn.disabled = !selected;
          unattendedBtn.classList.toggle("active", Boolean(selected && on));
          if (
            selected &&
            s &&
            !unattendedNumberDirty.cooldown_minutes &&
            Number.isInteger(s.unattended_cooldown_minutes) &&
            s.unattended_cooldown_minutes >= 1
          ) {
            unattendedCfg.cooldown_minutes = s.unattended_cooldown_minutes;
          }
          if (
            selected &&
            s &&
            !unattendedNumberDirty.remaining_injections &&
            Number.isInteger(s.unattended_remaining_injections) &&
            s.unattended_remaining_injections >= 0
          ) {
            unattendedCfg.remaining_injections = s.unattended_remaining_injections;
          }
          syncUnattendedNumberDraftsFromCfg();
          if (unattendedMenuOpen) {
            const enabledEl = $("#unattendedEnabled");
            syncUnattendedNumberInputs();
            if (enabledEl) enabledEl.checked = Boolean(selected && on);
          }
          syncAttachButtonState();
          fileBtn.disabled = !selected;
          copyConversationBtn.disabled = !selected;
          chatSearchBtn.disabled = !selected;
          sessionContextBar.style.display = selected ? "flex" : "none";
          chatNavRail.style.display = selected ? "flex" : "none";
          if (unattendedMenuOpen && (!selected || unattendedMenuSessionId !== selected)) hideUnattendedMenu();
          if (!selected && loadedChatSearchSnapshot().open) closeChatSearch();
          updateChatNavButtons();
          syncQueueSubmitState();
          syncSendButtonState();
          diagBtn.disabled = !selected;
        }
           async function loadUnattendedCfgForSelected({ sid = selected, openToken = null } = {}) {
             if (!sid) return;
             sid = String(sid);
              const d = await api(`/api/sessions/${sid}/unattended`);
              if (selected !== sid) return;
              if (openToken !== null && (unattendedMenuToken !== openToken || unattendedMenuSessionId !== sid || !unattendedMenuOpen)) return;
              if (!d || typeof d !== "object") throw new Error("invalid unattended response");
              if (typeof d.enabled !== "boolean") throw new Error("invalid unattended.enabled");
              if (typeof d.request !== "string") throw new Error("invalid unattended.request");
              if (!Number.isInteger(d.cooldown_minutes) || d.cooldown_minutes < 1) throw new Error("invalid unattended.cooldown_minutes");
              if (!Number.isInteger(d.remaining_injections) || d.remaining_injections < 0) throw new Error("invalid unattended.remaining_injections");
              unattendedCfg = {
                enabled: d.enabled,
                request: d.request,
                cooldown_minutes: d.cooldown_minutes,
                remaining_injections: d.remaining_injections,
              };
             unattendedNumberDirty.cooldown_minutes = false;
             unattendedNumberDirty.remaining_injections = false;
             syncUnattendedNumberDraftsFromCfg();
             const enabledEl = $("#unattendedEnabled");
             const requestEl = $("#unattendedRequest");
             if (enabledEl) enabledEl.checked = unattendedCfg.enabled;
             syncUnattendedNumberInputs();
             if (requestEl) requestEl.value = unattendedCfg.request;
           }
			        function validateUnattendedPayload(data) {
          if (!data || typeof data !== "object") throw new Error("invalid unattended response");
          if (typeof data.enabled !== "boolean") throw new Error("invalid unattended.enabled");
          if (typeof data.request !== "string") throw new Error("invalid unattended.request");
          if (!Number.isInteger(data.cooldown_minutes) || data.cooldown_minutes < 1) throw new Error("invalid unattended.cooldown_minutes");
          if (!Number.isInteger(data.remaining_injections) || data.remaining_injections < 0) throw new Error("invalid unattended.remaining_injections");
        }

        function unattendedSaveSnapshot(patch = {}) {
          const out = {};
          const has = (name) => Object.prototype.hasOwnProperty.call(patch, name);
          if (has("request")) out.request = String(patch.request || "");
          if (has("cooldown_minutes")) out.cooldown_minutes = patch.cooldown_minutes;
          if (has("remaining_injections")) {
            const remaining = Number(patch.remaining_injections);
            out.remaining_injections = remaining;
            if (Number.isFinite(remaining) && remaining <= 0) out.enabled = false;
          }
          if (has("enabled")) {
            const remaining = has("remaining_injections") ? Number(out.remaining_injections) : Number(unattendedCfg.remaining_injections);
            out.enabled = Boolean(patch.enabled) && Number.isFinite(remaining) && remaining > 0;
          }
          return out;
        }

        function applySavedUnattendedCfg(saved, sid) {
          if (selected !== sid) return;
          if (unattendedMenuOpen && unattendedMenuSessionId !== sid) return;
          unattendedCfg = {
            enabled: saved.enabled,
            request: saved.request,
            cooldown_minutes: saved.cooldown_minutes,
            remaining_injections: saved.remaining_injections,
          };
          const s = sessionIndex.get(sid);
          if (s) {
            s.unattended_enabled = Boolean(saved.enabled);
            s.unattended_cooldown_minutes = saved.cooldown_minutes;
            s.unattended_remaining_injections = saved.remaining_injections;
          }
          finalizeUnattendedNumberDraft("cooldown_minutes");
          finalizeUnattendedNumberDraft("remaining_injections");
          syncUnattendedNumberDraftsFromCfg();
          syncUnattendedNumberInputs();
          const enabledEl = $("#unattendedEnabled");
          if (enabledEl) enabledEl.checked = Boolean(saved.enabled);
          const requestEl = $("#unattendedRequest");
          if (requestEl) requestEl.value = String(saved.request || "");
        }

        async function flushUnattendedSave(sid) {
          if (!sid || appDisposed || unattendedSaveInFlight.get(sid)) return;
          const snapshot = unattendedSavePending.get(sid);
          if (!snapshot) return;
          unattendedSavePending.delete(sid);
          unattendedSaveInFlight.set(sid, true);
          try {
            const saved = await api(`/api/sessions/${sid}/unattended`, {
              method: "POST",
              body: snapshot,
            });
            validateUnattendedPayload(saved);
            if (appDisposed) return;
            if (!unattendedSavePending.has(sid)) applySavedUnattendedCfg(saved, sid);
            await refreshSessions();
          } catch (e) {
            if (e && e.status === 401) {
              handleAppAuthLoss();
              return;
            }
            console.error("save unattended mode failed", e);
            if (!appDisposed && selected === sid) setToast(`unattended save error: ${e && e.message ? e.message : "unknown error"}`);
          } finally {
            unattendedSaveInFlight.delete(sid);
            if (!appDisposed && unattendedSavePending.has(sid)) void flushUnattendedSave(sid);
            else if (!appDisposed && selected === sid) updateUnattendedBtnState();
          }
        }

        function scheduleUnattendedSave(patch = {}) {
          if (!selected) return;
          const sid = selected;
          const snapshot = unattendedSaveSnapshot(patch);
          if (!Object.keys(snapshot).length) return;
          unattendedSavePending.set(sid, { ...(unattendedSavePending.get(sid) || {}), ...snapshot });
          const existing = unattendedSaveTimers.get(sid);
          if (existing) clearTimeout(existing);
          const timer = setTimeout(() => {
            unattendedSaveTimers.delete(sid);
            void flushUnattendedSave(sid);
          }, 450);
          unattendedSaveTimers.set(sid, timer);
        }
        function setUnattendedMenuExpanded(open) {
          unattendedMenuOpen = Boolean(open);
          unattendedMenu.style.display = unattendedMenuOpen ? "block" : "none";
          unattendedBtn.setAttribute("aria-expanded", unattendedMenuOpen ? "true" : "false");
        }

        function restoreUnattendedFocus() {
          const target = unattendedReturnFocusEl;
          unattendedReturnFocusEl = null;
          restoreModalFocus(target, () => unattendedMenuOpen);
        }

        function focusUnattendedInitialControl() {
          requestAnimationFrame(() => {
            if (!unattendedMenuOpen) return;
            const target = $("#unattendedEnabled") || unattendedMenu;
            try {
              target.focus({ preventScroll: true });
            } catch {}
          });
        }

        function hideUnattendedMenu({ restoreFocus = false } = {}) {
          const wasOpen = unattendedMenuOpen;
          unattendedMenuToken += 1;
          unattendedMenuSessionId = null;
          setUnattendedMenuExpanded(false);
          if (restoreFocus && wasOpen) restoreUnattendedFocus();
          else unattendedReturnFocusEl = null;
        }

        async function showUnattendedMenu({ opener = null } = {}) {
          if (!selected) return;
          const sid = selected;
          const openToken = unattendedMenuToken + 1;
          unattendedMenuToken = openToken;
          unattendedMenuSessionId = sid;
          unattendedReturnFocusEl = opener instanceof HTMLElement ? opener : document.activeElement instanceof HTMLElement ? document.activeElement : null;
          setUnattendedControlsDisabled(true);
          setUnattendedMenuExpanded(true);
          const rect = unattendedBtn.getBoundingClientRect();
          const top = Math.min(window.innerHeight - 12, rect.bottom + 8);
          unattendedMenu.style.top = `${top}px`;
          unattendedMenu.style.left = "12px";
          unattendedMenu.style.right = "auto";
          const w = unattendedMenu.offsetWidth || 320;
          const left = Math.max(12, Math.min(window.innerWidth - 12 - w, rect.right - w));
          unattendedMenu.style.left = `${left}px`;
          try {
            await loadUnattendedCfgForSelected({ sid, openToken });
            if (unattendedMenuOpen && unattendedMenuToken === openToken && unattendedMenuSessionId === sid && selected === sid) {
              setUnattendedControlsDisabled(false);
              focusUnattendedInitialControl();
            }
          } catch (e) {
            if (unattendedMenuToken !== openToken || unattendedMenuSessionId !== sid || selected !== sid) return;
            console.error("load unattended mode failed", e);
            setToast(`unattended load error: ${e && e.message ? e.message : "unknown error"}`);
            setUnattendedControlsDisabled(false);
            hideUnattendedMenu({ restoreFocus: true });
          }
        }

        function toggleUnattendedMenu({ opener = null } = {}) {
          if (unattendedMenuOpen) hideUnattendedMenu({ restoreFocus: true });
          else showUnattendedMenu({ opener });
        }

        unattendedBtn.onclick = (e) => {
          e.preventDefault();
          e.stopPropagation();
          toggleUnattendedMenu({ opener: e.currentTarget });
        };
        unattendedMenu.onclick = (e) => e.stopPropagation();
        const onUnattendedKeydown = (e) => {
          if (e.key !== "Escape" || !unattendedMenuOpen) return;
          e.preventDefault();
          e.stopPropagation();
          hideUnattendedMenu({ restoreFocus: true });
        };
        const onDocClick = () => {
          if (unattendedMenuOpen) hideUnattendedMenu();
        };
        const onResize = () => {
          if (unattendedMenuOpen) hideUnattendedMenu();
        };
        addAppEvent(document, "keydown", onUnattendedKeydown, true);
        addAppEvent(document, "click", onDocClick);
        addAppEvent(window, "resize", onResize);
        const unattendedEnabledEl = $("#unattendedEnabled");
			        const unattendedCooldownEl = $("#unattendedCooldownMinutes");
			        const unattendedRemainingEl = $("#unattendedRemainingInjections");
			        const unattendedRequestEl = $("#unattendedRequest");
			        if (unattendedEnabledEl)
			          unattendedEnabledEl.onchange = (e) => {
			            if (!selected) return;
                  const requested = Boolean(e.target.checked);
                  unattendedCfg.enabled = requested && Number(unattendedCfg.remaining_injections) > 0;
                  if (requested && !unattendedCfg.enabled) setToast("increase injections before enabling unattended mode");
                  e.target.checked = unattendedCfg.enabled;
			            const s = sessionIndex.get(selected);
			            if (s) {
                  s.unattended_enabled = unattendedCfg.enabled;
                }
			            updateUnattendedBtnState();
			            scheduleUnattendedSave({ enabled: unattendedCfg.enabled });
			          };
        if (unattendedCooldownEl)
          unattendedCooldownEl.oninput = (e) => {
            if (!selected) return;
            unattendedNumberDraft.cooldown_minutes = String(e.target.value ?? "");
            unattendedNumberDirty.cooldown_minutes = true;
            const value = parseUnattendedDraftInt("cooldown_minutes");
            if (value === null) return;
            unattendedCfg.cooldown_minutes = value;
            scheduleUnattendedSave({ cooldown_minutes: value });
          };
        if (unattendedCooldownEl)
          unattendedCooldownEl.onblur = () => {
            if (parseUnattendedDraftInt("cooldown_minutes") !== null) return;
            restoreUnattendedNumberDraft("cooldown_minutes");
          };
        if (unattendedRemainingEl)
          unattendedRemainingEl.oninput = (e) => {
            if (!selected) return;
            unattendedNumberDraft.remaining_injections = String(e.target.value ?? "");
            unattendedNumberDirty.remaining_injections = true;
            const value = parseUnattendedDraftInt("remaining_injections");
            if (value === null) return;
            unattendedCfg.remaining_injections = value;
            const s = sessionIndex.get(selected);
            if (s) {
              s.unattended_remaining_injections = value;
              if (value <= 0) {
                unattendedCfg.enabled = false;
                const enabledEl = $("#unattendedEnabled");
                if (enabledEl) enabledEl.checked = false;
                s.unattended_enabled = false;
              }
            }
            updateUnattendedBtnState();
            scheduleUnattendedSave({ remaining_injections: value, ...(value <= 0 ? { enabled: false } : {}) });
          };
        if (unattendedRemainingEl)
          unattendedRemainingEl.onblur = () => {
            if (parseUnattendedDraftInt("remaining_injections") !== null) return;
            restoreUnattendedNumberDraft("remaining_injections");
          };
        if (unattendedRequestEl)
          unattendedRequestEl.oninput = (e) => {
            if (!selected) return;
            unattendedCfg.request = String(e.target.value ?? "");
            scheduleUnattendedSave({ request: unattendedCfg.request });
          };

        function voiceAnnouncementsEnabled() {
          return !!localAnnouncementEnabled;
        }

        function notificationsEnabledLocally() {
          return !!localNotificationEnabled;
        }

        function setAnnouncementEnabled(enabled) {
          localAnnouncementEnabled = !!enabled;
          if (localAnnouncementEnabled) storageSetItem("codoxear.announcementEnabled", "1");
          else storageRemoveItem("codoxear.announcementEnabled");
          if (!localAnnouncementEnabled) {
            stopAnnouncementHeartbeat();
            stopLiveAudioWatchdog();
            if (liveAudioRetryTimer) clearTimeout(liveAudioRetryTimer);
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
          if (localNotificationEnabled) {
            notificationFeedSinceTs = Date.now() / 1000;
          }
          updateVoiceUi();
        }

        function currentVoiceStreamUrl() {
          const streamUrl = voiceSettings && voiceSettings.audio && typeof voiceSettings.audio.stream_url === "string" && voiceSettings.audio.stream_url
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
          if (shouldPreferNativeLiveAudioPlayback()) {
            destroyLiveAudioHls();
            if (resetSource || liveAudioSourceUrl !== nextSrc || liveAudio.currentSrc !== nextSrc) {
              liveAudio.src = nextSrc;
              liveAudioSourceUrl = nextSrc;
            }
            return;
          }
          if (!browserSupportsMseLiveAudioPlayback()) {
            throw new Error("this browser does not support HLS audio playback in this app");
          }
          const HlsCtor = window.Hls;
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
          if (announcementHeartbeatTimer) clearInterval(announcementHeartbeatTimer);
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
          if (!browserSupportsLiveAudioPlayback()) return;
          if (!liveAudioHasReadySegments()) return;
          const now = Date.now();
          if ((now - liveAudioLastRestartTs) < LIVE_AUDIO_RESTART_THROTTLE_MS) return;
          liveAudioLastRestartTs = now;
          liveAudioStarted = false;
          liveAudioSuspectSinceTs = 0;
          updateVoiceUi();
          scheduleLiveAudioRetry(150, { resetSource: true });
        }

        function runLiveAudioWatchdog() {
          if (!localAnnouncementEnabled) return;
          if (!browserSupportsLiveAudioPlayback()) return;
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
          if ((now - baselineTs) < LIVE_AUDIO_STALL_GRACE_MS) return;
          queueLiveAudioHardRestart("watchdog");
        }

        function stopLiveAudioWatchdog() {
          if (liveAudioWatchdogTimer) clearInterval(liveAudioWatchdogTimer);
          liveAudioWatchdogTimer = null;
        }

        function startLiveAudioWatchdog() {
          runLiveAudioWatchdog();
          if (liveAudioWatchdogTimer) clearInterval(liveAudioWatchdogTimer);
          liveAudioWatchdogTimer = setInterval(() => {
            runLiveAudioWatchdog();
          }, LIVE_AUDIO_WATCHDOG_MS);
        }

        function startAnnouncementHeartbeat() {
          void sendAnnouncementHeartbeat(true);
          if (announcementHeartbeatTimer) clearInterval(announcementHeartbeatTimer);
          announcementHeartbeatTimer = setInterval(() => {
            void sendAnnouncementHeartbeat(true);
          }, 15000);
        }

        function resumeAnnouncementRuntime({ resetSource = false } = {}) {
          if (!localAnnouncementEnabled) return;
          startAnnouncementHeartbeat();
          startLiveAudioWatchdog();
          if (!liveAudioStarted && browserSupportsLiveAudioPlayback() && liveAudioHasReadySegments()) {
            scheduleLiveAudioRetry(150, { resetSource });
          }
        }

        function scheduleLiveAudioRetry(delayMs = 1200, { resetSource = true } = {}) {
          if (!localAnnouncementEnabled) return;
          if (liveAudioRetryTimer) clearTimeout(liveAudioRetryTimer);
          liveAudioRetryTimer = setTimeout(async () => {
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
          if (!browserSupportsLiveAudioPlayback()) return;
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
            notificationDeviceClass() === "mobile" &&
            notificationState.push_supported &&
            notificationState.permission === "granted" &&
            notificationState.notifications_enabled &&
            notificationState.endpoint
          );
        }

        function activeNotificationTransport() {
          if (!localNotificationEnabled) return "none";
          if (notificationDeviceClass() === "mobile") {
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
            if (typeof window.focus === "function") window.focus();
          } catch {}
          if (!sid) return;
          if (sessionIdFromHash() !== sid) setSessionHash(sid);
          void selectSessionFromHash({ refreshIfMissing: true, deferIfMissing: true }).catch((e) => {
            if (e && e.status === 401) handleAppAuthLoss();
            else console.error("desktop notification session select failed", e);
          });
        }

        function showDesktopNotification({ messageId, title, body, sessionId }) {
          if (!desktopNotificationsEnabled()) return;
          const id = String(messageId || "").trim();
          if (id && deliveredDesktopNotificationIds.has(id)) return;
          const sid = String(sessionId || "").trim();
          const safeTitle = String(title || "Session").trim() || "Session";
          const safeBody = String(body || "").replace(/\s+/g, " ").trim();
          if (!safeBody) return;
          try {
            const notification = new Notification(safeTitle, {
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
          } catch (e) {
            console.error("desktop notification failed", e);
          }
        }

        async function pollNotificationFeed({ prime = false } = {}) {
          if (appDisposed || !desktopNotificationsEnabled()) return;
          let maxSeen = notificationFeedSinceTs;
          try {
            const data = await api(`/api/notifications/feed?since=${encodeURIComponent(notificationFeedSinceTs)}`);
            if (appDisposed) return;
            const items = Array.isArray(data.items) ? data.items : [];
            for (const item of items) {
              const updatedTs = Number(item && item.updated_ts ? item.updated_ts : 0);
              if (updatedTs > maxSeen) maxSeen = updatedTs;
              if (prime) continue;
              showDesktopNotification({
                messageId: item && item.message_id,
                title: item && item.session_display_name,
                body: item && item.notification_text,
                sessionId: item && item.session_id,
              });
            }
          } catch (e) {
            console.error("notification feed poll failed", e);
            return;
          }
          notificationFeedSinceTs = maxSeen;
        }

        function maybeShowDesktopNotification(ev) {
          if (!ev || ev.role !== "assistant" || ev.pending) return;
          if (ev.message_class !== "final_response") return;
          if (!desktopNotificationsEnabled()) return;
          const messageId = typeof ev.message_id === "string" ? ev.message_id : "";
          const sessionId = typeof ev.session_id === "string" && ev.session_id.trim() ? ev.session_id.trim() : (selected || "");
          if (messageId && !ev.notification_text) {
            scheduleDesktopNotificationResolve({ ...ev, session_id: sessionId });
            return;
          }
          const s = sessionId ? sessionIndex.get(sessionId) : null;
          const title = s ? sessionDisplayName(s) : "Session";
          const body = String(ev.notification_text || ev.text || "").replace(/\s+/g, " ").trim();
          if (!body) return;
          showDesktopNotification({ messageId, title, body, sessionId });
        }

        function scheduleDesktopNotificationResolve(ev) {
          const messageId = typeof ev.message_id === "string" ? ev.message_id : "";
          if (!messageId || desktopNotificationTimers.has(messageId)) return;
          let attempts = 0;
          const tick = async () => {
            if (appDisposed || !desktopNotificationsEnabled()) {
              desktopNotificationTimers.delete(messageId);
              return;
            }
            attempts += 1;
            try {
              const data = await api(`/api/notifications/message?message_id=${encodeURIComponent(messageId)}`);
              if (appDisposed) {
                desktopNotificationTimers.delete(messageId);
                return;
              }
              const text = String(data.notification_text || "").trim();
              const summaryStatus = String(data.summary_status || "");
              if (text && (summaryStatus === "sent" || summaryStatus === "skipped" || summaryStatus === "error")) {
                desktopNotificationTimers.delete(messageId);
                maybeShowDesktopNotification({ ...ev, notification_text: text });
                return;
              }
            } catch (e) {
              if (!(e && e.status === 404)) console.error("desktop notification resolve failed", e);
            }
            if (attempts >= 20) {
              desktopNotificationTimers.delete(messageId);
              return;
            }
            const nextTimer = setTimeout(() => {
              desktopNotificationTimers.delete(messageId);
              void tick();
            }, 800);
            desktopNotificationTimers.set(messageId, nextTimer);
          };
          const firstTimer = setTimeout(() => {
            desktopNotificationTimers.delete(messageId);
            void tick();
          }, 800);
          desktopNotificationTimers.set(messageId, firstTimer);
        }

        function voiceSettingsDialogOpen() {
          return Boolean(voiceSettingsViewer && voiceSettingsViewer.style.display === "flex");
        }

        function syncVoiceSettingsFormFromState() {
          if (voiceBaseUrlInput) voiceBaseUrlInput.value = String(voiceSettings.tts_base_url || "");
          if (voiceApiKeyInput && !voiceApiKeyInput.matches(":focus")) {
            voiceApiKeyInput.value = "";
            voiceApiKeyInput.placeholder = voiceSettings.has_tts_api_key ? "Saved API key (leave blank to keep)" : "Enter API key";
          }
          if (voiceClearApiKeyToggle) voiceClearApiKeyToggle.checked = false;
          if (narrationSettingToggle) narrationSettingToggle.checked = !!voiceSettings.tts_enabled_for_narration;
        }

        function updateVoiceUi() {
          announceBtn.classList.toggle("active", voiceAnnouncementsEnabled());
          announceBtn.title = voiceAnnouncementsEnabled() ? "Announcements on" : "Announcements off";
          announceBtn.setAttribute("aria-label", announceBtn.title);
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
          if (!voiceSettingsDialogOpen()) syncVoiceSettingsFormFromState();
          notificationState.permission = typeof Notification === "undefined" ? "unsupported" : Notification.permission;
        }

        async function loadVoiceSettings() {
          const data = await api("/api/settings/voice");
          if (appDisposed) return data;
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
          if (localAnnouncementEnabled && !liveAudioStarted && browserSupportsLiveAudioPlayback() && liveAudioHasReadySegments()) {
            scheduleLiveAudioRetry(100, { resetSource: false });
          }
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
          if (voiceSaveTimer) clearTimeout(voiceSaveTimer);
          voiceSaveTimer = setTimeout(async () => {
            try {
              await saveVoiceSettings();
            } catch (e) {
              console.error("save voice settings failed", e);
              setToast(`voice settings error: ${e && e.message ? e.message : "unknown error"}`);
              try {
                await loadVoiceSettings();
              } catch (_error) {}
            }
          }, 250);
        }

        function versionedShellAssetPath(path) {
          const version = String(window.CODOXEAR_ASSET_VERSION || "").trim();
          if (!version) return path;
          return `${path}?v=${encodeURIComponent(version)}`;
        }

        async function ensureVoiceServiceWorker() {
          if (!("serviceWorker" in navigator) || !("PushManager" in window) || typeof Notification === "undefined") {
            throw new Error("push notifications are not supported in this browser");
          }
          if (!swRegistration) {
            swRegistration = await navigator.serviceWorker.register(resolveAppUrl(versionedShellAssetPath("/service-worker.js")), { scope: resolveAppUrl("/") });
          }
          return swRegistration;
        }

        async function syncNotificationState(serverSnapshot) {
          if (appDisposed) return;
          notificationState.desktop_supported = !!(window.isSecureContext && typeof Notification !== "undefined");
          notificationState.push_supported = !!(notificationState.desktop_supported && "serviceWorker" in navigator && "PushManager" in window);
          notificationState.permission = typeof Notification === "undefined" ? "unsupported" : Notification.permission;
          notificationState.desktop_enabled = storageGetItem("codoxear.desktopNotificationsEnabled") === "1";
          let snapshot = serverSnapshot;
          if (!snapshot) {
            try {
              snapshot = await api("/api/notifications/subscription");
            } catch (e) {
              if (!(e && e.status === 404)) throw e;
            }
          }
          if (appDisposed) return;
          let endpoint = "";
          if (notificationDeviceClass() === "mobile" && notificationState.push_supported) {
            try {
              const reg = await ensureVoiceServiceWorker();
              if (appDisposed) return;
              const sub = await reg.pushManager.getSubscription();
              if (appDisposed) return;
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
          if (Notification.permission !== "granted") {
            const permission = await Notification.requestPermission();
            if (permission !== "granted") {
              throw new Error(`notification permission ${permission}`);
            }
          }
          if (notificationDeviceClass() === "desktop") {
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
              applicationServerKey: base64UrlToUint8Array(publicKey),
            });
          }
          const snapshot = await api("/api/notifications/subscription", {
            method: "POST",
            body: {
              subscription: sub.toJSON(),
              user_agent: navigator.userAgent,
              device_label: "current-device",
              device_class: notificationDeviceClass(),
            },
          });
          await syncNotificationState(snapshot);
        }

        async function toggleCurrentDeviceNotifications(enabled) {
          if (!notificationState.desktop_supported) {
            throw new Error("notifications require HTTPS or localhost");
          }
          if (notificationDeviceClass() === "desktop") {
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
          if (!browserSupportsLiveAudioPlayback()) {
            throw new Error("this browser does not support HLS audio playback in this app");
          }
          if (!liveAudioHasReadySegments()) {
            throw new Error("no live audio segments are available yet; wait for the first announcement and try again");
          }
          const nextSrc = currentVoiceStreamUrl();
          await ensureLiveAudioPlaybackSource(nextSrc, { resetSource });
          await liveAudio.play();
          liveAudioStarted = true;
          markLiveAudioProgress();
          updateVoiceUi();
        }

        function describeLiveAudioStartError(error) {
          const message = error && error.message ? String(error.message) : "";
          if (/unsupported/i.test(message)) {
            if (!browserSupportsLiveAudioPlayback()) {
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
          voiceSettingsReturnFocusEl = document.activeElement instanceof HTMLElement ? document.activeElement : null;
          voiceSettingsBackdrop.style.display = "block";
          voiceSettingsViewer.style.display = "flex";
          updateVoiceUi();
          syncVoiceSettingsFormFromState();
          if (!voiceSettingsViewer.open) voiceSettingsViewer.showModal();
          afterModalVisibilityChanged();
        }

        function hideVoiceSettingsDialog() {
          const focusTarget = voiceSettingsReturnFocusEl;
          voiceSettingsReturnFocusEl = null;
          voiceSettingsBackdrop.style.display = "none";
          voiceSettingsViewer.style.display = "none";
          voiceSettingsStatus.textContent = "";
          if (voiceSettingsViewer.open) voiceSettingsViewer.close();
          afterModalVisibilityChanged();
          if (focusTarget && document.contains(focusTarget) && typeof focusTarget.focus === "function") {
            requestAnimationFrame(() => focusTarget.focus({ preventScroll: true }));
          }
        }

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
          const pending = notificationsEnabledLocally() && activeNotificationTransport() === "none";
          const next = pending ? true : !notificationsEnabledLocally();
          try {
            if (next) {
              setNotificationEnabledLocal(true);
              await enableNotificationsOnDevice();
            } else {
              await toggleCurrentDeviceNotifications(false);
              setNotificationEnabledLocal(false);
            }
          } catch (err) {
            console.error("notification toggle failed", err);
            setNotificationEnabledLocal(false);
            setToast(`notification error: ${err && err.message ? err.message : "unknown error"}`);
          }
        };
        liveAudio.addEventListener("error", () => {
          liveAudioStarted = false;
          liveAudioSuspectSinceTs = 0;
          updateVoiceUi();
          scheduleLiveAudioRetry(1200, { resetSource: true });
        });
        liveAudio.addEventListener("playing", () => {
          liveAudioStarted = true;
          markLiveAudioProgress();
          updateVoiceUi();
        });
        liveAudio.addEventListener("timeupdate", () => {
          markLiveAudioProgress();
        });
        liveAudio.addEventListener("waiting", () => {
          noteLiveAudioPotentialStall("waiting");
          runLiveAudioWatchdog();
        });
        liveAudio.addEventListener("stalled", () => {
          noteLiveAudioPotentialStall("stalled");
          runLiveAudioWatchdog();
        });
        liveAudio.addEventListener("suspend", () => {
          noteLiveAudioPotentialStall("suspend");
          runLiveAudioWatchdog();
        });
        liveAudio.addEventListener("ended", () => {
          liveAudioStarted = false;
          liveAudioSuspectSinceTs = 0;
          updateVoiceUi();
          scheduleLiveAudioRetry(500, { resetSource: true });
        });
        liveAudio.addEventListener("pause", () => {
          liveAudioStarted = false;
          liveAudioSuspectSinceTs = 0;
          updateVoiceUi();
        });
        narrationSettingToggle.onchange = (e) => {
          voiceSettings.tts_enabled_for_narration = Boolean(e.target.checked);
          scheduleVoiceSave();
        };
        voiceSettingsCloseBtn.onclick = hideVoiceSettingsDialog;
        $("#voiceSettingsCancelBtn").onclick = hideVoiceSettingsDialog;
        voiceSettingsBackdrop.onclick = hideVoiceSettingsDialog;
        voiceSettingsViewer.addEventListener("cancel", (e) => {
          e.preventDefault();
          hideVoiceSettingsDialog();
        });
        $("#voiceSettingsSaveBtn").onclick = async () => {
          try {
            voiceSettingsStatus.textContent = "Saving...";
            await saveVoiceSettings();
            await syncNotificationState();
            voiceSettingsStatus.textContent = "";
            hideVoiceSettingsDialog();
          } catch (e) {
            console.error("save voice settings failed", e);
            voiceSettingsStatus.textContent = `save error: ${e && e.message ? e.message : "unknown error"}`;
          }
        };
        function renderRecentCwdOptions() {
          const out = [];
          const seen = new Set();
          for (const raw of recentCwds) {
            const cwd = typeof raw === "string" ? raw.trim() : "";
            if (!cwd || seen.has(cwd)) continue;
            seen.add(cwd);
            out.push(cwd);
          }
          return out;
        }

        function filteredRecentCwdOptions() {
          const items = renderRecentCwdOptions();
          const query = String(newSessionCwdInput.value || "").trim();
          if (!query) return items.slice(0, 10).map((cwd, idx) => ({ cwd, idx, score: 1000 - idx }));
          return items
            .map((cwd, idx) => ({ cwd, idx, score: fuzzyRecentCwdScore(cwd, query) }))
            .filter((item) => item.score >= 0)
            .sort((a, b) => b.score - a.score || a.idx - b.idx || a.cwd.localeCompare(b.cwd))
            .slice(0, 10);
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
          newSessionCwdInput.value = String(cwd || "");
          setNewSessionCwdError("");
          syncNewSessionNamePlaceholder();
          newSessionCwdMenuOpen = false;
          newSessionCwdMenuFocus = -1;
          applyDialogMenus();
          scheduleNewSessionResumeLoad();
          newSessionCwdInput.focus();
          const end = newSessionCwdInput.value.length;
          try {
            newSessionCwdInput.setSelectionRange(end, end);
          } catch {}
        }

        function renderRecentCwdMenu() {
          newSessionCwdMenu.innerHTML = "";
          const raw = String(newSessionCwdInput.value || "").trim();
          const items = filteredRecentCwdOptions();
          if (newSessionCwdMenuFocus >= items.length) newSessionCwdMenuFocus = items.length ? items.length - 1 : -1;
          if (!items.length) {
            const emptyText = raw ? "No matching recent directories. Start still uses the typed path." : "No recent directories";
            newSessionCwdMenu.appendChild(el("div", { class: "pickerEmpty", text: emptyText }));
            newSessionCwdInput.removeAttribute("aria-activedescendant");
            return items;
          }
          for (const [idx, item] of items.entries()) {
            const cwd = item.cwd;
            const active = newSessionCwdMenuFocus === idx || (newSessionCwdMenuFocus < 0 && raw === cwd);
            const btn = el("button", {
              id: `newSessionCwdOption-${idx}`,
              class: "fileMenuItem" + (active ? " active" : ""),
              type: "button",
              role: "option",
              "aria-selected": active ? "true" : "false",
              title: cwd,
            });
            btn.appendChild(el("span", { class: "fileMenuPath", text: cwd }));
            btn.onmousedown = (e) => e.preventDefault();
            btn.onclick = () => applyNewSessionCwdSuggestion(cwd);
            newSessionCwdMenu.appendChild(btn);
          }
          if (newSessionCwdMenuFocus >= 0) newSessionCwdInput.setAttribute("aria-activedescendant", `newSessionCwdOption-${newSessionCwdMenuFocus}`);
          else newSessionCwdInput.removeAttribute("aria-activedescendant");
          return items;
        }

        function syncNewSessionNamePlaceholder() {
          const fallback = baseName(String(newSessionCwdInput.value || "").trim());
          newSessionNameInput.placeholder = fallback || "session-name";
        }

        function newSessionResumeLabel(item) {
          if (!item || typeof item !== "object") return "Start fresh";
          const alias = typeof item.alias === "string" ? item.alias.trim() : "";
          const firstUser = typeof item.first_user_message === "string" ? item.first_user_message.trim() : "";
          const primary = alias || firstUser || shortSessionId(item.session_id);
          const ts = Number(item.updated_ts || 0);
          const age = ts > 0 ? fmtRelativeAge(Math.max(0, Date.now() / 1000 - ts)) : "";
          return `${age ? `${age} | ` : ""}${primary}`;
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
          newSessionResumeSelection = item && typeof item === "object" ? item : null;
          setPickerButtonContent(
            newSessionResumeBtn,
            newSessionResumeSelection ? newSessionResumeLabel(newSessionResumeSelection) : "Start fresh",
            "",
            !newSessionResumeSelection
          );
          syncNewSessionWorktreeUi();
        }

        function renderNewSessionBackendTabs() {
          newSessionBackendTabs.innerHTML = "";
          for (const backend of ["codex", "pi", "cc"]) {
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
        }

        function newSessionProviderChoices() {
          return providerChoicesForBackend(newSessionBackend);
        }

        function newSessionHasProviderChoices() {
          return newSessionProviderChoices().length > 0;
        }

        function defaultNewSessionProviderChoice() {
          const choices = newSessionProviderChoices();
          if (!choices.length) return "";
          const defaults = defaultsForAgentBackend(newSessionBackend);
          const configured = typeof defaults.provider_choice === "string" ? defaults.provider_choice.trim() : "";
          const remembered = loadRememberedProviderChoice(newSessionBackend);
          if (remembered && choices.includes(remembered)) return remembered;
          if (configured && choices.includes(configured)) return configured;
          if (newSessionProvider && choices.includes(newSessionProvider)) return newSessionProvider;
          return choices[0] || "";
        }

        function newSessionProviderModelDisplay(model, providerChoice = "") {
          return providerModelDisplay(model, providerChoice, {
            hasProviderChoices: newSessionHasProviderChoices(),
            allowCustomProvider: newSessionAllowsCustomProvider(),
          });
        }

        function newSessionAllowsCustomProvider() {
          return newSessionBackend === "pi";
        }

        function parseNewSessionProviderModelInput(value = newSessionModelInput.value) {
          const raw = String(value || "").trim();
          const choices = newSessionProviderChoices();
          const allowCustomProvider = newSessionAllowsCustomProvider();
          const hasProviders = choices.length > 0 || allowCustomProvider;
          const defaults = defaultsForAgentBackend(newSessionBackend);
          const fallbackModel = typeof defaults.model === "string" && defaults.model.trim() ? defaults.model.trim() : "default";
          let providerChoice = hasProviders ? defaultNewSessionProviderChoice() : "";
          let model = raw || fallbackModel;
          let providerError = "";
          const providerAbsent = Boolean(newSessionLaunchPresetProviderAbsent && raw && raw === newSessionLiteralModelInputValue);
          if (providerAbsent) providerChoice = "";
          if (hasProviders && raw.includes("/") && raw !== newSessionLiteralModelInputValue) {
            const slash = raw.indexOf("/");
            const typedProvider = raw.slice(0, slash).trim();
            const typedModel = raw.slice(slash + 1).trim();
            if (typedProvider && (choices.includes(typedProvider) || allowCustomProvider)) {
              providerChoice = typedProvider;
            } else if (typedProvider) {
              providerError = `Provider must be one of ${choices.join(", ")}.`;
            }
            model = typedModel || fallbackModel;
          }
          return { providerChoice, model: model || "default", providerError, providerAbsent };
        }

        function rememberedNewSessionProviderModelChoice() {
          const remembered = loadRememberedProviderModelChoice(newSessionBackend);
          if (!remembered) return null;
          const absent = rememberedProviderModelAbsentChoice(remembered);
          if (absent) return absent;
          const parsed = parseNewSessionProviderModelInput(remembered);
          if (parsed.providerError) return null;
          const choices = newSessionProviderChoices();
          if (choices.length && parsed.providerChoice && !choices.includes(parsed.providerChoice) && !newSessionAllowsCustomProvider()) return null;
          return parsed;
        }

        function newSessionDefaultsWarningText() {
          const warnings = newSessionDefaults && typeof newSessionDefaults === "object" && newSessionDefaults.warnings && typeof newSessionDefaults.warnings === "object" ? newSessionDefaults.warnings : null;
          if (!warnings) return "";
          const names = Object.keys(warnings).map(agentBackendDisplayName).filter(Boolean);
          if (!names.length) return "";
          return `Launch defaults degraded for ${names.join(", ")}; using safe defaults.`;
        }

        function clearNewSessionProviderModelError() {
          newSessionModelField.classList.remove("error");
          if (String(newSessionStatus.textContent || "").startsWith("Provider must be one of ")) {
            newSessionStatus.textContent = newSessionDefaultsWarningText();
          }
        }

        function syncNewSessionProviderFromModelInput() {
          const parsed = parseNewSessionProviderModelInput();
          newSessionModelField.classList.toggle("error", Boolean(parsed.providerError));
          if (!parsed.providerError) clearNewSessionProviderModelError();
          if (parsed.providerChoice && !parsed.providerError && parsed.providerChoice !== newSessionProvider) {
            setNewSessionProvider(parsed.providerChoice);
          }
          return parsed;
        }

        function currentNewSessionModelForCapabilities() {
          const parsed = parseNewSessionProviderModelInput();
          const model = parsed.model;
          return model && model.toLowerCase() !== "default" ? model : null;
        }

        function currentReasoningChoices() {
          const parsed = parseNewSessionProviderModelInput();
          return reasoningChoicesForBackend(newSessionBackend, {
            provider: parsed.providerAbsent ? "" : parsed.providerChoice || newSessionProvider,
            model: currentNewSessionModelForCapabilities(),
          });
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
          const options = providerChoicesForBackend(newSessionBackend);
          const fallback = String(defaultsForAgentBackend(newSessionBackend).provider_choice || "").trim();
          const next = String(value || "").trim();
          newSessionProvider = options.includes(next) || (next && newSessionAllowsCustomProvider()) ? next : (fallback && options.includes(fallback) ? fallback : options[0] || "");
          rememberProviderChoice(newSessionBackend, newSessionProvider);
          setNewSessionReasoningEffort(newSessionReasoningEffort);
          renderNewSessionReasoningMenu();
        }

        function newSessionModelOption(model, { providerChoice = "", recent = false, configured = false, providerAbsent = false } = {}) {
          const cleanModel = String(model || "").trim() || "default";
          const cleanProvider = providerAbsent ? "" : String(providerChoice || "").trim();
          const displayText = newSessionProviderModelDisplay(cleanModel, cleanProvider);
          return {
            model: cleanModel,
            providerChoice: cleanProvider,
            providerAbsent: !!providerAbsent,
            recent: !!recent,
            configured: !!configured,
            displayText,
            searchText: cleanProvider ? `${cleanProvider}/${cleanModel} ${cleanModel}` : cleanModel,
          };
        }

        function addNewSessionModelOption(out, seen, model, opts = {}) {
          const cleanModel = String(model || "").trim();
          if (!cleanModel) return;
          const cleanProvider = String(opts.providerChoice || "").trim();
          const key = `${cleanProvider}|${cleanModel}`;
          if (seen.has(key)) return;
          seen.add(key);
          out.push(newSessionModelOption(cleanModel, opts));
        }

        function sessionModelOptions() {
          const seen = new Set();
          const out = [];
          const defaults = defaultsForAgentBackend(newSessionBackend);
          const providerChoices = newSessionProviderChoices();
          const configuredDefault = typeof defaults.model === "string" ? defaults.model.trim() : "";
          const activeProvider = providerChoices.length ? defaultNewSessionProviderChoice() : "";
          if (configuredDefault) addNewSessionModelOption(out, seen, configuredDefault, { providerChoice: activeProvider, configured: true });
          for (const item of latestSessions) {
            if (sessionAgentBackend(item) !== newSessionBackend) continue;
            const model = typeof item.model === "string" ? item.model.trim() : "";
            if (!model) continue;
            const provider = sessionProviderChoice(item);
            const providerChoice = providerChoices.includes(provider) || (provider && newSessionAllowsCustomProvider()) ? provider : "";
            const providerAbsent = newSessionBackend === "pi" && !providerChoice && !(typeof item.model_provider === "string" && item.model_provider.trim());
            addNewSessionModelOption(out, seen, model, { providerChoice, providerAbsent, recent: true });
          }
          const configuredModels = Array.isArray(defaults.models) ? defaults.models : [];
          if (providerChoices.length) {
            for (const providerChoice of providerChoices) {
              for (const value of configuredModels) addNewSessionModelOption(out, seen, value, { providerChoice, configured: true });
            }
          } else {
            for (const value of configuredModels) addNewSessionModelOption(out, seen, value, { configured: true });
          }
          if (!out.length) addNewSessionModelOption(out, seen, "default", { providerChoice: activeProvider, configured: true });
          return out;
        }

        function filteredNewSessionModelOptions() {
          const query = String(newSessionModelInput.value || "").trim().toLowerCase();
          const options = sessionModelOptions();
          if (!query) return options.slice(0, 12);
          const exact = options.filter((item) => String(item.model || "").toLowerCase() === query || String(item.searchText || "").toLowerCase() === query);
          const prefix = options.filter((item) => !exact.includes(item) && String(item.searchText || item.model || "").toLowerCase().startsWith(query));
          const contains = options.filter((item) => !exact.includes(item) && !prefix.includes(item) && modelOptionMatches(item, query));
          return exact.concat(prefix, contains).slice(0, 12);
        }

        function setNewSessionReasoningEffort(value) {
          const choices = currentReasoningChoices();
          const next = String(value || "").trim().toLowerCase();
          const fallback = String(defaultsForAgentBackend(newSessionBackend).reasoning_effort || "").trim().toLowerCase();
          newSessionReasoningEffort = choices.includes(next) ? next : (choices.includes(fallback) ? fallback : choices[0] || "high");
          setPickerButtonContent(newSessionReasoningBtn, newSessionReasoningEffort);
        }

        function setNewSessionFast(value) {
          newSessionFast = !!value;
          newSessionFastToggle.checked = newSessionFast;
        }

        function syncNewSessionCwdHint() {
          const errorText = String(newSessionCwdError || "").trim();
          const hintText = !errorText && newSessionCwdInfo && newSessionCwdInfo.will_create ? "Directory will be created when you start the session." : "";
          const text = errorText || hintText;
          newSessionCwdField.classList.toggle("error", !!errorText);
          newSessionCwdHint.classList.toggle("danger", !!errorText);
          newSessionCwdHint.textContent = text;
        }

        function setNewSessionCwdError(message) {
          newSessionCwdError = String(message || "").trim();
          syncNewSessionCwdHint();
        }

        function clearNewSessionCwdInfo() {
          newSessionCwdInfo = { exists: false, will_create: false, git_repo: false, git_root: "", git_branch: "" };
          syncNewSessionCwdHint();
        }

        function syncNewSessionTmuxUi() {
          if (!tmuxAvailable) newSessionTmuxToggle.checked = false;
          newSessionTmuxToggle.disabled = !tmuxAvailable;
          newSessionTmuxField.style.opacity = tmuxAvailable ? "1" : "0.58";
        }

        function syncNewSessionWorktreeUi() {
          const canOffer = !!(newSessionCwdInfo && newSessionCwdInfo.git_repo) && !newSessionResumeSelection;
          if (!canOffer) newSessionWorktreeToggle.checked = false;
          const enabled = canOffer && !!newSessionWorktreeToggle.checked;
          newSessionWorktreeField.style.display = canOffer ? "" : "none";
          newSessionWorktreeInput.disabled = !enabled;
          newSessionWorktreeInput.style.display = enabled ? "" : "none";
          if (newSessionResumeSelection) newSessionStartBtn.textContent = "Resume session";
          else if (enabled) newSessionStartBtn.textContent = "Create worktree session";
          else newSessionStartBtn.textContent = "Start session";
        }

        function renderNewSessionResumeMenu() {
          newSessionResumeMenu.innerHTML = "";
          const freshBtn = el("button", {
            class: "fileMenuItem" + (!newSessionResumeSelection ? " active" : ""),
            type: "button",
            title: "Start a new conversation",
          });
          freshBtn.appendChild(el("span", { class: "fileMenuPath", text: "Start fresh" }));
          freshBtn.onclick = () => {
            setNewSessionResumeSelection(null);
            newSessionResumeMenuOpen = false;
            applyDialogMenus();
          };
          newSessionResumeMenu.appendChild(freshBtn);
          if (!newSessionResumeCandidates.length) {
            newSessionResumeMenu.appendChild(el("div", { class: "pickerEmpty", text: "No matching sessions" }));
            return;
          }
          for (const item of newSessionResumeCandidates) {
            const btn = el("button", {
              class: "fileMenuItem" + (newSessionResumeSelection && newSessionResumeSelection.session_id === item.session_id ? " active" : ""),
              type: "button",
              title: newSessionResumeLabel(item),
            });
            btn.appendChild(el("span", { class: "fileMenuPath", text: newSessionResumeLabel(item) }));
            btn.onclick = () => {
              setNewSessionResumeSelection(item);
              newSessionResumeMenuOpen = false;
              applyDialogMenus();
            };
            newSessionResumeMenu.appendChild(btn);
          }
        }

        function selectNewSessionModel(option) {
          newSessionLiteralModelInputValue = "";
          newSessionLaunchPresetProviderAbsent = false;
          const item = option && typeof option === "object" ? option : newSessionModelOption(option || "default");
          const selectedProvider = item.providerAbsent ? "" : item.providerChoice || newSessionProvider;
          if (item.providerChoice && !item.providerAbsent && newSessionProviderChoices().includes(item.providerChoice)) {
            setNewSessionProvider(item.providerChoice);
          }
          newSessionModelInput.value = newSessionProviderModelDisplay(item.model || "default", selectedProvider);
          if (item.providerAbsent) {
            newSessionLiteralModelInputValue = newSessionModelInput.value;
            newSessionLaunchPresetProviderAbsent = true;
          }
          rememberProviderModelChoice(newSessionBackend, selectedProvider, item.model || "default", { providerAbsent: Boolean(item.providerAbsent) });
          newSessionModelField.classList.remove("error");
          newSessionModelMenuOpen = false;
          newSessionModelMenuFocus = -1;
          setNewSessionReasoningEffort(newSessionReasoningEffort);
          renderNewSessionReasoningMenu();
          applyDialogMenus();
          newSessionModelInput.focus();
          const end = newSessionModelInput.value.length;
          try {
            newSessionModelInput.setSelectionRange(end, end);
          } catch (_) {}
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
              title,
            });
            btn.appendChild(el("span", { class: "fileMenuPath", text: title }));
            if (item.providerChoice) btn.appendChild(el("span", { class: "fileMenuHint", text: item.recent ? "Recent" : item.configured ? "Configured" : item.providerChoice }));
            btn.onmousedown = (e) => e.preventDefault();
            btn.onclick = () => selectNewSessionModel(item);
            newSessionModelMenu.appendChild(btn);
          }
          if (newSessionModelMenuFocus >= 0) newSessionModelInput.setAttribute("aria-activedescendant", `newSessionModelOption-${newSessionModelMenuFocus}`);
          else newSessionModelInput.removeAttribute("aria-activedescendant");
          return items;
        }

        async function loadNewSessionResumeCandidates(cwd) {
          const raw = String(cwd || "").trim();
          const seq = ++newSessionResumeLoadSeq;
          const backend = newSessionBackend;
          if (!raw) {
            setNewSessionCwdError("");
            newSessionResumeCandidates = [];
            setNewSessionResumeSelection(null);
            clearNewSessionCwdInfo();
            renderNewSessionResumeMenu();
            syncNewSessionWorktreeUi();
            return;
          }
          try {
            const res = await api(`/api/session_resume_candidates?cwd=${encodeURIComponent(raw)}&agent_backend=${encodeURIComponent(backend)}`);
            if (seq !== newSessionResumeLoadSeq) return;
            newSessionCwdInfo = {
              exists: !!(res && res.exists),
              will_create: !!(res && res.will_create),
              git_repo: !!(res && res.git_repo),
              git_root: res && typeof res.git_root === "string" ? res.git_root : "",
              git_branch: res && typeof res.git_branch === "string" ? res.git_branch : "",
            };
            setNewSessionCwdError("");
            const items = Array.isArray(res && res.sessions) ? res.sessions.filter((item) => item && typeof item === "object" && typeof item.session_id === "string") : [];
            newSessionResumeCandidates = items;
            const currentId = newSessionResumeSelection && typeof newSessionResumeSelection.session_id === "string" ? newSessionResumeSelection.session_id : "";
            const next = currentId ? items.find((item) => item.session_id === currentId) || null : null;
            setNewSessionResumeSelection(next);
            renderNewSessionResumeMenu();
            syncNewSessionWorktreeUi();
          } catch (e) {
            if (seq !== newSessionResumeLoadSeq) return;
            newSessionResumeCandidates = [];
            setNewSessionResumeSelection(null);
            clearNewSessionCwdInfo();
            if (e && e.obj && e.obj.field === "cwd") setNewSessionCwdError(e.message);
            renderNewSessionResumeMenu();
            syncNewSessionWorktreeUi();
          }
        }

        function scheduleNewSessionResumeLoad() {
          if (newSessionResumeLoadTimer) clearTimeout(newSessionResumeLoadTimer);
          const cwd = String(newSessionCwdInput.value || "").trim();
          newSessionResumeLoadTimer = setTimeout(() => {
            newSessionResumeLoadTimer = null;
            void loadNewSessionResumeCandidates(cwd);
          }, 180);
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
          if (!s || typeof s !== "object") return "";
          const backend = sessionAgentBackend(s);
          const provider = typeof s.model_provider === "string" ? s.model_provider.trim() : "";
          if (backend === "pi") return provider;
          if (backend === "cc") return "";
          const explicit = typeof s.provider_choice === "string" ? s.provider_choice.trim() : "";
          if (explicit) return explicit;
          if (!provider) return "";
          if (backend === "codex" && provider === "openai") {
            const auth = typeof s.preferred_auth_method === "string" ? s.preferred_auth_method.trim() : "";
            return auth === "chatgpt" ? "chatgpt" : "openai-api";
          }
          return provider;
        }

        function applyNewSessionLaunchPreset(sessionInfo) {
          const s = sessionInfo && typeof sessionInfo === "object" ? sessionInfo : null;
          if (!s) return false;
          const backend = sessionAgentBackend(s);
          if (backend !== newSessionBackend) setNewSessionBackend(backend, { resetSelections: true });
          const provider = launchPresetProviderChoice(s);
          const providerChoices = newSessionProviderChoices();
          const acceptsProvider = Boolean(provider && (providerChoices.includes(provider) || newSessionAllowsCustomProvider()));
          if (acceptsProvider) setNewSessionProvider(provider);
          const model = typeof s.model === "string" && s.model.trim() ? s.model.trim() : "";
          const providerAbsent = backend === "pi" && !provider;
          newSessionLiteralModelInputValue = "";
          newSessionLaunchPresetProviderAbsent = false;
          if (model || providerAbsent || acceptsProvider) {
            newSessionModelInput.value = newSessionProviderModelDisplay(model || "default", acceptsProvider ? provider : "");
            if (!acceptsProvider) {
              newSessionLiteralModelInputValue = newSessionModelInput.value;
              newSessionLaunchPresetProviderAbsent = providerAbsent;
            }
          }
          clearNewSessionProviderModelError();
          const reasoning = typeof s.reasoning_effort === "string" ? s.reasoning_effort.trim().toLowerCase() : "";
          if (reasoning) setNewSessionReasoningEffort(reasoning);
          const defaults = defaultsForAgentBackend(newSessionBackend);
          if (defaults && defaults.supports_fast) setNewSessionFast(String(s.service_tier || "").trim().toLowerCase() === "fast");
          if (tmuxAvailable) newSessionTmuxToggle.checked = Boolean(s.transport === "tmux" || s.tmux_session || s.tmux_window);
          renderNewSessionReasoningMenu();
          renderNewSessionModelMenu();
          return true;
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
          newSessionResumeCandidates = [];
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
          const resumeSessionId = newSessionResumeSelection && newSessionResumeSelection.session_id ? newSessionResumeSelection.session_id : null;
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
          identityHint: (entry, duplicatePaths, options) => filePickerIdentityHint(entry, duplicatePaths, options),
          titleForEntry: (entry, hint) => filePickerTitle(entry, hint),
          normalizeFileApiPath: (value) => normalizeFileApiPath(value),
          activeIdentity: () => currentActiveFileIdentity(),
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
          confirmReload: (message) => window.confirm(message),
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
          renderMonacoDiff: (rel, originalText, modifiedText, lineNumber, request) => renderMonacoDiff(rel, originalText, modifiedText, lineNumber, request),
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
          if (e.key !== "Escape") return;
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
            hideSendChoice({ restoreFocus: true });
            return;
          }
          if (queueViewer.style.display === "flex") hideQueueViewer();
          if (helpViewer.style.display === "flex") hideHelpViewer();
          if (diagViewer.style.display === "flex") hideDiagViewer();
          if (voiceSettingsViewer.style.display === "flex") hideVoiceSettingsDialog();
          if (editViewer.style.display === "flex") hideEditSession();
          if (newSessionViewer.style.display === "flex") hideNewSessionDialog();
        });

        let sendChoicePending = null;
        let sendChoiceReturnFocusEl = null;
        function syncSendChoiceAttachmentPolicy() {
          const laterBtn = $("#sendChoiceLater");
          if (!laterBtn) return;
          const hasAttachments = Boolean(sendChoicePending && sendChoicePending.attachmentCount > 0);
          const laterLabel = hasAttachments ? "Attachments cannot be queued; send now or wait until idle" : "Send after current";
          laterBtn.disabled = hasAttachments;
          laterBtn.title = laterLabel;
          laterBtn.setAttribute("aria-label", laterLabel);
        }
        function focusSendChoiceInitial() {
          requestAnimationFrame(() => {
            if (sendChoice.style.display !== "flex") return;
            const laterBtn = $("#sendChoiceLater");
            const nowBtn = $("#sendChoiceNow");
            const cancelBtn = $("#sendChoiceCancel");
            const target = laterBtn && !laterBtn.disabled ? laterBtn : nowBtn && !nowBtn.disabled ? nowBtn : cancelBtn;
            if (!target || typeof target.focus !== "function") return;
            try {
              target.focus({ preventScroll: true });
            } catch {}
          });
        }
        function showSendChoice(raw, { opener = null } = {}) {
          prepareModalOpen();
          sendChoiceReturnFocusEl = opener instanceof HTMLElement ? opener : document.activeElement instanceof HTMLElement ? document.activeElement : null;
          sendChoicePending = { sid: selected, text: raw, attachmentCount: attachedFiles };
          syncSendChoiceAttachmentPolicy();
          sendChoiceBackdrop.style.display = "block";
          sendChoice.style.display = "flex";
          afterModalVisibilityChanged();
          focusSendChoiceInitial();
        }
        function hideSendChoice({ restoreFocus = false } = {}) {
          const target = sendChoiceReturnFocusEl;
          sendChoiceReturnFocusEl = null;
          sendChoicePending = null;
          syncSendChoiceAttachmentPolicy();
          sendChoiceBackdrop.style.display = "none";
          sendChoice.style.display = "none";
          afterModalVisibilityChanged();
          if (restoreFocus) restoreModalFocus(target, () => sendChoice.style.display === "flex");
        }
        const sendChoiceNowBtn = $("#sendChoiceNow");
        const sendChoiceLaterBtn = $("#sendChoiceLater");
        const sendChoiceCancelBtn = $("#sendChoiceCancel");
        if (sendChoiceNowBtn)
          sendChoiceNowBtn.onclick = async () => {
            const raw = sendChoicePending && sendChoicePending.text;
            const sid = sendChoicePending && sendChoicePending.sid;
            hideSendChoice({ restoreFocus: true });
            if (!raw || !sid) return;
            const ok = await sendText(raw, { sid });
            if (ok && sid === selected && $("#msg").value === raw) clearComposer();
          };
        if (sendChoiceLaterBtn)
          sendChoiceLaterBtn.onclick = async () => {
            const raw = sendChoicePending && sendChoicePending.text;
            const sid = sendChoicePending && sendChoicePending.sid;
            const hasAttachments = Boolean(sendChoicePending && sendChoicePending.attachmentCount > 0);
            if (hasAttachments) {
              setToast("attachments can only be sent now; wait until idle to queue text with files");
              return;
            }
            hideSendChoice({ restoreFocus: true });
            if (!raw || !sid) return;
            const ok = await enqueueComposerText(raw, { sid });
            if (ok && sid === selected && $("#msg").value === raw) clearComposer();
          };
        if (sendChoiceCancelBtn)
          sendChoiceCancelBtn.onclick = () => {
            hideSendChoice({ restoreFocus: true });
          };
        sendChoiceBackdrop.onclick = () => hideSendChoice({ restoreFocus: true });

        const queueUpdateTimers = new Map();
        const queueMutationLocks = new Set();
        const queuePendingDeletes = new Set();
        const queueDraftTexts = new Map();
        let queueLastEditMs = 0;
        let queueSubmitBusy = false;
        let queueViewerSid = null;
        let queueViewerItems = [];

        function selectedSessionHasUnknownSend() {
          return sessionHasUnknownSend(selected ? sessionIndex.get(selected) : null);
        }

        function selectedSessionIsOrphanRecovery() {
          return sessionIsOrphanRecovery(selected ? sessionIndex.get(selected) : null);
        }

        function selectedSessionHasOrphanQueueRecovery() {
          return sessionHasOrphanQueueRecovery(selected ? sessionIndex.get(selected) : null);
        }

        function selectedSessionLaunchFailed() {
          return sessionLaunchFailed(selected ? sessionIndex.get(selected) : null);
        }

        function syncQueueSubmitState() {
          const queueControl = $("#queueBtn");
          if (!queueControl) return;
          const unknownSend = selectedSessionHasUnknownSend();
          const orphanQueueRecovery = selectedSessionHasOrphanQueueRecovery();
          const launchFailed = selectedSessionLaunchFailed();
          queueControl.disabled = !!queueSubmitBusy || !selected || launchFailed || (unknownSend && !orphanQueueRecovery);
          const queueLabel = !selected
            ? "Select a session to view queued messages"
            : launchFailed
              ? "Failed launch cannot receive queued messages"
              : orphanQueueRecovery
                ? "Review preserved queued recovery items"
                : unknownSend
                  ? "Resolve the unknown send before queueing"
                  : "Queued messages";
          queueControl.title = queueLabel;
          queueControl.setAttribute("aria-label", queueLabel);
        }

        function syncSendButtonState() {
          const sendControl = $("#sendBtn");
          if (!sendControl) return;
          const unknownSend = selectedSessionHasUnknownSend();
          const orphanRecovery = selectedSessionIsOrphanRecovery();
          const recoveryQueue = selectedSessionHasOrphanQueueRecovery();
          const launchFailed = selectedSessionLaunchFailed();
          sendControl.disabled = !!sending || !selected || launchFailed || unknownSend || orphanRecovery || recoveryQueue;
          const sendLabel = !selected ? "Select a session to send" : launchFailed ? "Failed launch cannot receive messages" : unknownSend ? "Resolve the unknown send before sending" : orphanRecovery ? "Missing session can only be reviewed" : recoveryQueue ? "Review preserved queued recovery items before sending" : "Send";
          sendControl.title = sendLabel;
          sendControl.setAttribute("aria-label", sendLabel);
        }

        async function enqueueComposerText(raw, { sid = null } = {}) {
          const sessionId = sid || selected;
          const text = String(raw || "");
          if (!sessionId || !text.trim()) return false;
          const sessionInfo = sessionIndex.get(sessionId) || null;
          if (sessionInfo && sessionLaunchFailed(sessionInfo)) {
            setToast("failed launch cannot receive queued messages");
            return false;
          }
          if (sessionInfo && sessionInfo.orphan_recovery) {
            setToast("missing session can only be reviewed");
            return false;
          }
          if (sessionInfo && sessionInfo.queue_recovery) {
            setToast("review preserved queue before queueing");
            return false;
          }
          if (sessionInfo && sessionInfo.commit_unknown_send) {
            setToast("resolve the unknown send before queueing");
            void clearCommitUnknownSend(sessionId, sessionInfo.commit_unknown_send_text || "");
            return false;
          }
          if (queueSubmitBusy) return false;
          queueSubmitBusy = true;
          syncQueueSubmitState();
          try {
            const res = await api(`/api/sessions/${sessionId}/enqueue`, { method: "POST", body: { text } });
            const qn = res && typeof res.queue_len === "number" ? res.queue_len : null;
            if (res && res.commit_unknown) setToast("send status unknown; queued item needs review");
            else if (res && res.queued) setToast(`queued (${qn ?? "?"})`);
            else setToast("sent");
            pollFastUntilMs = Date.now() + 5000;
            kickPoll(0);
            await refreshSessions();
            updateQueueBadge();
            syncRecoveryUiForSession(sessionId);
            if (queueViewer.style.display === "flex" && (queueViewerSid || selected) === sessionId) {
              await refreshQueueViewer();
            }
            return true;
          } catch (e) {
            if (e && e.status === 401) {
              handleAppAuthLoss();
              return false;
            }
            setToast(`queue error: ${e && e.message ? e.message : "unknown error"}`);
            return false;
          } finally {
            queueSubmitBusy = false;
            syncQueueSubmitState();
          }
        }

        async function deleteQueueItem(sid, itemId) {
          const key = String(itemId || "");
          if (!sid || !key) return;
          const item = queueViewerItems.find((candidate) => String(candidate && candidate.id || "") === key) || null;
          const commitUnknown = Boolean(item && item.commitUnknown);
          const orphanRecovery = Boolean(item && item.orphanRecovery);
          if (commitUnknown || orphanRecovery) {
            const text = String(item && item.text || "").trim();
            const suffix = text ? `\n\nQueued prompt: ${text.slice(0, 240)}${text.length > 240 ? "..." : ""}` : "";
            const confirmed = window.confirm(
              `Delete this recovery item only after checking the transcript or terminal.${commitUnknown ? " This may allow later queued prompts to send." : ""}${suffix}`
            );
            if (!confirmed) return;
          }
          const timerKey = `${sid}:${key}`;
          const pendingUpdate = queueUpdateTimers.get(timerKey);
          if (pendingUpdate) {
            clearTimeout(pendingUpdate);
            queueUpdateTimers.delete(timerKey);
          }
          if (queueMutationLocks.has(key)) {
            queuePendingDeletes.add(key);
            setToast("delete queued");
            return;
          }
          queueLastEditMs = 0;
          queuePendingDeletes.delete(key);
          queueMutationLocks.add(key);
          queueViewerItems = queueViewerItems.filter((item) => String(item.id || "") !== key);
          queueDraftTexts.delete(key);
          renderQueueList();
          try {
            await api(`/api/sessions/${sid}/queue/delete`, { method: "POST", body: { id: key, allow_commit_unknown: commitUnknown, allow_orphan_recovery: orphanRecovery } });
            await refreshSessions();
            updateQueueBadge();
            syncRecoveryUiForSession(sid);
            if (queueViewer.style.display === "flex") {
              const refreshedSession = sessionIndex.get(sid);
              if (refreshedSession && Number(refreshedSession.queue_len || 0) > 0) await refreshQueueViewer();
              else hideQueueViewer();
            }
          } catch (e) {
            if (e && e.status === 401) {
              handleAppAuthLoss();
              return;
            }
            await refreshQueueViewer();
            setToast(`queue delete error: ${e && e.message ? e.message : "unknown error"}`);
          } finally {
            queueMutationLocks.delete(key);
          }
        }

        async function moveQueueItem(sid, itemId, toIndex) {
          const key = String(itemId || "");
          if (!sid || !key) return;
          if (queueMutationLocks.has(key)) {
            setToast("queue item busy; retry in a moment");
            return;
          }
          queueMutationLocks.add(key);
          try {
            await api(`/api/sessions/${sid}/queue/move`, { method: "POST", body: { id: key, to_index: toIndex } });
            await refreshQueueViewer();
            await refreshSessions();
            updateQueueBadge();
            syncRecoveryUiForSession(sid);
          } catch (e) {
            if (e && e.status === 401) {
              handleAppAuthLoss();
              return;
            }
            setToast(`queue move error: ${e && e.message ? e.message : "unknown error"}`);
          } finally {
            queueMutationLocks.delete(key);
          }
        }

        function scheduleQueueUpdate(sid, itemId, text) {
          if (!sid) return;
          const itemKey = String(itemId || "");
          if (!itemKey) return;
          if (!String(text || "").trim()) {
            const key0 = `${sid}:${itemKey}`;
            const existing0 = queueUpdateTimers.get(key0);
            if (existing0) clearTimeout(existing0);
            queueUpdateTimers.delete(key0);
            return;
          }
          const key = `${sid}:${itemKey}`;
          const existing = queueUpdateTimers.get(key);
          if (existing) clearTimeout(existing);
          const t = setTimeout(async () => {
            queueUpdateTimers.delete(key);
            if (appDisposed) return;
            queueMutationLocks.add(itemKey);
            try {
              await api(`/api/sessions/${sid}/queue/update`, { method: "POST", body: { id: itemKey, text } });
              if (appDisposed) return;
              queueLastEditMs = 0;
              queueDraftTexts.set(itemKey, text);
              await refreshQueueViewer();
              if (appDisposed) return;
              await refreshSessions();
              if (appDisposed) return;
              updateQueueBadge();
              syncRecoveryUiForSession(sid);
            } catch (e) {
              if (appDisposed) return;
              if (e && e.status === 401) {
                handleAppAuthLoss();
                return;
              }
              setToast(`queue update error: ${e && e.message ? e.message : "unknown error"}`);
            } finally {
              queueMutationLocks.delete(itemKey);
              if (!appDisposed && queuePendingDeletes.has(itemKey)) {
                queuePendingDeletes.delete(itemKey);
                void deleteQueueItem(sid, itemKey);
              }
            }
          }, 350);
          queueUpdateTimers.set(key, t);
        }

        function renderQueueList() {
          queueList.innerHTML = "";
          const sid = queueViewerSid || selected;
          if (!sid) {
            queueEmpty.style.display = "block";
            return;
          }
          const q = Array.isArray(queueViewerItems) ? queueViewerItems : [];
          queueEmpty.style.display = q.length ? "none" : "block";
          if (!q.length) return;
          const autosizeQueueText = (ta) => {
            if (!ta) return;
            ta.style.height = "0px";
            ta.style.height = `${Math.max(58, Math.min(220, ta.scrollHeight))}px`;
          };
          const queueMoveCrossesBarrier = (fromIdx, toIdx) => {
            const lo = Math.min(fromIdx, toIdx);
            const hi = Math.max(fromIdx, toIdx);
            for (let i = lo; i <= hi; i += 1) {
              if (i === fromIdx) continue;
              const candidate = q[i];
              if (candidate && (candidate.sending || candidate.commitUnknown || candidate.orphanRecovery)) return true;
            }
            return false;
          };
          q.forEach((item, idx) => {
            const itemId = String(item.id || "");
            const sending = !!item.sending;
            const commitUnknown = !!item.commitUnknown;
            const orphanRecovery = !!item.orphanRecovery;
            const locked = sending || commitUnknown || orphanRecovery || queueMutationLocks.has(itemId);
            const row = el("div", { class: "queueItem" });
            const editorShell = el("div", { class: "queueEditorShell" });
            const ta = el("textarea", { class: "queueText", "aria-label": `Queued message ${idx + 1}` });
            ta.value = queueDraftTexts.has(itemId) ? String(queueDraftTexts.get(itemId) || "") : String(item.text || "");
            ta.disabled = locked;
            ta.oninput = () => {
              queueLastEditMs = Date.now();
              const nextText = String(ta.value || "");
              queueDraftTexts.set(itemId, nextText);
              autosizeQueueText(ta);
              scheduleQueueUpdate(sid, itemId, nextText);
            };
            autosizeQueueText(ta);
            editorShell.appendChild(ta);
            const actions = el("div", { class: "queueActionRail" });
            if (sending) actions.appendChild(el("div", { class: "queueSendingTag muted", text: "Sending" }));
            if (commitUnknown) actions.appendChild(el("div", { class: "queueSendingTag warning", text: "Commit unknown" }));
            else if (orphanRecovery) actions.appendChild(el("div", { class: "queueSendingTag warning", text: "Recovery" }));
            const up = el("button", { class: "icon-btn queueIconBtn", title: "Move up", "aria-label": "Move up", type: "button", html: iconSvg("up") });
            up.disabled = locked || idx <= 0 || queueMoveCrossesBarrier(idx, idx - 1);
            up.onclick = (e) => {
              e.preventDefault();
              e.stopPropagation();
              void moveQueueItem(sid, itemId, idx - 1);
            };
            const down = el("button", { class: "icon-btn queueIconBtn", title: "Move down", "aria-label": "Move down", type: "button", html: iconSvg("down") });
            down.disabled = locked || idx >= q.length - 1 || queueMoveCrossesBarrier(idx, idx + 1);
            down.onclick = (e) => {
              e.preventDefault();
              e.stopPropagation();
              void moveQueueItem(sid, itemId, idx + 1);
            };
            const del = el("button", { class: "icon-btn queueIconBtn danger", title: "Delete", "aria-label": "Delete", type: "button", html: iconSvg("trash") });
            del.disabled = sending || queueMutationLocks.has(itemId);
            del.onclick = async (e) => {
              e.preventDefault();
              e.stopPropagation();
              await deleteQueueItem(sid, itemId);
            };
            actions.appendChild(up);
            actions.appendChild(down);
            actions.appendChild(del);
            row.appendChild(editorShell);
            row.appendChild(actions);
            queueList.appendChild(row);
          });
        }

        async function refreshQueueViewer() {
          const sid = queueViewerSid || selected;
          if (!sid) return;
          if (queueViewer.style.display === "flex" && Date.now() - queueLastEditMs < 900) return;
          queueEmpty.textContent = "Loading...";
          try {
            const data = await api(`/api/sessions/${sid}/queue`);
            if (queueViewerSid && queueViewerSid !== sid) return;
            const q = normalizeQueueItems(data);
            const nextDrafts = new Map();
            q.forEach((item) => {
              const itemId = String(item.id || "");
              if (!itemId) return;
              if (queueDraftTexts.has(itemId)) {
                const draft = String(queueDraftTexts.get(itemId) || "");
                if (draft.trim()) {
                  item.text = draft;
                  nextDrafts.set(itemId, draft);
                  return;
                }
              }
              nextDrafts.set(itemId, String(item.text || ""));
            });
            queueDraftTexts.clear();
            nextDrafts.forEach((value, key) => queueDraftTexts.set(key, value));
            queueViewerSid = sid;
            queueViewerItems = q;
            queueEmpty.textContent = "No queued messages.";
            renderQueueList();
          } catch (e) {
            if (e && e.status === 401) {
              handleAppAuthLoss();
              return;
            }
            if (queueViewerSid && queueViewerSid !== sid) return;
            queueViewerSid = sid;
            queueViewerItems = [];
            queueEmpty.textContent = `Queue unavailable: ${e && e.message ? e.message : "unknown error"}`;
            setToast(`queue load error: ${e && e.message ? e.message : "unknown error"}`);
            renderQueueList();
          }
        }

        function showQueueViewer({ opener = null } = {}) {
          if (!selected) return;
          queueReturnFocusEl = opener instanceof HTMLElement ? opener : document.activeElement instanceof HTMLElement ? document.activeElement : null;
          prepareModalOpen();
          queueViewerSid = selected;
          queueBackdrop.style.display = "block";
          queueViewer.style.display = "flex";
          afterModalVisibilityChanged();
          focusModalCloseButton(queueViewer, queueCloseBtn);
          void refreshQueueViewer();
        }

        function hideQueueViewer() {
          const wasOpen = isModalTargetOpen(queueViewer);
          const focusTarget = queueReturnFocusEl;
          queueReturnFocusEl = null;
          queueBackdrop.style.display = "none";
          queueViewer.style.display = "none";
          queueViewerSid = null;
          queueViewerItems = [];
          afterModalVisibilityChanged();
          if (wasOpen) {
            const fallback = document.querySelector(".recovery-panel .icon-btn") || queueBtn || null;
            restoreModalFocus(focusTarget && focusTarget.isConnected ? focusTarget : fallback, () => isModalTargetOpen(queueViewer));
          }
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

        diagNewLikeBtn.onclick = (e) => {
          e.preventDefault();
          e.stopPropagation();
          if (!diagNewLikeSession) {
            setToast("details not loaded");
            return;
          }
          const preset = diagNewLikeSession;
          const returnFocusEl = diagReturnFocusEl && diagReturnFocusEl.isConnected ? diagReturnFocusEl : null;
          hideDiagViewer({ restoreFocus: false });
          openNewSessionDialog({ likeSession: preset, statusText: "Review copied launch settings before starting.", returnFocusEl });
        };

        diagCopyBtn.onclick = async (e) => {
          e.preventDefault();
          e.stopPropagation();
          if (!diagCopyText) {
            setToast("details not loaded");
            return;
          }
          try {
            await copyToClipboard(diagCopyText);
            setToast("Copied details");
          } catch (err) {
            setToast(`copy failed: ${err && err.message ? err.message : "unknown error"}`);
          }
        };

        async function showDiagViewer({ opener = null } = {}) {
          const sid = selected;
          if (!sid) return;
          diagReturnFocusEl = opener instanceof HTMLElement ? opener : document.activeElement instanceof HTMLElement ? document.activeElement : null;
          prepareModalOpen();
          diagContent.innerHTML = "";
          diagCopyText = "";
          diagNewLikeSession = null;
          diagNewLikeBtn.disabled = true;
          diagCopyBtn.disabled = true;
          diagStatus.textContent = "Loading...";
          diagBackdrop.style.display = "block";
          diagViewer.style.display = "flex";
          afterModalVisibilityChanged();
          focusModalCloseButton(diagViewer, diagCloseBtn);
          try {
            const d = await api(`/api/sessions/${sid}/diagnostics`);
            if (selected !== sid) return;
            diagStatus.textContent = "";
            const now = Date.now() / 1000;
            const diagRows = [];
            const addRow = (label, value, { mono = false } = {}) => {
              const cleanLabel = String(label || "");
              const v = value == null || value === "" ? "-" : String(value);
              diagRows.push([cleanLabel, v]);
              const row = el("div", { class: "detailsRow" });
              row.appendChild(el("div", { class: "detailsLabel", text: cleanLabel }));
              row.appendChild(el("div", { class: mono ? "detailsValue mono" : "detailsValue", text: v }));
              diagContent.appendChild(row);
            };
            const age = (ts) => {
              const t = Number(ts);
              if (!Number.isFinite(t) || t <= 0) return "";
              const s = Math.max(0, Math.floor(now - t));
              return fmtRelativeAge(s);
            };
            addRow("Session", d && d.session_id ? d.session_id : "-");
            addRow("Thread", d && d.thread_id ? d.thread_id : "-");
            addRow("Owned", d ? sessionLaunchLabel(d).replace("-owned", "") : "-");
            addRow("Busy", d && typeof d.busy === "boolean" ? (d.busy ? "busy" : "idle") : "-");
            addRow("Queue", d && typeof d.queue_len === "number" ? String(d.queue_len) : "-");
            addRow("CWD", d && d.cwd ? d.cwd : "-", { mono: true });
            addRow("Started", d && typeof d.start_ts === "number" ? `${fmtTs(d.start_ts)}${age(d.start_ts) ? " (" + age(d.start_ts) + ")" : ""}` : "-");
            addRow(
              "Updated",
              d && typeof d.updated_ts === "number" ? `${fmtTs(d.updated_ts)}${age(d.updated_ts) ? " (" + age(d.updated_ts) + ")" : ""}` : "-"
            );
            addRow("Broker PID", d && typeof d.broker_pid === "number" ? String(d.broker_pid) : "-");
            addRow("Agent", d ? agentBackendDisplayName(d.agent_backend) : "-");
            addRow("Agent PID", d && typeof d.codex_pid === "number" ? String(d.codex_pid) : "-");
	            addRow("Log", d && d.log_path ? d.log_path : "-", { mono: true });
	            addRow("tmux", d && d.tmux_session ? `${d.tmux_session}${d.tmux_window ? ":" + d.tmux_window : ""}` : "-");
	            addRow("Branch", d && d.git_branch ? d.git_branch : "-");
	            addRow("Provider", diagnosticsProviderDisplay(d));
	            addRow("Model", d && d.model ? d.model : "-");
	            addRow("Reasoning", d && d.reasoning_effort ? d.reasoning_effort : "-");
	            addRow("Service tier", d && d.service_tier ? d.service_tier : "-");
	            addRow("Priority", d && typeof d.final_priority === "number" ? Number(d.final_priority).toFixed(4) : "-");
	            addRow("Priority offset", d && typeof d.priority_offset === "number" ? formatPriorityOffset(d.priority_offset) : "-");
	            addRow("Snooze", d && typeof d.snooze_until === "number" ? fmtTs(d.snooze_until) : "-");
	            addRow("Depends on", d && d.dependency_session_id ? d.dependency_session_id : "-");
	            addRow("UI", UI_VERSION);
	            const tok = d && d.token && typeof d.token === "object" ? d.token : null;
	            if (tok) {
	              const ctx = Number(tok.context_window);
	              const used = Number(tok.tokens_in_context);
	              const pct = Number(tok.percent_remaining);
              if (Number.isFinite(ctx) && Number.isFinite(used) && ctx > 0 && used >= 0) {
                const p = Number.isFinite(pct) ? Math.max(0, Math.min(100, Math.round(pct))) : null;
                const maxInput = Number(tok.max_input_tokens);
                const reserved = Number(tok.reserved_tokens);
                const effectiveMaxInput = Number.isFinite(maxInput) && maxInput >= 0 ? maxInput : ctx;
                const effectiveReserved = Number.isFinite(reserved) && reserved >= 0 ? reserved : Math.max(ctx - effectiveMaxInput, 0);
                const txt = p === null ? `${used}/${effectiveMaxInput}` : `${used}/${effectiveMaxInput} (${p}% left; ${effectiveReserved} reserved)`;
                addRow("Context", txt);
              }
            }
            diagCopyText = diagnosticsCopyText(sid, diagRows);
            diagNewLikeSession = d && typeof d === "object" ? {
              session_id: d.session_id,
              cwd: d.cwd,
              agent_backend: d.agent_backend,
              provider_choice: d.provider_choice,
              model_provider: d.model_provider,
              preferred_auth_method: d.preferred_auth_method,
              model: d.model,
              reasoning_effort: d.reasoning_effort,
              service_tier: d.service_tier,
              transport: d.transport,
              tmux_session: d.tmux_session,
              tmux_window: d.tmux_window,
            } : null;
            diagNewLikeBtn.disabled = !diagNewLikeSession;
            diagCopyBtn.disabled = !diagCopyText;
          } catch (e) {
            if (selected !== sid) return;
            diagCopyText = "";
            diagNewLikeSession = null;
            diagNewLikeBtn.disabled = true;
            diagCopyBtn.disabled = true;
            diagStatus.textContent = `error: ${e && e.message ? e.message : "unknown error"}`;
          }
        }
        function hideDiagViewer({ restoreFocus = true } = {}) {
          const wasOpen = isModalTargetOpen(diagViewer);
          const focusTarget = diagReturnFocusEl;
          diagReturnFocusEl = null;
          diagBackdrop.style.display = "none";
          diagViewer.style.display = "none";
          afterModalVisibilityChanged();
          if (restoreFocus && wasOpen) restoreModalFocus(focusTarget, () => isModalTargetOpen(diagViewer));
        }

        const queueBtn = $("#queueBtn");
        if (queueBtn) {
          queueBtn.onclick = (e) => {
            e.preventDefault();
            e.stopPropagation();
            const selectedInfo = sessionIndex.get(selected);
            if (sessionLaunchFailed(selectedInfo)) {
              setToast("failed session cannot receive messages");
              return;
            }
            if (selectedInfo && (selectedInfo.queue_recovery || selectedInfo.orphan_recovery) && Number(selectedInfo.queue_len || 0) > 0) {
              showQueueViewer({ opener: e.currentTarget });
              return;
            }
            const raw = $("#msg") ? $("#msg").value : "";
            if (raw && raw.trim()) {
              if (!selected) return;
              const sid = selected;
              void enqueueComposerText(raw, { sid }).then((ok) => {
                if (ok && selected === sid && $("#msg").value === raw) clearComposer();
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
        const composerStopBtn = $("#composerStopBtn");
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
	          const cur = chat.scrollTop;
	          const d = cur - lastScrollTop;
	          lastScrollTop = cur;
	          if (d < 0) autoScroll = false;
          else if (isNearBottom()) autoScroll = true;
          if (olderLoadRuntime.shouldCancelOnScroll() && cur > OLDER_CANCEL_PX) invalidateOlderLoad();
          if (cur <= OLDER_TOP_TRIGGER_PX && d <= 0) maybeAutoLoadOlder();
          syncJumpButton();
        });
        chat.addEventListener(
          "wheel",
          (e) => {
            if (e.deltaY < 0) {
              autoScroll = false;
              syncJumpButton();
              maybeAutoLoadOlder();
            }
          },
          { passive: true }
        );
        let touchY = null;
        chat.addEventListener(
          "touchstart",
          (e) => {
            const t = e.touches && e.touches[0];
            touchY = t ? t.clientY : null;
          },
          { passive: true }
        );
        chat.addEventListener(
          "touchmove",
          (e) => {
            const t = e.touches && e.touches[0];
            if (!t || touchY === null) return;
            const dy = t.clientY - touchY;
            touchY = t.clientY;
            // Finger moves down -> content scrolls up.
            if (dy > 0) {
              autoScroll = false;
              syncJumpButton();
              maybeAutoLoadOlder();
            }
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

         const textarea = $("#msg");
         const msgPh = $("#msgPh");
         const imgInput = $("#imgInput");
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
	             if (preserveChatBottom && (autoScroll || isNearBottom())) scrollToBottom();
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
	               if (autoScroll || isNearBottom()) requestAnimationFrame(() => scrollToBottom());
	             }
	           };
	           addAppEvent(window.visualViewport, "resize", onViewportShift);
	           addAppEvent(window.visualViewport, "scroll", onViewportShift);
	         }
         const attachBtn = $("#attachBtn");
         if (!attachBadgeEl) {
           attachBadgeEl = el("span", { class: "attachBadge", id: "attachBadge" });
           attachBtn.appendChild(attachBadgeEl);
         }
        if (!queueBadgeEl && queueBtn) {
          queueBadgeEl = el("span", { class: "attachBadge queueBadge", id: "queueBadge" });
          queueBtn.appendChild(queueBadgeEl);
        }
        const setAttachCount = (n) => {
          const next = Math.max(0, Number(n) || 0);
          attachedFiles = next;
          if (!attachBadgeEl) return;
          if (next > 0) {
            attachBadgeEl.textContent = String(next);
            attachBadgeEl.style.display = "inline-flex";
          } else {
            attachBadgeEl.textContent = "";
            attachBadgeEl.style.display = "none";
          }
        };
        function syncAttachButtonState() {
          const attachControl = $("#attachBtn");
          if (!attachControl) return;
          let attachLabel = `Attach file (max ${fmtBytes(ATTACH_UPLOAD_MAX_BYTES)})`;
          let disabled = false;
          if (!selected) {
            attachLabel = "Select a session to attach a file";
            disabled = true;
          } else if (selectedSessionLaunchFailed()) {
            attachLabel = "Failed launch cannot receive file attachments";
            disabled = true;
          } else if (selectedSessionHasUnknownSend()) {
            attachLabel = "Resolve the unknown send before attaching a file";
            disabled = true;
          } else if (selectedSessionIsOrphanRecovery()) {
            attachLabel = "Missing session can only be reviewed";
            disabled = true;
          } else if (selectedSessionHasOrphanQueueRecovery()) {
            attachLabel = "Review preserved queued recovery items before attaching a file";
            disabled = true;
          } else if (currentRunning) {
            attachLabel = "Wait for the current response to finish before attaching a file";
            disabled = true;
          } else if (sending) {
            attachLabel = "Wait for the current send to finish before attaching a file";
            disabled = true;
          }
          attachControl.disabled = disabled;
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
        syncSendButtonState();
	        function autoGrow() {
	          const basePx = parseFloat(getComputedStyle(textarea).minHeight || "0") || 32;
	          const maxPx = 180;
	          const hasNewline = textarea.value.includes("\n");
	          if (msgPh) msgPh.style.display = textarea.value ? "none" : "flex";
	          textarea.style.height = `${basePx}px`;
	          let h = textarea.scrollHeight;
	          const needsMultiline = hasNewline || h > basePx + 1;
	          form.classList.toggle("multiline", needsMultiline);
	          textarea.style.height = needsMultiline ? "auto" : `${basePx}px`;
	          h = textarea.scrollHeight;
	          const next = needsMultiline ? Math.min(h, maxPx) : basePx;
	          textarea.style.height = `${next}px`;
	          textarea.style.overflowY = h > maxPx ? "auto" : "hidden";
	          if (autoScroll) requestAnimationFrame(() => scrollToBottom());
	        }
	        textarea.addEventListener("input", autoGrow);
	          textarea.addEventListener(
	            "focus",
	            () => {
	              const wasNear = isNearBottom();
              if (wasNear) {
                autoScroll = true;
                syncJumpButton();
              }
	              if (isIOS) runIOSViewportGuard({ preserveChatBottom: wasNear, durationMs: 1800 });
	              else {
	                const tick = () => {
	                  updateAppHeightVar();
	                  if (wasNear) scrollToBottom();
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
          if (e.key !== "Enter") return;
          if (e.isComposing) return;
          if (!(e.ctrlKey || e.metaKey)) return;
          e.preventDefault();
          form.requestSubmit();
        });
        addAppEvent(window, "resize", () => {
          if (autoScroll) requestAnimationFrame(() => scrollToBottom());
        });

	        attachBtn.onclick = () => {
	          if (!selected) return;
	          if (selectedSessionLaunchFailed()) {
	            setToast("failed launch cannot receive file attachments");
	            return;
	          }
	          if (currentRunning) {
	            setToast("wait for the current response before attaching a file");
	            return;
	          }
	          if (sending) return;
	          imgInput.value = "";
	          imgInput.click();
	        };
		        imgInput.addEventListener("change", async () => {
		          const sid = selected;
		          if (!sid) return;
		          const sessionInfo = sessionIndex.get(sid) || null;
		          if (sessionInfo && sessionLaunchFailed(sessionInfo)) {
		            imgInput.value = "";
		            if (selected === sid) setToast("failed launch cannot receive file attachments");
		            return;
		          }
		          const attachmentIndex = attachedFiles + 1;
		          const f = imgInput.files && imgInput.files[0];
		          if (!f) return;
		          if (currentRunning) {
		            if (selected === sid) setToast("wait for the current response before attaching a file");
		            return;
		          }
		          if (sending) return;
		          try {
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

	            setToast("uploading file...");
	            const maxBytes = ATTACH_UPLOAD_MAX_BYTES;
	            let uploadBlob = f;
	            let uploadName = f.name || "file";
	            if (looksLikeImage(f) && (f.size > maxBytes || isLikelyHeic(f))) {
	              setToast("compressing image...");
	              const stem = safeAttachmentStem(f.name);
	              uploadName = `${stem}.jpg`;
	              // Try a few (dim, quality) pairs until it fits.
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
			            const res = await api(`/api/sessions/${sid}/inject_file`, {
		              method: "POST",
		              body: { filename: uploadName, data_b64: b64, attachment_index: attachmentIndex },
		            });
		            if (selected === sid) {
		              if (res && res.ok) {
		                setToast("file attached");
		                setAttachCount(attachmentIndex);
		              }
		              pollFastUntilMs = Date.now() + 4000;
		              kickPoll(0);
		            }
		          } catch (e) {
              if (e && e.status === 401) {
                handleAppAuthLoss();
                return;
              }
	            if (selected === sid) {
	              const commitUnknown = Boolean(e && e.obj && e.obj.commit_unknown);
	              if (commitUnknown) {
	                setToast("attachment status unknown; check before retrying");
	                pollFastUntilMs = Date.now() + 4000;
	                kickPoll(0);
	                void refreshSessions().catch((refreshErr) => {
                  if (refreshErr && refreshErr.status === 401) handleAppAuthLoss();
                  else console.error("refreshSessions failed", refreshErr);
                });
	              } else {
	                setToast(`attach error: ${e.message}`);
	              }
	            }
	          }
	        });

        function clearComposer() {
          $("#msg").value = "";
          autoGrow();
        }

        async function sendText(raw, { sid = null } = {}) {
          const sessionId = sid || selected;
          if (!sessionId) return false;
          if (!raw || !raw.trim()) return false;
          if (sending) return false;
          const renderHere = sessionId === selected;
          const renewsTranscript = isTranscriptRenewalCommand(raw, sessionId);
          const sessionInfo = sessionIndex.get(sessionId) || null;
          if (sessionInfo && sessionLaunchFailed(sessionInfo)) {
            setToast("failed launch cannot receive messages");
            return false;
          }
          if (sessionInfo && sessionInfo.orphan_recovery) {
            setToast("missing session can only be reviewed");
            return false;
          }
          if (sessionInfo && sessionInfo.queue_recovery) {
            setToast("review preserved queue before sending");
            return false;
          }
          if (sessionInfo && sessionInfo.commit_unknown_send) {
            setToast("resolve the unknown send before sending again");
            void clearCommitUnknownSend(sessionId, sessionInfo.commit_unknown_send_text || "");
            return false;
          }
          const localAttachmentCount = typeof attachedFiles === "number" ? attachedFiles : 0;
          let allowPendingAttachment = Boolean(renderHere && localAttachmentCount > 0);
          if (!allowPendingAttachment && sessionInfo && sessionInfo.pending_attachment) {
            const confirmed = window.confirm("This session has a pending file attachment. Send it with this message?");
            if (!confirmed) return false;
            allowPendingAttachment = true;
          }
          sending = true;
          syncSendButtonState();
          syncAttachButtonState();
          setToast("sending...");

          const localId = ++localEchoSeq;
          const t0 = Date.now() / 1000;
          if (renderHere && !renewsTranscript) {
            if (!renderedAtLiveTail) {
              clearTranscriptDom();
              clearRenderedTranscriptRange();
              setOlderState({ hasMore: false, isLoading: false });
            }
            const slot = getSessionTranscriptSlot(sessionId);
            pendingUser.push({ id: localId, sessionId, epoch: slot.epoch, key: pendingMatchKey(raw), loose: normalizeTextForPendingMatch(raw), t0, text: raw });
            appendEvent({ role: "user", text: raw, pending: true, localId, ts: t0 });
            turnOpen = true;
            currentRunning = true;
          }
          try {
            const res = await api(`/api/sessions/${sessionId}/send`, { method: "POST", body: { text: raw, allow_pending_attachment: allowPendingAttachment } });
            if (renderHere && renewsTranscript) {
              sessionTailCache.delete(sessionId);
              beginTranscriptRenewal(sessionId);
              liveCursor = null;
              clearRenderedTranscriptRange();
              invalidateOlderLoad();
              renderPendingTranscriptSlot(sessionId);
              turnOpen = true;
              currentRunning = true;
            }
            if (res.queued) setToast(`queued (queue ${res.queue_len})`);
            else setToast("sent");
            setAttachCount(0);
            pollFastUntilMs = Date.now() + 5000;
            kickPoll(0);
            void refreshSessions().catch((e) => {
              if (e && e.status === 401) handleAppAuthLoss();
              else console.error("refreshSessions failed", e);
            });
            return true;
          } catch (e2) {
            if (e2 && e2.status === 401) {
              handleAppAuthLoss();
              return false;
            }
            const commitUnknown = Boolean(e2 && e2.obj && e2.obj.commit_unknown);
            if (commitUnknown) {
              setToast("send status unknown; check transcript before retrying");
              const currentSessionInfo = sessionIndex.get(sessionId) || sessionInfo;
              if (currentSessionInfo) {
                currentSessionInfo.commit_unknown_send = true;
                currentSessionInfo.commit_unknown_send_text = raw;
                currentSessionInfo.commit_unknown_send_ts = Date.now() / 1000;
                sessionIndex.set(sessionId, currentSessionInfo);
              }
              syncSendButtonState();
              syncQueueSubmitState();
              syncAttachButtonState();
              pollFastUntilMs = Date.now() + 4000;
              kickPoll(0);
              void refreshSessions().catch((e) => {
                if (e && e.status === 401) handleAppAuthLoss();
                else console.error("refreshSessions failed", e);
              });
            } else setToast(`send error: ${e2.message}`);
            if (!commitUnknown && sessionInfo && sessionInfo.pending_attachment && /broker must be restarted/i.test(String(e2 && e2.message ? e2.message : ""))) {
              const clearPending = window.confirm("This session has a pending attachment but the current broker cannot confirm sends. Clear the browser pending-attachment state only if you already handled it in the terminal?");
              if (clearPending) {
                try {
                  await api(`/api/sessions/${sessionId}/pending_attachment/clear`, { method: "POST", body: {} });
                  setToast("pending attachment state cleared");
                  void refreshSessions().catch((e) => {
                    if (e && e.status === 401) handleAppAuthLoss();
                    else console.error("refreshSessions failed", e);
                  });
                } catch (clearErr) {
                  if (clearErr && clearErr.status === 401) {
                    handleAppAuthLoss();
                    return false;
                  }
                  setToast(`clear pending attachment error: ${clearErr && clearErr.message ? clearErr.message : "unknown error"}`);
                }
              }
            }
            if (renderHere) {
              for (let i = pendingUser.length - 1; i >= 0; i -= 1) {
                const pending = pendingUser[i];
                if (pending && pending.id === localId && pending.sessionId === sessionId) pendingUser.splice(i, 1);
              }
              const pendingEl = chatInner.querySelector(`.msg.user[data-local-id="${localId}"]`);
              if (pendingEl) {
                const pendingRow = pendingEl.closest(".msg-row");
                if (pendingRow) pendingRow.remove();
                else pendingEl.remove();
              }
              if (!pendingUser.some((pending) => pending && pending.sessionId === sessionId)) {
                turnOpen = false;
                currentRunning = false;
              }
              if (commitUnknown) syncRecoveryUiForSession(sessionId);
            }
            return false;
          } finally {
            sending = false;
            syncSendButtonState();
            syncAttachButtonState();
          }
        }

        form.onsubmit = async (e) => {
          e.preventDefault();
          if (!selected) {
            setToast("select a session first");
            return;
          }
          if (sessionLaunchFailed(sessionIndex.get(selected))) {
            setToast("failed session cannot receive messages");
            return;
          }
          const raw = $("#msg").value;
          if (!raw || !raw.trim()) return;
          if (sending) return;
          if (currentRunning) {
            showSendChoice(raw, { opener: document.activeElement instanceof HTMLElement ? document.activeElement : textarea });
            return;
          }
          const ok = await sendText(raw);
          if (ok && $("#msg").value === raw) clearComposer();
        };

        activeAppCleanup = cleanupApp;

	        (async () => {
	          if (storageGetItem("codexweb.sidebarCollapsed") === "1") setSidebarCollapsed(true);
	          if (storageGetItem("codexweb.sidebarOpen") === "1") setSidebarOpen(true);

	          try {
              await loadVoiceSettings();
              await syncNotificationState();
              if (localAnnouncementEnabled) {
                resumeAnnouncementRuntime({ resetSource: false });
              }
              if (notificationsEnabledLocally()) await pollNotificationFeed({ prime: true });
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
	            autoGrow();

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
