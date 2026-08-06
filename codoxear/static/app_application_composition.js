/* Application composition owns concrete lifecycle, controller assembly, and UI behavior.
 * app_application_runtime.js remains the stable bootstrap facade. */
(function installCodoxearApplicationComposition(global) {
  "use strict";

  function createEventBindings(options = {}) {
    const addEvent = options.addEvent;
    if (typeof addEvent !== "function") throw new TypeError("event bindings dependency missing: addEvent");

    function on(target, type, handler, listenerOptions) {
      if (!target || typeof handler !== "function") throw new TypeError(`event binding requires ${type} target and handler`);
      return addEvent(target, type, handler, listenerOptions);
    }

    function onClick(target, handler, listenerOptions) {
      return on(target, "click", handler, listenerOptions);
    }

    return Object.freeze({ on, onClick });
  }

  function createToastController(options = {}) {
    function requireToastNode(value, name) {
      if (!value || typeof value !== "object" || !("textContent" in value)) {
        throw new TypeError(`toast dependency missing: ${name}`);
      }
      return value;
    }

    if (!options || typeof options !== "object") throw new TypeError("toast dependency missing: options");
    const toast = requireToastNode(options.toast, "toast");
    const setTimeoutFn = typeof options.setTimeout === "function" ? options.setTimeout : setTimeout;
    const dismissAfterMs = Number.isFinite(options.dismissAfterMs) ? Math.max(0, options.dismissAfterMs) : 2200;

    function show(value) {
      const text = value ? String(value) : "";
      toast.textContent = text;
      if (!text) return "";
      setTimeoutFn(() => {
        if (toast.textContent === text) toast.textContent = "";
      }, dismissAfterMs);
      return text;
    }

    return Object.freeze({ show });
  }

  function createApplicationComposition(deps = {}) {
    const {
      window, document, navigator, HTMLElement, EventSource, AbortController, getComputedStyle,
      requestAnimationFrame, setTimeout, clearTimeout, $, UI_VERSION, ATTACH_UPLOAD_MAX_BYTES,
      isTextEntryElement, updateAppHeightVar,
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
      cleanupActiveApp, renderLogin, setActiveAppCleanup, clearActiveAppCleanup
    } = deps;
    if (typeof cleanupActiveApp !== "function" || typeof setActiveAppCleanup !== "function" || typeof clearActiveAppCleanup !== "function") {
      throw new Error("Codoxear application runtime requires cleanup lifecycle dependencies");
    }
    let newSessionDefaults = { default_backend: "pi", backends: { codex: null, pi: null, cc: null } };
    let latestSessions = [];

      function renderApp() {
            cleanupActiveApp();
	        const root = $("#root");
        const codoxearWiring = window.CodoxearWiring;
        if (!codoxearWiring || typeof codoxearWiring.createWiring !== "function")
          throw new Error("Codoxear wiring factories failed to load");
        const wiring = codoxearWiring.createWiring();
        const shellDOM = codoxearShell.createShellDOM(wiring.createShellDOMOptions({
          root,
          el,
          iconSvg,
          resolveAppUrl,
          versionedShellAssetPath,
        }));
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
          networkBanner,
          toggleSidebarBtn,
          unattendedBtn,
          diagBtn,
          prevUserBtn,
          nextUserBtn,
          chatSearchBtn,
          fileBtn,
          composer,
          form,
          textarea,
          msgPh,
          modelPicker,
          imgInput,
          attachBtn,
          queueBtn,
          sendBtn,
        } = shellDOM.elements;
        const networkStatus = codoxearNetwork.createNetworkStatusController(wiring.createNetworkStatusOptions({
          banner: networkBanner,
          navigatorLike: typeof navigator === "undefined" ? undefined : navigator,
        }));
        const codoxearUnattendedDom = window.CodoxearUnattended;
        if (!codoxearUnattendedDom || typeof codoxearUnattendedDom.createUnattendedDom !== "function")
          throw new Error("Codoxear unattended DOM failed to load");
        const unattendedDom = codoxearUnattendedDom.createUnattendedDom(wiring.createUnattendedDomOptions({ el, iconSvg, unattendedBtn }));
        const { unattendedMenu, enabledEl: unattendedEnabledEl, cooldownEl: unattendedCooldownEl, remainingEl: unattendedRemainingEl, requestEl: unattendedRequestEl } = unattendedDom;
        root.appendChild(unattendedMenu);
        let pendingHashSessionId = "";
        let pendingHashSessionSelectInFlight = false;
        const INIT_PAGE_LIMIT = 24;
        const OLDER_PAGE_LIMIT = 60;
        const CHAT_DOM_WINDOW = 260;
        const CHAT_DOM_WINDOW_WITH_HISTORY_SLACK = CHAT_DOM_WINDOW + OLDER_PAGE_LIMIT;
        const OLDER_TOP_TRIGGER_PX = 1;
        const OLDER_CANCEL_PX = 48;
        const OLDER_AUTO_COOLDOWN_MS = 450;
        let pollGen = 0;
        let turnOpen = false;
	         let sessionsTimer = null;
         let secondaryPollTimer = null;
         let sessionsPollingEnabled = true;
         let secondaryPollingEnabled = true;
         let sessionsPollErrorStreak = 0;
         let secondaryPollErrorStreak = 0;
         let currentRunning = false;
        let selected = null; // selected session_id (null until chosen)
	        let sessionIndex = new Map(); // session_id -> session info
        let recentCwds = [];
	        let sending = false;
        let attachmentsController = null;
        let composerController = null;
        let messageFlowController = null;
        let sessionLifecycleController = null;
        let sessionRefreshController = null;
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
        let sessionEditController = null;
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
        let appDisposed = false;
        const appEventCleanups = [];
        function addAppEvent(target, type, handler, options) {
          if (!target || typeof target.addEventListener !== "function") return handler;
          target.addEventListener(type, handler, options);
          appEventCleanups.push(() => target.removeEventListener(type, handler, options));
          return handler;
        }
        const codoxearEventBindings = window.CodoxearEventBindings;
        if (!codoxearEventBindings || typeof codoxearEventBindings.createEventBindings !== "function")
          throw new Error("Codoxear event bindings failed to load");
        const eventBindings = codoxearEventBindings.createEventBindings(wiring.createEventBindingsOptions({ addEvent: addAppEvent }));
        function stopMessagePolling() {
          selected = null;
          pollGen += 1;
          if (messageFlowController) messageFlowController.stop();
          turnOpen = false;
        }
        function cleanupApp() {
          if (appDisposed) return;
          appDisposed = true;
          sessionsPollingEnabled = false;
          secondaryPollingEnabled = false;
          stopMessagePolling();
          stopAllPolling();
          if (newSessionDialogController) newSessionDialogController.close();
          if (voiceController) voiceController.dispose();
          if (unattendedController) unattendedController.dispose();
          filePickerSearchState.dispose();
          if (iosViewportController) iosViewportController.dispose();
          if (chatSearchController) chatSearchController.dispose();
          if (queueController) queueController.dispose();
          if (diagController) diagController.dispose();
          if (chatNavigationController) chatNavigationController.dispose();
          if (hintModeController) hintModeController.dispose();
          olderLoadRuntime.invalidate();
          fileViewerController.abortPendingFileOpenTransport();
          hideUnattendedMenu();
          hideFilePasteDialog();
          fileUnsavedController.hideFileUnsavedDialog("cancel");
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
          clearActiveAppCleanup(cleanupApp);
        }
        function handleAppAuthLoss() {
          if (appDisposed) return;
          cleanupApp();
          renderLogin(renderApp);
        }
        function sessionsPollDelayMs() {
          return codoxearPolling.networkRetryDelayMs({
            normalDelayMs: codoxearPolling.sessionsPollDelayMs(document.visibilityState),
            offline: browserOffline(),
            errorStreak: sessionsPollErrorStreak,
          });
        }
        function secondaryPollDelayMs() {
          return codoxearPolling.networkRetryDelayMs({
            normalDelayMs: codoxearPolling.secondaryPollDelayMs(document.visibilityState),
            offline: browserOffline(),
            errorStreak: secondaryPollErrorStreak,
          });
        }
        function browserOffline() {
          return codoxearPolling.browserOffline(typeof navigator === "undefined" ? undefined : navigator);
        }
        function markSessionsPollSuccess() {
          sessionsPollErrorStreak = 0;
          networkStatus.reportSuccess();
        }
        function markSessionsPollFailure(transportFailed = true) {
          sessionsPollErrorStreak = Math.min(sessionsPollErrorStreak + 1, 20);
          if (transportFailed) networkStatus.reportFailure();
        }
        function markSecondaryPollSuccess() {
          secondaryPollErrorStreak = 0;
          networkStatus.reportSuccess();
        }
        function markSecondaryPollFailure(transportFailed = true) {
          secondaryPollErrorStreak = Math.min(secondaryPollErrorStreak + 1, 20);
          if (transportFailed) networkStatus.reportFailure();
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
            markSessionsPollSuccess();
          } catch (e2) {
            if (e2 && e2.status === 401) {
              handleAppAuthLoss();
              return;
            }
            markSessionsPollFailure(!(e2 && typeof e2.status === "number"));
            console.error("refreshSessions timer failed", e2);
          }
          scheduleSessionsPoll();
        }
        async function runSecondaryPollTick() {
          if (appDisposed || !secondaryPollingEnabled) return;
          try {
            await refreshVoiceBackgroundState();
            markSecondaryPollSuccess();
          } catch (e2) {
            if (e2 && e2.status === 401) {
              handleAppAuthLoss();
              return;
            }
            markSecondaryPollFailure(!(e2 && typeof e2.status === "number"));
            console.error("secondary poll failed", e2);
          }
          secondaryPollController.scheduleSecondaryPoll();
        }
        function scheduleSessionsPoll(delayMs = sessionsPollDelayMs()) {
          if (appDisposed || !sessionsPollingEnabled) return;
          stopSessionsPolling();
          sessionsTimer = setTimeout(() => {
            sessionsTimer = null;
            void runSessionsPollTick();
          }, Math.max(0, Number(delayMs) || 0));
        }
        const secondaryPollController = codoxearSecondaryPoll.createSecondaryPollController(wiring.createSecondaryPollOptions({
          isDisposed: () => appDisposed,
          isPollingEnabled: () => secondaryPollingEnabled,
          stopPolling: stopSecondaryPolling,
          setTimer: (timer) => {
            secondaryPollTimer = timer;
          },
          setTimeout,
          runTick: runSecondaryPollTick,
          delayForPoll: secondaryPollDelayMs,
        }));

        const codoxearSessionTitle = window.CodoxearSessionTitle;
        if (!codoxearSessionTitle || typeof codoxearSessionTitle.createSessionTitleController !== "function")
          throw new Error("Codoxear session title controller failed to load");
        const sessionTitleController = codoxearSessionTitle.createSessionTitleController(wiring.createSessionTitleOptions({
          titleLabel,
          getSelected: () => selected,
          openEditSession: (sessionId) => sessionEditController.openEditSession(sessionId),
        }));

        let helpReturnFocusEl = null;
        const applicationModalDOM = codoxearShell.createApplicationModalDOM(wiring.createApplicationModalDOMOptions({
          root, el, iconSvg, windowTarget: window, codoxearVoice, voiceHost: shellDOM.elements.voiceHost,
        }));
        const {
          fileBackdrop, fileCloseBtn, fileStatus, filePickerInput, filePickerMenu, filePickerField,
      fileModeDiffBtn, fileModePreviewBtn, fileEditBtn, fileVideoPreviewBtn, fileDownloadBtn,
      fileTouchSelectBtn, fileTouchCopyBtn, fileTouchPasteBtn, fileTouchUpBtn, fileTouchLeftBtn,
      fileTouchDownBtn, fileTouchRightBtn, fileTouchDpad, fileTouchActions, fileTouchToolbar,
      fileDiff, fileImage, fileVideo, fileViewer, fileUnsavedBackdrop, fileUnsavedDialog,
      filePasteBackdrop, filePasteInput, filePasteDialog, sendChoiceBackdrop, sendChoice,
      appConfirmBackdrop, appConfirmTitle, appConfirmMessage, appConfirmConfirmBtn,
      appConfirmCancelBtn, appConfirm, queueBackdrop, queueCloseBtn, queueList, queueEmpty,
      queueViewer, helpBackdrop, helpCloseBtn, helpViewer, diagBackdrop, diagCopyConversationBtn,
      diagCopyBtn, diagCloseBtn, diagStatus, diagContent, diagViewer, editCloseBtn, editStatus,
      editNameInput, editPriorityRange, editPriorityValue, editPriorityResetBtn,
      editSnoozeModeButtons, editSnoozeButtons, editSnoozeCustomDate, editSnoozeCustomTime,
      editSnoozeCustomRow, editDependencyBtn, editDependencyMenu, editDependencyField,
      editSaveBtn, editViewer, announceBtn, notificationBtn, liveAudio, voiceSettingsBackdrop,
      voiceSettingsCloseBtn, voiceSettingsStatus, voiceBaseUrlInput, voiceApiKeyInput,
      voiceClearApiKeyToggle, narrationSettingToggle, unattendedPromptInput,
      unattendedPromptResetBtn, voiceSettingsViewer, voiceSettingsCancelBtn, voiceSettingsSaveBtn
        } = applicationModalDOM;
        const codoxearModal = window.CodoxearModal;
        if (
          !codoxearModal ||
          typeof codoxearModal.isModalTargetOpen !== "function" ||
          typeof codoxearModal.syncModalIsolation !== "function" ||
          typeof codoxearModal.restoreModalFocus !== "function" ||
          typeof codoxearModal.focusModalCloseButton !== "function" ||
          typeof codoxearModal.createModalKeyboardHandler !== "function"
        )
          throw new Error("Codoxear modal helpers failed to load");

        function setPickerButtonContent(button, primaryText, secondaryText = "", placeholder = false) {
          if (!button) return;
          button.innerHTML = "";
          const textWrap = el("span", { class: `pickerButtonText${placeholder ? " placeholder" : ""}` });
          textWrap.appendChild(el("span", { class: "pickerButtonPrimary", text: String(primaryText || "") }));
          if (secondaryText) textWrap.appendChild(el("span", { class: "pickerButtonSecondary", text: String(secondaryText) }));
          button.appendChild(textWrap);
          button.appendChild(el("span", { class: "pickerButtonChevron", html: iconSvg("chevronDown") }));
        }

        const codoxearDialogMenu = window.CodoxearDialogMenu;
        if (!codoxearDialogMenu || typeof codoxearDialogMenu.createDialogMenuController !== "function")
          throw new Error("Codoxear dialog menu controller failed to load");
        const dialogMenuController = codoxearDialogMenu.createDialogMenuController(wiring.createDialogMenuOptions({ windowTarget: window }));

        const newSessionDialogController = codoxearNewSession.createNewSessionDialogController(wiring.createNewSessionDialogOptions({
          root,
          el,
          iconSvg,
          document,
          window,
          addEvent: addAppEvent,
          defaultsSource: () => newSessionDefaults,
          latestSessions: () => latestSessions,
          recentCwds: () => recentCwds,
          tmuxAvailable: () => tmuxAvailable,
          selectedSession: () => selected,
          sessionForId: (sessionId) => sessionIndex.get(sessionId),
          isMobile: () => codoxearViewport.isMobile(),
          prepareModalOpen,
          afterModalVisibilityChanged,
          isModalTargetOpen,
          applyDialogMenus: () => dialogMenusController.applyDialogMenus(),
          positionDialogMenu: (menu, anchorBtn) => dialogMenuController.positionDialogMenu(menu, anchorBtn),
          setPickerButtonContent,
          fetchResumeCandidates: (cwd, backend) => api(`/api/session_resume_candidates?cwd=${encodeURIComponent(cwd)}&agent_backend=${encodeURIComponent(backend)}`),
          spawnSession: (...args) => sessionLifecycleController.spawnSessionWithCwd(...args),
        }));

        const modalIsolationTargets = [
          fileUnsavedDialog,
          filePasteDialog,
          fileViewer,
          sendChoice,
          appConfirm,
          queueViewer,
          helpViewer,
          diagViewer,
          editViewer,
          voiceSettingsViewer,
          newSessionDialogController.viewer,
        ];

        function isModalTargetOpen(node) {
          return codoxearModal.isModalTargetOpen(node);
        }

        function syncModalIsolation() {
          return codoxearModal.syncModalIsolation(app, modalIsolationTargets);
        }

        function closeFilePickerMenu() {
          if (!fileOpsController) return;
          fileOpsController.closeFilePickerMenu({ restoreInput: false });
        }

        function closeTransientOverlays({ closeSearch = false } = {}) {
          if (unattendedController.isOpen()) hideUnattendedMenu();
          if (closeSearch && chatSearchController.isOpen()) closeChatSearch();
          if (document.body.classList.contains("sidebar-open")) setSidebarOpen(false);
          closeFilePickerMenu();
          newSessionDialogController.closeMenus();
          sessionEditController.closeDependencyMenu();
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

        eventBindings.on(appConfirmConfirmBtn, 'click', () => resolveAppConfirm(true));
        eventBindings.on(appConfirmCancelBtn, 'click', () => resolveAppConfirm(false));
        eventBindings.on(appConfirmBackdrop, 'click', () => resolveAppConfirm(false));

        const codoxearClipboard = window.CodoxearClipboard;
        if (!codoxearClipboard || typeof codoxearClipboard.copyToClipboard !== "function")
          throw new Error("Codoxear clipboard helpers failed to load");
        const codoxearCodeCopy = window.CodoxearCodeCopy;
        if (!codoxearCodeCopy || typeof codoxearCodeCopy.createCodeBlockCopyRuntime !== "function")
          throw new Error("Codoxear code copy helpers failed to load");

        const toastController = createToastController({ toast, setTimeout });
        function setToast(text) {
          return toastController.show(text);
        }

        async function copyToClipboard(text) {
          return codoxearClipboard.copyToClipboard(text);
        }

        const codeBlockCopyRuntime = codoxearCodeCopy.createCodeBlockCopyRuntime(wiring.createCodeBlockCopyOptions({
          copyToClipboard,
          setToast,
          setTimeout,
          clearTimeout,
        }));

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
        let currentSubagentsRunning = 0;
        const codoxearSessionDisplay = window.CodoxearSessionDisplay;
        if (!codoxearSessionDisplay || typeof codoxearSessionDisplay.createSessionDisplayController !== "function")
          throw new Error("Codoxear session display controller failed to load");
        const sessionDisplayController = codoxearSessionDisplay.createSessionDisplayController(wiring.createSessionDisplayOptions({
          getSelected: () => selected,
          getRunning: () => currentRunning,
          setRunning: (value) => { currentRunning = Boolean(value); },
          getQueueLen: () => currentQueueLen,
          setQueueLen: (value) => { currentQueueLen = value; },
          getSubagentsRunning: () => currentSubagentsRunning,
          getAttachmentsController: () => attachmentsController,
          updateQueueBadge: () => updateQueueBadge(),
          setToast, statusChip, interruptBtn, ctxChip, eventBindings,
        }));
        const { renderStatusChip, setStatus, setContext } = sessionDisplayController;
        let fileOpsController = null;

        const codoxearChatInteraction = window.CodoxearChatInteraction;
        if (!codoxearChatInteraction || typeof codoxearChatInteraction.createChatInteractionController !== "function")
          throw new Error("Codoxear chat interaction controller failed to load");
        const chatInteractionController = codoxearChatInteraction.createChatInteractionController(wiring.createChatInteractionOptions({
          ...deps,
          wiring,
          INIT_PAGE_LIMIT, OLDER_PAGE_LIMIT, OLDER_AUTO_COOLDOWN_MS, OLDER_TOP_TRIGGER_PX, OLDER_CANCEL_PX, CHAT_DOM_WINDOW, CHAT_DOM_WINDOW_WITH_HISTORY_SLACK,
          window, document, navigator, HTMLElement, EventSource, AbortController, getComputedStyle,
          requestAnimationFrame, setTimeout, clearTimeout, $, el, iconSvg,
          getSelected: () => selected,
          getPollGeneration: () => pollGen,
          getSessionIndex: () => sessionIndex,
          getSessionLifecycleController: () => sessionLifecycleController,
          getSessionRefreshController: () => sessionRefreshController,
          getSending: () => sending,
          setSending: (value) => { sending = Boolean(value); },
          getTurnOpen: () => turnOpen,
          setTurnOpen: (value) => { turnOpen = Boolean(value); },
          getCurrentRunning: () => currentRunning,
          setCurrentRunning: (value) => { currentRunning = Boolean(value); },
          getCurrentSubagentsRunning: () => currentSubagentsRunning,
          setCurrentSubagentsRunning: (value) => { currentSubagentsRunning = value; },
          isAppDisposed: () => appDisposed,
          setStatus, setContext,
          addAppEvent, eventBindings, app, root, chat, chatInner, olderWrap, olderBtn, olderRetryBtn,
          olderError, olderErrorText, bottomSentinel, jumpBtn, chatTimeChip, chatSearchInput,
          chatSearchPrevBtn, chatSearchNextBtn, chatSearchCloseBtn, chatSearchStatus, chatSearchAllHintEl,
          chatSearchBar, chatSearchBtn, prevUserBtn, nextUserBtn, textarea, statusChip, ctxChip,
          interruptBtn, toast, titleLabel, sessionsWrap, sidebarEmptyHint, queueViewer, helpViewer, diagViewer, editViewer,
          fileViewer, appConfirm, sendChoice, composer, attachBtn, imgInput, codeBlockCopyRuntime,
          networkStatus, Node: window.Node, resizeComposer, renderStatusChip,
          syncComposerSendButton, syncQueueSubmitState, updateUnattendedBtnState: () => updateUnattendedBtnState(),
          updateQueueBadge: () => updateQueueBadge(),
          refreshQueueViewer,
          codoxearCodeCopy: window.CodoxearCodeCopy,
          codoxearTranscriptRender: window.CodoxearTranscriptRender,
          codoxearMessageHistory: window.CodoxearMessageHistory,
          codoxearSendLifecycle: window.CodoxearSendLifecycle,
          codoxearPendingUser, codoxearNavigationPulse,
          codoxearModal, codoxearViewport, codoxearDisplay,
          codoxearMessageFlow, codoxearAttachments, codoxearComposer, codoxearConversationCopy,
          codoxearPolling, codoxearNetwork, codoxearSessionHelpers,
          modalIsolationTargets, newSessionDialogController,
          getSessionEditController: () => sessionEditController,
          getQueueController: () => queueController,
          handleAppAuthLoss, confirmApp, copyToClipboard, setToast, closeTransientOverlays,
          prepareModalOpen, afterModalVisibilityChanged, restoreModalFocus, isModalTargetOpen,
          normalizeAppConfirmOptions, appConfirmFocusableControls, focusAppConfirmInitial, resolveAppConfirm,
          sessionLaunchFailed, sessionAgentBackend, sessionTitleWithId, normalizeAgentBackendName,
          providerChoiceToSettings, backendSupportsFast, sessionHasUnknownSend, sessionIsOrphanRecovery,
          sessionHasOrphanQueueRecovery, normalizeQueueItems, api, resolveAppUrl, clearApiCache,
          chatMarkdownHtmlCached,
          upgradeCandidateFileRefs: (...args) => fileOpsController.fileReferenceRuntime.upgradeCandidateRefs(...args),
          versionedShellAssetPath, performance,
          isFileViewerOpen: () => fileOpsController.isFileViewerOpen(),
        }));
        ({ attachmentsController, messageFlowController } = chatInteractionController);
        const {
          chatSearchController, chatNavigationController, hintModeController, sidebarController, transcriptSlotRuntime, typingRowRuntime,
          transcriptScrollRuntime, transcriptDomRuntime, transcriptEventRuntime, olderLoadRuntime,
          resetChatRenderState, clearOlderLoadError, updateChatNavButtons,
          closeChatSearch, clearRenderedTranscriptRange, initPageLimit, dropPendingUserRows,
          updateSessionTranscriptSlot, tailCacheMatchesSession, applySessionListTranscriptIdentity,
          updateQueueBadge, updateTypingStatsFromSession, setTyping, messagePollDelayMs, kickPoll,
          setPollFastUntilMs, openMessageEventSource, isMobile, useDesktopSessionActions,
          useTouchFileEditorControls, setSidebarOpen, setSidebarCollapsed, clearCommitUnknownSend,
          refreshSessions, loadOlderMessages, applySessionRuntimeFromTail, renderSessionTail,
          recoveryDetailsText, syncRecoveryUiForSession, renderPendingTranscriptSlot,
          renderTranscriptLoading, renderTranscriptLoadError, applyCachedTail, jumpToLatest,
          rememberPendingHashSession, maybeSelectPendingHashSession,
        } = chatInteractionController;
        const unattendedController = (function instantiateUnattendedController() {
          const codoxearUnattended = window.CodoxearUnattended;
          if (!codoxearUnattended || typeof codoxearUnattended.createUnattendedController !== "function")
            throw new Error("Codoxear unattended controller failed to load");
          return codoxearUnattended.createUnattendedController(wiring.createUnattendedOptions({
            unattendedBtn,
            unattendedMenu,
            enabledEl: unattendedEnabledEl,
            cooldownEl: unattendedCooldownEl,
            remainingEl: unattendedRemainingEl,
            requestEl: unattendedRequestEl,
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
            storageGetItem,
            storageSetItem,
            storageRemoveItem,
          }));
        })();

        // App-shell button projection. The unattended-specific projection
        // (button disabled/title/active, cfg cache sync from session fields,
        // number-input draft sync, menu enabled-checkbox sync, and the
        // close-menu-when-selected-changes guard) is delegated to the
        // controller. Everything else (title edit, attach/file/send/queue/diag
        // buttons, context bar, chat nav, chat-search close) stays here.
        function updateUnattendedBtnState() {
          sessionTitleController.syncTitleEditState();
          unattendedController.syncButtonState();
          attachmentsController.syncAttachButtonState();
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
        // announcement state through thin wrappers below. app_voice.js owns
        // voice DOM construction, state, handlers, timers, and HLS lifecycle;
        // app.js supplies shell/runtime dependencies and keeps event wiring that
        // feeds voice from the poll/SSE orchestration.
        let voiceController;
        function instantiateVoiceController() {
          return codoxearVoice.createVoiceController(wiring.createVoiceOptions({
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
            voiceSettingsCancelBtn,
            voiceSettingsSaveBtn,
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
              void sessionLifecycleController.selectSessionFromHash({ refreshIfMissing: true, deferIfMissing: true }).catch((e) => {
                if (e && e.status === 401) handleAppAuthLoss();
                else console.error("desktop notification session select failed", e);
              });
            },
          }));
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
        function refreshVoiceBackgroundState(options) {
          return voiceController.refreshBackgroundState(options);
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
        const codoxearFileOps = window.CodoxearFileOps;
        if (!codoxearFileOps || typeof codoxearFileOps.createFileOpsController !== "function")
          throw new Error("Codoxear file operations controller failed to load");
        fileOpsController = codoxearFileOps.createFileOpsController(wiring.createFileOpsOptions({
          wiring, document, window, HTMLElement, requestAnimationFrame, setTimeout,
          $, el, iconSvg, resolveAppUrl, api, setToast, confirmApp, addAppEvent,
          sessionLaunchFailed, normalizeLineNumber, markdownPreviewHtml,
          blockedFileMessage, listFromFilesField, listFromFileRecords, baseName,
          codoxearFilePicker, codoxearFilePickerOps: window.CodoxearFilePickerOps,
          codoxearFileViewer, codoxearFileEditor, codoxearFileEditorOps: window.CodoxearFileEditorOps,
          codoxearFileEditMode,
          codoxearFileTouch, codoxearDialogMenus,
          prepareModalOpen, afterModalVisibilityChanged, focusModalCloseButton, restoreModalFocus,
          isModalTargetOpen, newSessionDialogController, eventBindings,
          codoxearFileHelpers, copyToClipboard, dialogMenuController, duplicateFilePickerPaths,
          editCloseBtn, editDependencyBtn, editDependencyMenu, editNameInput, editPriorityRange,
          editPriorityResetBtn, editPriorityValue, editSaveBtn, editSnoozeCustomDate,
          editSnoozeCustomRow, editSnoozeCustomTime, editSnoozeModeButtons, editStatus, editViewer,
          fileBtn, filePickerIdentityHint, filePickerSectionLabel, filePickerTitle, fmtBytes,
          formatPriorityOffset, handleAppAuthLoss, isDiffableFileKind, isMarkdownPreviewable,
          isTextEntryElement, isTextFileKind, modalIsolationTargets, normalizeDraftFilePath,
          parseLocalFileRef, rawByteDuplicatePaths, refreshSessions: () => sessionRefreshController.refreshSessions(), selectedSessionLaunchFailed,
          sessionDisplayName, sessionTitleWithId, setPickerButtonContent, storageGetItem,
          storageSetItem, stripPathLocationSuffix, titleLabel,
          useTouchFileEditorControls: () => codoxearViewport.useTouchFileEditorControls(),
          filePickerField, filePickerMenu, filePickerInput, fileStatus, fileDiff, fileImage,
          fileVideo, fileVideoPreviewBtn, fileTouchToolbar, fileTouchActions, fileTouchDpad,
          fileTouchCopyBtn, fileTouchPasteBtn, fileTouchSelectBtn, fileTouchUpBtn, fileTouchLeftBtn,
          fileTouchDownBtn, fileTouchRightBtn, fileModeDiffBtn, fileModePreviewBtn, fileDownloadBtn,
          fileBackdrop, fileViewer, fileCloseBtn, fileUnsavedBackdrop, fileUnsavedDialog,
          filePasteBackdrop, filePasteDialog, filePasteInput, fileEditBtn, chatInner,
          codeBlockCopyRuntime, appConfirm, appConfirmFocusableControls, resolveAppConfirm,
          sendChoice, closeSendChoiceDialog, queueViewer, hideQueueViewer, helpViewer,
          hideHelpViewer, diagViewer, hideDiagViewer, voiceController, hideVoiceSettingsDialog,
          getSelected: () => selected,
          getSessionIndex: () => sessionIndex,
          getSessionLifecycleController: () => sessionLifecycleController,
        }));
        const {
          dialogMenusController, sessionEditController: fileOpsSessionEditController,
          filePickerSearchState, fileViewerController, fileUnsavedController, fileReferenceRuntime,
          hideFilePasteDialog, currentFileViewerSessionId, ensureCurrentFileViewerSession,
          currentFileDirty, isFileViewerOpen, handleFileViewerSessionUnavailable, refreshFileCandidates,
        } = fileOpsController;
        sessionEditController = fileOpsSessionEditController;
        const queueController = (function instantiateQueueController() {
          const codoxearQueue = window.CodoxearQueue;
          if (!codoxearQueue || typeof codoxearQueue.createQueueController !== "function")
            throw new Error("Codoxear queue controller failed to load");
          return codoxearQueue.createQueueController(wiring.createQueueOptions({
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
            getQueueLen: () => currentQueueLen,
            getComposerText: () => (textarea ? textarea.value : ""),
            clearComposerInput,
            syncRecoveryUiForSession,
            kickPoll,
            setPollFastUntilMs,
            handleAppAuthLoss,
            prepareModalOpen,
            afterModalVisibilityChanged,
            el,
            iconSvg,
            confirmAction: (options) => confirmApp(options),
            recoveryPanelFocusFallback: () => null,
          }));
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
        // Copy conversation / Copy details / show / hide
        // behavior live in the CodoxearDiagnostics controller
        // (codoxear/static/app_diagnostics.js).
        // app.js owns DOM construction for the diag nodes and the thin
        // delegating wrappers below; all diag rendering authority is delegated.
        const diagController = (function instantiateDiagnosticsController() {
          const codoxearDiagnostics = window.CodoxearDiagnostics;
          if (!codoxearDiagnostics || typeof codoxearDiagnostics.createDiagnosticsController !== "function")
            throw new Error("Codoxear diagnostics controller failed to load");
          return codoxearDiagnostics.createDiagnosticsController(wiring.createDiagnosticsOptions({
            diagBackdrop,
            diagViewer,
            diagContent,
            diagStatus,
            diagCloseBtn,
            diagCopyConversationBtn,
            diagCopyBtn,
            getSelected: () => selected,
            getSessionInfo: (sid) => sessionIndex.get(sid),
            api,
            setToast,
            copyToClipboard,
            copyConversation,
            recoveryDetailsText,
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
          }));
        })();

        eventBindings.on(diagCopyConversationBtn, 'click', (e) => void diagController.onCopyConversationClick(e));
        eventBindings.on(diagCopyBtn, 'click', (e) => diagController.onCopyClick(e));

        async function showDiagViewer(opts) {
          return diagController.show(opts);
        }

        function hideDiagViewer(opts) {
          return diagController.hide(opts);
        }

        syncQueueSubmitState();

        const codoxearSessionLifecycle = window.CodoxearSessionLifecycle;
        if (!codoxearSessionLifecycle || typeof codoxearSessionLifecycle.createSessionLifecycleController !== "function")
          throw new Error("Codoxear session lifecycle controller failed to load");
        sessionLifecycleController = codoxearSessionLifecycle.createSessionLifecycleController(wiring.createSessionLifecycleOptions({
          nextPollGeneration: () => { pollGen += 1; return pollGen; },
          incrementPollGeneration: () => { pollGen += 1; },
          prepareSessionOpen: () => messageFlowController.prepareSessionOpen(),
          getSelected: () => selected,
          setSelected: (sessionId) => { selected = sessionId; },
          setActiveSession: (sessionId) => {
            sessionsWrap.querySelectorAll(".session.active").forEach((element) => element.classList.remove("active"));
            const active = sessionsWrap.querySelector(`.session[data-session-id="${sessionId}"]`);
            if (active) active.classList.add("active");
          },
          saveComposerDraft: saveSelectedComposerDraft,
          loadComposerDraft: loadSelectedComposerDraft,
          closeUnattendedForOtherSession: (sessionId) => {
            if (unattendedController.isOpen() && unattendedController.menuSessionId() !== sessionId) hideUnattendedMenu();
          },
          persistSelected: (sessionId) => storageSetItem("codexweb.selected", sessionId),
          removePersistedSelected: () => storageRemoveItem("codexweb.selected"),
          setSessionHash,
          resetTranscriptForSession: () => {
            transcriptSlotRuntime.setActivePending();
            clearRenderedTranscriptRange();
            turnOpen = false;
          },
          clearTranscriptForRemovedSession: clearRenderedTranscriptRange,
          syncAttachments: () => attachmentsController.syncStagedAttachmentsFromSelectedSession(),
          clearAttachments: () => attachmentsController.setStagedAttachments([]),
          syncAttachmentButton: () => attachmentsController.syncAttachButtonState(),
          updateQueueBadge,
          setStatus,
          setContext,
          setTyping,
          resetChatRenderState,
          getSession: (sessionId) => sessionIndex.get(sessionId),
          isCurrent: (sessionId, generation) => selected === sessionId && pollGen === generation,
          setTitle: (session, sessionId) => { titleLabel.textContent = session ? sessionTitleWithId(session) : sessionId ? String(sessionId) : "No session selected"; },
          setNoSessionTitle: () => { titleLabel.textContent = "No session selected"; },
          markClickLoad: () => { clickLoadT0 = performance.now(); clickMetricPending = true; },
          setTurnOpen: (value) => { turnOpen = Boolean(value); },
          updateTypingStats: updateTypingStatsFromSession,
          beginFileViewerSync: () => {
            const started = Boolean(isFileViewerOpen() && !currentFileDirty());
            if (started) void ensureCurrentFileViewerSession().catch((error) => console.error("file viewer session sync failed after selection", error));
            return started;
          },
          finishFileViewerSync: (sessionId, started, refreshCandidates) => {
            if (isFileViewerOpen() && !currentFileDirty() && !started) void ensureCurrentFileViewerSession();
            else if (isFileViewerOpen() && !currentFileDirty() && currentFileViewerSessionId() === sessionId) {
              void refreshCandidates({ sessionId }).catch((error) => console.error("file candidates refresh failed after transcript load", error));
            }
          },
          handleFileViewerSessionUnavailable,
          getTailCache: (sessionId) => transcriptSlotRuntime.getTailCache(sessionId),
          tailCacheMatchesSession,
          applyCachedTail,
          renderTranscriptLoading,
          renderTranscriptLoadError,
          messageFlow: () => messageFlowController,
          api,
          initPageLimit: () => INIT_PAGE_LIMIT,
          handleAuthLoss: handleAppAuthLoss,
          refreshSessions,
          isDisposed: () => appDisposed,
          kickPoll,
          messagePollDelayMs,
          updateTranscriptSlot: updateSessionTranscriptSlot,
          renderPendingTranscriptSlot,
          applySessionRuntimeFromTail,
          renderSessionTail,
          openMessageEventSource,
          isMobile,
          closeSidebar: () => setSidebarOpen(false),
          updateUnattendedButton: updateUnattendedBtnState,
          refreshFileCandidates,
          isUnattendedOpen: () => unattendedController.isOpen(),
          hideUnattendedMenu,
          syncComposerSendButton,
          syncQueueSubmitState,
          saveSessionScrollPosition: (sessionId) => transcriptScrollRuntime.saveSessionScrollPosition(sessionId),
          restoreSessionScrollPosition: (sessionId) => transcriptScrollRuntime.restoreSessionScrollPosition(sessionId),
          clearSessionScrollPosition: (sessionId) => transcriptScrollRuntime.clearSessionScrollPosition(sessionId),
          setActiveTranscriptPending: () => transcriptSlotRuntime.setActivePending(),
          deleteTranscriptSession: (sessionId) => transcriptSlotRuntime.deleteSession(sessionId),
          dropPendingUserRows: (sessionId) => dropPendingUserRows(sessionId, () => true),
          sessionIdFromHash,
          rememberPendingHashSession,
          sessionSelectable,
          normalizeAgentBackendName,
          providerChoiceToSettings,
          backendSupportsFast,
          setToast,
          confirmAction: (options) => confirmApp(options),
          syncRecoveryUiForSession,
          sleep: (ms) => new Promise((resolve) => setTimeout(resolve, ms)),
          consoleError: (...args) => console.error(...args),
        }));

        const codoxearSessionRefresh = window.CodoxearSessionRefresh;
        if (!codoxearSessionRefresh || typeof codoxearSessionRefresh.createSessionRefreshController !== "function")
          throw new Error("Codoxear session refresh controller failed to load");
        sessionRefreshController = codoxearSessionRefresh.createSessionRefreshController(wiring.createSessionRefreshOptions({
          api,
          isDisposed: () => appDisposed,
          apiResponseNotModified,
          getLatestSessions: () => latestSessions,
          setLatestSessions: (sessions) => { latestSessions = sessions; },
          setNewSessionDefaults: (defaults) => { newSessionDefaults = defaults; },
          emptyDefaults: () => ({
            default_backend: "pi",
            backends: { codex: legacyCodexLaunchDefaults(), pi: emptyPiLaunchDefaults(), cc: emptyCcLaunchDefaults() },
          }),
          setTmuxAvailable: (available) => { tmuxAvailable = Boolean(available); },
          setRecentCwds: (cwds) => { recentCwds = cwds; },
          refreshNewSessionDefaults: () => {
            if (newSessionDialogController.isOpen()) newSessionDialogController.refreshDefaults();
          },
          clearFileDiscoveryCaches: () => fileReferenceRuntime.clearDiscoveryCaches(),
          useDesktopSessionActions,
          setSessionIndex: (index) => { sessionIndex = index; },
          getSelected: () => selected,
          clearSelectedSessionAfterRemoval: (...args) => sessionLifecycleController.clearSelectedSessionAfterRemoval(...args),
          applySessionListTranscriptIdentity,
          syncRecoveryUiForSession,
          syncAttachments: () => attachmentsController.syncStagedAttachmentsFromSelectedSession(),
          clearAttachments: () => attachmentsController.setStagedAttachments([]),
          renderSessions: (sessions, options) => sidebarController.renderSessions(sessions, options),
          hasDeferredRefresh: () => sidebarController.hasDeferredRefresh(),
          setTitle: (title) => { titleLabel.textContent = title; },
          sessionTitle: sessionTitleWithId,
          updateTypingStats: updateTypingStatsFromSession,
          updateUnattendedButton: updateUnattendedBtnState,
          updateQueueBadge,
          syncComposerSendButton,
          syncQueueSubmitState,
          maybeSelectPendingHashSession,
        }));

        eventBindings.on($("#helpBtnSide"), 'click', (e) => {
          e.preventDefault();
          e.stopPropagation();
          showHelpViewer({ opener: e.currentTarget });
        });
        eventBindings.on($("#settingsBtnSide"), 'click', (e) => {
          e.preventDefault();
          e.stopPropagation();
          showVoiceSettingsDialog();
        });
        eventBindings.on(helpCloseBtn, 'click', (e) => {
          e.preventDefault();
          e.stopPropagation();
          hideHelpViewer();
        });
        eventBindings.on(helpBackdrop, 'click', () => hideHelpViewer());

        eventBindings.on(diagBtn, 'click', (e) => {
          e.preventDefault();
          e.stopPropagation();
          void showDiagViewer({ opener: e.currentTarget });
        });
        eventBindings.on(diagCloseBtn, 'click', (e) => {
          e.preventDefault();
          e.stopPropagation();
          hideDiagViewer();
        });
        eventBindings.on(diagBackdrop, 'click', () => hideDiagViewer());
        eventBindings.on($("#newBtn"), 'click', async () => {
          newSessionDialogController.open();
        });
        eventBindings.on($("#chatEmptyNewBtn"), 'click', async () => {
          newSessionDialogController.open();
        });
        const interruptController = codoxearInterrupt.createInterruptController(wiring.createInterruptOptions({
          selectedSessionId: () => selected,
          setToast,
          api,
          now: Date.now,
          setPollFastUntilMs,
          kickPoll,
        }));
        eventBindings.on(interruptBtn, 'click', (e) => {
          e.preventDefault();
          e.stopPropagation();
          void interruptController.interruptSelectedSession();
        });

        eventBindings.on($("#logoutBtnSide"), 'click', async () => {
          try {
            await api("/api/logout", { method: "POST" });
          } catch (e) {
            console.error("logout failed", e);
          } finally {
            if (appDisposed) return;
            cleanupApp();
            renderLogin(renderApp);
          }
        });

        eventBindings.on(toggleSidebarBtn, 'click', () => {
          if (isMobile()) {
            setSidebarOpen(!document.body.classList.contains("sidebar-open"));
            return;
          }
          setSidebarCollapsed(!document.body.classList.contains("sidebar-collapsed"));
        });
	        eventBindings.on(backdrop, 'click', () => setSidebarOpen(false));

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
        eventBindings.on(jumpBtn, 'click', () => {
          void jumpToLatest();
        });
        eventBindings.on(olderBtn, 'click', () => {
          void loadOlderMessages({ auto: false });
        });
        eventBindings.on(olderRetryBtn, 'click', () => {
          clearOlderLoadError();
          void loadOlderMessages({ auto: false });
        });

        const codoxearIOSViewport = window.CodoxearIOSViewport;
        if (!codoxearIOSViewport || typeof codoxearIOSViewport.createIOSViewportController !== "function")
          throw new Error("Codoxear iOS viewport controller failed to load");
        const iosViewportController = codoxearIOSViewport.createIOSViewportController(wiring.createIOSViewportOptions({
          windowTarget: window,
          documentTarget: document,
          navigatorTarget: navigator,
          textarea,
          isTextEntryElement,
          updateAppHeightVar,
          transcriptScrollRuntime,
          addAppEvent,
          requestAnimationFrame,
          setTimeout,
          clearTimeout,
        }));
        updateQueueBadge();
        syncQueueSubmitState();
        syncComposerSendButton();
        composerController = codoxearComposer.createComposerController(wiring.createComposerOptions({
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
          sessionLaunchFailed,
          getSending: () => sending,
          getCurrentRunning: () => currentRunning,
          getStagedAttachments: () => attachmentsController.getStagedAttachments(),
          isModalOpen: () => modalIsolationTargets.some(isModalTargetOpen),
          api,
          setToast,
          setPollFastUntilMs,
          kickPoll,
          sendText: (raw, options) => messageFlowController.sendText(raw, options),
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
        }));

        setActiveAppCleanup(cleanupApp);
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
	            if (pick) await sessionLifecycleController.selectSession(pick);
              void (async () => {
                try {
                  await refreshVoiceBackgroundState({ force: true, primeNotifications: true });
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
            secondaryPollController.scheduleSecondaryPoll();
              addAppEvent(window, "hashchange", async () => {
                await sessionLifecycleController.selectSessionFromHash({ refreshIfMissing: true, deferIfMissing: true });
              });
              addAppEvent(window, "beforeunload", () => {
                cleanupApp();
              });
              addAppEvent(document, "visibilitychange", () => {
                if (appDisposed) return;
                if (document.visibilityState === "visible") {
                  if (selected) messageFlowController.resumeLiveDelivery();
                  scheduleSessionsPoll(0);
                  secondaryPollController.scheduleSecondaryPoll(0);
                  return;
                }
                if (selected) kickPoll(messagePollDelayMs());
                scheduleSessionsPoll(sessionsPollDelayMs());
                secondaryPollController.scheduleSecondaryPoll(secondaryPollDelayMs());
              });
              addAppEvent(window, "online", () => {
                if (appDisposed) return;
                networkStatus.reportSuccess();
                messageFlowController.resetMessagePollBackoff();
                sessionsPollErrorStreak = 0;
                secondaryPollErrorStreak = 0;
                if (selected) {
                  messageFlowController.resumeLiveDelivery();
                  kickPoll(0);
                }
                scheduleSessionsPoll(0);
                secondaryPollController.scheduleSecondaryPoll(0);
              });
              addAppEvent(window, "offline", () => {
                if (appDisposed) return;
                networkStatus.sync();
                messageFlowController.closeMessageEventSource();
                if (selected) kickPoll(messagePollDelayMs());
                scheduleSessionsPoll(sessionsPollDelayMs());
                secondaryPollController.scheduleSecondaryPoll(secondaryPollDelayMs());
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

    return Object.freeze({ renderApp });
  }


  global.CodoxearEventBindings = Object.freeze({ createEventBindings });
  global.CodoxearToast = Object.freeze({ createToastController });
  global.CodoxearApplicationComposition = Object.freeze({ createApplicationComposition });
})(window);
