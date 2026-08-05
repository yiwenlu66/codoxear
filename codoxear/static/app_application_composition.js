/* Application composition owns concrete lifecycle, controller assembly, and UI behavior.
 * app_application_runtime.js remains the stable bootstrap facade. */
(function installCodoxearApplicationComposition(global) {
  "use strict";

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
          isMobile,
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

        function closeTransientOverlays({ closeSearch = false } = {}) {
          if (unattendedController.isOpen()) hideUnattendedMenu();
          if (closeSearch && chatSearchController.isOpen()) closeChatSearch();
          if (document.body.classList.contains("sidebar-open")) setSidebarOpen(false);
          filePickerMenuState.close();
          filePickerMenu.classList.remove("open");
          filePickerInput.setAttribute("aria-expanded", "false");
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
        function renderStatusChip() {
          const q = currentQueueLen;
          const base = currentRunning ? "Busy" : q ? `Queue ${q}` : "Idle";
          statusChip.style.display = "inline-flex";
          statusChip.textContent = currentSubagentsRunning > 0 ? `${base} · ▸${currentSubagentsRunning}` : base;
        }

        function setStatus({ running, queueLen }) {
          const q = Math.max(0, Number(queueLen) || 0);
          const wasRunning = currentRunning;
          currentRunning = Boolean(running);
          currentQueueLen = q;
          renderStatusChip();
          const canInterrupt = Boolean(running && selected);
          interruptBtn.style.display = canInterrupt ? "inline-flex" : "none";
          interruptBtn.disabled = !canInterrupt;
          if (wasRunning && !currentRunning) {
            // no-op placeholder; keep transition boundary for future UI behavior
          }
          attachmentsController.syncAttachButtonState();
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
        eventBindings.on(ctxChip, 'click', () => {
          if (!lastToken) return;
          setToast(`ctx ${lastToken.used}/${lastToken.ctx} (${lastToken.pct ?? "?"}% left)`);
        });

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
        const codoxearTranscriptView = window.CodoxearTranscriptView;
        if (
          !codoxearMessageRows ||
          !codoxearTranscriptView ||
          typeof codoxearTranscriptView.createTranscriptViewController !== "function" ||
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
          throw new Error("Codoxear transcript view helpers failed to load");

        let transcriptViewController = null;

        function transcriptView() {
          if (!transcriptViewController) throw new Error("transcript view controller is not initialized");
          return transcriptViewController;
        }

        const messageCopyNavigationRuntime = codoxearMessageRows.createMessageCopyNavigationRuntime(wiring.createMessageCopyNavigationOptions({ root: chatInner }));

        function renderedMessageRows() {
          return transcriptView().renderedMessageRows();
        }

        function loadedUserMessageRows() {
          return transcriptView().loadedUserMessageRows();
        }

        function loadedCopyMessageRows() {
          return transcriptView().loadedCopyMessageRows();
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

        function applyChatSearchMarks(matches, currentRow, query) {
          return transcriptView().applyChatSearchMarks(matches, currentRow, query);
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

        function isTouchCopyMode() {
          return window.matchMedia("(max-width: 700px), (pointer: coarse)").matches;
        }

        addAppEvent(chatInner, "click", (e) => {
          if (!isTouchCopyMode() || window.getSelection().toString()) return;
          const target = e.target && typeof e.target.closest === "function" ? e.target : null;
          if (!target || target.closest("a, button, input, select, textarea, [role='link'], mark")) return;
          const pre = codoxearCodeCopy.codePreFromTarget(target);
          if (pre && chatInner.contains(pre)) {
            codeBlockCopyRuntime.toggleTouchPre(pre, chatInner);
            return;
          }
          const bubble = target.closest(".msg");
          const row = bubble && bubble.closest(".msg-row");
          if (!row || !chatInner.contains(row) || (target !== bubble && !target.closest(".md"))) return;
          messageCopyNavigationRuntime.toggleTouchRow(row);
        });

        function prefersReducedMotion() {
          return codoxearViewport.prefersReducedMotion();
        }

        const navigationPulseController = codoxearNavigationPulse.createNavigationPulseController(wiring.createNavigationPulseOptions({
          setActiveRow: setActiveMessageCopyRow,
          activeElementIsCopyButton: activeElementIsMessageCopyButton,
          setTimeout,
        }));

        const hintModeController = (function instantiateHintModeController() {
          const codoxearHintMode = window.CodoxearHintMode;
          if (!codoxearHintMode || typeof codoxearHintMode.createHintModeController !== "function")
            throw new Error("Codoxear hint mode controller failed to load");
          return codoxearHintMode.createHintModeController(wiring.createHintModeOptions({
            documentTarget: document,
            isTextEntryElement,
            isMobile,
            modalIsolationTargets,
            isModalTargetOpen,
            addAppEvent,
            shellHints: Array.from(document.querySelectorAll("[data-hint]")).map((element) => ({
              label: element.getAttribute("data-hint"),
              element,
            })),
          }));
        })();

        const activateModalButtonForKey = codoxearModal.createModalKeyboardHandler(wiring.createModalKeyboardHandlerOptions({
          modalIsolationTargets,
          isTextEntryElement,
        }));
        addAppEvent(document, "keydown", activateModalButtonForKey);

        // --- Direct (no-leader) Vimium-style shortcuts ---
        // These fire when not in a text-entry element, no modal is open, and
        // hint mode is not active. They don't conflict with hint-mode letters
        // (which require `f` leader first) — the context disambiguates.
        addAppEvent(document, "keydown", (e) => {
          if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey) return;
          if (e.isComposing) return;
          if (hintModeController && hintModeController.isActive()) return;
          if (isTextEntryElement(document.activeElement)) return;
          if (isModalTargetOpen(appConfirm) || isModalTargetOpen(sendChoice) || isModalTargetOpen(queueViewer) || isModalTargetOpen(helpViewer) || isModalTargetOpen(diagViewer) || isModalTargetOpen(editViewer) || newSessionDialogController.isOpen() || isFileViewerOpen()) return;
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
                  sessionLifecycleController.clearDeletedSessionClientState(sid);
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
          return codoxearChatNavigation.createChatNavigationController(wiring.createChatNavigationOptions({
            prevUserBtn,
            nextUserBtn,
            getSelected: () => selected,
            getPollGen: () => pollGen,
            api,
            loadTranscriptWindowAtCursor,
            loadOlderMessages,
            loadedUserMessageRows,
            loadedCopyMessageRows,
            loadedUserJumpTarget,
            loadedCopyJumpTarget,
            getScrollTop: () => chat.scrollTop,
            prefersReducedMotion,
            pulseNavigatedRow: (row) => navigationPulseController.pulseNavigatedRow(row),
            setToast,
            openChatSearch,
            handleAppAuthLoss,
            isTextEntryElement,
            modalIsolationTargets,
            isModalTargetOpen,
            addAppEvent,
            documentTarget: document,
          }));
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
          transcriptView().clearChatSearchMarks();
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
          return transcriptView().oldestRenderedHistoryCursor() || activeTailHistoryCursor;
        }

        function clearRenderedTranscriptRange() {
          activeTailHistoryCursor = null;
          setOlderState({ hasMore: false, isLoading: false });
          transcriptScrollRuntime.markLiveTail();
        }

        function initPageLimit() {
          return INIT_PAGE_LIMIT;
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
          typeof codoxearTranscript.hasHumanOriginatedUserEvent !== "function" ||
          typeof codoxearTranscript.createTranscriptRenderRuntime !== "function" ||
          typeof codoxearTranscript.createTranscriptDomRuntime !== "function" ||
          typeof codoxearTranscript.createTranscriptScrollRuntime !== "function" ||
          typeof codoxearTranscript.createTranscriptEventRuntime !== "function" ||
          typeof codoxearTranscript.createOlderLoadRuntime !== "function" ||
          typeof codoxearTranscript.createLoadedChatSearchRuntime !== "function" ||
          typeof codoxearTranscript.createChatSearchAllRuntime !== "function"
        )
          throw new Error("Codoxear transcript helpers failed to load");

        const olderLoadRuntime = codoxearTranscript.createOlderLoadRuntime(wiring.createOlderLoadOptions({
          olderWrap,
          olderButton: olderBtn,
          olderError,
          olderErrorText,
          AbortControllerCtor: AbortController,
          nowMs: () => performance.now(),
          autoCooldownMs: OLDER_AUTO_COOLDOWN_MS,
        }));

        chatSearchController = (function instantiateChatSearchController() {
          const codoxearChatSearch = window.CodoxearChatSearch;
          if (!codoxearChatSearch || typeof codoxearChatSearch.createChatSearchController !== "function")
            throw new Error("Codoxear chat search controller failed to load");
          return codoxearChatSearch.createChatSearchController(wiring.createChatSearchOptions({
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
            loadTranscriptWindowAtCursor,
            handleAppAuthLoss,
            syncVisibleTimeIndicator: () => transcriptScrollRuntime.syncVisibleTimeIndicator(),
            renderedMessageRows,
            rowSearchText,
            clearChatSearchMarks,
            applyChatSearchMarks,
            pulseNavigatedRow: (row) => navigationPulseController.pulseNavigatedRow(row),
            prefersReducedMotion,
          }));
        })();

        const transcriptSlotRuntime = codoxearTranscript.createTranscriptSlotRuntime(wiring.createTranscriptSlotOptions({
          getSession: (sessionId) => sessionIndex.get(sessionId) || null,
          maxTailEvents: INIT_PAGE_LIMIT,
        }));

        function activeTranscriptSnapshot() {
          return transcriptSlotRuntime.activeSnapshot();
        }

        const typingRowRuntime = codoxearTranscript.createTypingRowRuntime(wiring.createTypingRowOptions({
          root: chatInner,
          bottomSentinel,
          el,
          shouldAutoScroll: () => transcriptScrollRuntime.snapshot().autoScroll,
          scheduleScrollToBottom: () => transcriptScrollRuntime.scheduleScrollToBottom(),
        }));

        const transcriptScrollRuntime = codoxearTranscript.createTranscriptScrollRuntime(wiring.createTranscriptScrollOptions({
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
        }));

        const transcriptDomRuntime = codoxearTranscript.createTranscriptDomRuntime(wiring.createTranscriptDomOptions({
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
        }));

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
          attachmentsController.setAttachCount(0);
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
          if (queueController) {
            queueController.updateQueueBadge();
            if (queueViewer.style.display === "flex") void refreshQueueViewer();
          }
        }

          function markClickFirstPaint() {
            if (!clickMetricPending) return;
            clickMetricPending = false;
            const dt = performance.now() - clickLoadT0;
            pushPerfSample("click_to_first_message_ms", dt);
          }

        function updateTypingStatsFromSession(session) {
          return messageFlowController.updateTypingStatsFromSession(session);
        }

        function setTyping(show) {
          typingRowRuntime.setVisible(show);
          typingRowRuntime.setSubagentVisible(!show);
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

      const transcriptEventRuntime = codoxearTranscript.createTranscriptEventRuntime(wiring.createTranscriptEventOptions({
        eventKey: codoxearMessageIdentity.eventKey,
        pendingMatchKey: codoxearMessageIdentity.pendingMatchKey,
        normalizePendingText: codoxearMessageIdentity.normalizeTextForPendingMatch,
        assistantDedupeKey: codoxearMessageIdentity.chatAssistantDedupeKey,
        maxRecentEventKeys: 320,
      }));

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

      const pendingUserController = codoxearPendingUser.createPendingUserController(wiring.createPendingUserOptions({
        selectedSessionId: () => selected,
        takePendingUserMatch,
        chatInner,
        markdownHtml: chatMarkdownHtmlCached,
        time24,
        rebuildDecorations,
        markEventSeen,
      }));

        transcriptViewController = codoxearTranscriptView.createTranscriptViewController(wiring.createTranscriptViewOptions({
          root: chatInner,
          bottomSentinel,
          document,
          el,
          messageRows: codoxearMessageRows,
          transcript: codoxearTranscript,
          getSelectedSessionId: () => selected,
          getMessageRowDeps: messageRowDeps,
          renderRuntime: {
            normalizeEvents: normalizedTranscriptEvents,
            consumePendingUserIfMatches: (event, sessionId) => pendingUserController.consumePendingUserIfMatches(event, sessionId),
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
          },
        }));

        attachmentsController = codoxearAttachments.createAttachmentsController(wiring.createAttachmentsOptions({
          attachBtn,
          imgInput,
          composer,
          textarea,
          getSelected: () => selected,
          getSessionInfo: (sessionId) => sessionIndex.get(sessionId) || null,
          patchSessionInfo: (sessionId, patch) => {
            const current = sessionIndex.get(sessionId);
            if (!current) return;
            Object.assign(current, patch || {});
            sessionIndex.set(sessionId, current);
          },
          getSending: () => sending,
          sessionLaunchFailed,
          sessionHasUnknownSend,
          sessionIsOrphanRecovery,
          sessionHasOrphanQueueRecovery,
          api,
          setToast,
          handleAppAuthLoss,
          refreshSessions,
          setPollFastUntilMs,
          kickPoll,
          resizeComposer,
          getTray: () => $("#stagedAttachments"),
          el,
          fmtBytes,
          safeAttachmentStem,
          isLikelyHeic,
          looksLikeImage,
          b64FromBytes,
          dataTransferHasFiles,
          extractFilesFromClipboardData,
          extractFilesFromDropData,
          addEventListener: addAppEvent,
          uploadMaxBytes: ATTACH_UPLOAD_MAX_BYTES,
        }));

        messageFlowController = codoxearMessageFlow.createMessageFlowController(wiring.createMessageFlowOptions({
          getSelected: () => selected,
          getGeneration: () => pollGen,
          isAppDisposed: () => appDisposed,
          getTurnOpen: () => turnOpen,
          setTurnOpen: (value) => { turnOpen = Boolean(value); },
          getSessionInfo: (sessionId) => sessionIndex.get(sessionId) || null,
          patchSessionInfo: (sessionId, patch) => {
            const current = sessionIndex.get(sessionId);
            if (!current) return;
            Object.assign(current, patch || {});
            sessionIndex.set(sessionId, current);
          },
          sessionLaunchFailed,
          api,
          resolveAppUrl,
          handleAppAuthLoss,
          refreshSessions,
          openSession: (...args) => sessionLifecycleController.openSession(...args),
          clearSelectedSessionAfterRemoval: (...args) => sessionLifecycleController.clearSelectedSessionAfterRemoval(...args),
          activeTranscriptSnapshot,
          updateSessionTranscriptSlot,
          renderPendingTranscriptSlot,
          renderSessionTail,
          applySessionRuntimeFromTail,
          resetChatRenderState,
          setAttachCount: (count) => attachmentsController.setAttachCount(count),
          setLiveCursor: (cursor) => transcriptSlotRuntime.setLiveCursor(cursor),
          appendEvent,
          appendTailSnapshotEvents,
          setStatus,
          setContext,
          setTyping,
          setSubagentsRunning: (value) => {
            currentSubagentsRunning = value;
            renderStatusChip();
          },
          updateSessionTitle: (session) => { titleLabel.textContent = sessionTitleWithId(session); },
          initPageLimit,
          typingRowRuntime,
          getSending: () => sending,
          setSending: (value) => { sending = Boolean(value); },
          getCurrentRunning: () => currentRunning,
          setCurrentRunning: (value) => { currentRunning = Boolean(value); },
          getStagedAttachments: () => attachmentsController.getStagedAttachments(),
          normalizedStagedAttachments: (list) => attachmentsController.normalizedStagedAttachments(list),
          setSelectedSessionPendingAttachment: (sessionId, value) => attachmentsController.setSelectedSessionPendingAttachment(sessionId, value),
          syncSendButtonState: syncComposerSendButton,
          syncAttachButtonState: () => attachmentsController.syncAttachButtonState(),
          syncQueueSubmitState,
          syncRecoveryUiForSession,
          confirmAction: (options) => confirmApp(options),
          setToast,
          isTranscriptRenewalCommand,
          nextLocalEchoId: () => transcriptEventRuntime.nextLocalEchoId(),
          renderedAtLiveTail: () => transcriptScrollRuntime.snapshot().renderedAtLiveTail,
          clearTranscriptDom,
          clearRenderedTranscriptRange,
          setOlderState,
          getSessionTranscriptSlot,
          addPendingUser: (pending) => transcriptEventRuntime.addPendingUser(pending),
          deleteTailCache: (sessionId) => transcriptSlotRuntime.deleteTailCache(sessionId),
          beginTranscriptRenewal,
          clearLiveCursor: () => transcriptSlotRuntime.clearLiveCursor(),
          invalidateOlderLoad,
          dropPendingUser: (sessionId, localId) => transcriptEventRuntime.dropPendingUsers(sessionId, (pending) => pending && pending.id === localId),
          removePendingUserRow: (localId) => {
            const pendingEl = chatInner.querySelector(`.msg.user[data-local-id="${localId}"]`);
            if (!pendingEl) return;
            const pendingRow = pendingEl.closest(".msg-row");
            if (pendingRow) pendingRow.remove();
            else pendingEl.remove();
          },
          hasPendingForSession: (sessionId) => transcriptEventRuntime.hasPendingForSession(sessionId),
          visibilityState: () => document.visibilityState,
          navigatorValue: () => (typeof navigator === "undefined" ? undefined : navigator),
          reportTransportSuccess: () => networkStatus.reportSuccess(),
          reportTransportFailure: () => networkStatus.reportFailure(),
          EventSource: typeof EventSource === "function" ? EventSource : null,
          AbortController: typeof AbortController === "function" ? AbortController : null,
          setTimeout: window.setTimeout.bind(window),
          clearTimeout: window.clearTimeout.bind(window),
          now: () => Date.now(),
          consoleWarn: (...args) => console.warn(...args),
          consoleError: (...args) => console.error(...args),
        }));

        function messagePollDelayMs(now = Date.now()) {
          return messageFlowController.messagePollDelayMs(now);
        }

        function kickPoll(ms = 0) {
          return messageFlowController.kickPoll(ms);
        }

        function setPollFastUntilMs(value) {
          messageFlowController.setPollFastUntilMs(value);
        }

        function openMessageEventSource(sessionId = selected, generation = pollGen) {
          return messageFlowController.openMessageEventSource(sessionId, generation);
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

        function clearCommitUnknownSend(sid, previewText = "") {
          return sessionLifecycleController.clearCommitUnknownSend(sid, previewText);
        }


         const sidebarController = codoxearSessions.createSessionsController(wiring.createSessionsOptions({
           sessionsWrap,
           sidebarEmptyHint,
           el,
           iconSvg,
           sidebarRenderSignature,
           sidebarSessionEntries,
           sessionDisplayName,
           sessionLaunchFailed,
           sessionLaunchPending,
           redactedLaunchErrorText,
           fmtRelativeAge,
           sidebarEffortCode,
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
           clearDeletedSessionClientState: (...args) => sessionLifecycleController.clearDeletedSessionClientState(...args),
           refreshSessions,
           setToast,
           openEditSession: (sid) => sessionEditController.openEditSession(sid),
           duplicateSession: async (session) => {
             const cwd = session && session.cwd && session.cwd !== "?" ? session.cwd : "";
             if (!cwd) {
               setToast("cwd unavailable");
               return;
             }
             await sessionLifecycleController.spawnSessionWithCwd(
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
           selectSession: (...args) => sessionLifecycleController.selectSession(...args),
           setSidebarOpen,
           now: () => Date.now(),
           performanceNow: () => performance.now(),
           consoleError: (...args) => console.error(...args),
         }));

        function refreshSessions() {
          return sessionRefreshController.refreshSessions();
        }


        function appendEvent(ev) {
          transcriptView().appendEvent(ev);
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
          return transcriptView().renderTranscript(events, { preserveScroll });
        }

        function renderDetachedTranscriptWindow(events, { hasMore = false } = {}) {
          return transcriptView().renderDetachedTranscriptWindow(events, { hasMore });
        }

        async function loadTranscriptWindowAtCursor(cursor) {
          const cleanCursor = String(cursor || "").trim();
          if (!selected || !cleanCursor) return null;
          const sid = selected;
          const gen = pollGen;
          invalidateOlderLoad();
          try {
            const data = await api(`/api/sessions/${sid}/messages/window?cursor=${encodeURIComponent(cleanCursor)}&before=30&after=30`);
            if (selected !== sid || pollGen !== gen) return null;
            const events = Array.isArray(data.events) ? data.events : [];
            activeTailHistoryCursor = usableOlderHistoryCursor(data);
            setOlderState({ hasMore: Boolean(activeTailHistoryCursor), isLoading: false });
            if (!renderDetachedTranscriptWindow(events, { hasMore: Boolean(activeTailHistoryCursor) })) return null;
            return data;
          } catch (error) {
            if (error && error.status === 401) handleAppAuthLoss();
            else if (selected === sid && pollGen === gen) showOlderLoadError();
            return null;
          }
        }

        function prependOlderEvents(allEvents, { preserveViewport = false } = {}) {
          return transcriptView().prependOlderEvents(allEvents, { preserveViewport });
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
              await sessionLifecycleController.openSession(sid, { useCache: false });
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
            sessionLifecycleController.clearDeletedSessionClientState(sessionId);
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
          attachmentsController.syncAttachButtonState();
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
          transcriptView().renderLoadingRow();
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
          transcriptView().renderLoadErrorRow({
            message: `Could not load transcript.${reason}`,
            onRetry: (e) => {
              e.preventDefault();
              e.stopPropagation();
              if (selected !== sessionId) return;
              void sessionLifecycleController.openSession(sessionId, { useCache: true });
            },
          });
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

        async function applyLiveMessageData(sid, gen, data) {
          return messageFlowController.applyLiveMessageData(sid, gen, data);
        }

        async function pollMessages(sid = selected, gen = pollGen) {
          return messageFlowController.pollMessages(sid, gen);
        }

        async function jumpToLatest() {
          if (!selected) return;
          const sid = selected;
          invalidateOlderLoad();
          transcriptScrollRuntime.enableAutoScroll();
          try {
            await sessionLifecycleController.openSession(sid, { useCache: false, fallbackToCacheOnFailure: true });
          } catch (e) {
            if (selected !== sid) return;
            setToast(`jump error: ${e && e.message ? e.message : "unknown error"}`);
          }
          if (selected !== sid) return;
          transcriptScrollRuntime.scheduleScrollToBottom({ syncJump: true });
          kickPoll(0);
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
          void sessionLifecycleController.selectSession(sid)
            .catch((e) => {
              if (e && e.status === 401) handleAppAuthLoss();
              else console.error("pending hash session select failed", e);
            })
            .finally(() => {
              pendingHashSessionSelectInFlight = false;
            });
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
        const dialogMenusController = codoxearDialogMenus.createDialogMenusController(wiring.createDialogMenusOptions({
          sessionEditController: () => sessionEditController,
          newSessionDialogController: () => newSessionDialogController,
        }));

        const FILE_CANDIDATE_CACHE_TTL_MS = 15000;
        const filePickerMenuState = codoxearFilePicker.createMenuState(wiring.createMenuStateOptions({
          normalizeLineNumber,
        }));
        const filePickerDomRuntime = codoxearFilePicker.createMenuDomRuntime(wiring.createMenuDomOptions({
          field: filePickerField,
          menu: filePickerMenu,
          input: filePickerInput,
          menuState: filePickerMenuState,
        }));
        const filePickerSearchState = codoxearFilePicker.createSearchState(wiring.createSearchStateOptions({
          blocked: () => blockUnavailableFileAction(),
          currentSessionId: () => currentFileViewerSessionId() || selected || "",
          api,
          inputValue: () => filePickerInput.value,
          isMenuOpen: () => filePickerMenuState.isOpen(),
          renderMenu: () => renderFilePickerMenu(),
          applyMenuState: () => applyFileMenuState(),
          normalizeFileApiPath: (value) => normalizeFileApiPath(value),
        }));
        const filePickerEntryRuntime = codoxearFilePicker.createEntryRuntime(wiring.createEntryOptions({
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
        }));
        const filePickerRenderRuntime = codoxearFilePicker.createMenuRenderRuntime(wiring.createMenuRenderOptions({
          menu: filePickerMenu,
          menuState: filePickerMenuState,
          inputValue: () => filePickerInput.value,
          visibleEntries: () => filePickerEntryRuntime.visibleEntries(),
          searchSnapshot: () => filePickerSearchSnapshot(),
          normalizeDraftFilePath: (query) => normalizeDraftFilePath(query),
          draftSuppressed: () => filePickerSearchState.draftSuppressed(filePickerInput.value),
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
        }));
        const filePickerInputRuntime = codoxearFilePicker.createInputRuntime(wiring.createInputOptions({
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
        }));
        const MONACO_LOADER_TIMEOUT_MS = 4000;
        const PDFJS_LOADER_TIMEOUT_MS = 6000;
        const fileEditorRuntime = codoxearFileEditor.createFileEditorRuntime();
        const fileEditorMonacoLoader = codoxearFileEditor.createMonacoLoader(wiring.createMonacoLoaderOptions({
          resolveAppUrl,
          timeoutMs: MONACO_LOADER_TIMEOUT_MS,
        }));
        const fileEditorRenderer = codoxearFileEditor.createFileEditorRenderer(wiring.createFileEditorRendererOptions({
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
        }));
        const filePdfLoader = codoxearFileViewer.createPdfLoader(wiring.createPdfLoaderOptions({
          resolveAppUrl,
          timeoutMs: PDFJS_LOADER_TIMEOUT_MS,
        }));
        const fileFallbackRuntime = codoxearFileViewer.createFileFallbackRuntime(wiring.createFileFallbackOptions({
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
        }));
        const fileDownloadRuntime = codoxearFileViewer.createFileDownloadRuntime(wiring.createFileDownloadOptions({
          resolveAppUrl,
          document,
        }));
        const filePdfRenderRuntime = codoxearFileViewer.createFilePdfRenderRuntime(wiring.createFilePdfRenderOptions({
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
        }));
        const filePasteDialogRuntime = codoxearFileViewer.createFilePasteDialogRuntime(wiring.createFilePasteDialogOptions({
          backdrop: filePasteBackdrop,
          dialog: filePasteDialog,
          input: filePasteInput,
          prepareModalOpen,
          afterModalVisibilityChanged,
          focusActiveEditor: () => fileEditorRuntime.focusActiveCodeEditor(currentFileEditorKind()),
          requestAnimationFrame: (callback) => requestAnimationFrame(callback),
        }));
        const fileRenderSurfaceRuntime = codoxearFileViewer.createFileRenderSurfaceRuntime(wiring.createFileRenderSurfaceOptions({
          diff: fileDiff,
          image: fileImage,
          video: fileVideo,
          videoPreviewButton: fileVideoPreviewBtn,
          clearActiveVideoFallback: () => fileViewerController.clearActiveVideoFallback(),
        }));
        const fileModeControlsRuntime = codoxearFileViewer.createFileModeControlsRuntime(wiring.createFileModeControlsOptions({
          diffButton: fileModeDiffBtn,
          previewButton: fileModePreviewBtn,
          downloadButton: fileDownloadBtn,
          videoPreviewButton: fileVideoPreviewBtn,
          hideFilePasteDialog: () => hideFilePasteDialog(),
          setFileEditMode: (mode) => fileEditModeController.setFileEditMode(mode),
          syncFileEditorReadOnly: () => syncFileEditorReadOnly(),
          updateFileEditButton: () => updateFileEditButton(),
        }));
        const fileTouchToolbarRuntime = codoxearFileViewer.createFileTouchToolbarRuntime(wiring.createFileTouchToolbarOptions({
          toolbar: fileTouchToolbar,
          actions: fileTouchActions,
          dpad: fileTouchDpad,
          copyButton: fileTouchCopyBtn,
          pasteButton: fileTouchPasteBtn,
          selectButton: fileTouchSelectBtn,
        }));
        const fileViewerModalRuntime = codoxearFileViewer.createFileViewerModalRuntime(wiring.createFileViewerModalOptions({
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
        }));
        const fileUnsavedDialogRuntime = codoxearFileViewer.createFileUnsavedDialogRuntime(wiring.createFileUnsavedDialogOptions({
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
        }));
        const codoxearFileUnsaved = window.CodoxearFileUnsaved;
        if (!codoxearFileUnsaved || typeof codoxearFileUnsaved.createFileUnsavedController !== "function")
          throw new Error("Codoxear file unsaved controller failed to load");
        const fileUnsavedController = codoxearFileUnsaved.createFileUnsavedController(wiring.createFileUnsavedOptions({
          documentTarget: document,
          ElementCtor: HTMLElement,
          dialogRuntime: fileUnsavedDialogRuntime,
          getFileViewerController: () => fileViewerController,
        }));

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

        const fileEditModeController = codoxearFileEditMode.createFileEditModeController(wiring.createFileEditModeOptions({
          fileViewerController: () => fileViewerController,
        }));

        const fileInspectRuntime = codoxearFileViewer.createFileInspectRuntime(wiring.createFileInspectOptions({
          currentSessionId: () => currentFileViewerSessionId(),
          selectedSessionId: () => selected,
          normalizeFileApiPath: (value) => normalizeFileApiPath(value),
          api: (url, options) => api(url, options),
        }));

        const fileViewerController = codoxearFileViewer.createFileViewerController(wiring.createFileViewerOptions({
          el,
          fileStatus,
          fileEditButton: fileEditBtn,
          iconSvg,
          currentSessionId: () => currentFileViewerSessionId(),
          currentFileSessionId: () => currentFileSessionId(),
          normalizeLineNumber,
          normalizeFileApiPath,
          isFileViewerOpen: () => isFileViewerOpen(),
          hideFileUnsavedDialog: (choice) => fileUnsavedController.hideFileUnsavedDialog(choice),
          resetFileSearchState: () => resetFileSearchState(),
          closeFilePickerMenu: (options) => closeFilePickerMenu(options),
          isTextFileKind: (kind) => isTextFileKind(kind),
          isDiffableFileKind: (kind) => isDiffableFileKind(kind),
          confirmReload: (message) => confirmApp({ title: "Reload file from disk?", message, confirmText: "Reload", cancelText: "Cancel", destructive: true }),
          promptUnsavedFileChoice: () => fileUnsavedController.promptFileUnsavedChoice(),
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
        }));
        sessionEditController = window.CodoxearSessionEdit.createSessionEditController(wiring.createSessionEditOptions({
          documentTarget: document,
          ElementCtor: HTMLElement,
          el,
          editCloseBtn,
          editStatus,
          editNameInput,
          editPriorityRange,
          editPriorityValue,
          editPriorityResetBtn,
          editSnoozeModeButtons,
          editSnoozeCustomDate,
          editSnoozeCustomTime,
          editSnoozeCustomRow,
          editDependencyBtn,
          editDependencyMenu,
          editSaveBtn,
          editCancelBtn: $("#editCancelBtn"),
          editViewer,
          getSessionInfo: (sid) => sessionIndex.get(sid),
          getSessions: () => Array.from(sessionIndex.values()),
          selectedSessionId: () => selected,
          sessionDisplayName,
          baseName,
          formatPriorityOffset,
          setPickerButtonContent,
          api,
          refreshSessions,
          setToast,
          setTitle: (_sid, session) => { if (session) titleLabel.textContent = sessionTitleWithId(session); },
          prepareModalOpen,
          afterModalVisibilityChanged,
          positionDialogMenu: (menu, anchorBtn) => dialogMenuController.positionDialogMenu(menu, anchorBtn),
          addAppEvent,
        }));
        const fileViewerPanelRuntime = codoxearFileViewer.createFileViewerPanelRuntime(wiring.createFileViewerPanelOptions({
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
        }));
        const fileViewerLifecycleRuntime = codoxearFileViewer.createFileViewerLifecycleRuntime(wiring.createFileViewerLifecycleOptions({
          controller: fileViewerController,
          beginHide: () => fileViewerModalRuntime.beginHide(),
          hideDisplay: () => fileViewerModalRuntime.hideDisplay(),
          finishHide: (state) => fileViewerModalRuntime.finishHide(state),
          hideFileUnsavedDialog: () => fileUnsavedController.hideFileUnsavedDialog(),
          hideFilePasteDialog: () => hideFilePasteDialog(),
          resetFileViewerPanel: () => resetFileViewerPanel(),
          closeFilePickerMenu: (options) => closeFilePickerMenu(options),
          resetFileSearchState: () => resetFileSearchState(),
          setFileSearchSessionId: (sessionId) => filePickerSearchState.setSessionId(sessionId),
          updateFileTouchToolbar: () => updateFileTouchToolbar(),
          isFileViewerOpen: () => isFileViewerOpen(),
          selectedSessionId: () => selected,
          maybeHandleUnsavedFileChanges: () => fileUnsavedController.maybeHandleUnsavedFileChanges(),
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
        }));
        const fileVideoPreviewRuntime = codoxearFileViewer.createFileVideoPreviewRuntime(wiring.createFileVideoPreviewOptions({
          controller: fileViewerController,
          fetchPreview: (url, options) => fetch(url, options),
          resolveAppUrl: (url) => resolveAppUrl(url),
          handleAuthLoss: () => handleAppAuthLoss(),
          errorText: (error) => codoxearFileHelpers.fileVideoPreviewErrorText(error),
          video: fileVideo,
        }));
        const fileLoadResultRuntime = codoxearFileViewer.createFileLoadResultRuntime(wiring.createFileLoadResultOptions({
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
        }));
        const fileCandidateRefreshRuntime = codoxearFileViewer.createFileCandidateRefreshRuntime(wiring.createFileCandidateRefreshOptions({
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
        }));
        const openedFileRuntime = codoxearFileViewer.createOpenedFileRuntime(wiring.createOpenedFileOptions({
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
        }));
        const fileReferenceRuntime = codoxearFileViewer.createFileReferenceRuntime(wiring.createFileReferenceOptions({
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
          selectSession: (sessionId) => sessionLifecycleController.selectSession(sessionId),
          openDirectorySession: (options) => newSessionDialogController.open(options),
          setToast: (message) => setToast(message),
          api: (url, options) => api(url, options),
          el,
        }));

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

        eventBindings.on(fileBtn, 'click', (e) => {
          e.preventDefault();
          e.stopPropagation();
          void showFileViewer();
        });
        eventBindings.on(filePickerInput, 'focus', () => filePickerInputRuntime.focus());
        eventBindings.on(filePickerInput, 'click', (e) => filePickerInputRuntime.click(e));
        eventBindings.on(filePickerInput, 'input', () => filePickerInputRuntime.input());
        eventBindings.on(filePickerInput, 'blur', () => filePickerInputRuntime.blur());
        eventBindings.on(filePickerInput, 'keydown', (e) => filePickerInputRuntime.keydown(e));
        eventBindings.on(fileModeDiffBtn, 'click', (e) => {
          e.preventDefault();
          e.stopPropagation();
          void handleFileDiffModeButtonPress();
        });
        eventBindings.on(fileModePreviewBtn, 'click', (e) => {
          e.preventDefault();
          e.stopPropagation();
          void handleFilePreviewModeButtonPress();
        });
        eventBindings.on(fileEditBtn, 'click', async (e) => {
          e.preventDefault();
          e.stopPropagation();
          await handleFileEditButtonPress();
        });
        eventBindings.on(fileVideoPreviewBtn, 'click', (e) => {
          e.preventDefault();
          e.stopPropagation();
          void fileVideoPreviewRuntime.handleButtonPress();
        });

        eventBindings.on(fileDownloadBtn, 'click', (e) => {
          e.preventDefault();
          e.stopPropagation();
          fileDownloadRuntime.download(activeFileDownloadApiPath());
        });
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
        eventBindings.on(fileCloseBtn, 'click', (e) => {
          e.preventDefault();
          e.stopPropagation();
          void requestHideFileViewer();
        });
        eventBindings.on(fileBackdrop, 'click', () => void requestHideFileViewer());
        eventBindings.on($("#fileUnsavedSaveBtn"), 'click', () => fileUnsavedController.handleFileUnsavedSaveChoice());
        eventBindings.on($("#fileUnsavedDiscardBtn"), 'click', () => fileUnsavedController.handleFileUnsavedDiscardChoice());
        eventBindings.on($("#fileUnsavedCancelBtn"), 'click', () => fileUnsavedController.handleFileUnsavedCancelChoice());
        eventBindings.on(fileUnsavedBackdrop, 'click', () => fileUnsavedController.handleFileUnsavedCancelChoice());
        eventBindings.on($("#filePasteInsertBtn"), 'click', () => {
          handleFilePasteInsert(filePasteInput.value);
        });
        eventBindings.on($("#filePasteCancelBtn"), 'click', () => hideFilePasteDialog({ restoreFocus: true }));
        eventBindings.on(filePasteBackdrop, 'click', () => hideFilePasteDialog({ restoreFocus: true }));
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
        });
        const fileTouchController = codoxearFileTouch.createFileTouchController(wiring.createFileTouchOptions({
          fileViewerController: () => fileViewerController,
        }));
        addAppEvent(document, "keydown", (event) => fileTouchController.handleFileTouchSelectionKeydown(event), true);
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
            fileUnsavedController.hideFileUnsavedDialog("cancel");
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
          if (sessionEditController.viewer.style.display === "flex" || sessionEditController.viewer.open) sessionEditController.hideEditSession();
          if (newSessionDialogController.isOpen()) newSessionDialogController.close();
        });

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
          initPageLimit,
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


  global.CodoxearApplicationComposition = Object.freeze({ createApplicationComposition });
})(window);
