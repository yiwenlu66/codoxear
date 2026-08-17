import * as CodoxearChatInteraction from "./app_chat_interaction.js";
import * as CodoxearClipboard from "./app_file_ops.js";
import * as CodoxearCodeCopy from "./app_code_copy.js";
import * as CodoxearDiagnostics from "./app_diagnostics.js";
import * as CodoxearDialogMenu from "./app_dialog_menu.js";
import * as CodoxearFileEditorOps from "./app_file_editor_ops.js";
import * as CodoxearFileOps from "./app_file_ops.js";
import * as CodoxearFilePickerOps from "./app_file_picker_ops.js";
import * as CodoxearIOSViewport from "./app_ios_viewport.js";
import * as CodoxearMessageHistory from "./app_message_history.js";
import * as CodoxearModal from "./app_modal.js";
import * as CodoxearQueue from "./app_queue.js";
import * as CodoxearSendLifecycle from "./app_send_lifecycle.js";
import * as CodoxearSessionLifecycle from "./app_session_lifecycle.js";
import * as CodoxearSessionCatalog from "./app_session_catalog.js";
import * as CodoxearSessionRefresh from "./app_session_refresh.js";
import * as CodoxearSessionState from "./app_session_state.js";
import * as CodoxearSessionTitle from "./app_session_title.js";
import * as CodoxearTranscriptRender from "./app_transcript_render.js";
import * as CodoxearUnattended from "./app_unattended.js";
import * as CodoxearWiring from "./app_wiring.js";


/* Application composition owns concrete lifecycle, controller assembly, and UI behavior.
 * app_application_runtime.js remains the stable bootstrap facade. */

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
      codoxearDom, el, codoxearShell, codoxearSessions, codoxearComposer, codoxearAttachments, codoxearTopbar,
      codoxearMessageFlow, codoxearInterrupt, codoxearDialogMenus,
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
      emptyPiLaunchDefaults, emptyCcLaunchDefaults, redactedLaunchErrorText, sessionLaunchLabel,
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

      function renderApp() {
            cleanupActiveApp();
	        const root = $("#root");
        const wiring = CodoxearWiring.createWiring();
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
          topMeta,
          topActions,
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
        const unattendedDom = CodoxearUnattended.createUnattendedDom(wiring.createUnattendedDomOptions({ el, iconSvg, unattendedBtn }));
        const { unattendedMenu, enabledEl: unattendedEnabledEl, cooldownEl: unattendedCooldownEl, remainingEl: unattendedRemainingEl, requestEl: unattendedRequestEl } = unattendedDom;
        root.appendChild(unattendedMenu);
        const INIT_PAGE_LIMIT = 24;
        const OLDER_PAGE_LIMIT = 60;
        const CHAT_DOM_WINDOW = 260;
        const CHAT_DOM_WINDOW_WITH_HISTORY_SLACK = CHAT_DOM_WINDOW + OLDER_PAGE_LIMIT;
        const OLDER_TOP_TRIGGER_PX = 1;
        const OLDER_CANCEL_PX = 48;
        const OLDER_AUTO_COOLDOWN_MS = 450;
        // Epoch is intentionally separate from polling: every async consumer
        // may compare it, while only this composition owns scheduling powers.
        const asyncEpoch = codoxearPolling.createAsyncEpoch();
        const pollingRuntime = codoxearPolling.createPollingRuntime({ setTimeout, clearTimeout });
        const sessionState = CodoxearSessionState.createSessionState({ consoleError: (...args) => console.error(...args) });
        const sessionCatalog = CodoxearSessionCatalog.createSessionCatalog({ consoleError: (...args) => console.error(...args) });
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
        let sessionEditController = null;
        let interruptController = null;
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
        const eventBindings = CodoxearEventBindings.createEventBindings(wiring.createEventBindingsOptions({ addEvent: addAppEvent }));
        function stopMessagePolling() {
          sessionState.set("selected", null);
          asyncEpoch.incrementGeneration();
          if (messageFlowController) messageFlowController.stop();
          sessionState.set("turnOpen", false);
        }
        function cleanupApp() {
          if (appDisposed) return;
          appDisposed = true;
          pollingRuntime.disable();
          stopMessagePolling();
          if (newSessionDialogController) {
            newSessionDialogController.close();
            newSessionDialogController.dispose();
          }
          if (voiceController) voiceController.dispose();
          if (unattendedController) unattendedController.dispose();
          filePickerSearchState.dispose();
          if (iosViewportController) iosViewportController.dispose();
          if (chatSearchController) chatSearchController.dispose();
          if (sessionTitleController) sessionTitleController.dispose();
          if (chatInteractionController) chatInteractionController.dispose();
          if (topbarController) topbarController.dispose();
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
            errorStreak: pollingRuntime.sessionsPollErrorStreak(),
          });
        }
        function secondaryPollDelayMs() {
          return codoxearPolling.networkRetryDelayMs({
            normalDelayMs: codoxearPolling.secondaryPollDelayMs(document.visibilityState),
            offline: browserOffline(),
            errorStreak: pollingRuntime.secondaryPollErrorStreak(),
          });
        }
        function browserOffline() {
          return codoxearPolling.browserOffline(typeof navigator === "undefined" ? undefined : navigator);
        }
        function markSessionsPollSuccess() {
          pollingRuntime.markSessionsPollSuccess();
          networkStatus.reportSuccess();
        }
        function markSessionsPollFailure(transportFailed = true) {
          pollingRuntime.markSessionsPollFailure();
          if (transportFailed) networkStatus.reportFailure();
        }
        function markSecondaryPollSuccess() {
          pollingRuntime.markSecondaryPollSuccess();
          networkStatus.reportSuccess();
        }
        function markSecondaryPollFailure(transportFailed = true) {
          pollingRuntime.markSecondaryPollFailure();
          if (transportFailed) networkStatus.reportFailure();
        }

        async function runSessionsPollTick() {
          if (appDisposed) return;
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
          if (appDisposed) return;
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
          scheduleSecondaryPoll();
        }
        function scheduleSessionsPoll(delayMs = sessionsPollDelayMs()) {
          if (appDisposed) return;
          pollingRuntime.scheduleSessions(delayMs, runSessionsPollTick);
        }
        function scheduleSecondaryPoll(delayMs = secondaryPollDelayMs()) {
          if (appDisposed) return;
          pollingRuntime.scheduleSecondary(delayMs, runSecondaryPollTick);
        }

        const sessionTitleController = CodoxearSessionTitle.createSessionTitleController(wiring.createSessionTitleOptions({
          titleLabel,
          sessionState,
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

        function setPickerButtonContent(button, primaryText, secondaryText = "", placeholder = false) {
          if (!button) return;
          button.innerHTML = "";
          const textWrap = el("span", { class: `pickerButtonText${placeholder ? " placeholder" : ""}` });
          textWrap.appendChild(el("span", { class: "pickerButtonPrimary", text: String(primaryText || "") }));
          if (secondaryText) textWrap.appendChild(el("span", { class: "pickerButtonSecondary", text: String(secondaryText) }));
          button.appendChild(textWrap);
          button.appendChild(el("span", { class: "pickerButtonChevron", html: iconSvg("chevronDown") }));
        }

        const dialogMenuController = CodoxearDialogMenu.createDialogMenuController(wiring.createDialogMenuOptions({ windowTarget: window }));

        const newSessionDialogController = codoxearNewSession.createNewSessionDialogController(wiring.createNewSessionDialogOptions({
          root,
          el,
          iconSvg,
          document,
          window,
          addEvent: addAppEvent,
          sessionCatalog,
          sessionState,
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
          return CodoxearModal.isModalTargetOpen(node);
        }

        function syncModalIsolation() {
          return CodoxearModal.syncModalIsolation(app, modalIsolationTargets);
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
          return CodoxearModal.restoreModalFocus(target, isStillOpen);
        }

        function focusModalCloseButton(viewer, closeBtn) {
          return CodoxearModal.focusModalCloseButton(viewer, closeBtn);
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


        const toastController = createToastController({ toast, setTimeout });
        function setToast(text) {
          return toastController.show(text);
        }

        const topbarController = codoxearTopbar.createTopbarController(wiring.createTopbarOptions({
          el,
          iconSvg,
          setToast,
          onInterrupt: () => interruptController && interruptController.interruptSelectedSession(),
          sessionState,
          topMeta,
          topActions,
          eventBindings,
        }));

        async function copyToClipboard(text) {
          return CodoxearClipboard.copyToClipboard(text);
        }

        const codeBlockCopyRuntime = CodoxearCodeCopy.createCodeBlockCopyRuntime(wiring.createCodeBlockCopyOptions({
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
          if (!sessionState.get("selected")) return;
          const sid = sessionState.get("selected");
          try {
            const data = await api(`/api/sessions/${sid}/messages/export`);
            if (sessionState.get("selected") !== sid) return;
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

        let fileOpsController = null;

        const chatInteractionController = CodoxearChatInteraction.createChatInteractionController(wiring.createChatInteractionOptions({
          currentGeneration: asyncEpoch.currentGeneration,
          sessionCatalog,
          getSessionLifecycleController: () => sessionLifecycleController,
          getSessionRefreshController: () => sessionRefreshController,
          sessionState,
          getSessionEditController: () => sessionEditController,
          getQueueController: () => queueController,
          codoxearTranscriptRender: CodoxearTranscriptRender,
          codoxearMessageHistory: CodoxearMessageHistory,
          codoxearSendLifecycle: CodoxearSendLifecycle,
          wiring: wiring,
          document: document,
          storageSetItem: storageSetItem,
          storageRemoveItem: storageRemoveItem,
          codoxearViewport: codoxearViewport,
          codoxearSessions: codoxearSessions,
          sessionsWrap: sessionsWrap,
          sidebarEmptyHint: sidebarEmptyHint,
          el: el,
          iconSvg: iconSvg,
          sidebarRenderSignature: sidebarRenderSignature,
          sidebarSessionEntries: sidebarSessionEntries,
          sessionDisplayName: sessionDisplayName,
          sessionLaunchFailed: sessionLaunchFailed,
          sessionLaunchPending: sessionLaunchPending,
          redactedLaunchErrorText: redactedLaunchErrorText,
          fmtRelativeAge: fmtRelativeAge,
          sidebarEffortCode: sidebarEffortCode,
          sidebarModelText: sidebarModelText,
          baseName: baseName,
          sessionIsFast: sessionIsFast,
          agentBackendLogoPath: agentBackendLogoPath,
          agentBackendDisplayName: agentBackendDisplayName,
          sessionAgentBackend: sessionAgentBackend,
          sessionLaunchIcon: sessionLaunchIcon,
          sessionLaunchLabel: sessionLaunchLabel,
          confirmApp: confirmApp,
          api: api,
          setToast: setToast,
          sessionProviderChoice: sessionProviderChoice,
          queueViewer: queueViewer,
          refreshQueueViewer: refreshQueueViewer,
          isAppDisposed: () => appDisposed,
          isFileViewerOpen: () => fileOpsController.isFileViewerOpen(),
          upgradeCandidateFileRefs: (...args) => fileOpsController.fileReferenceRuntime.upgradeCandidateRefs(...args),
          $: $,
          ATTACH_UPLOAD_MAX_BYTES: ATTACH_UPLOAD_MAX_BYTES,
          AbortController: AbortController,
          CHAT_DOM_WINDOW: CHAT_DOM_WINDOW,
          CHAT_DOM_WINDOW_WITH_HISTORY_SLACK: CHAT_DOM_WINDOW_WITH_HISTORY_SLACK,
          EventSource: EventSource,
          INIT_PAGE_LIMIT: INIT_PAGE_LIMIT,
          Node: window.Node,
          OLDER_AUTO_COOLDOWN_MS: OLDER_AUTO_COOLDOWN_MS,
          OLDER_CANCEL_PX: OLDER_CANCEL_PX,
          OLDER_PAGE_LIMIT: OLDER_PAGE_LIMIT,
          OLDER_TOP_TRIGGER_PX: OLDER_TOP_TRIGGER_PX,
          addAppEvent: addAppEvent,
          appConfirm: appConfirm,
          attachBtn: attachBtn,
          b64FromBytes: b64FromBytes,
          bottomSentinel: bottomSentinel,
          chat: chat,
          chatInner: chatInner,
          chatMarkdownHtmlCached: chatMarkdownHtmlCached,
          chatSearchAllHintEl: chatSearchAllHintEl,
          chatSearchBar: chatSearchBar,
          chatSearchBtn: chatSearchBtn,
          chatSearchCloseBtn: chatSearchCloseBtn,
          chatSearchInput: chatSearchInput,
          chatSearchNextBtn: chatSearchNextBtn,
          chatSearchPrevBtn: chatSearchPrevBtn,
          chatSearchStatus: chatSearchStatus,
          chatTimeChip: chatTimeChip,
          codeBlockCopyRuntime: codeBlockCopyRuntime,
          codoxearAttachments: codoxearAttachments,
          codoxearCodeCopy: CodoxearCodeCopy,
          codoxearDisplay: codoxearDisplay,
          codoxearMessageFlow: codoxearMessageFlow,
          codoxearModal: CodoxearModal,
          codoxearNavigationPulse: codoxearNavigationPulse,
          codoxearPendingUser: codoxearPendingUser,
          composer: composer,
          copyToClipboard: copyToClipboard,
          dataTransferHasFiles: dataTransferHasFiles,
          diagViewer: diagViewer,
          editViewer: editViewer,
          extractFilesFromClipboardData: extractFilesFromClipboardData,
          extractFilesFromDropData: extractFilesFromDropData,
          fmtBytes: fmtBytes,
          handleAppAuthLoss: handleAppAuthLoss,
          helpViewer: helpViewer,
          imgInput: imgInput,
          isLikelyHeic: isLikelyHeic,
          isModalTargetOpen: isModalTargetOpen,
          isTextEntryElement: isTextEntryElement,
          jumpBtn: jumpBtn,
          looksLikeImage: looksLikeImage,
          modalIsolationTargets: modalIsolationTargets,
          navigator: navigator,
          networkStatus: networkStatus,
          newSessionDialogController: newSessionDialogController,
          nextUserBtn: nextUserBtn,
          olderBtn: olderBtn,
          olderError: olderError,
          olderErrorText: olderErrorText,
          olderWrap: olderWrap,
          performance: performance,
          prevUserBtn: prevUserBtn,
          pushPerfSample: pushPerfSample,
          requestAnimationFrame: requestAnimationFrame,
          resizeComposer: resizeComposer,
          resolveAppUrl: resolveAppUrl,
          safeAttachmentStem: safeAttachmentStem,
          sendChoice: sendChoice,
          sessionHasOrphanQueueRecovery: sessionHasOrphanQueueRecovery,
          sessionHasUnknownSend: sessionHasUnknownSend,
          sessionIdFromHash: sessionIdFromHash,
          sessionIsOrphanRecovery: sessionIsOrphanRecovery,
          sessionSelectable: sessionSelectable,
          sessionTitleWithId: sessionTitleWithId,
          setTimeout: setTimeout,
          syncComposerSendButton: syncComposerSendButton,
          syncQueueSubmitState: syncQueueSubmitState,
          textarea: textarea,
          titleLabel: titleLabel,
          updateUnattendedBtnState: () => updateUnattendedBtnState(),
          window: window,
          updateQueueBadge: () => updateQueueBadge(),
        }));
        ({ attachmentsController, messageFlowController } = chatInteractionController);
        const {
          chatSearchController, chatNavigationController, hintModeController, sidebarController, transcriptSlotRuntime, typingRowRuntime,
          transcriptScrollRuntime, transcriptDomRuntime, transcriptEventRuntime, transcriptView, markClickLoad, olderLoadRuntime,
          resetChatRenderState, clearOlderLoadError, updateChatNavButtons,
          closeChatSearch, clearRenderedTranscriptRange, initPageLimit, dropPendingUserRows,
          updateSessionTranscriptSlot, tailCacheMatchesSession, applySessionListTranscriptIdentity,
          updateQueueBadge, updateTypingStatsFromSession, messagePollDelayMs, kickPoll,
          setPollFastUntilMs, openMessageEventSource, isMobile, useDesktopSessionActions,
          useTouchFileEditorControls, setSidebarOpen, setSidebarCollapsed, clearCommitUnknownSend,
          refreshSessions, loadOlderMessages, applySessionRuntimeFromTail, renderSessionTail,
          recoveryDetailsText, syncRecoveryUiForSession, renderPendingTranscriptSlot,
          renderTranscriptLoading, renderTranscriptLoadError, applyCachedTail, jumpToLatest,
          rememberPendingHashSession, maybeSelectPendingHashSession,
        } = chatInteractionController;
        const unattendedController = (function instantiateUnattendedController() {
          return CodoxearUnattended.createUnattendedController(wiring.createUnattendedOptions({
            unattendedBtn,
            unattendedMenu,
            enabledEl: unattendedEnabledEl,
            cooldownEl: unattendedCooldownEl,
            remainingEl: unattendedRemainingEl,
            requestEl: unattendedRequestEl,
            sessionState,
            getSessionInfo: (sid) => sessionCatalog.get("sessionIndex").get(sid),
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
          unattendedController.syncButtonState();
          attachmentsController.syncAttachButtonState();
          const fileViewerBlocked = Boolean(sessionState.get("selected") && selectedSessionLaunchFailed());
          const fileViewerLabel = !sessionState.get("selected") ? "Select a session to view files" : fileViewerBlocked ? "Failed launch has no file browser" : "View file";
          fileBtn.disabled = !sessionState.get("selected") || fileViewerBlocked;
          fileBtn.title = fileViewerLabel;
          fileBtn.setAttribute("aria-label", fileViewerLabel);
          chatSearchBtn.disabled = !sessionState.get("selected");
          chatNavRail.style.display = sessionState.get("selected") ? "flex" : "none";
          chatEmptyState.style.display = sessionState.get("selected") ? "none" : "flex";
          if (!sessionState.get("selected") && chatSearchController.isOpen()) closeChatSearch();
          updateChatNavButtons();
          syncQueueSubmitState();
          syncComposerSendButton();
          diagBtn.disabled = !sessionState.get("selected");
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
        fileOpsController = CodoxearFileOps.createFileOpsController(wiring.createFileOpsOptions({
          wiring, document, window, HTMLElement, requestAnimationFrame, setTimeout,
          $, el, iconSvg, resolveAppUrl, api, setToast, confirmApp, addAppEvent,
          sessionLaunchFailed, normalizeLineNumber, markdownPreviewHtml,
          blockedFileMessage, listFromFilesField, listFromFileRecords, baseName,
          codoxearFilePicker, codoxearFilePickerOps: CodoxearFilePickerOps,
          codoxearFileViewer, codoxearFileEditor, codoxearFileEditorOps: CodoxearFileEditorOps,
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
          sessionState,
          sessionCatalog,
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
          return CodoxearQueue.createQueueController(wiring.createQueueOptions({
            queueBackdrop,
            queueCloseBtn,
            queueList,
            queueEmpty,
            queueViewer,
            queueBtn: $("#queueBtn"),
            sessionState,
            getSessionInfo: (sid) => sessionCatalog.get("sessionIndex").get(sid),
            isAppDisposed: () => appDisposed,
            api,
            setToast,
            clearCommitUnknownSend,
            refreshSessions,
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
          return sessionLaunchFailed(sessionState.get("selected") ? sessionCatalog.get("sessionIndex").get(sessionState.get("selected")) : null);
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
          return CodoxearDiagnostics.createDiagnosticsController(wiring.createDiagnosticsOptions({
            diagBackdrop,
            diagViewer,
            diagContent,
            diagStatus,
            diagCloseBtn,
            diagCopyConversationBtn,
            diagCopyBtn,
            sessionState,
            getSessionInfo: (sid) => sessionCatalog.get("sessionIndex").get(sid),
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

        sessionLifecycleController = CodoxearSessionLifecycle.createSessionLifecycleController(wiring.createSessionLifecycleOptions({
          asyncEpoch,
          prepareSessionOpen: () => messageFlowController.prepareSessionOpen(),
          sessionState,
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
            sessionState.set("turnOpen", false);
          },
          clearTranscriptForRemovedSession: clearRenderedTranscriptRange,
          syncAttachments: () => attachmentsController.syncStagedAttachmentsFromSelectedSession(),
          clearAttachments: () => attachmentsController.setStagedAttachments([]),
          syncAttachmentButton: () => attachmentsController.syncAttachButtonState(),
          resetChatRenderState,
          getSession: (sessionId) => sessionCatalog.get("sessionIndex").get(sessionId),
          isCurrent: (sessionId, generation) => sessionState.get("selected") === sessionId && asyncEpoch.currentGeneration() === generation,
          setTitle: (session, sessionId) => { titleLabel.textContent = session ? sessionTitleWithId(session) : sessionId ? String(sessionId) : "No session selected"; },
          setNoSessionTitle: () => { titleLabel.textContent = "No session selected"; },
          markClickLoad,
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
          invalidateOlderLoad: () => olderLoadRuntime.invalidate(),
          renderPendingTranscriptSlot,
          applySessionRuntimeFromTail,
          renderSessionTail,
          replaceWith: (events, options) => transcriptView().replaceWith(events, options),
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
          sessionCatalog,
          backendSupportsFastForDefaults: (backend, defaults) => codoxearLaunch.backendSupportsFast(backend, defaults),
          setToast,
          confirmAction: (options) => confirmApp(options),
          syncRecoveryUiForSession,
          sleep: (ms) => new Promise((resolve) => setTimeout(resolve, ms)),
          consoleError: (...args) => console.error(...args),
        }));

        sessionRefreshController = CodoxearSessionRefresh.createSessionRefreshController(wiring.createSessionRefreshOptions({
          api,
          isDisposed: () => appDisposed,
          apiResponseNotModified,
          sessionCatalog,
          emptyDefaults: () => ({
            default_backend: "pi",
            backends: { codex: legacyCodexLaunchDefaults(), pi: emptyPiLaunchDefaults(), cc: emptyCcLaunchDefaults() },
          }),
          clearFileDiscoveryCaches: () => fileReferenceRuntime.clearDiscoveryCaches(),
          useDesktopSessionActions,
          sessionState,
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
        interruptController = codoxearInterrupt.createInterruptController(wiring.createInterruptOptions({
          sessionState,
          setToast,
          api,
          now: Date.now,
          setPollFastUntilMs,
          kickPoll,
        }));

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
          transcriptView().observeScroll("handleScroll");
        });
        chat.addEventListener(
          "wheel",
          (e) => {
            transcriptView().observeScroll("handleWheel", e);
          },
          { passive: true }
        );
        chat.addEventListener(
          "touchstart",
          (e) => {
            transcriptView().observeScroll("handleTouchStart", e);
          },
          { passive: true }
        );
        chat.addEventListener(
          "touchmove",
          (e) => {
            // Finger moves down -> content scrolls up.
            transcriptView().observeScroll("handleTouchMove", e);
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

        const iosViewportController = CodoxearIOSViewport.createIOSViewportController(wiring.createIOSViewportOptions({
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
          sessionState,
          sessionCatalog,
          sessionLaunchFailed,
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
	              hashed && sessionSelectable(sessionCatalog.get("sessionIndex").get(hashed))
	                ? hashed
	                : remembered && sessionSelectable(sessionCatalog.get("sessionIndex").get(remembered))
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
            scheduleSecondaryPoll();
              addAppEvent(window, "hashchange", async () => {
                await sessionLifecycleController.selectSessionFromHash({ refreshIfMissing: true, deferIfMissing: true });
              });
              addAppEvent(window, "beforeunload", () => {
                cleanupApp();
              });
              addAppEvent(document, "visibilitychange", () => {
                if (appDisposed) return;
                if (document.visibilityState === "visible") {
                  if (sessionState.get("selected")) messageFlowController.resumeLiveDelivery();
                  scheduleSessionsPoll(0);
                  scheduleSecondaryPoll(0);
                  return;
                }
                if (sessionState.get("selected")) kickPoll(messagePollDelayMs());
                scheduleSessionsPoll(sessionsPollDelayMs());
                scheduleSecondaryPoll(secondaryPollDelayMs());
              });
              addAppEvent(window, "online", () => {
                if (appDisposed) return;
                networkStatus.reportSuccess();
                messageFlowController.resetMessagePollBackoff();
                pollingRuntime.resetStreaks();
                if (sessionState.get("selected")) {
                  messageFlowController.resumeLiveDelivery();
                  kickPoll(0);
                }
                scheduleSessionsPoll(0);
                scheduleSecondaryPoll(0);
              });
              addAppEvent(window, "offline", () => {
                if (appDisposed) return;
                networkStatus.sync();
                messageFlowController.closeMessageEventSource();
                if (sessionState.get("selected")) kickPoll(messagePollDelayMs());
                scheduleSessionsPoll(sessionsPollDelayMs());
                scheduleSecondaryPoll(secondaryPollDelayMs());
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

const CodoxearEventBindings = { createEventBindings, createToastController, createApplicationComposition };

export { createEventBindings, createToastController, createApplicationComposition };
