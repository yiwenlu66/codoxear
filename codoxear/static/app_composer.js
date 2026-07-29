(function () {
  "use strict";

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`composer controller dependency missing: ${name}`);
    return value;
  }

  function requireNode(value, name) {
    if (!value || typeof value !== "object" || typeof value.addEventListener !== "function")
      throw new TypeError(`composer controller dependency missing: ${name}`);
    return value;
  }

  function createComposerController(options = {}) {
    if (!options || typeof options !== "object") throw new TypeError("composer controller dependency missing: options");
    const form = requireNode(options.form, "form");
    const textarea = requireNode(options.textarea, "textarea");
    const msgPh = requireNode(options.msgPh, "msgPh");
    const sendBtn = requireNode(options.sendBtn, "sendBtn");
    const sendChoice = requireNode(options.sendChoice, "sendChoice");
    const sendChoiceBackdrop = requireNode(options.sendChoiceBackdrop, "sendChoiceBackdrop");
    const sendChoiceNowBtn = requireNode(options.sendChoiceNowBtn, "sendChoiceNowBtn");
    const sendChoiceLaterBtn = requireNode(options.sendChoiceLaterBtn, "sendChoiceLaterBtn");
    const sendChoiceCancelBtn = requireNode(options.sendChoiceCancelBtn, "sendChoiceCancelBtn");
    const getSelected = requireFunction(options.getSelected, "getSelected");
    const getSessionInfo = requireFunction(options.getSessionInfo, "getSessionInfo");
    const patchSessionInfo = requireFunction(options.patchSessionInfo, "patchSessionInfo");
    const sessionLaunchFailed = requireFunction(options.sessionLaunchFailed, "sessionLaunchFailed");
    const getSending = requireFunction(options.getSending, "getSending");
    const setSending = requireFunction(options.setSending, "setSending");
    const getCurrentRunning = requireFunction(options.getCurrentRunning, "getCurrentRunning");
    const setCurrentRunning = requireFunction(options.setCurrentRunning, "setCurrentRunning");
    const setTurnOpen = requireFunction(options.setTurnOpen, "setTurnOpen");
    const getStagedAttachments = requireFunction(options.getStagedAttachments, "getStagedAttachments");
    const normalizedStagedAttachments = requireFunction(options.normalizedStagedAttachments, "normalizedStagedAttachments");
    const setSelectedSessionPendingAttachment = requireFunction(options.setSelectedSessionPendingAttachment, "setSelectedSessionPendingAttachment");
    const setAttachCount = requireFunction(options.setAttachCount, "setAttachCount");
    const syncAttachButtonState = requireFunction(options.syncAttachButtonState, "syncAttachButtonState");
    const syncQueueSubmitState = requireFunction(options.syncQueueSubmitState, "syncQueueSubmitState");
    const syncRecoveryUiForSession = requireFunction(options.syncRecoveryUiForSession, "syncRecoveryUiForSession");
    const confirmAction = requireFunction(options.confirmAction, "confirmAction");
    const api = requireFunction(options.api, "api");
    const setToast = requireFunction(options.setToast, "setToast");
    const handleAppAuthLoss = requireFunction(options.handleAppAuthLoss, "handleAppAuthLoss");
    const refreshSessions = requireFunction(options.refreshSessions, "refreshSessions");
    const setPollFastUntilMs = requireFunction(options.setPollFastUntilMs, "setPollFastUntilMs");
    const kickPoll = requireFunction(options.kickPoll, "kickPoll");
    const isTranscriptRenewalCommand = requireFunction(options.isTranscriptRenewalCommand, "isTranscriptRenewalCommand");
    const nextLocalEchoId = requireFunction(options.nextLocalEchoId, "nextLocalEchoId");
    const renderedAtLiveTail = requireFunction(options.renderedAtLiveTail, "renderedAtLiveTail");
    const clearTranscriptDom = requireFunction(options.clearTranscriptDom, "clearTranscriptDom");
    const clearRenderedTranscriptRange = requireFunction(options.clearRenderedTranscriptRange, "clearRenderedTranscriptRange");
    const setOlderState = requireFunction(options.setOlderState, "setOlderState");
    const getSessionTranscriptSlot = requireFunction(options.getSessionTranscriptSlot, "getSessionTranscriptSlot");
    const addPendingUser = requireFunction(options.addPendingUser, "addPendingUser");
    const appendEvent = requireFunction(options.appendEvent, "appendEvent");
    const deleteTailCache = requireFunction(options.deleteTailCache, "deleteTailCache");
    const beginTranscriptRenewal = requireFunction(options.beginTranscriptRenewal, "beginTranscriptRenewal");
    const clearLiveCursor = requireFunction(options.clearLiveCursor, "clearLiveCursor");
    const invalidateOlderLoad = requireFunction(options.invalidateOlderLoad, "invalidateOlderLoad");
    const renderPendingTranscriptSlot = requireFunction(options.renderPendingTranscriptSlot, "renderPendingTranscriptSlot");
    const dropPendingUser = requireFunction(options.dropPendingUser, "dropPendingUser");
    const removePendingUserRow = requireFunction(options.removePendingUserRow, "removePendingUserRow");
    const hasPendingForSession = requireFunction(options.hasPendingForSession, "hasPendingForSession");
    const enqueueComposerText = requireFunction(options.enqueueComposerText, "enqueueComposerText");
    const prepareModalOpen = requireFunction(options.prepareModalOpen, "prepareModalOpen");
    const afterModalVisibilityChanged = requireFunction(options.afterModalVisibilityChanged, "afterModalVisibilityChanged");
    const restoreModalFocus = requireFunction(options.restoreModalFocus, "restoreModalFocus");
    const storageGetItem = requireFunction(options.storageGetItem, "storageGetItem");
    const storageSetItem = requireFunction(options.storageSetItem, "storageSetItem");
    const storageRemoveItem = requireFunction(options.storageRemoveItem, "storageRemoveItem");
    const onAutoGrow = typeof options.onAutoGrow === "function" ? options.onAutoGrow : () => {};
    const requestFrame = typeof options.requestFrame === "function" ? options.requestFrame : (callback) => requestAnimationFrame(callback);
    const getComputedStyleFn = typeof options.getComputedStyle === "function" ? options.getComputedStyle : (node) => getComputedStyle(node);
    const activeElement = typeof options.activeElement === "function" ? options.activeElement : () => document.activeElement;
    const isHTMLElement = typeof options.isHTMLElement === "function" ? options.isHTMLElement : (value) => typeof HTMLElement === "function" && value instanceof HTMLElement;
    const now = typeof options.now === "function" ? options.now : () => Date.now();
    const consoleError = typeof options.consoleError === "function" ? options.consoleError : () => {};
    const windowTarget = options.windowTarget && typeof options.windowTarget.addEventListener === "function" ? options.windowTarget : null;

    const cleanups = [];
    const listen = (target, type, handler, eventOptions) => {
      target.addEventListener(type, handler, eventOptions);
      cleanups.push(() => target.removeEventListener(type, handler, eventOptions));
    };
    const sessionDraftKey = (sessionId) => `codexweb.draft.${sessionId}`;
    let sendChoicePending = null;
    let sendChoiceReturnFocusEl = null;

    function selectedSessionLaunchFailed() {
      const sessionId = getSelected();
      return sessionLaunchFailed(sessionId ? getSessionInfo(sessionId) : null);
    }

    function syncComposerState() {
      const sessionId = getSelected();
      const launchFailed = selectedSessionLaunchFailed();
      const blocked = !sessionId || launchFailed;
      const label = !sessionId ? "Select a session to send" : launchFailed ? "Failed launch cannot receive messages" : "Enter your instructions here";
      textarea.disabled = blocked;
      textarea.setAttribute("aria-label", label);
      textarea.title = blocked ? label : "";
      msgPh.textContent = label;
    }

    function syncSendButtonState() {
      const sessionId = getSelected();
      const launchFailed = selectedSessionLaunchFailed();
      const label = !sessionId ? "Select a session to send" : launchFailed ? "Failed launch cannot receive messages" : "Send";
      sendBtn.disabled = Boolean(getSending() || !sessionId || launchFailed);
      sendBtn.title = label;
      sendBtn.setAttribute("aria-label", label);
      syncComposerState();
    }

    function autoGrow() {
      const basePx = parseFloat(getComputedStyleFn(textarea).minHeight || "0") || 32;
      const maxPx = 180;
      const stagedCount = getStagedAttachments().length;
      msgPh.style.display = textarea.value || stagedCount ? "none" : "flex";
      textarea.style.height = `${basePx}px`;
      let height = textarea.scrollHeight;
      const multiline = textarea.value.includes("\n") || height > basePx + 1;
      form.classList.toggle("multiline", multiline);
      textarea.style.height = multiline ? "auto" : `${basePx}px`;
      height = textarea.scrollHeight;
      textarea.style.height = `${multiline ? Math.min(height, maxPx) : basePx}px`;
      textarea.style.overflowY = height > maxPx ? "auto" : "hidden";
      onAutoGrow();
    }

    function saveSessionDraft(sessionId) {
      if (!sessionId) return;
      const value = String(textarea.value || "");
      if (value) storageSetItem(sessionDraftKey(sessionId), value);
      else storageRemoveItem(sessionDraftKey(sessionId));
    }

    function loadSessionDraft(sessionId) {
      textarea.value = sessionId ? storageGetItem(sessionDraftKey(sessionId)) || "" : "";
      autoGrow();
    }

    function clearSessionDraft(sessionId) {
      if (sessionId) storageRemoveItem(sessionDraftKey(sessionId));
    }

    function clearComposer() {
      textarea.value = "";
      clearSessionDraft(getSelected());
      autoGrow();
    }

    function syncSendChoiceAttachmentPolicy() {
      const hasAttachments = Boolean(sendChoicePending && sendChoicePending.attachmentCount > 0);
      const label = hasAttachments ? "Attachments cannot be queued; send now or wait until idle" : "Send after current";
      sendChoiceLaterBtn.disabled = hasAttachments;
      sendChoiceLaterBtn.title = label;
      sendChoiceLaterBtn.setAttribute("aria-label", label);
    }

    function focusSendChoiceInitial() {
      requestFrame(() => {
        if (sendChoice.style.display !== "flex") return;
        const target = !sendChoiceLaterBtn.disabled ? sendChoiceLaterBtn : !sendChoiceNowBtn.disabled ? sendChoiceNowBtn : sendChoiceCancelBtn;
        if (!target || typeof target.focus !== "function") return;
        try { target.focus({ preventScroll: true }); } catch (_) {}
      });
    }

    function showSendChoice(raw, { opener = null } = {}) {
      prepareModalOpen();
      const focused = activeElement();
      sendChoiceReturnFocusEl = isHTMLElement(opener) ? opener : isHTMLElement(focused) ? focused : null;
      sendChoicePending = { sid: getSelected(), text: raw, attachmentCount: getStagedAttachments().length };
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

    async function sendText(raw, { sid = null } = {}) {
      const sessionId = sid || getSelected();
      if (!sessionId || !raw || !raw.trim() || getSending()) return false;
      const renderHere = sessionId === getSelected();
      const renewsTranscript = isTranscriptRenewalCommand(raw, sessionId);
      const sessionInfo = getSessionInfo(sessionId) || null;
      if (sessionInfo && sessionLaunchFailed(sessionInfo)) {
        setToast("failed launch cannot receive messages");
        return false;
      }
      const stagedAttachments = getStagedAttachments();
      const localAttachmentCount = renderHere ? stagedAttachments.length : normalizedStagedAttachments(sessionInfo && sessionInfo.staged_attachments).length;
      let allowPendingAttachment = localAttachmentCount > 0;
      if (!allowPendingAttachment && sessionInfo && sessionInfo.pending_attachment) {
        const confirmed = await confirmAction({
          title: "Send pending attachment?",
          message: "This session has a pending file attachment. Send it with this message?",
          confirmText: "Send with attachment",
          cancelText: "Cancel",
        });
        if (!confirmed) return false;
        allowPendingAttachment = true;
      }
      setSending(true);
      syncSendButtonState();
      syncAttachButtonState();
      setToast("sending...");

      const localId = nextLocalEchoId();
      const startedAt = now() / 1000;
      if (renderHere && !renewsTranscript) {
        if (!renderedAtLiveTail()) {
          clearTranscriptDom();
          clearRenderedTranscriptRange();
          setOlderState({ hasMore: false, isLoading: false });
        }
        const slot = getSessionTranscriptSlot(sessionId);
        addPendingUser({ id: localId, sessionId, epoch: slot.epoch, text: raw, t0: startedAt });
        appendEvent({ role: "user", text: raw, pending: true, localId, ts: startedAt });
        setTurnOpen(true);
        setCurrentRunning(true);
      }
      try {
        const response = await api(`/api/sessions/${sessionId}/send`, { method: "POST", body: { text: raw, allow_pending_attachment: allowPendingAttachment } });
        if (renderHere && renewsTranscript) {
          deleteTailCache(sessionId);
          beginTranscriptRenewal(sessionId);
          clearLiveCursor();
          clearRenderedTranscriptRange();
          invalidateOlderLoad();
          renderPendingTranscriptSlot(sessionId);
          setTurnOpen(true);
          setCurrentRunning(true);
        }
        const attachmentCleanupError = response && (response.attachment_cleanup_error || response.attachments_cleanup_error) ? String(response.attachment_cleanup_error || response.attachments_cleanup_error) : "";
        const sendStateCleanupError = response && response.send_state_cleanup_error ? String(response.send_state_cleanup_error) : "";
        const deliveredToast = response.queued ? `queued (queue ${response.queue_len})` : "sent";
        const cleanupWarnings = [];
        if (attachmentCleanupError) cleanupWarnings.push(`attachment cleanup failed: ${attachmentCleanupError}`);
        if (sendStateCleanupError) cleanupWarnings.push(`send state cleanup failed: ${sendStateCleanupError}`);
        setToast(cleanupWarnings.length ? `${deliveredToast}; ${cleanupWarnings.join("; ")}` : deliveredToast);
        if (allowPendingAttachment && !attachmentCleanupError) {
          setSelectedSessionPendingAttachment(sessionId, false);
          setAttachCount(0);
        }
        setPollFastUntilMs(now() + 5000);
        kickPoll(0);
        void refreshSessions().catch((error) => {
          if (error && error.status === 401) handleAppAuthLoss();
          else consoleError("refreshSessions failed", error);
        });
        return true;
      } catch (error) {
        if (error && error.status === 401) {
          handleAppAuthLoss();
          return false;
        }
        const commitUnknown = Boolean(error && error.obj && error.obj.commit_unknown);
        if (commitUnknown) {
          setToast("send status unknown; check transcript before retrying");
          patchSessionInfo(sessionId, {
            commit_unknown_send: true,
            commit_unknown_send_text: raw,
            commit_unknown_send_ts: now() / 1000,
          });
          syncSendButtonState();
          syncQueueSubmitState();
          syncAttachButtonState();
          setPollFastUntilMs(now() + 4000);
          kickPoll(0);
          void refreshSessions().catch((refreshError) => {
            if (refreshError && refreshError.status === 401) handleAppAuthLoss();
            else consoleError("refreshSessions failed", refreshError);
          });
        } else {
          setToast(`send error: ${error && error.message ? error.message : "unknown error"}`);
        }
        if (!commitUnknown && sessionInfo && sessionInfo.pending_attachment && /broker must be restarted/i.test(String(error && error.message ? error.message : ""))) {
          const clearPending = await confirmAction({
            title: "Clear pending attachment state?",
            message: "This session has a pending attachment but the current broker cannot confirm sends. Clear the browser pending-attachment state only if you already handled it in the terminal?",
            confirmText: "Clear state",
            cancelText: "Cancel",
            destructive: true,
          });
          if (clearPending) {
            try {
              await api(`/api/sessions/${sessionId}/pending_attachment/clear`, { method: "POST", body: {} });
              setToast("pending attachment state cleared");
              if (getSelected() === sessionId) setSelectedSessionPendingAttachment(sessionId, false);
              void refreshSessions().catch((refreshError) => {
                if (refreshError && refreshError.status === 401) handleAppAuthLoss();
                else consoleError("refreshSessions failed", refreshError);
              });
            } catch (clearError) {
              if (clearError && clearError.status === 401) {
                handleAppAuthLoss();
                return false;
              }
              setToast(`clear pending attachment error: ${clearError && clearError.message ? clearError.message : "unknown error"}`);
            }
          }
        }
        if (renderHere) {
          dropPendingUser(sessionId, localId);
          removePendingUserRow(localId);
          if (!hasPendingForSession(sessionId)) {
            setTurnOpen(false);
            setCurrentRunning(false);
          }
          if (commitUnknown) syncRecoveryUiForSession(sessionId);
        }
        return false;
      } finally {
        setSending(false);
        syncSendButtonState();
        syncAttachButtonState();
      }
    }

    listen(textarea, "input", () => { autoGrow(); saveSessionDraft(getSelected()); });
    listen(textarea, "keydown", (event) => {
      if (event.key !== "Enter" || event.isComposing || !(event.ctrlKey || event.metaKey)) return;
      event.preventDefault();
      form.requestSubmit();
    });
    if (windowTarget) listen(windowTarget, "resize", onAutoGrow);

    form.onsubmit = async (event) => {
      event.preventDefault();
      const sessionId = getSelected();
      if (!sessionId) { setToast("select a session first"); return; }
      if (sessionLaunchFailed(getSessionInfo(sessionId))) { setToast("failed session cannot receive messages"); return; }
      const raw = textarea.value;
      if (!raw || !raw.trim() || getSending()) return;
      if (getCurrentRunning()) {
        const focused = activeElement();
        showSendChoice(raw, { opener: isHTMLElement(focused) ? focused : textarea });
        return;
      }
      const ok = await sendText(raw);
      if (ok && textarea.value === raw) clearComposer();
    };

    sendChoiceNowBtn.onclick = async () => {
      const raw = sendChoicePending && sendChoicePending.text;
      const sessionId = sendChoicePending && sendChoicePending.sid;
      hideSendChoice({ restoreFocus: true });
      if (!raw || !sessionId) return;
      const ok = await sendText(raw, { sid: sessionId });
      if (ok && sessionId === getSelected() && textarea.value === raw) clearComposer();
    };
    sendChoiceLaterBtn.onclick = async () => {
      const raw = sendChoicePending && sendChoicePending.text;
      const sessionId = sendChoicePending && sendChoicePending.sid;
      const hasAttachments = Boolean(sendChoicePending && sendChoicePending.attachmentCount > 0);
      if (hasAttachments) { setToast("attachments can only be sent now; wait until idle to queue text with files"); return; }
      hideSendChoice({ restoreFocus: true });
      if (!raw || !sessionId) return;
      const ok = await enqueueComposerText(raw, { sid: sessionId });
      if (ok && sessionId === getSelected() && textarea.value === raw) clearComposer();
    };
    sendChoiceCancelBtn.onclick = () => hideSendChoice({ restoreFocus: true });
    sendChoiceBackdrop.onclick = () => hideSendChoice({ restoreFocus: true });

    syncSendButtonState();
    autoGrow();

    return Object.freeze({
      autoGrow,
      clearComposer,
      clearSessionDraft,
      loadSessionDraft,
      saveSessionDraft,
      sendText,
      showSendChoice,
      hideSendChoice,
      isSendChoiceOpen: () => sendChoice.style.display === "flex",
      syncComposerState,
      syncSendButtonState,
      dispose() {
        form.onsubmit = null;
        sendChoiceNowBtn.onclick = null;
        sendChoiceLaterBtn.onclick = null;
        sendChoiceCancelBtn.onclick = null;
        sendChoiceBackdrop.onclick = null;
        while (cleanups.length) cleanups.pop()();
      },
    });
  }

  window.CodoxearComposer = Object.freeze({ createComposerController });
})();
