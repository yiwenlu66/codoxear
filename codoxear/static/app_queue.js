(function () {
  "use strict";

  // Queue orchestration authority. Owns every piece of queue state that used to
  // live as app.js locals (timers, mutation locks, pending deletes, draft text,
  // submit-busy flag, viewer sid/items, modal return-focus) plus the queue
  // button submit-state projection, the enqueue/move/delete/update decision
  // logic, the queue viewer modal behavior, and the rendered queue list.
  //
  // Pure helpers (normalizeQueueItems + the launch/recovery predicates) come
  // from window.CodoxearSessionHelpers; modal focus/isolation helpers come from
  // window.CodoxearModal. Everything that touches app-level runtime state
  // (selected session, session index, polling, recovery UI, toasts, auth loss,
  // modal open/close coordination, DOM element factory) is injected through
  // createQueueController(options) so the controller has no hidden coupling to
  // app.js globals and can be exercised in a VM with fakes.

  const codoxearSessionHelpers = window.CodoxearSessionHelpers;
  if (
    !codoxearSessionHelpers ||
    typeof codoxearSessionHelpers.normalizeQueueItems !== "function" ||
    typeof codoxearSessionHelpers.sessionLaunchFailed !== "function" ||
    typeof codoxearSessionHelpers.sessionHasUnknownSend !== "function" ||
    typeof codoxearSessionHelpers.sessionIsOrphanRecovery !== "function" ||
    typeof codoxearSessionHelpers.sessionHasOrphanQueueRecovery !== "function"
  )
    throw new Error("Codoxear session helpers failed to load");

  const codoxearModal = window.CodoxearModal;
  if (
    !codoxearModal ||
    typeof codoxearModal.isModalTargetOpen !== "function" ||
    typeof codoxearModal.focusModalCloseButton !== "function" ||
    typeof codoxearModal.restoreModalFocus !== "function"
  )
    throw new Error("Codoxear modal helpers failed to load");

  const normalizeQueueItems = codoxearSessionHelpers.normalizeQueueItems;
  const sessionLaunchFailed = codoxearSessionHelpers.sessionLaunchFailed;
  const sessionHasUnknownSend = codoxearSessionHelpers.sessionHasUnknownSend;
  const sessionIsOrphanRecovery = codoxearSessionHelpers.sessionIsOrphanRecovery;
  const sessionHasOrphanQueueRecovery = codoxearSessionHelpers.sessionHasOrphanQueueRecovery;
  const isModalTargetOpen = codoxearModal.isModalTargetOpen;
  const focusModalCloseButton = codoxearModal.focusModalCloseButton;
  const restoreModalFocus = codoxearModal.restoreModalFocus;

  const QUEUE_UPDATE_DEBOUNCE_MS = 350;
  const QUEUE_REFRESH_EDIT_GUARD_MS = 900;

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`queue controller dependency missing: ${name}`);
    return value;
  }

  function requireNode(value, name) {
    if (!value || typeof value !== "object" || !value.style) throw new TypeError(`queue controller dependency missing: ${name}`);
    return value;
  }

  function createQueueController(options = {}) {
    if (!options || typeof options !== "object") throw new TypeError("queue controller dependency missing: options");

    // DOM nodes (created and owned by app.js).
    const queueBackdrop = requireNode(options.queueBackdrop, "queueBackdrop");
    const queueCloseBtn = requireNode(options.queueCloseBtn, "queueCloseBtn");
    const queueList = requireNode(options.queueList, "queueList");
    const queueEmpty = requireNode(options.queueEmpty, "queueEmpty");
    const queueViewer = requireNode(options.queueViewer, "queueViewer");
    const queueBtn = requireNode(options.queueBtn, "queueBtn");

    // App-level runtime state accessors.
    const getSelected = requireFunction(options.getSelected, "getSelected");
    const getSessionInfo = requireFunction(options.getSessionInfo, "getSessionInfo");
    const isAppDisposed = requireFunction(options.isAppDisposed, "isAppDisposed");
    const api = requireFunction(options.api, "api");
    const setToast = requireFunction(options.setToast, "setToast");
    const clearCommitUnknownSend = requireFunction(options.clearCommitUnknownSend, "clearCommitUnknownSend");
    const refreshSessions = requireFunction(options.refreshSessions, "refreshSessions");
    const updateQueueBadge = requireFunction(options.updateQueueBadge, "updateQueueBadge");
    const syncRecoveryUiForSession = requireFunction(options.syncRecoveryUiForSession, "syncRecoveryUiForSession");
    const kickPoll = requireFunction(options.kickPoll, "kickPoll");
    const setPollFastUntilMs = requireFunction(options.setPollFastUntilMs, "setPollFastUntilMs");
    const handleAppAuthLoss = requireFunction(options.handleAppAuthLoss, "handleAppAuthLoss");
    const prepareModalOpen = requireFunction(options.prepareModalOpen, "prepareModalOpen");
    const afterModalVisibilityChanged = requireFunction(options.afterModalVisibilityChanged, "afterModalVisibilityChanged");
    const el = requireFunction(options.el, "el");
    const iconSvg = requireFunction(options.iconSvg, "iconSvg");
    const recoveryPanelFocusFallback = requireFunction(options.recoveryPanelFocusFallback, "recoveryPanelFocusFallback");

    const requestFrame = typeof options.requestFrame === "function" ? options.requestFrame : requestAnimationFrame;
    const setTimeoutFn = typeof options.setTimeout === "function" ? options.setTimeout : setTimeout;
    const clearTimeoutFn = typeof options.clearTimeout === "function" ? options.clearTimeout : clearTimeout;
    const nowFn = typeof options.now === "function" ? options.now : () => Date.now();

    // Queue state owned by this controller.
    const queueUpdateTimers = new Map();
    const queueMutationLocks = new Set();
    const queuePendingDeletes = new Set();
    const queueDraftTexts = new Map();
    let queueLastEditMs = 0;
    let queueSubmitBusy = false;
    let queueViewerSid = null;
    let queueViewerItems = [];
    let queueReturnFocusEl = null;

    function selectedSessionHasUnknownSend() {
      const selected = getSelected();
      return sessionHasUnknownSend(selected ? getSessionInfo(selected) : null);
    }

    function selectedSessionIsOrphanRecovery() {
      const selected = getSelected();
      return sessionIsOrphanRecovery(selected ? getSessionInfo(selected) : null);
    }

    function selectedSessionHasOrphanQueueRecovery() {
      const selected = getSelected();
      return sessionHasOrphanQueueRecovery(selected ? getSessionInfo(selected) : null);
    }

    function selectedSessionLaunchFailed() {
      const selected = getSelected();
      return sessionLaunchFailed(selected ? getSessionInfo(selected) : null);
    }

    function syncQueueSubmitState() {
      if (!queueBtn) return;
      const selected = getSelected();
      const unknownSend = selectedSessionHasUnknownSend();
      const orphanQueueRecovery = selectedSessionHasOrphanQueueRecovery();
      const launchFailed = selectedSessionLaunchFailed();
      queueBtn.disabled = !!queueSubmitBusy || !selected || launchFailed || (unknownSend && !orphanQueueRecovery);
      const queueLabel = !selected
        ? "Select a session to view queued messages"
        : launchFailed
          ? "Failed launch cannot receive queued messages"
          : orphanQueueRecovery
            ? "Review preserved queued recovery items"
            : unknownSend
              ? "Resolve the unknown send before queueing"
              : "Queued messages";
      queueBtn.title = queueLabel;
      queueBtn.setAttribute("aria-label", queueLabel);
    }

    async function enqueueComposerText(raw, { sid = null } = {}) {
      const selected = getSelected();
      const sessionId = sid || selected;
      const text = String(raw || "");
      if (!sessionId || !text.trim()) return false;
      const sessionInfo = getSessionInfo(sessionId) || null;
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
        setPollFastUntilMs(nowFn() + 5000);
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
      const item = queueViewerItems.find((candidate) => String((candidate && candidate.id) || "") === key) || null;
      const commitUnknown = Boolean(item && item.commitUnknown);
      const orphanRecovery = Boolean(item && item.orphanRecovery);
      if (commitUnknown || orphanRecovery) {
        const text = String((item && item.text) || "").trim();
        const suffix = text ? `\n\nQueued prompt: ${text.slice(0, 240)}${text.length > 240 ? "..." : ""}` : "";
        const confirmed = window.confirm(
          `Delete this recovery item only after checking the transcript or terminal.${commitUnknown ? " This may allow later queued prompts to send." : ""}${suffix}`
        );
        if (!confirmed) return;
      }
      const timerKey = `${sid}:${key}`;
      const pendingUpdate = queueUpdateTimers.get(timerKey);
      if (pendingUpdate) {
        clearTimeoutFn(pendingUpdate);
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
      queueViewerItems = queueViewerItems.filter((entry) => String(entry.id || "") !== key);
      queueDraftTexts.delete(key);
      renderQueueList();
      try {
        await api(`/api/sessions/${sid}/queue/delete`, { method: "POST", body: { id: key, allow_commit_unknown: commitUnknown, allow_orphan_recovery: orphanRecovery } });
        await refreshSessions();
        updateQueueBadge();
        syncRecoveryUiForSession(sid);
        if (queueViewer.style.display === "flex") {
          const refreshedSession = getSessionInfo(sid);
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
        if (existing0) clearTimeoutFn(existing0);
        queueUpdateTimers.delete(key0);
        return;
      }
      const key = `${sid}:${itemKey}`;
      const existing = queueUpdateTimers.get(key);
      if (existing) clearTimeoutFn(existing);
      const t = setTimeoutFn(async () => {
        queueUpdateTimers.delete(key);
        if (isAppDisposed()) return;
        queueMutationLocks.add(itemKey);
        try {
          await api(`/api/sessions/${sid}/queue/update`, { method: "POST", body: { id: itemKey, text } });
          if (isAppDisposed()) return;
          queueLastEditMs = 0;
          queueDraftTexts.set(itemKey, text);
          await refreshQueueViewer();
          if (isAppDisposed()) return;
          await refreshSessions();
          if (isAppDisposed()) return;
          updateQueueBadge();
          syncRecoveryUiForSession(sid);
        } catch (e) {
          if (isAppDisposed()) return;
          if (e && e.status === 401) {
            handleAppAuthLoss();
            return;
          }
          setToast(`queue update error: ${e && e.message ? e.message : "unknown error"}`);
        } finally {
          queueMutationLocks.delete(itemKey);
          if (!isAppDisposed() && queuePendingDeletes.has(itemKey)) {
            queuePendingDeletes.delete(itemKey);
            void deleteQueueItem(sid, itemKey);
          }
        }
      }, QUEUE_UPDATE_DEBOUNCE_MS);
      queueUpdateTimers.set(key, t);
    }

    function renderQueueList() {
      queueList.innerHTML = "";
      const sid = queueViewerSid || getSelected();
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
          queueLastEditMs = nowFn();
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
      const sid = queueViewerSid || getSelected();
      if (!sid) return;
      if (queueViewer.style.display === "flex" && nowFn() - queueLastEditMs < QUEUE_REFRESH_EDIT_GUARD_MS) return;
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
      const selected = getSelected();
      if (!selected) return;
      queueReturnFocusEl = opener instanceof HTMLElement ? opener : document.activeElement instanceof HTMLElement ? document.activeElement : null;
      prepareModalOpen();
      queueViewerSid = selected;
      queueBackdrop.style.display = "block";
      queueViewer.style.display = "flex";
      afterModalVisibilityChanged();
      focusModalCloseButton(queueViewer, queueCloseBtn, requestFrame);
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
        const fallback = recoveryPanelFocusFallback() || queueBtn || null;
        restoreModalFocus(focusTarget && focusTarget.isConnected ? focusTarget : fallback, () => isModalTargetOpen(queueViewer), requestFrame);
      }
    }

    function dispose() {
      queueUpdateTimers.forEach((timer) => clearTimeoutFn(timer));
      queueUpdateTimers.clear();
      queueMutationLocks.clear();
      queuePendingDeletes.clear();
      queueDraftTexts.clear();
      queueLastEditMs = 0;
      queueSubmitBusy = false;
      queueViewerSid = null;
      queueViewerItems = [];
      queueReturnFocusEl = null;
    }

    return Object.freeze({
      syncQueueSubmitState,
      enqueueComposerText,
      deleteQueueItem,
      moveQueueItem,
      scheduleQueueUpdate,
      renderQueueList,
      refreshQueueViewer,
      showQueueViewer,
      hideQueueViewer,
      dispose,
    });
  }

  window.CodoxearQueue = Object.freeze({ createQueueController });
})();
