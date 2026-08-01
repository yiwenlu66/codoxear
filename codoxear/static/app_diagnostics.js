(function () {
  "use strict";

  // Details/diagnostics modal authority. Owns every piece of Details/diagnostics
  // state that used to live as app.js locals (return-focus element and copy text)
  // plus the diag Copy conversation / Copy details click behavior, show/hide modal
  // behavior, and the rendering decisions for failed-launch (local recovery rows,
  // no API), live sessions (fetch /diagnostics, ignore stale responses), and the
  // error path.
  //
  // Pure helpers (sessionLaunchFailed) come from window.CodoxearSessionHelpers;
  // modal focus/isolation helpers come from window.CodoxearModal. Everything
  // that touches app-level runtime state (selected session, session index, API,
  // clipboard, toasts, recovery-text helpers, DOM element factory, modal
  // open/close coordination) is injected through createDiagnosticsController(options)
  // so the controller has no hidden coupling to app.js globals and can be exercised
  // in a VM with fakes.

  const codoxearSessionHelpers = window.CodoxearSessionHelpers;
  if (
    !codoxearSessionHelpers ||
    typeof codoxearSessionHelpers.sessionLaunchFailed !== "function"
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

  const sessionLaunchFailed = codoxearSessionHelpers.sessionLaunchFailed;
  const isModalTargetOpen = codoxearModal.isModalTargetOpen;
  const focusModalCloseButton = codoxearModal.focusModalCloseButton;
  const restoreModalFocus = codoxearModal.restoreModalFocus;

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`diagnostics controller dependency missing: ${name}`);
    return value;
  }

  function requireNode(value, name) {
    if (!value || typeof value !== "object" || !value.style) throw new TypeError(`diagnostics controller dependency missing: ${name}`);
    return value;
  }

  function requireString(value, name) {
    if (typeof value !== "string") throw new TypeError(`diagnostics controller dependency missing: ${name}`);
    return value;
  }

  function createDiagnosticsController(options = {}) {
    if (!options || typeof options !== "object") throw new TypeError("diagnostics controller dependency missing: options");

    // DOM nodes (created and owned by app.js).
    const diagBackdrop = requireNode(options.diagBackdrop, "diagBackdrop");
    const diagViewer = requireNode(options.diagViewer, "diagViewer");
    const diagContent = requireNode(options.diagContent, "diagContent");
    const diagStatus = requireNode(options.diagStatus, "diagStatus");
    const diagCloseBtn = requireNode(options.diagCloseBtn, "diagCloseBtn");
    const diagCopyConversationBtn = requireNode(options.diagCopyConversationBtn, "diagCopyConversationBtn");
    const diagCopyBtn = requireNode(options.diagCopyBtn, "diagCopyBtn");

    // App-level runtime state accessors and effects.
    const getSelected = requireFunction(options.getSelected, "getSelected");
    const getSessionInfo = requireFunction(options.getSessionInfo, "getSessionInfo");
    const api = requireFunction(options.api, "api");
    const setToast = requireFunction(options.setToast, "setToast");
    const copyToClipboard = requireFunction(options.copyToClipboard, "copyToClipboard");
    const copyConversation = requireFunction(options.copyConversation, "copyConversation");
    const recoveryDetailsText = requireFunction(options.recoveryDetailsText, "recoveryDetailsText");
    const redactedLaunchErrorText = requireFunction(options.redactedLaunchErrorText, "redactedLaunchErrorText");
    const sessionLaunchLabel = requireFunction(options.sessionLaunchLabel, "sessionLaunchLabel");
    const agentBackendDisplayName = requireFunction(options.agentBackendDisplayName, "agentBackendDisplayName");
    const diagnosticsProviderDisplay = requireFunction(options.diagnosticsProviderDisplay, "diagnosticsProviderDisplay");
    const diagnosticsCopyText = requireFunction(options.diagnosticsCopyText, "diagnosticsCopyText");
    const fmtTs = requireFunction(options.fmtTs, "fmtTs");
    const fmtRelativeAge = requireFunction(options.fmtRelativeAge, "fmtRelativeAge");
    const formatPriorityOffset = requireFunction(options.formatPriorityOffset, "formatPriorityOffset");
    const prepareModalOpen = requireFunction(options.prepareModalOpen, "prepareModalOpen");
    const afterModalVisibilityChanged = requireFunction(options.afterModalVisibilityChanged, "afterModalVisibilityChanged");
    const el = requireFunction(options.el, "el");
    const uiVersion = requireString(options.uiVersion, "uiVersion");

    const requestFrame = typeof options.requestFrame === "function" ? options.requestFrame : requestAnimationFrame;

    // Details/diagnostics state owned by this controller.
    let diagReturnFocusEl = null;
    let diagCopyText = "";
    let diagConversationCopyReady = false;

    function resetActionButtonState() {
      diagCopyConversationBtn.disabled = true;
      diagCopyBtn.disabled = true;
    }

    function applyActionButtonState() {
      diagCopyConversationBtn.disabled = !diagConversationCopyReady;
      diagCopyBtn.disabled = !diagCopyText;
    }

    function addRowTo(content, rows, label, value, { mono = false } = {}) {
      const cleanLabel = String(label || "");
      const v = value == null || value === "" ? "-" : String(value);
      if (rows) rows.push([cleanLabel, v]);
      const row = el("div", { class: "detailsRow" });
      row.appendChild(el("div", { class: "detailsLabel", text: cleanLabel }));
      row.appendChild(el("div", { class: mono ? "detailsValue mono" : "detailsValue", text: v }));
      content.appendChild(row);
    }

    function renderFailedLaunchRows(sid, selectedInfo) {
      diagStatus.textContent = "";
      const addRecoveryRow = (label, value, opts = {}) => addRowTo(diagContent, null, label, value, opts);
      addRecoveryRow("Session", sid);
      addRecoveryRow("State", "launch failed");
      addRecoveryRow("Stage", selectedInfo.launch_stage || "-");
      addRecoveryRow("Error", redactedLaunchErrorText(selectedInfo.launch_error || "-"));
      addRecoveryRow("CWD", selectedInfo.cwd || "-", { mono: true });
      addRecoveryRow("Agent", agentBackendDisplayName(selectedInfo.agent_backend));
      addRecoveryRow("Provider", diagnosticsProviderDisplay(selectedInfo));
      addRecoveryRow("Model", selectedInfo.model || "-");
      addRecoveryRow("Reasoning", selectedInfo.reasoning_effort || "-");
      addRecoveryRow(
        "tmux",
        selectedInfo.tmux_session
          ? `${selectedInfo.tmux_session}${selectedInfo.tmux_window ? ":" + selectedInfo.tmux_window : ""}`
          : "-"
      );
      diagCopyText = recoveryDetailsText(sid, selectedInfo);
      diagConversationCopyReady = true;
      applyActionButtonState();
    }

    function renderLiveRows(sid, d) {
      diagStatus.textContent = "";
      const now = Date.now() / 1000;
      const diagRows = [];
      const addRow = (label, value, opts = {}) => addRowTo(diagContent, diagRows, label, value, opts);
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
      addRow("UI", uiVersion);
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
      diagConversationCopyReady = true;
      applyActionButtonState();
    }

    function showErrorState(e) {
      diagCopyText = "";
      diagConversationCopyReady = false;
      resetActionButtonState();
      diagStatus.textContent = `error: ${e && e.message ? e.message : "unknown error"}`;
    }

    async function show({ opener = null } = {}) {
      const sid = getSelected();
      if (!sid) return;
      diagReturnFocusEl = opener instanceof HTMLElement ? opener : document.activeElement instanceof HTMLElement ? document.activeElement : null;
      prepareModalOpen();
      diagContent.innerHTML = "";
      diagCopyText = "";
      diagConversationCopyReady = false;
      resetActionButtonState();
      diagStatus.textContent = "Loading...";
      diagBackdrop.style.display = "block";
      diagViewer.style.display = "flex";
      afterModalVisibilityChanged();
      focusModalCloseButton(diagViewer, diagCloseBtn, requestFrame);
      const selectedInfo = getSessionInfo(sid) || null;
      if (sessionLaunchFailed(selectedInfo)) {
        renderFailedLaunchRows(sid, selectedInfo);
        return;
      }
      try {
        const d = await api(`/api/sessions/${sid}/diagnostics`);
        if (getSelected() !== sid) return;
        renderLiveRows(sid, d);
      } catch (e) {
        if (getSelected() !== sid) return;
        showErrorState(e);
      }
    }

    function hide({ restoreFocus = true } = {}) {
      const wasOpen = isModalTargetOpen(diagViewer);
      const focusTarget = diagReturnFocusEl;
      diagReturnFocusEl = null;
      diagBackdrop.style.display = "none";
      diagViewer.style.display = "none";
      afterModalVisibilityChanged();
      if (restoreFocus && wasOpen) restoreModalFocus(focusTarget, () => isModalTargetOpen(diagViewer), requestFrame);
    }

    async function onCopyConversationClick(e) {
      if (e && typeof e.preventDefault === "function") {
        e.preventDefault();
        e.stopPropagation();
      }
      if (!diagConversationCopyReady) {
        setToast("details not loaded");
        return;
      }
      diagCopyConversationBtn.disabled = true;
      try {
        await copyConversation();
      } finally {
        applyActionButtonState();
      }
    }

    async function onCopyClick(e) {
      if (e && typeof e.preventDefault === "function") {
        e.preventDefault();
        e.stopPropagation();
      }
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
    }

    function dispose() {
      diagReturnFocusEl = null;
      diagCopyText = "";
      diagConversationCopyReady = false;
      resetActionButtonState();
    }

    return Object.freeze({
      show,
      hide,
      onCopyConversationClick,
      onCopyClick,
      dispose,
    });
  }

  window.CodoxearDiagnostics = Object.freeze({ createDiagnosticsController });
})();
