(function () {
  "use strict";

  // Recovery panel authority. Owns every piece of recovery-panel rendering,
  // action handling, and focus-preservation state that used to live as app.js
  // locals (pendingRecoveryFocusDescriptor plus the render/focus helpers and
  // the recovery-panel DOM construction for Launch failed / Recovery needed).
  //
  // Pure helpers come from window.CodoxearSessionHelpers (sessionLaunchFailed).
  // Everything that touches app-level runtime state (chat container, queue
  // button, typing-row anchor, session index, recovery text/preset factories,
  // queue viewer opener, commit-unknown clear, new-session dialog opener,
  // failed-launch dismiss, clipboard, toasts, DOM element factory, animation
  // frame) is injected through createRecoveryPanelController(options) so the
  // controller has no hidden coupling to app.js globals and can be exercised
  // in a VM with fakes.

  const codoxearSessionHelpers = window.CodoxearSessionHelpers;
  if (
    !codoxearSessionHelpers ||
    typeof codoxearSessionHelpers.sessionLaunchFailed !== "function"
  )
    throw new Error("Codoxear session helpers failed to load");

  const sessionLaunchFailed = codoxearSessionHelpers.sessionLaunchFailed;

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`recovery controller dependency missing: ${name}`);
    return value;
  }

  function requireNode(value, name) {
    if (!value || typeof value !== "object" || !value.style) throw new TypeError(`recovery controller dependency missing: ${name}`);
    return value;
  }

  function createRecoveryPanelController(options = {}) {
    if (!options || typeof options !== "object") throw new TypeError("recovery controller dependency missing: options");

    // DOM nodes (created and owned by app.js).
    const chatInner = requireNode(options.chatInner, "chatInner");
    const queueBtn = requireNode(options.queueBtn, "queueBtn");

    // App-level runtime state accessors and effects.
    const typingRowAnchor = requireFunction(options.typingRowAnchor, "typingRowAnchor");
    const getSessionInfo = requireFunction(options.getSessionInfo, "getSessionInfo");
    const el = requireFunction(options.el, "el");
    const recoveryPromptPreview = requireFunction(options.recoveryPromptPreview, "recoveryPromptPreview");
    const redactedLaunchErrorText = requireFunction(options.redactedLaunchErrorText, "redactedLaunchErrorText");
    const recoveryDetailsText = requireFunction(options.recoveryDetailsText, "recoveryDetailsText");
    const launchPresetFromSessionInfo = requireFunction(options.launchPresetFromSessionInfo, "launchPresetFromSessionInfo");
    const showQueueViewer = requireFunction(options.showQueueViewer, "showQueueViewer");
    const clearCommitUnknownSend = requireFunction(options.clearCommitUnknownSend, "clearCommitUnknownSend");
    const openNewSessionDialog = requireFunction(options.openNewSessionDialog, "openNewSessionDialog");
    const dismissFailedLaunchRecord = requireFunction(options.dismissFailedLaunchRecord, "dismissFailedLaunchRecord");
    const copyToClipboard = requireFunction(options.copyToClipboard, "copyToClipboard");
    const setToast = requireFunction(options.setToast, "setToast");

    const requestFrame = typeof options.requestFrame === "function" ? options.requestFrame : requestAnimationFrame;

    // Recovery-panel focus-preservation state owned by this controller.
    let pendingRecoveryFocusDescriptor = null;

    function recoverySessionInfo(sessionId) {
      const s = getSessionInfo(sessionId);
      if (!s || (!sessionLaunchFailed(s) && !s.orphan_recovery && !s.queue_recovery && !s.commit_unknown_send)) return null;
      return s;
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
      requestFrame(() => {
        try {
          if (target.isConnected && !target.disabled) {
            target.focus({ preventScroll: true });
            pendingRecoveryFocusDescriptor = null;
          }
        } catch {}
      });
      return true;
    }

    function focusFallbackCandidate() {
      const fallback = chatInner.querySelector(".recovery-panel .icon-btn");
      return fallback || null;
    }

    function focusRecoveryFallback(descriptor) {
      if (!descriptor) return;
      const fallback = focusFallbackCandidate() || queueBtn || null;
      if (!fallback || typeof fallback.focus !== "function" || fallback.disabled) {
        pendingRecoveryFocusDescriptor = null;
        return;
      }
      pendingRecoveryFocusDescriptor = descriptor;
      requestFrame(() => {
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
        const launchStage = String(s.launch_stage || "").trim();
        const launchSummary = launchStage.endsWith("_after_log_bind")
          ? "This web-owned session stopped after binding a transcript log, before the turn completed."
          : "This web-owned session failed before a usable session log was bound.";
        list.appendChild(el("li", { text: launchSummary }));
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
      chatInner.insertBefore(row, typingRowAnchor());
      if (focusDescriptor && !focusRecoveryAction(row, focusDescriptor)) focusRecoveryFallback(focusDescriptor);
      return true;
    }

    function dispose() {
      pendingRecoveryFocusDescriptor = null;
    }

    return Object.freeze({
      render: renderRecoveryPanelIfNeeded,
      renderRecoveryPanelIfNeeded,
      focusFallbackCandidate,
      dispose,
    });
  }

  window.CodoxearRecovery = Object.freeze({ createRecoveryPanelController });
})();
