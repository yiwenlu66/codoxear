(function () {
  "use strict";

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`shell dependency missing: ${name}`);
    return value;
  }

  function requireNode(value, name) {
    if (!value || typeof value !== "object" || typeof value.appendChild !== "function")
      throw new TypeError(`shell dependency missing: ${name}`);
    return value;
  }

  function createShellDOM(options = {}) {
    if (!options || typeof options !== "object") throw new TypeError("shell dependency missing: options");
    const root = requireNode(options.root, "root");
    const el = requireFunction(options.el, "el");
    const iconSvg = requireFunction(options.iconSvg, "iconSvg");
    const resolveAppUrl = requireFunction(options.resolveAppUrl, "resolveAppUrl");
    const versionedShellAssetPath = requireFunction(options.versionedShellAssetPath, "versionedShellAssetPath");

    root.innerHTML = "";
    const backdrop = el("div", { class: "backdrop", id: "backdrop" });
    const app = el("div", { class: "app" });
    const sidebar = el("div", { class: "sidebar" });
    const sessionsWrap = el("div", { class: "sessions", id: "sessions" });
    const sidebarEmptyHint = el("div", { class: "sidebarEmptyHint muted", text: "No sessions yet" });
    const sidebarFooter = el("footer", {}, [
      el("button", { id: "helpBtnSide", type: "button", title: "Help", "aria-label": "Help", html: iconSvg("help") + "Help" }),
      el("button", { id: "settingsBtnSide", type: "button", title: "Settings", "aria-label": "Settings", html: iconSvg("settings") + "Settings" }),
      el("button", { id: "logoutBtnSide", type: "button", title: "Log out", "aria-label": "Log out", html: iconSvg("logout") + "Log out" }),
    ]);
    const main = el("div", { class: "main" });
    const chatWrap = el("div", { class: "chatWrap", id: "chatWrap" });
    const chatEmptyState = el("div", { class: "chatEmptyState", id: "chatEmptyState" }, [
      el("div", { class: "chatEmptyCopy muted", text: "Start a session to begin a conversation." }),
      el("button", { id: "chatEmptyNewBtn", class: "icon-btn text-btn", type: "button", title: "New session", "aria-label": "New session", text: "New session" }),
    ]);
    const chat = el("div", { class: "chat", id: "chat" });
    const chatInner = el("div", { class: "chatInner", id: "chatInner" });
    const olderWrap = el("div", { class: "olderWrap", id: "olderWrap" });
    const olderBtn = el("button", { class: "olderBtn", id: "olderBtn", type: "button", text: "Load older messages" });
    const olderErrorText = el("span", { class: "olderErrorText", text: "" });
    const olderRetryBtn = el("button", { class: "olderRetryBtn", type: "button", text: "Retry" });
    const olderError = el("div", { class: "olderError", id: "olderError", role: "status" }, [olderErrorText, olderRetryBtn]);
    olderWrap.append(olderBtn, olderError);
    const bottomSentinel = el("div", { id: "bottomSentinel" });
    const jumpBtn = el("button", { class: "jumpBtn", id: "jumpBtn", title: "Jump to latest message", "aria-label": "Jump to latest message", html: iconSvg("down") });
    const chatTimeChip = el("div", { id: "chatTimeChip", class: "chatTimeChip", "aria-hidden": "true" });
    const chatSearchInput = el("input", { id: "chatSearchInput", class: "chatSearchInput", type: "search", placeholder: "Search loaded chat", "aria-label": "Search loaded chat messages", autocomplete: "off" });
    const chatSearchPrevBtn = el("button", { id: "chatSearchPrevBtn", class: "icon-btn", type: "button", title: "Previous match", "aria-label": "Previous match", html: iconSvg("up") });
    const chatSearchNextBtn = el("button", { id: "chatSearchNextBtn", class: "icon-btn", type: "button", title: "Next match", "aria-label": "Next match", html: iconSvg("down") });
    const chatSearchCloseBtn = el("button", { id: "chatSearchCloseBtn", class: "icon-btn", type: "button", title: "Close search", "aria-label": "Close search", html: iconSvg("x") });
    const chatSearchStatus = el("span", { id: "chatSearchStatus", class: "chatSearchStatus", text: "Loaded" });
    const chatSearchAllHintEl = el("span", { id: "chatSearchAllHint", class: "chatSearchAllHint", text: "" });
    const chatSearchBar = el("div", { id: "chatSearchBar", class: "chatSearchBar", role: "search", "aria-label": "Search loaded chat messages" }, [chatSearchInput, chatSearchStatus, chatSearchAllHintEl, chatSearchPrevBtn, chatSearchNextBtn, chatSearchCloseBtn]);
    chatSearchBar.style.display = "none";
    chatInner.append(olderWrap, bottomSentinel);
    chat.appendChild(chatInner);
    chatWrap.append(chat, chatEmptyState, jumpBtn, chatTimeChip, chatSearchBar);

    const titleLabel = el("div", { id: "threadTitle", text: "No session selected" });
    const statusChip = el("span", { class: "status-chip", id: "statusChip", text: "Idle" });
    const ctxChip = el("button", { class: "status-chip", id: "ctxChip", text: "", type: "button", "aria-label": "Context usage details" });
    ctxChip.style.display = "none";
    ctxChip.disabled = true;
    const interruptBtn = el("button", { id: "interruptBtn", class: "icon-btn", title: "Interrupt (Esc)", "aria-label": "Interrupt (Esc)", type: "button", html: iconSvg("stop") });
    interruptBtn.style.display = "none";
    const toast = el("div", { class: "muted toast", id: "toast", role: "status", "aria-live": "polite" });
    const toggleSidebarBtn = el("button", { id: "toggleSidebarBtn", class: "icon-btn", title: "Toggle sidebar", "aria-label": "Toggle sidebar", html: iconSvg("menu") });
    const unattendedBtn = el("button", { id: "unattendedBtn", class: "icon-btn", title: "Unattended mode", "aria-label": "Unattended mode", "aria-controls": "unattendedMenu", "aria-expanded": "false", "aria-haspopup": "dialog", type: "button", html: iconSvg("unattended") });
    unattendedBtn.disabled = true;
    const announceBtn = el("button", { id: "announceBtn", class: "icon-btn", title: "Voice announcements", "aria-label": "Voice announcements", type: "button", html: iconSvg("volume") });
    const notificationBtn = el("button", { id: "notificationBtn", class: "icon-btn", title: "Notifications", "aria-label": "Notifications", type: "button", html: iconSvg("bell") });
    const diagBtn = el("button", { id: "diagBtn", class: "icon-btn", title: "Details", "aria-label": "Details", type: "button", html: iconSvg("info") });
    diagBtn.disabled = true;
    const prevUserBtn = el("button", { id: "prevUserBtn", class: "icon-btn", title: "Previous user message (Alt+↑)", "aria-label": "Previous user message (Alt+↑)", type: "button", html: iconSvg("up") });
    const nextUserBtn = el("button", { id: "nextUserBtn", class: "icon-btn", title: "Next user message (Alt+↓)", "aria-label": "Next user message (Alt+↓)", type: "button", html: iconSvg("down") });
    const chatSearchBtn = el("button", { id: "chatSearchBtn", class: "icon-btn", title: "Search loaded messages", "aria-label": "Search loaded messages", type: "button", html: iconSvg("search") });
    const fileBtn = el("button", { id: "fileBtn", class: "icon-btn", title: "View file", "aria-label": "View file", type: "button", html: iconSvg("file") });
    prevUserBtn.disabled = nextUserBtn.disabled = chatSearchBtn.disabled = fileBtn.disabled = true;
    const unattendedMenu = el("div", { id: "unattendedMenu", class: "unattendedMenu", role: "dialog", "aria-label": "Unattended mode settings" }, [
      el("div", { class: "row" }, [el("label", {}, [el("input", { type: "checkbox", id: "unattendedEnabled" }), el("span", { text: "Unattended mode" })])]),
      el("div", { class: "unattendedGrid" }, [
        el("div", {}, [el("div", { class: "label", text: "Cooldown time (minutes)" }), el("input", { id: "unattendedCooldownMinutes", type: "number", min: "1", step: "1", inputmode: "numeric", "aria-label": "Unattended cooldown time in minutes" })]),
        el("div", {}, [el("div", { class: "label", text: "Number of injections" }), el("input", { id: "unattendedRemainingInjections", type: "number", min: "0", step: "1", inputmode: "numeric", "aria-label": "Unattended remaining injections" })]),
      ]),
      el("div", { class: "label", text: "Additional request to append (optional; per session)" }),
      el("textarea", { id: "unattendedRequest", "aria-label": "Additional request for unattended prompt" }),
    ]);
    const liveAudio = el("audio", { id: "liveAudio", preload: "none", playsinline: "true" });
    liveAudio.style.display = "none";
    const topMeta = el("div", { class: "topMeta" }, [ctxChip]);
    const titleRow = el("div", { class: "titleRow" }, [titleLabel, topMeta]);
    const titleWrap = el("div", { class: "titleWrap" }, [titleRow]);
    const chatMessageNavControls = el("div", { class: "chatMessageNavControls", role: "group", "aria-label": "User message navigation" }, [prevUserBtn, nextUserBtn]);
    const chatNavRail = el("div", { class: "chatNavRail", id: "chatNavRail", "aria-label": "Loaded chat navigation" }, [chatSearchBtn, chatMessageNavControls]);
    chatWrap.appendChild(chatNavRail);
    const topbar = el("div", { class: "topbar" }, [el("div", { class: "pill" }, [toggleSidebarBtn, titleWrap]), el("div", { class: "actions topActions" }, [fileBtn, diagBtn, unattendedBtn, interruptBtn])]);
    const composer = el("div", { class: "composer" });
    const stagedTray = el("div", { class: "stagedAttachments", id: "stagedAttachments", "aria-live": "polite" });
    const textarea = el("textarea", { id: "msg", placeholder: "", "aria-label": "Enter your instructions here" });
    const msgPh = el("div", { class: "ph", id: "msgPh", text: "Enter your instructions here" });
    const imgInput = el("input", { id: "imgInput", type: "file", accept: "image/*,video/*,*/*", multiple: "multiple", style: "display:none" });
    const attachBtn = el("button", { class: "icon-btn", id: "attachBtn", type: "button", title: "Attach file", "aria-label": "Attach file", html: iconSvg("paperclip") });
    const queueBtn = el("button", { class: "icon-btn", id: "queueBtn", type: "button", title: "Queued messages", "aria-label": "Queued messages", html: iconSvg("queue") });
    const composerStopBtn = el("button", { class: "icon-btn composerStopBtn", id: "composerStopBtn", type: "button", title: "Stop current response", "aria-label": "Stop current response", html: iconSvg("stop") });
    const sendBtn = el("button", { class: "icon-btn primary", id: "sendBtn", type: "submit", title: "Send", "aria-label": "Send", html: iconSvg("send") });
    const form = el("form", {}, [attachBtn, el("div", { class: "inputWrap" }, [stagedTray, textarea, msgPh]), imgInput, queueBtn, composerStopBtn, sendBtn]);
    composer.appendChild(form);
    sidebar.appendChild(el("header", {}, [el("div", { class: "title", html: `<img class="sidebarLogo" src="${resolveAppUrl(versionedShellAssetPath("/static/codoxear-icon.png"))}" alt="" />Codoxear` }), el("div", { class: "actions" }, [el("button", { id: "newBtn", class: "icon-btn", title: "New session", "aria-label": "New session", html: iconSvg("plus") }), notificationBtn, announceBtn])]));
    sidebar.append(sessionsWrap, sidebarFooter);
    main.append(topbar, toast, chatWrap, composer);
    app.append(sidebar, main, backdrop);
    root.append(app, unattendedMenu, liveAudio);

    const elements = Object.freeze({ root, app, backdrop, sidebar, sessionsWrap, sidebarEmptyHint, main, chatWrap, chatEmptyState, chat, chatInner, olderWrap, olderBtn, olderRetryBtn, olderError, olderErrorText, bottomSentinel, jumpBtn, chatTimeChip, chatSearchInput, chatSearchPrevBtn, chatSearchNextBtn, chatSearchCloseBtn, chatSearchStatus, chatSearchAllHintEl, chatSearchBar, chatNavRail, titleLabel, statusChip, ctxChip, interruptBtn, toast, toggleSidebarBtn, unattendedBtn, announceBtn, notificationBtn, diagBtn, prevUserBtn, nextUserBtn, chatSearchBtn, fileBtn, unattendedMenu, liveAudio, composer, form, textarea, msgPh, imgInput, attachBtn, queueBtn, composerStopBtn, sendBtn, stagedTray });
    return Object.freeze({ elements, cleanup() { root.innerHTML = ""; } });
  }

  function createSidebarController(options = {}) {
    if (!options || typeof options !== "object") throw new TypeError("shell dependency missing: sidebar options");
    const sessionsWrap = requireNode(options.sessionsWrap, "sessionsWrap");
    const sidebarEmptyHint = requireNode(options.sidebarEmptyHint, "sidebarEmptyHint");
    const el = requireFunction(options.el, "el");
    const iconSvg = requireFunction(options.iconSvg, "iconSvg");
    const sidebarRenderSignature = requireFunction(options.sidebarRenderSignature, "sidebarRenderSignature");
    const sessionDisplayName = requireFunction(options.sessionDisplayName, "sessionDisplayName");
    const sessionLaunchFailed = requireFunction(options.sessionLaunchFailed, "sessionLaunchFailed");
    const sessionLaunchPending = requireFunction(options.sessionLaunchPending, "sessionLaunchPending");
    const redactedLaunchErrorText = requireFunction(options.redactedLaunchErrorText, "redactedLaunchErrorText");
    const fmtRelativeAge = requireFunction(options.fmtRelativeAge, "fmtRelativeAge");
    const reasoningEffortMarker = requireFunction(options.reasoningEffortMarker, "reasoningEffortMarker");
    const sidebarModelText = requireFunction(options.sidebarModelText, "sidebarModelText");
    const baseName = requireFunction(options.baseName, "baseName");
    const sessionIsFast = requireFunction(options.sessionIsFast, "sessionIsFast");
    const agentBackendLogoPath = requireFunction(options.agentBackendLogoPath, "agentBackendLogoPath");
    const agentBackendDisplayName = requireFunction(options.agentBackendDisplayName, "agentBackendDisplayName");
    const sessionAgentBackend = requireFunction(options.sessionAgentBackend, "sessionAgentBackend");
    const sessionLaunchIcon = requireFunction(options.sessionLaunchIcon, "sessionLaunchIcon");
    const sessionLaunchLabel = requireFunction(options.sessionLaunchLabel, "sessionLaunchLabel");
    const confirmAction = requireFunction(options.confirmAction, "confirmAction");
    const api = requireFunction(options.api, "api");
    const clearDeletedSessionClientState = requireFunction(options.clearDeletedSessionClientState, "clearDeletedSessionClientState");
    const refreshSessions = requireFunction(options.refreshSessions, "refreshSessions");
    const setToast = requireFunction(options.setToast, "setToast");
    const openEditSession = requireFunction(options.openEditSession, "openEditSession");
    const duplicateSession = requireFunction(options.duplicateSession, "duplicateSession");
    const selectSession = requireFunction(options.selectSession, "selectSession");
    const setSidebarOpen = requireFunction(options.setSidebarOpen, "setSidebarOpen");
    const now = typeof options.now === "function" ? options.now : () => Date.now();
    const performanceNow = typeof options.performanceNow === "function" ? options.performanceNow : () => performance.now();
    const consoleError = typeof options.consoleError === "function" ? options.consoleError : () => {};

    let openSwipeContent = null;
    let openSwipeSessionId = null;
    let openSwipeTargetX = 0;
    let refreshDeferred = false;
    let lastRenderSignature = "";

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

    function closeOpenSwipe() {
      if (!openSwipeContent) return;
      openSwipeContent.style.transform = "translate3d(0px, 0, 0)";
      openSwipeContent.dataset.swipeX = "0";
      openSwipeContent = null;
      openSwipeSessionId = null;
      openSwipeTargetX = 0;
      if (refreshDeferred) void refreshSessions().catch((error) => consoleError("refreshSessions failed after swipe close", error));
    }

    function bindSwipe(content, sessionId, { leftMax, rightMax }) {
      let startX = null;
      let startY = 0;
      let startSwipe = 0;
      let lastMoveTs = 0;
      let lastMoveX = 0;
      let swipeVelocity = 0;
      let dragging = false;
      content.addEventListener("pointerdown", (event) => {
        if (event.pointerType === "mouse" && event.button !== 0) return;
        startX = event.clientX;
        startY = event.clientY;
        startSwipe = Number(content.dataset.swipeX || 0);
        lastMoveTs = performanceNow();
        lastMoveX = event.clientX;
        swipeVelocity = 0;
        dragging = false;
        if (openSwipeContent && openSwipeContent !== content) closeOpenSwipe();
        try { content.setPointerCapture(event.pointerId); } catch (_) {}
      });
      content.addEventListener("pointermove", (event) => {
        if (startX === null) return;
        const dx = event.clientX - startX;
        const dy = event.clientY - startY;
        const moveTs = performanceNow();
        const dt = Math.max(moveTs - lastMoveTs, 1);
        swipeVelocity = ((event.clientX - lastMoveX) / dt) * 1000;
        lastMoveTs = moveTs;
        lastMoveX = event.clientX;
        if (!dragging) {
          if (Math.abs(dx) < 4 || Math.abs(dx) < Math.abs(dy) * 0.7) return;
          dragging = true;
          content.style.transition = "none";
        }
        event.preventDefault();
        const x = Math.min(leftMax, Math.max(-rightMax, startSwipe + dx));
        content.style.transform = `translate3d(${x}px, 0, 0)`;
        content.dataset.swipeX = String(x);
      });
      const finishSwipe = (event) => {
        if (startX === null) return;
        try { if (event && event.pointerId != null) content.releasePointerCapture(event.pointerId); } catch (_) {}
        startX = null;
        if (!dragging) return;
        dragging = false;
        content.style.transition = "";
        const x = Number(content.dataset.swipeX || 0);
        const commitLeft = leftMax > 0 && (x > leftMax * 0.28 || swipeVelocity > 420);
        const commitRight = rightMax > 0 && (-x > rightMax * 0.28 || swipeVelocity < -420);
        const target = commitLeft ? leftMax : commitRight ? -rightMax : 0;
        content.style.transform = `translate3d(${target}px, 0, 0)`;
        content.dataset.swipeX = String(target);
        if (target !== 0) {
          openSwipeContent = content;
          openSwipeSessionId = sessionId;
          openSwipeTargetX = target;
        } else if (openSwipeContent === content) {
          openSwipeContent = null;
          openSwipeSessionId = null;
          openSwipeTargetX = 0;
        }
      };
      content.addEventListener("pointerup", finishSwipe);
      content.addEventListener("pointercancel", finishSwipe);
    }

    function render(entries, { selectedId = "", swipeActions = false } = {}) {
      const sidebarEntries = Array.isArray(entries) ? entries : [];
      if (swipeActions && openSwipeSessionId && sessionsWrap.childElementCount > 0) {
        refreshDeferred = true;
        return false;
      }
      const applyingDeferredRefresh = refreshDeferred && !openSwipeSessionId;
      const signature = sidebarRenderSignature(sidebarEntries, { selectedId, swipeActions });
      const unchanged = !applyingDeferredRefresh && sessionsWrap.childElementCount > 0 && signature === lastRenderSignature;
      if (applyingDeferredRefresh) refreshDeferred = false;
      if (!unchanged) {
        sessionsWrap.innerHTML = "";
        openSwipeContent = null;
        lastRenderSignature = signature;
        for (const entry of sidebarEntries) {
          if (entry.type === "header") {
            sessionsWrap.appendChild(renderSessionGroupHeader(entry));
            continue;
          }
          const session = entry.session;
          const sessionId = session.session_id;
          const card = el("div", { class: `session${selectedId === sessionId ? " active" : ""}`, "data-session-id": sessionId, role: "link", tabindex: "0" });
          const title = sessionDisplayName(session);
          const badges = [];
          const launchFailed = sessionLaunchFailed(session);
          const launchPending = sessionLaunchPending(session);
          const launchRow = launchFailed || launchPending;
          if (launchFailed) badges.push(el("span", { class: "badge launchFailed", text: "failed", title: redactedLaunchErrorText(session.launch_error) || "Session launch failed" }));
          if (launchPending) badges.push(el("span", { class: "badge launchPending", text: "starting", title: "Session is still starting" }));
          if (session.unattended_enabled) badges.push(el("span", { class: "badge unattended", text: "unattended", title: "Unattended mode enabled" }));
          if (session.queue_len) badges.push(el("span", { class: "badge queue", text: `queue ${session.queue_len}` }));

          const updatedTs = typeof session.updated_ts === "number" && Number.isFinite(session.updated_ts) ? session.updated_ts : session.start_ts;
          const ageSeconds = updatedTs ? Math.max(0, now() / 1000 - updatedTs) : 0;
          const effortText = String(session.reasoning_effort || "").trim().toLowerCase();
          const effortMarker = reasoningEffortMarker(effortText);
          const stateText = launchPending ? "starting" : fmtRelativeAge(ageSeconds);
          const modelText = sidebarModelText(session);
          const branchText = typeof session.git_branch === "string" ? session.git_branch.trim() : "";

          const doDelete = async (event) => {
            if (event) { event.preventDefault(); event.stopPropagation(); }
            closeOpenSwipe();
            const confirmed = await confirmAction({
              title: launchRow ? "Dismiss launch record?" : "Delete session?",
              message: launchRow ? "Dismiss this launch record?" : "Delete this session?",
              confirmText: launchRow ? "Dismiss" : "Delete",
              cancelText: "Cancel",
              destructive: true,
            });
            if (!confirmed) return;
            try {
              await api(`/api/sessions/${sessionId}/delete`, { method: "POST", body: {} });
              clearDeletedSessionClientState(sessionId);
              if (launchRow && card.parentNode) card.remove();
              await refreshSessions();
            } catch (error) {
              setToast(`delete error: ${error.message}`);
            }
          };
          const renameBtn = el("button", { class: "icon-btn", title: "Edit conversation", "aria-label": "Edit conversation", type: "button", html: iconSvg("edit") });
          renameBtn.onclick = (event) => { event.preventDefault(); event.stopPropagation(); closeOpenSwipe(); openEditSession(sessionId); };
          const duplicateBtn = el("button", { class: "icon-btn", title: "Duplicate session", "aria-label": "Duplicate session", type: "button", html: iconSvg("duplicate") });
          duplicateBtn.onclick = async (event) => {
            event.preventDefault();
            event.stopPropagation();
            closeOpenSwipe();
            if (launchRow) {
              if (launchFailed) void selectSession(sessionId);
              setToast(launchFailed ? "review failed launch before retrying" : "session still starting");
              return;
            }
            await duplicateSession(session);
          };
          const deleteBtn = el("button", { class: "icon-btn danger sessionDel", title: launchRow ? "Dismiss launch record" : "Delete session", "aria-label": launchRow ? "Dismiss launch record" : "Delete session", type: "button", html: iconSvg("trash") });
          deleteBtn.onclick = (event) => void doDelete(event);
          const stateDot = el("span", { class: `stateDot${launchPending ? " pending" : session.snoozed || session.blocked ? " suppressed" : session.busy ? " busy" : " idle"}` });
          const titleRow = el("div", { class: "sessionTitleRow" }, [
            stateDot,
            el("div", { class: "titleLine", title: session.cwd || "" }, [
              el("span", { class: "titleText", text: title }),
              sessionIsFast(session) ? el("span", { class: "sessionFastIcon", html: iconSvg("lightning"), title: "Fast session" }) : null,
            ].filter(Boolean)),
          ]);
          const badgesWrap = el("div", { class: "sessionBadges" }, badges);
          const backend = sessionAgentBackend(session);
          const metaItems = [
            el("img", { class: "sessionBackendStatusIcon", src: agentBackendLogoPath(backend), alt: `${agentBackendDisplayName(backend)} logo`, width: "12", height: "12" }),
            el("span", { class: `ownerBadge ownerIconBadge ${session.transport === "tmux" ? "owner-tmux" : session.owned ? "owner-web" : "owner-terminal"}`, html: iconSvg(sessionLaunchIcon(session)), title: sessionLaunchLabel(session) }),
          ];
          if (effortMarker) metaItems.push(el("span", { class: `effortMark effort-${effortText}`, text: effortMarker, title: `reasoning effort ${effortText}` }));
          metaItems.push(el("span", { class: "metaText", text: [stateText, modelText, baseName(session.cwd), branchText].filter(Boolean).join(" | ") }));
          const meta = el("div", { class: "muted subLine sessionMetaLine" }, metaItems);
          if (launchFailed) meta.title = redactedLaunchErrorText(session.launch_error) || "Session launch failed";
          if (launchPending) meta.title = "Session is still starting";
          const editActions = launchRow ? [] : [renameBtn, duplicateBtn];

          if (swipeActions) {
            const content = el("div", { class: "sessionContent" }, [el("div", { class: "sessionInner" }, [el("div", { class: "row" }, [titleRow, badgesWrap]), meta])]);
            content.dataset.swipeX = "0";
            card.appendChild(el("div", { class: "sessionSwipe" }, [el("div", { class: "sessionActions left" }, [deleteBtn]), el("div", { class: "sessionActions right" }, editActions), content]));
            if (openSwipeSessionId === sessionId && openSwipeTargetX !== 0) {
              content.style.transform = `translate3d(${openSwipeTargetX}px, 0, 0)`;
              content.dataset.swipeX = String(openSwipeTargetX);
              openSwipeContent = content;
            }
            bindSwipe(content, sessionId, { leftMax: 72, rightMax: editActions.length ? 104 : 0 });
            card.onclick = () => {
              if (Math.abs(Number(content.dataset.swipeX || 0)) > 2) { closeOpenSwipe(); return; }
              setSidebarOpen(false);
              if (launchPending) { setToast("session still starting"); return; }
              void selectSession(sessionId);
            };
          } else {
            card.classList.add("desktop");
            card.appendChild(el("div", { class: "sessionInner sessionDesktopLayout" }, [
              el("div", { class: "sessionMain" }, [el("div", { class: "sessionTitleWithBadges" }, [titleRow, badgesWrap]), meta]),
              el("div", { class: "sessionActionsInline" }, [...editActions, deleteBtn]),
            ]));
            card.onclick = () => {
              if (launchPending) { setToast("session still starting"); return; }
              void selectSession(sessionId);
            };
          }
          sessionsWrap.appendChild(card);
        }
      }
      if (sessionsWrap.childElementCount === 0) {
        if (!sidebarEmptyHint.parentElement) sessionsWrap.appendChild(sidebarEmptyHint);
      } else if (sidebarEmptyHint.parentElement) sidebarEmptyHint.remove();
      const renderedIds = new Set(sidebarEntries.filter((entry) => entry && entry.type === "session" && entry.session).map((entry) => entry.session.session_id));
      if (openSwipeSessionId && !renderedIds.has(openSwipeSessionId)) {
        openSwipeSessionId = null;
        openSwipeTargetX = 0;
        openSwipeContent = null;
      }
      return true;
    }

    return Object.freeze({
      render,
      closeOpenSwipe,
      hasDeferredRefresh: () => refreshDeferred,
      dispose() {
        openSwipeContent = null;
        openSwipeSessionId = null;
        openSwipeTargetX = 0;
        refreshDeferred = false;
        lastRenderSignature = "";
      },
    });
  }

  window.CodoxearShell = Object.freeze({ createShellDOM, createSidebarController });
})();
