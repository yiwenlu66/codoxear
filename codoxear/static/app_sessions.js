
  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`sessions controller dependency missing: ${name}`);
    return value;
  }

  function requireNode(value, name) {
    if (!value || typeof value !== "object" || typeof value.appendChild !== "function")
      throw new TypeError(`sessions controller dependency missing: ${name}`);
    return value;
  }

  function createSessionsController(options = {}) {
    if (!options || typeof options !== "object") throw new TypeError("sessions controller dependency missing: options");
    const sessionsWrap = requireNode(options.sessionsWrap, "sessionsWrap");
    const sidebarEmptyHint = requireNode(options.sidebarEmptyHint, "sidebarEmptyHint");
    const el = requireFunction(options.el, "el");
    const iconSvg = requireFunction(options.iconSvg, "iconSvg");
    const sidebarRenderSignature = requireFunction(options.sidebarRenderSignature, "sidebarRenderSignature");
    const sidebarSessionEntries = requireFunction(options.sidebarSessionEntries, "sidebarSessionEntries");
    const sessionDisplayName = requireFunction(options.sessionDisplayName, "sessionDisplayName");
    const sessionLaunchFailed = requireFunction(options.sessionLaunchFailed, "sessionLaunchFailed");
    const sessionLaunchPending = requireFunction(options.sessionLaunchPending, "sessionLaunchPending");
    const redactedLaunchErrorText = requireFunction(options.redactedLaunchErrorText, "redactedLaunchErrorText");
    const fmtRelativeAge = requireFunction(options.fmtRelativeAge, "fmtRelativeAge");
    const sidebarEffortCode = requireFunction(options.sidebarEffortCode, "sidebarEffortCode");
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
          const lost = !!session.lost;
          const launchRow = launchFailed || launchPending;
          if (launchFailed) badges.push(el("span", { class: "badge launchFailed", text: lost ? "lost" : "failed", title: lost ? "Broker stopped; this session can no longer accept control." : redactedLaunchErrorText(session.launch_error) || "Session launch failed" }));
          if (launchPending) badges.push(el("span", { class: "badge launchPending", text: "starting", title: "Session is still starting" }));
          if (session.unattended_enabled) badges.push(el("span", { class: "badge unattended", text: "unattended", title: "Unattended mode enabled" }));
          if (session.queue_len) badges.push(el("span", { class: "badge queue", text: `queue ${session.queue_len}` }));
          if (Number(session.unread_count) > 0) badges.push(el("span", { class: "badge unread", text: `unread ${session.unread_count}`, title: "Unread messages" }));

          const updatedTs = typeof session.updated_ts === "number" && Number.isFinite(session.updated_ts) ? session.updated_ts : session.start_ts;
          const ageSeconds = updatedTs ? Math.max(0, now() / 1000 - updatedTs) : 0;
          const effortText = String(session.reasoning_effort || "").trim().toLowerCase();
          const effortCode = sidebarEffortCode(effortText, session.agent_backend);
          const stateText = lost ? "lost" : launchPending ? "starting" : fmtRelativeAge(ageSeconds);
          const modelText = sidebarModelText(session);
          const branchText = typeof session.git_branch === "string" ? session.git_branch.trim() : "";

          const doDelete = async (event) => {
            if (event) { event.preventDefault(); event.stopPropagation(); }
            closeOpenSwipe();
            const confirmed = await confirmAction({
              title: lost ? "Dismiss lost session?" : launchRow ? "Dismiss launch record?" : "Delete session?",
              message: lost ? "Dismiss this lost session?" : launchRow ? "Dismiss this launch record?" : "Delete this session?",
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
              setToast(lost ? "broker stopped" : launchFailed ? "review failed launch before retrying" : "session still starting");
              return;
            }
            await duplicateSession(session);
          };
          const deleteBtn = el("button", { class: "icon-btn danger sessionDel", title: lost ? "Dismiss lost session" : launchRow ? "Dismiss launch record" : "Delete session", "aria-label": lost ? "Dismiss lost session" : launchRow ? "Dismiss launch record" : "Delete session", type: "button", html: iconSvg("trash") });
          deleteBtn.onclick = (event) => void doDelete(event);
          const stateDot = el("span", { class: `stateDot${launchPending ? " pending" : session.busy ? " busy" : " idle"}` });
          const subagentsRunning = Number(session.subagents_running);
          const subagentMarker = Number.isFinite(subagentsRunning) && subagentsRunning > 0
            ? el("span", { class: "muted subagentMarker", text: `▸${Math.floor(subagentsRunning)}` })
            : null;
          const titleRow = el("div", { class: "sessionTitleRow" }, [
            stateDot,
            subagentMarker,
            el("div", { class: "titleLine", title: session.cwd || "" }, [
              el("span", { class: "titleText", text: title }),
              sessionIsFast(session) ? el("span", { class: "sessionFastIcon", html: iconSvg("lightning"), title: "Fast session" }) : null,
            ].filter(Boolean)),
          ].filter(Boolean));
          const badgesWrap = el("div", { class: "sessionBadges" }, badges);
          const backend = sessionAgentBackend(session);
          const metaItems = [
            el("img", { class: "sessionBackendStatusIcon", src: agentBackendLogoPath(backend), alt: `${agentBackendDisplayName(backend)} logo`, width: "12", height: "12" }),
            el("span", { class: `ownerBadge ownerIconBadge ${session.transport === "tmux" ? "owner-tmux" : session.owned ? "owner-web" : "owner-terminal"}`, html: iconSvg(sessionLaunchIcon(session)), title: sessionLaunchLabel(session) }),
          ];
          const metadataSegments = [
            el("span", { class: "sidebarMetaLabel", text: stateText }),
            modelText
              ? el("span", { class: "sidebarMetaData" }, [
                  el("span", { text: modelText }),
                  effortCode ? el("span", { text: ` ·${effortCode}` }) : null,
                ].filter(Boolean))
              : effortCode
                ? el("span", { class: "sidebarMetaData", text: `·${effortCode}` })
                : null,
            el("span", { class: "sidebarMetaLabel", text: baseName(session.cwd) }),
            branchText ? el("span", { class: "sidebarMetaLabel", text: branchText }) : null,
          ].filter(Boolean);
          const metaText = el("span", { class: "metaText" });
          metadataSegments.forEach((segment, index) => {
            if (index) metaText.appendChild(el("span", { class: "sidebarMetaSeparator", text: " | " }));
            metaText.appendChild(segment);
          });
          metaItems.push(metaText);
          const meta = el("div", { class: "muted subLine sessionMetaLine" }, metaItems);
          if (lost) meta.title = "Broker stopped; stale control sidecar was removed.";
          else if (launchFailed) meta.title = redactedLaunchErrorText(session.launch_error) || "Session launch failed";
          if (launchPending) meta.title = "Session is still starting";
          const editActions = launchRow ? [] : [renameBtn, duplicateBtn];

          if (swipeActions) {
            // TOUCH BRANCH: swipe-revealed left/right action groups
            const swipeHint = el("span", { class: "swipeHint", "aria-hidden": "true", text: "‹" });
            const content = el("div", { class: "sessionContent" }, [
              el("div", { class: "sessionInner" }, [el("div", { class: "row" }, [titleRow, badgesWrap]), meta]),
              swipeHint,
            ]);
            card.appendChild(el("div", { class: "sessionSwipe" }, [
              el("div", { class: "sessionActions left" }, [deleteBtn]),
              el("div", { class: "sessionActions right" }, editActions),
              content,
            ]));
            content.dataset.swipeX = "0";
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
            // DESKTOP BRANCH: single inline hover-revealed action group
            // (LOCKED: do not unify with touch swipe DOM per AGENTS.md)
            const inlineActions = el("div", { class: "sessionActionsInline" }, [...editActions, deleteBtn]);
            const content = el("div", { class: "sessionContent" }, [
              el("div", { class: "sessionInner" }, [el("div", { class: "row" }, [titleRow, badgesWrap]), meta]),
            ]);
            card.appendChild(content);
            card.appendChild(inlineActions);
            card.classList.add("desktop");
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
      // The deploy browser smoke check uses this post-render marker. It proves
      // the first session-list projection completed even when there are no
      // session cards to count.
      sessionsWrap.dataset.codoxearSessionsRendered = "true";
      return true;
    }

    function renderSessions(sessions, options = {}) {
      return render(sidebarSessionEntries(sessions), options);
    }

    return Object.freeze({
      render,
      renderSessions,
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

export { createSessionsController };
