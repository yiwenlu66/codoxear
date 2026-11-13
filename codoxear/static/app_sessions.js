
import * as CodoxearSidebarDrag from "./app_sidebar_drag.js";

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
    const sessionState = options.sessionState;
    if (!sessionState || typeof sessionState.get !== "function" || typeof sessionState.subscribe !== "function") {
      throw new TypeError("sessions controller dependency missing: sessionState");
    }
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

    // --- Drag: reorder inside Now, drop on Later to snooze indefinitely, drop
    // on Now to wake. One gesture session at a time. Geometry and drop decisions
    // are pure functions in app_sidebar_drag.js; the server owns order and
    // computes the dragged row's offset, so a drop resolves to ids only.
    let draggingSessionId = null;
    let dragSession = null;
    let suppressClickUntil = 0;

    function dragBlocks() {
      const blocks = [];
      let current = null;
      for (const node of sessionsWrap.children) {
        const classes = node.classList;
        if (!classes) continue;
        if (classes.contains("sessionGroupHeader")) {
          current = { group: (node.dataset && node.dataset.sessionGroup) || "", header: node, cards: [] };
          blocks.push(current);
          continue;
        }
        if (classes.contains("session") && node.dataset && node.dataset.sessionId) {
          if (!current) { current = { group: "now", header: null, cards: [] }; blocks.push(current); }
          const rect = node.getBoundingClientRect();
          current.cards.push({ id: node.dataset.sessionId, el: node, top: rect.top, height: rect.height });
        }
      }
      return blocks;
    }

    function dragGroupOf(blocks, sessionId) {
      for (const block of blocks) {
        if (block.cards.some((card) => card.id === sessionId)) return block.group;
      }
      return "";
    }

    function dragBlockSpan(blocks, index, container) {
      const block = blocks[index];
      const headerRect = block.header ? block.header.getBoundingClientRect() : null;
      const firstTop = block.cards.length ? block.cards[0].top : container.bottom;
      const start = headerRect ? headerRect.top : firstTop;
      const next = blocks[index + 1];
      let end = container.bottom;
      if (next) {
        const nextHeader = next.header ? next.header.getBoundingClientRect() : null;
        end = nextHeader ? nextHeader.top : (next.cards.length ? next.cards[0].top : container.bottom);
      }
      return { start, end };
    }

    function dragTargetAt(blocks, pointerY, container) {
      if (pointerY < container.top || pointerY > container.bottom) return { group: "", block: null };
      for (let index = 0; index < blocks.length; index += 1) {
        const span = dragBlockSpan(blocks, index, container);
        if (pointerY >= span.start && pointerY < span.end) return { group: blocks[index].group, block: blocks[index] };
      }
      // Past the whole list: below it is the Later zone, above it the Now zone.
      // That is what keeps each cross-block target reachable when the opposite
      // block has no rows yet.
      const lastSpan = blocks.length ? dragBlockSpan(blocks, blocks.length - 1, container) : null;
      if (lastSpan && pointerY >= lastSpan.end) return { group: "later", block: null };
      return { group: "now", block: null };
    }

    function dragEnsureHeader(group) {
      if (!dragSession || !group) return null;
      for (const node of sessionsWrap.children) {
        if (node.classList && node.classList.contains("sessionGroupHeader") && node.dataset.sessionGroup === group) return node;
      }
      const header = renderSessionGroupHeader({ key: group, label: group === "later" ? "Later" : "Now", count: 0 });
      header.dataset.ephemeral = "true";
      if (group === "later") sessionsWrap.appendChild(header);
      else sessionsWrap.insertBefore(header, sessionsWrap.firstElementChild);
      dragSession.ephemeral.push(header);
      return header;
    }

    function dragSetDropState(group, kind) {
      for (const node of sessionsWrap.children) {
        if (!node.classList || !node.classList.contains("sessionGroupHeader")) continue;
        const nodeGroup = node.dataset.sessionGroup || "";
        const state = nodeGroup === group && (kind === "snooze" || kind === "wake" || kind === "refuse") ? kind : "";
        if (state) node.dataset.drop = state;
        else delete node.dataset.drop;
      }
    }

    function dragShowLine(lineY, containerTop) {
      if (!dragSession) return;
      let line = dragSession.line;
      if (!line) {
        const doc = sessionsWrap.ownerDocument;
        line = doc.createElement("div");
        line.className = "dropIndicator";
        sessionsWrap.appendChild(line);
        dragSession.line = line;
      }
      line.hidden = false;
      line.style.top = `${Math.round(lineY - containerTop + sessionsWrap.scrollTop)}px`;
    }

    function dragHideLine() {
      if (dragSession && dragSession.line) dragSession.line.hidden = true;
    }

    function updateDrag(pointerX, pointerY) {
      if (!dragSession) return;
      dragSession.pointerX = pointerX;
      dragSession.pointerY = pointerY;
      const container = sessionsWrap.getBoundingClientRect();
      const blocks = dragBlocks();
      const target = dragTargetAt(blocks, pointerY, container);
      const block = target.block;
      const payload = CodoxearSidebarDrag.dragPayload({
        sourceGroup: dragSession.sourceGroup,
        targetGroup: target.group,
        overHeader: Boolean(block && (!block.cards.length || pointerY < block.cards[0].top)),
        sourceId: dragSession.sessionId,
        targetIds: block ? block.cards.map((card) => card.id) : [],
        cards: block ? block.cards : [],
        pointerY,
      });
      dragSession.payload = payload;
      if (payload.kind === "reorder" || payload.kind === "wake") {
        const others = block ? block.cards.filter((card) => card.id !== dragSession.sessionId) : [];
        const headerBottom = block && block.header ? block.header.getBoundingClientRect().bottom : container.top;
        dragShowLine(CodoxearSidebarDrag.insertionLineY(others, payload.index === null ? 0 : payload.index, headerBottom), container.top);
      } else {
        dragHideLine();
      }
      dragSetDropState(payload.kind === "snooze" || payload.kind === "wake" || payload.kind === "refuse" ? target.group : "", payload.kind);
      if (payload.kind === "snooze") dragEnsureHeader("later");
      if (payload.kind === "wake") dragEnsureHeader("now");
    }

    function dragTeardown() {
      const session = dragSession;
      draggingSessionId = null;
      dragSession = null;
      if (!session) return;
      if (session.raf) session.cancelRaf(session.raf);
      for (const [type, handler, options] of session.listeners) session.doc.removeEventListener(type, handler, options);
      if (session.line && session.line.parentNode) session.line.parentNode.removeChild(session.line);
      for (const node of session.ephemeral) if (node.parentNode) node.parentNode.removeChild(node);
      delete session.card.dataset.dragging;
      delete session.doc.body.dataset.dragging;
      dragSetDropState("", "none");
      if (refreshDeferred) void refreshSessions().catch((error) => consoleError("refreshSessions failed after drag", error));
    }

    async function commitDrop(payload, sessionId) {
      try {
        const needsNeighbours = payload.kind === "reorder" || payload.kind === "wake";
        if (needsNeighbours && (payload.above !== null || payload.below !== null)) {
          const result = await api(`/api/sessions/${sessionId}/reorder`, {
            method: "POST",
            body: { above: payload.above, below: payload.below },
          });
          if (result && result.applied === false) setToast("landed at the nearest free slot");
        }
        if (payload.kind === "snooze") {
          await api(`/api/sessions/${sessionId}/snooze`, { method: "POST", body: { indefinite: true } });
        } else if (payload.kind === "wake") {
          await api(`/api/sessions/${sessionId}/snooze`, { method: "POST", body: { indefinite: false } });
        }
      } catch (error) {
        setToast(`move failed: ${error && error.message ? error.message : "unknown error"}`);
      } finally {
        await refreshSessions().catch(() => {});
      }
    }

    function endDrag(cancelled) {
      const session = dragSession;
      if (!session) return;
      const payload = cancelled ? { kind: "none" } : (session.payload || { kind: "none" });
      const writes = payload.kind === "reorder" || payload.kind === "snooze" || payload.kind === "wake";
      dragTeardown();
      if (!writes) return;
      suppressClickUntil = performanceNow() + 500;
      void commitDrop(payload, session.sessionId);
    }

    function beginDrag(sessionId, card, pointerX, pointerY) {
      if (dragSession) return;
      const doc = sessionsWrap.ownerDocument;
      if (!doc) return;
      const sourceGroup = dragGroupOf(dragBlocks(), sessionId);
      if (sourceGroup !== "now" && sourceGroup !== "later") return;
      const view = doc.defaultView || {};
      const raf = typeof view.requestAnimationFrame === "function" ? view.requestAnimationFrame.bind(view) : null;
      const cancelRaf = typeof view.cancelAnimationFrame === "function" ? view.cancelAnimationFrame.bind(view) : () => {};
      const listen = (type, handler, options) => {
        doc.addEventListener(type, handler, options);
        return [type, handler, options];
      };
      closeOpenSwipe();
      draggingSessionId = sessionId;
      card.dataset.dragging = "true";
      doc.body.dataset.dragging = "true";
      const session = { sessionId, sourceGroup, card, pointerX, pointerY, payload: { kind: "none" }, line: null, ephemeral: [], raf: 0, listeners: [], doc, raf0: raf, cancelRaf };
      dragSession = session;
      session.listeners.push(listen("pointermove", (event) => { event.preventDefault(); updateDrag(event.clientX, event.clientY); }, true));
      session.listeners.push(listen("pointerup", () => endDrag(false), true));
      session.listeners.push(listen("pointercancel", () => endDrag(true), true));
      session.listeners.push(listen("keydown", (event) => { if (event.key === "Escape") { event.preventDefault(); endDrag(true); } }, true));
      // A pointer event's preventDefault does not stop scrolling; a touch scroll
      // that has not started yet is what this cancels.
      session.listeners.push(listen("touchmove", (event) => { if (event.cancelable) event.preventDefault(); }, { passive: false }));
      dragEnsureHeader(sourceGroup === "now" ? "later" : "now");
      if (raf) {
        const tick = () => {
          if (!dragSession) return;
          const rect = sessionsWrap.getBoundingClientRect();
          const delta = CodoxearSidebarDrag.autoScrollStep(rect.top, rect.bottom, dragSession.pointerY);
          if (delta) {
            const before = sessionsWrap.scrollTop;
            sessionsWrap.scrollTop = before + delta;
            if (sessionsWrap.scrollTop !== before) updateDrag(dragSession.pointerX, dragSession.pointerY);
          }
          dragSession.raf = raf(tick);
        };
        session.raf = raf(tick);
      }
      updateDrag(pointerX, pointerY);
    }

    // Activation: a still hold on touch/pen (the list scrolls, so movement first
    // is a scroll), a short vertical pull on mouse. Horizontal intent stays with
    // the swipe's own gate.
    function bindCardDrag(card, pressTarget, sessionId) {
      let pending = null;
      const clearPending = () => {
        if (pending && pending.timer) pending.clearTimer(pending.timer);
        pending = null;
      };
      pressTarget.addEventListener("pointerdown", (event) => {
        if (event.pointerType === "mouse" && event.button !== 0) return;
        if (dragSession) return;
        const doc = sessionsWrap.ownerDocument;
        const view = doc && doc.defaultView ? doc.defaultView : {};
        const setTimer = typeof view.setTimeout === "function" ? view.setTimeout.bind(view) : null;
        const clearTimer = typeof view.clearTimeout === "function" ? view.clearTimeout.bind(view) : () => {};
        clearPending();
        pending = { x: event.clientX, y: event.clientY, timer: null, setTimer, clearTimer, pointerType: event.pointerType };
        if (event.pointerType !== "mouse" && setTimer) {
          pending.timer = setTimer(() => {
            const start = pending;
            pending = null;
            if (start && !dragSession) beginDrag(sessionId, card, start.x, start.y);
          }, 380);
        }
      });
      pressTarget.addEventListener("pointermove", (event) => {
        if (dragSession) return;
        if (!pending) return;
        const dx = event.clientX - pending.x;
        const dy = event.clientY - pending.y;
        if (Math.abs(dx) > 6 && Math.abs(dx) > Math.abs(dy) * 0.7) { clearPending(); return; }
        if (pending.pointerType === "mouse") {
          if (Math.abs(dy) >= 6 && Math.abs(dy) > Math.abs(dx) * 1.2) {
            clearPending();
            beginDrag(sessionId, card, event.clientX, event.clientY);
          }
          return;
        }
        if (Math.abs(dx) > 8 || Math.abs(dy) > 8) clearPending();
      });
      pressTarget.addEventListener("pointerup", clearPending);
      pressTarget.addEventListener("pointercancel", clearPending);
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

    function applyActiveClass(card = null) {
      const cards = card
        ? [card]
        : typeof sessionsWrap.querySelectorAll === "function"
          ? sessionsWrap.querySelectorAll(".session[data-session-id]")
          : [];
      const selectedId = sessionState.get("selected");
      cards.forEach((candidate) => {
        const active = candidate.dataset.sessionId === selectedId;
        if (typeof candidate.classList.toggle === "function") candidate.classList.toggle("active", active);
        else if (active) candidate.classList.add("active");
        else if (typeof candidate.classList.remove === "function") candidate.classList.remove("active");
      });
    }

    function render(entries, { swipeActions = false } = {}) {
      const selectedId = sessionState.get("selected");
      const sidebarEntries = Array.isArray(entries) ? entries : [];
      // A live gesture owns its nodes: neither a swipe nor a drag may be wiped by
      // a poll. The flush happens when the gesture ends.
      if ((openSwipeSessionId || draggingSessionId) && sessionsWrap.childElementCount > 0) {
        refreshDeferred = true;
        return false;
      }
      const applyingDeferredRefresh = refreshDeferred && !openSwipeSessionId && !draggingSessionId;
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
          const card = el("div", { class: "session", "data-session-id": sessionId, role: "link", tabindex: "0" });
          applyActiveClass(card);
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
            bindCardDrag(card, content, sessionId);
            card.onclick = () => {
              if (performanceNow() < suppressClickUntil) return;
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
            bindCardDrag(card, card, sessionId);
            card.onclick = () => {
              if (performanceNow() < suppressClickUntil) return;
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

    const unsubscribeSelected = sessionState.subscribe("selected", () => applyActiveClass());

    return Object.freeze({
      render,
      renderSessions,
      closeOpenSwipe,
      hasDeferredRefresh: () => refreshDeferred,
      dispose() {
        unsubscribeSelected();
        endDrag(true);
        openSwipeContent = null;
        openSwipeSessionId = null;
        openSwipeTargetX = 0;
        refreshDeferred = false;
        lastRenderSignature = "";
      },
    });
  }

export { createSessionsController };
