(() => {
  "use strict";

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`message row dependency missing: ${name}`);
    return value;
  }

  function requireRoot(value, name) {
    if (!value || typeof value.querySelectorAll !== "function") throw new TypeError(`message row dependency missing: ${name}`);
    return value;
  }

  function makeRow(ev, { ts, pending }, deps) {
    const el = requireFunction(deps && deps.el, "el");
    const chatMarkdownHtmlCached = requireFunction(deps && deps.chatMarkdownHtmlCached, "chatMarkdownHtmlCached");
    const upgradeCandidateFileRefs = requireFunction(deps && deps.upgradeCandidateFileRefs, "upgradeCandidateFileRefs");
    const time24 = requireFunction(deps && deps.time24, "time24");
    const iconSvg = requireFunction(deps && deps.iconSvg, "iconSvg");
    const copyToClipboard = requireFunction(deps && deps.copyToClipboard, "copyToClipboard");
    const setToast = requireFunction(deps && deps.setToast, "setToast");
    const chatAssistantDedupeKey = requireFunction(deps && deps.chatAssistantDedupeKey, "chatAssistantDedupeKey");
    const setTimeoutFn = requireFunction(deps && deps.setTimeout, "setTimeout");
    const selectedSessionId = deps ? deps.selectedSessionId : undefined;

    const role = ev.role === "user" ? "user" : "assistant";
    const row = el("div", { class: `msg-row ${role}` });
    row.dataset.role = role;
    if (typeof ts === "number" && Number.isFinite(ts)) row.dataset.ts = String(ts);
    if (!pending && typeof ev.history_cursor === "string" && ev.history_cursor) row.dataset.historyCursor = ev.history_cursor;
    const messageClass = typeof ev.message_class === "string" ? ev.message_class : "";
    const assistantDedupeKey = role === "assistant" ? chatAssistantDedupeKey(ev) : "";
    if (assistantDedupeKey) row.dataset.assistantDedupeKey = assistantDedupeKey;

    const bubble = el("div", { class: role === "user" ? "msg user" : "msg assistant" });
    if (role === "assistant" && (messageClass === "error" || messageClass === "warning")) {
      bubble.classList.add(messageClass);
    }
    const md = el("div", { class: "md", html: chatMarkdownHtmlCached(ev.text, selectedSessionId) });
    bubble.appendChild(md);
    void upgradeCandidateFileRefs(md);
    if (typeof ts === "number" && Number.isFinite(ts)) bubble.appendChild(el("div", { class: "ts", text: time24(new Date(ts * 1000)) }));

    if (pending) {
      bubble.style.opacity = "0.72";
      bubble.setAttribute("data-pending", "1");
      if (ev.localId) bubble.setAttribute("data-local-id", String(ev.localId));
    }

    const shell = el("div", { class: `msg-shell ${role}` });
    shell.appendChild(bubble);
    if (typeof ev.text === "string" && ev.text.length) {
      const copyBtn = el("button", {
        class: "icon-btn msg-copy-btn",
        type: "button",
        title: "Copy raw markdown",
        "aria-label": "Copy raw markdown",
        tabindex: "-1",
        disabled: "true",
        "aria-hidden": "true",
        html: iconSvg("copy"),
      });
      copyBtn.onclick = async (e) => {
        e.preventDefault();
        e.stopPropagation();
        try {
          await copyToClipboard(ev.text);
          copyBtn.classList.add("copied");
          setTimeoutFn(() => copyBtn.classList.remove("copied"), 1200);
          setToast("Copied markdown");
        } catch (err) {
          setToast(`copy failed: ${err && err.message ? err.message : "unknown error"}`);
        }
      };
      shell.appendChild(copyBtn);
    }

    row.appendChild(shell);
    return { row, bubble };
  }

  function safeMakeRow(ev, opts, deps) {
    const el = requireFunction(deps && deps.el, "el");
    const time24 = requireFunction(deps && deps.time24, "time24");
    const chatAssistantDedupeKey = requireFunction(deps && deps.chatAssistantDedupeKey, "chatAssistantDedupeKey");
    const consoleError = requireFunction(deps && deps.consoleError, "consoleError");
    try {
      return makeRow(ev, opts, deps);
    } catch (err) {
      consoleError("makeRow failed", err);
      const role = ev && ev.role === "user" ? "user" : "assistant";
      const ts = opts && typeof opts.ts === "number" && Number.isFinite(opts.ts) ? opts.ts : null;
      const pending = Boolean(opts && opts.pending);
      const row = el("div", { class: `msg-row ${role}` });
      row.dataset.role = role;
      if (ts !== null) row.dataset.ts = String(ts);
      if (!pending && typeof ev?.history_cursor === "string" && ev.history_cursor) row.dataset.historyCursor = ev.history_cursor;
      const messageClass = typeof ev?.message_class === "string" ? ev.message_class : "";
      const assistantDedupeKey = role === "assistant" ? chatAssistantDedupeKey(ev) : "";
      if (assistantDedupeKey) row.dataset.assistantDedupeKey = assistantDedupeKey;
      const bubble = el("div", { class: role === "user" ? "msg user" : "msg assistant" });
      if (role === "assistant" && (messageClass === "error" || messageClass === "warning")) {
        bubble.classList.add(messageClass);
      }
      const md = el("div", { class: "md" });
      md.textContent = typeof ev?.text === "string" ? ev.text : String(ev?.text ?? "");
      bubble.appendChild(md);
      if (ts !== null) bubble.appendChild(el("div", { class: "ts", text: time24(new Date(ts * 1000)) }));
      if (pending) {
        bubble.style.opacity = "0.72";
        bubble.setAttribute("data-pending", "1");
        if (ev && ev.localId) bubble.setAttribute("data-local-id", String(ev.localId));
      }
      const shell = el("div", { class: `msg-shell ${role}` });
      shell.appendChild(bubble);
      row.appendChild(shell);
      return { row, bubble };
    }
  }

  function messageCopyButtonForRow(row) {
    return row && typeof row.querySelector === "function" ? row.querySelector(".msg-copy-btn") : null;
  }

  function renderedMessageRows(chatInner) {
    return Array.from(chatInner.querySelectorAll(".msg-row")).filter((row) => !row.classList.contains("typing-row") && !row.classList.contains("recovery-panel-row"));
  }

  function loadedUserMessageRows(chatInner) {
    return renderedMessageRows(chatInner).filter((row) => row.dataset.role === "user");
  }

  function loadedCopyMessageRows(chatInner) {
    return renderedMessageRows(chatInner).filter((row) => messageCopyButtonForRow(row));
  }

  function activeElementIsMessageCopyButton(documentLike) {
    return Boolean(documentLike.activeElement && documentLike.activeElement.classList && documentLike.activeElement.classList.contains("msg-copy-btn"));
  }

  function rowSearchText(row) {
    const md = row ? row.querySelector(".md") : null;
    return String((md || row || {}).textContent || "");
  }

  function compareRowsInDomOrder(a, b, nodeLike) {
    if (a === b) return 0;
    if (!a || !b || !a.compareDocumentPosition) return 0;
    const pos = a.compareDocumentPosition(b);
    const nodeConstants = nodeLike || window.Node;
    if (pos & nodeConstants.DOCUMENT_POSITION_FOLLOWING) return -1;
    if (pos & nodeConstants.DOCUMENT_POSITION_PRECEDING) return 1;
    return 0;
  }

  function loadedUserJumpTarget(rows, direction, threshold) {
    if (!rows.length) return { reason: "none", target: null };
    // Use getBoundingClientRect for coordinate-system-independent comparison.
    // offsetTop is relative to offsetParent, which may not match the scroll
    // container's coordinate space, causing repeated clicks to land on the
    // same message instead of advancing.
    const chatEl = (rows[0].closest("#chat") || rows[0].closest(".chatWrap") || null);
    const chatTop = chatEl ? chatEl.getBoundingClientRect().top + 1 : threshold;
    if (direction < 0) {
      for (let i = rows.length - 1; i >= 0; i -= 1) {
        if (rows[i].getBoundingClientRect().top < chatTop) return { reason: "target", target: rows[i] };
      }
      return { reason: "first", target: null };
    }
    for (const row of rows) {
      if (row.getBoundingClientRect().top > chatTop + 2) return { reason: "target", target: row };
    }
    return { reason: "last", target: null };
  }

  function loadedCopyJumpTarget(rows, activeRow, direction, threshold) {
    if (!rows.length) return { reason: "none", target: null };
    let idx = activeRow && activeRow.isConnected ? rows.indexOf(activeRow) : -1;
    if (idx < 0) {
      if (direction < 0) {
        for (let i = rows.length - 1; i >= 0; i -= 1) {
          if (rows[i].offsetTop < threshold) {
            idx = i + 1;
            break;
          }
        }
      } else {
        for (let i = 0; i < rows.length; i += 1) {
          if (rows[i].offsetTop > threshold) {
            idx = i - 1;
            break;
          }
        }
      }
      if (idx < 0) idx = direction < 0 ? rows.length : -1;
    }
    const nextIndex = idx + (direction < 0 ? -1 : 1);
    if (nextIndex < 0) return { reason: "first", target: null };
    if (nextIndex >= rows.length) return { reason: "last", target: null };
    return { reason: "target", target: rows[nextIndex] };
  }

  function createMessageCopyNavigationRuntime(options = {}) {
    const root = requireRoot(options.root, "root");
    let activeRow = null;

    function copyButtons() {
      return Array.from(root.querySelectorAll(".msg-copy-btn"));
    }

    function activeRowSnapshot() {
      return activeRow && activeRow.isConnected ? activeRow : null;
    }

    function syncTabStops(rows = renderedMessageRows(root)) {
      const buttons = copyButtons();
      let activeBtn = messageCopyButtonForRow(activeRowSnapshot());
      if (!activeBtn || !activeBtn.isConnected) {
        activeBtn = null;
        for (let i = rows.length - 1; i >= 0; i -= 1) {
          const candidate = messageCopyButtonForRow(rows[i]);
          if (candidate) {
            activeBtn = candidate;
            break;
          }
        }
      }
      activeRow = activeBtn && typeof activeBtn.closest === "function" ? activeBtn.closest(".msg-row") : null;
      for (const btn of buttons) {
        const active = btn === activeBtn;
        btn.tabIndex = active ? 0 : -1;
        btn.disabled = !active;
        if (active) btn.removeAttribute("aria-hidden");
        else btn.setAttribute("aria-hidden", "true");
      }
      return activeRowSnapshot();
    }

    function setActiveRow(row, { focusCopy = false } = {}) {
      activeRow = row && row.isConnected && messageCopyButtonForRow(row) ? row : null;
      const current = syncTabStops();
      if (focusCopy && current) {
        const btn = messageCopyButtonForRow(current);
        if (btn && btn.tabIndex >= 0 && typeof btn.focus === "function") btn.focus({ preventScroll: true });
      }
      return current;
    }

    function jumpTarget(rows, direction, threshold) {
      return loadedCopyJumpTarget(rows, activeRowSnapshot(), direction, threshold);
    }

    function reset() {
      activeRow = null;
      return syncTabStops();
    }

    return Object.freeze({
      activeRow: activeRowSnapshot,
      jumpTarget,
      reset,
      setActiveRow,
      syncTabStops,
    });
  }

  function clearChatSearchMarks(rows) {
    for (const row of rows) row.classList.remove("chat-search-hit", "chat-search-current");
  }

  function applyChatSearchMarks(matches, currentRow) {
    for (const match of matches) match.classList.add("chat-search-hit");
    if (currentRow) currentRow.classList.add("chat-search-current");
  }

  function oldestRenderedHistoryCursor(rows) {
    for (const row of rows) {
      const cursor = typeof row.dataset.historyCursor === "string" ? row.dataset.historyCursor : "";
      if (cursor) return cursor;
    }
    return null;
  }

  function firstVisibleMessageRow(rows, viewportTop) {
    for (const row of rows) {
      if ((row.offsetTop + row.offsetHeight) > viewportTop) return row;
    }
    return rows.length ? rows[rows.length - 1] : null;
  }

  function trimRenderedRowTargets(rows, fromTop, maxRows, defaultMaxRows) {
    const allowedRows = Number.isFinite(Number(maxRows))
      ? Math.max(1, Math.floor(Number(maxRows)))
      : defaultMaxRows;
    if (rows.length <= allowedRows) return [];
    const extra = rows.length - allowedRows;
    return fromTop ? rows.slice(0, extra) : rows.slice(rows.length - extra);
  }

  function trimRowsBeforeViewportTargets(rows, maxRows, defaultMaxRows, viewportTop) {
    const allowedRows = Number.isFinite(Number(maxRows))
      ? Math.max(defaultMaxRows, Math.floor(Number(maxRows)))
      : defaultMaxRows;
    if (rows.length <= allowedRows) return [];
    const extra = rows.length - allowedRows;
    let firstVisible = 0;
    while (firstVisible < rows.length) {
      const row = rows[firstVisible];
      if ((row.offsetTop + row.offsetHeight) > viewportTop) break;
      firstVisible += 1;
    }
    const removable = Math.min(extra, firstVisible);
    if (removable <= 0) return [];
    return rows.slice(0, removable);
  }

  window.CodoxearMessageRows = Object.freeze({
    makeRow,
    safeMakeRow,
    messageCopyButtonForRow,
    renderedMessageRows,
    loadedUserMessageRows,
    loadedCopyMessageRows,
    activeElementIsMessageCopyButton,
    createMessageCopyNavigationRuntime,
    rowSearchText,
    compareRowsInDomOrder,
    loadedUserJumpTarget,
    loadedCopyJumpTarget,
    clearChatSearchMarks,
    applyChatSearchMarks,
    oldestRenderedHistoryCursor,
    firstVisibleMessageRow,
    trimRenderedRowTargets,
    trimRowsBeforeViewportTargets,
  });
})();
