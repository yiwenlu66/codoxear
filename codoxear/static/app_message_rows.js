(() => {
  "use strict";

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`message row dependency missing: ${name}`);
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
    if (direction < 0) {
      for (let i = rows.length - 1; i >= 0; i -= 1) {
        if (rows[i].offsetTop < threshold) return { reason: "target", target: rows[i] };
      }
      return { reason: "first", target: null };
    }
    for (const row of rows) {
      if (row.offsetTop > threshold) return { reason: "target", target: row };
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

  window.CodoxearMessageRows = Object.freeze({
    makeRow,
    safeMakeRow,
    messageCopyButtonForRow,
    renderedMessageRows,
    loadedUserMessageRows,
    loadedCopyMessageRows,
    activeElementIsMessageCopyButton,
    rowSearchText,
    compareRowsInDomOrder,
    loadedUserJumpTarget,
    loadedCopyJumpTarget,
  });
})();
