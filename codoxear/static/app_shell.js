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
    const navRail = el("aside", { class: "navRail", "aria-label": "Primary navigation" }, [
      el("div", { class: "navRailBrand", title: "Codoxear", "aria-label": "Codoxear", html: iconSvg("terminal") }),
      el("nav", { class: "navRailNav", "aria-label": "Workspace" }, [
        el("div", { class: "navRailItem", "aria-current": "page", title: "Sessions" }, [
          el("span", { class: "navRailIcon", "aria-hidden": "true", html: iconSvg("terminal") }),
          el("span", { class: "navRailItemLabel", text: "Sessions" }),
        ]),
      ]),
    ]);
    const sidebar = el("div", { class: "sidebar" });
    const sessionsWrap = el("div", { class: "sessions", id: "sessions" });
    const sidebarEmptyHint = el("div", { class: "sidebarEmptyHint muted", text: "No sessions yet" });
    const sidebarFooter = el("footer", {}, [
      el("button", { id: "helpBtnSide", type: "button", title: "Help", "aria-label": "Help", "data-hint": "h", html: iconSvg("help") + "Help" }),
      el("button", { id: "settingsBtnSide", type: "button", title: "Settings", "aria-label": "Settings", "data-hint": "w", html: iconSvg("settings") + "Settings" }),
      el("button", { id: "logoutBtnSide", type: "button", title: "Log out", "aria-label": "Log out", "data-hint": "l", html: iconSvg("logout") + "Log out" }),
    ]);
    const main = el("div", { class: "main" });
    const chatWrap = el("div", { class: "chatWrap", id: "chatWrap" });
    const chatEmptyState = el("div", { class: "chatEmptyState", id: "chatEmptyState" }, [
      el("div", { class: "chatEmptyCopy muted", text: "Start a session to begin a conversation." }),
      el("button", { id: "chatEmptyNewBtn", class: "icon-btn text-btn", type: "button", title: "New session", "aria-label": "New session", "data-hint": "m", text: "New session" }),
    ]);
    const chat = el("div", { class: "chat", id: "chat" });
    const chatInner = el("div", { class: "chatInner", id: "chatInner" });
    const olderWrap = el("div", { class: "olderWrap", id: "olderWrap" });
    const olderBtn = el("button", { class: "olderBtn", id: "olderBtn", type: "button", "data-hint": "o", text: "Load older messages" });
    const olderErrorText = el("span", { class: "olderErrorText", text: "" });
    const olderRetryBtn = el("button", { class: "olderRetryBtn", id: "olderRetryBtn", type: "button", "data-hint": "r", text: "Retry" });
    const olderError = el("div", { class: "olderError", id: "olderError", role: "status" }, [olderErrorText, olderRetryBtn]);
    olderWrap.append(olderBtn, olderError);
    const bottomSentinel = el("div", { id: "bottomSentinel" });
    const jumpBtn = el("button", { class: "jumpBtn", id: "jumpBtn", title: "Jump to latest message", "aria-label": "Jump to latest message", "data-hint": "g", html: iconSvg("down-bar") });
    const chatTimeChip = el("div", { id: "chatTimeChip", class: "chatTimeChip", "aria-hidden": "true" });
    const chatSearchInput = el("input", { id: "chatSearchInput", class: "chatSearchInput", type: "search", placeholder: "Search conversation", "aria-label": "Search conversation", "data-hint-excluded": "native-text-entry", autocomplete: "off" });
    const chatSearchGlyph = el("span", { class: "chatSearchGlyph", "aria-hidden": "true", html: iconSvg("search") });
    const chatSearchPrevBtn = el("button", { id: "chatSearchPrevBtn", class: "icon-btn", type: "button", title: "Previous match", "aria-label": "Previous match", "data-hint": "v", html: iconSvg("up") });
    const chatSearchNextBtn = el("button", { id: "chatSearchNextBtn", class: "icon-btn", type: "button", title: "Next match", "aria-label": "Next match", "data-hint": "k", html: iconSvg("down") });
    const chatSearchCloseBtn = el("button", { id: "chatSearchCloseBtn", class: "icon-btn", type: "button", title: "Close search", "aria-label": "Close search", "data-hint": "x", html: iconSvg("x") });
    const chatSearchStatus = el("span", { id: "chatSearchStatus", class: "chatSearchStatus", text: "Search conversation" });
    const chatSearchAllHintEl = el("span", { id: "chatSearchAllHint", class: "chatSearchAllHint", text: "" });
    const chatSearchBar = el("div", { id: "chatSearchBar", class: "chatSearchBar", role: "search", "aria-label": "Search conversation" }, [chatSearchGlyph, chatSearchInput, chatSearchStatus, chatSearchAllHintEl, chatSearchPrevBtn, chatSearchNextBtn, chatSearchCloseBtn]);
    chatSearchBar.style.display = "none";
    chatInner.append(olderWrap, bottomSentinel);
    chat.appendChild(chatInner);

    const titleLabel = el("div", { id: "threadTitle", "data-hint": "t", text: "No session selected" });
    const statusChip = el("span", { class: "status-chip", id: "statusChip", text: "Idle" });
    const ctxChip = el("button", { class: "status-chip", id: "ctxChip", text: "", type: "button", "aria-label": "Context usage details", "data-hint": "y" });
    ctxChip.style.display = "none";
    ctxChip.disabled = true;
    const interruptBtn = el("button", { id: "interruptBtn", class: "icon-btn", title: "Interrupt (Esc)", "aria-label": "Interrupt (Esc)", "data-hint": "z", type: "button", html: iconSvg("stop") });
    interruptBtn.style.display = "none";
    const toast = el("div", { class: "muted toast", id: "toast", role: "status", "aria-live": "polite" });
    const networkBanner = el("div", { class: "networkBanner", id: "networkBanner", role: "status", "aria-live": "polite", "aria-atomic": "true", "aria-hidden": "true" });
    networkBanner.hidden = true;
    const toggleSidebarBtn = el("button", { id: "toggleSidebarBtn", class: "icon-btn", title: "Toggle sidebar", "aria-label": "Toggle sidebar", "data-hint": "s", html: iconSvg("menu") });
    const unattendedBtn = el("button", { id: "unattendedBtn", class: "icon-btn", title: "Unattended mode", "aria-label": "Unattended mode", "data-hint": "u", "aria-controls": "unattendedMenu", "aria-expanded": "false", "aria-haspopup": "dialog", type: "button", html: iconSvg("unattended") });
    unattendedBtn.disabled = true;
    const sidebarHeaderActions = el("div", { class: "actions" });
    const newBtn = el("button", { id: "newBtn", class: "icon-btn", title: "New session", "aria-label": "New session", "data-hint": "c", html: iconSvg("plus") });
    sidebarHeaderActions.appendChild(newBtn);
    const diagBtn = el("button", { id: "diagBtn", class: "icon-btn", title: "Details", "aria-label": "Details", "data-hint": "d", type: "button", html: iconSvg("info") });
    diagBtn.disabled = true;
    const prevUserBtn = el("button", { id: "prevUserBtn", class: "icon-btn", title: "Previous user message", "aria-label": "Previous user message", "data-hint": "p", type: "button", html: iconSvg("up") });
    const nextUserBtn = el("button", { id: "nextUserBtn", class: "icon-btn", title: "Next user message", "aria-label": "Next user message", "data-hint": "n", type: "button", html: iconSvg("down") });
    const chatSearchBtn = el("button", { id: "chatSearchBtn", class: "icon-btn", title: "Search conversation", "aria-label": "Search conversation", "data-hint": "/", type: "button", html: iconSvg("search") });
    const fileBtn = el("button", { id: "fileBtn", class: "icon-btn", title: "View file", "aria-label": "View file", "data-hint": "b", type: "button", html: iconSvg("file") });
    prevUserBtn.disabled = nextUserBtn.disabled = chatSearchBtn.disabled = fileBtn.disabled = true;
    const voiceHost = el("div", { class: "actions voiceActions", id: "voiceActions" });
    sidebarHeaderActions.appendChild(voiceHost);
    const topMeta = el("div", { class: "topMeta" }, [ctxChip]);
    const titleRow = el("div", { class: "titleRow" }, [titleLabel, topMeta]);
    const titleWrap = el("div", { class: "titleWrap" }, [titleRow]);
    const chatMessageNavControls = el("div", { class: "chatMessageNavControls", role: "group", "aria-label": "User message navigation" }, [prevUserBtn, nextUserBtn]);
    const chatNavRail = el("div", { class: "chatNavRail", id: "chatNavRail", "aria-label": "Loaded chat navigation" }, [chatSearchBtn, chatMessageNavControls]);
    const chatHeader = el("div", { class: "chatHeader", id: "chatHeader" }, [chatTimeChip, chatNavRail]);
    chatWrap.append(chatHeader, chat, chatEmptyState, jumpBtn, chatSearchBar);
    const topbar = el("div", { class: "topbar" }, [el("div", { class: "pill" }, [toggleSidebarBtn, titleWrap]), el("div", { class: "actions topActions" }, [fileBtn, diagBtn, unattendedBtn, interruptBtn])]);
    const composer = el("div", { class: "composer" });
    const stagedTray = el("div", { class: "stagedAttachments", id: "stagedAttachments", "aria-live": "polite" });
    const textarea = el("textarea", { id: "msg", placeholder: "", "aria-label": "Message", "data-hint": "i" });
    const msgPh = el("div", { class: "ph", id: "msgPh", text: "Message" });
    const modelPicker = el("div", { class: "modelPicker", id: "modelPicker", role: "listbox", "aria-label": "Available Pi models" });
    modelPicker.style.display = "none";
    const imgInput = el("input", { id: "imgInput", type: "file", accept: "image/*,video/*,*/*", multiple: "multiple", "data-hint-excluded": "hidden-file-input", style: "display:none" });
    const attachBtn = el("button", { class: "icon-btn", id: "attachBtn", type: "button", title: "Attach file", "aria-label": "Attach file", "data-hint": "a", html: iconSvg("paperclip") });
    const queueBtn = el("button", { class: "icon-btn", id: "queueBtn", type: "button", title: "Queued messages", "aria-label": "Queued messages", "data-hint": "q", html: iconSvg("queue") });
    const sendBtn = el("button", { class: "icon-btn primary", id: "sendBtn", type: "submit", title: "Send", "aria-label": "Send", "data-hint": "e", html: iconSvg("send") });
    const form = el("form", {}, [
      stagedTray,
      el("div", { class: "composerInputRow" }, [attachBtn, el("div", { class: "inputWrap" }, [textarea, msgPh]), queueBtn, sendBtn]),
      modelPicker,
      imgInput,
    ]);
    composer.appendChild(form);
    sidebar.appendChild(el("header", {}, [el("div", { class: "title", html: `<svg class="sidebarLogo" viewBox="0 0 24 24" data-logo-motif="dog-ear-terminal" aria-hidden="true" focusable="false"><path d="M4 2h10l6 6v14H4z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="miter"/><path d="M14 2v6h6" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="miter"/><path d="m7 13 3 3-3 3M12 19h4" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="square" stroke-linejoin="miter"/></svg>Codoxear` }), sidebarHeaderActions]));
    sidebar.append(sessionsWrap, sidebarFooter);
    main.append(topbar, networkBanner, toast, chatWrap, composer);
    app.append(navRail, sidebar, main, backdrop);
    root.append(app);

    const elements = Object.freeze({ root, app, navRail, backdrop, sidebar, sessionsWrap, sidebarEmptyHint, main, chatWrap, chatHeader, chatEmptyState, chat, chatInner, olderWrap, olderBtn, olderRetryBtn, olderError, olderErrorText, bottomSentinel, jumpBtn, chatTimeChip, chatSearchInput, chatSearchPrevBtn, chatSearchNextBtn, chatSearchCloseBtn, chatSearchStatus, chatSearchAllHintEl, chatSearchBar, chatNavRail, titleLabel, statusChip, ctxChip, interruptBtn, toast, networkBanner, toggleSidebarBtn, unattendedBtn, sidebarHeaderActions, voiceHost, diagBtn, prevUserBtn, nextUserBtn, chatSearchBtn, fileBtn, composer, form, textarea, msgPh, modelPicker, imgInput, attachBtn, queueBtn, sendBtn, stagedTray });
    return Object.freeze({ elements, cleanup() { root.innerHTML = ""; } });
  }


  window.CodoxearShell = Object.freeze({ createShellDOM });
})();
