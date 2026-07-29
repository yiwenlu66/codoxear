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

  // The persistent application chrome is deliberately constructed in one
  // place. Controllers receive the returned nodes rather than reaching into
  // the document for ownership; IDs remain stable for legacy integrations.
  function createShellDOM(options = {}) {
    if (!options || typeof options !== "object") throw new TypeError("shell dependency missing: options");
    const root = requireNode(options.root, "root");
    const el = requireFunction(options.el, "el");
    const iconSvg = requireFunction(options.iconSvg, "iconSvg");
    const resolveAppUrl = requireFunction(options.resolveAppUrl, "resolveAppUrl");
    const versionedShellAssetPath = requireFunction(options.versionedShellAssetPath, "versionedShellAssetPath");

    // During the staged migration app.js can hand the already-mounted shell
    // back to this factory. This keeps its lifecycle boundary authoritative
    // while old modal construction is still being moved behind the factory.
    if (options.reuseExisting) {
      const byId = (id) => {
        const node = root.querySelector(`#${id}`);
        if (!node) throw new Error(`shell DOM missing required element: ${id}`);
        return node;
      };
      const elements = Object.freeze({
        root, app: root.querySelector(".app"), backdrop: byId("backdrop"), sidebar: root.querySelector(".sidebar"), sessionsWrap: byId("sessions"), main: root.querySelector(".main"), chatWrap: byId("chatWrap"), chat: byId("chat"), chatInner: byId("chatInner"), olderWrap: byId("olderWrap"), olderBtn: byId("olderBtn"), olderRetryBtn: root.querySelector(".olderRetryBtn"), olderError: byId("olderError"), olderErrorText: root.querySelector(".olderErrorText"), bottomSentinel: byId("bottomSentinel"), jumpBtn: byId("jumpBtn"), chatTimeChip: byId("chatTimeChip"), chatSearchInput: byId("chatSearchInput"), chatSearchPrevBtn: byId("chatSearchPrevBtn"), chatSearchNextBtn: byId("chatSearchNextBtn"), chatSearchCloseBtn: byId("chatSearchCloseBtn"), chatSearchStatus: byId("chatSearchStatus"), chatSearchAllHintEl: byId("chatSearchAllHint"), chatSearchBar: byId("chatSearchBar"), titleLabel: byId("threadTitle"), statusChip: byId("statusChip"), ctxChip: byId("ctxChip"), interruptBtn: byId("interruptBtn"), toast: byId("toast"), toggleSidebarBtn: byId("toggleSidebarBtn"), unattendedBtn: byId("unattendedBtn"), announceBtn: byId("announceBtn"), notificationBtn: byId("notificationBtn"), diagBtn: byId("diagBtn"), prevUserBtn: byId("prevUserBtn"), nextUserBtn: byId("nextUserBtn"), chatSearchBtn: byId("chatSearchBtn"), fileBtn: byId("fileBtn"), unattendedMenu: byId("unattendedMenu"), liveAudio: byId("liveAudio"), composer: root.querySelector(".composer"), form: root.querySelector(".composer form"), textarea: byId("msg"), msgPh: byId("msgPh"), imgInput: byId("imgInput"), attachBtn: byId("attachBtn"), queueBtn: byId("queueBtn"), composerStopBtn: byId("composerStopBtn"), sendBtn: byId("sendBtn"), stagedTray: byId("stagedAttachments"),
      });
      return Object.freeze({ elements, cleanup() {} });
    }

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
    const form = el("form", {}, [
      el("button", { class: "icon-btn", id: "attachBtn", type: "button", title: "Attach file", "aria-label": "Attach file", html: iconSvg("paperclip") }),
      el("div", { class: "inputWrap" }, [el("div", { class: "stagedAttachments", id: "stagedAttachments", "aria-live": "polite" }), el("textarea", { id: "msg", placeholder: "", "aria-label": "Enter your instructions here" }), el("div", { class: "ph", id: "msgPh", text: "Enter your instructions here" })]),
      el("input", { id: "imgInput", type: "file", accept: "image/*,video/*,*/*", multiple: "multiple", style: "display:none" }),
      el("button", { class: "icon-btn", id: "queueBtn", type: "button", title: "Queued messages", "aria-label": "Queued messages", html: iconSvg("queue") }),
      el("button", { class: "icon-btn composerStopBtn", id: "composerStopBtn", type: "button", title: "Stop current response", "aria-label": "Stop current response", html: iconSvg("stop") }),
      el("button", { class: "icon-btn primary", id: "sendBtn", type: "submit", title: "Send", "aria-label": "Send", html: iconSvg("send") }),
    ]);
    composer.appendChild(form);
    sidebar.appendChild(el("header", {}, [el("div", { class: "title", html: `<img class="sidebarLogo" src="${resolveAppUrl(versionedShellAssetPath("/static/codoxear-icon.png"))}" alt="" />Codoxear` }), el("div", { class: "actions" }, [el("button", { id: "newBtn", class: "icon-btn", title: "New session", "aria-label": "New session", html: iconSvg("plus") }), notificationBtn, announceBtn])]));
    sidebar.append(sessionsWrap, sidebarFooter);
    main.append(topbar, toast, chatWrap, composer);
    app.append(sidebar, main, backdrop);
    root.append(app, unattendedMenu, liveAudio);

    const elements = Object.freeze({ root, app, backdrop, sidebar, sessionsWrap, sidebarEmptyHint, main, chatWrap, chat, chatInner, olderWrap, olderBtn, olderRetryBtn, olderError, olderErrorText, bottomSentinel, jumpBtn, chatTimeChip, chatSearchInput, chatSearchPrevBtn, chatSearchNextBtn, chatSearchCloseBtn, chatSearchStatus, chatSearchAllHintEl, chatSearchBar, titleLabel, statusChip, ctxChip, interruptBtn, toast, toggleSidebarBtn, unattendedBtn, announceBtn, notificationBtn, diagBtn, prevUserBtn, nextUserBtn, chatSearchBtn, fileBtn, unattendedMenu, liveAudio, composer, form, textarea: form.querySelector("#msg"), msgPh: form.querySelector("#msgPh"), imgInput: form.querySelector("#imgInput"), attachBtn: form.querySelector("#attachBtn"), queueBtn: form.querySelector("#queueBtn"), composerStopBtn: form.querySelector("#composerStopBtn"), sendBtn: form.querySelector("#sendBtn") });
    return Object.freeze({ elements, cleanup() { root.innerHTML = ""; } });
  }

  window.CodoxearShell = Object.freeze({ createShellDOM });
})();
