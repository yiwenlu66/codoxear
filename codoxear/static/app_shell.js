import * as CodoxearQueue from "./app_queue.js";


function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`shell dependency missing: ${name}`);
    return value;
  }

  function requireNode(value, name) {
    if (!value || typeof value !== "object" || typeof value.appendChild !== "function")
      throw new TypeError(`shell dependency missing: ${name}`);
    return value;
  }

  function createComposerDOM(options = {}) {
    if (!options || typeof options !== "object") throw new TypeError("shell composer dependency missing: options");
    const el = requireFunction(options.el, "el");
    const iconSvg = requireFunction(options.iconSvg, "iconSvg");
    const idPrefix = typeof options.idPrefix === "string" ? options.idPrefix : "";
    const id = (name) => `${idPrefix}${name}`;
    const composer = el("div", { class: "composer" });
    const stagedTray = el("div", { class: "stagedAttachments", id: id("stagedAttachments"), "aria-live": "polite" });
    const textarea = el("textarea", { id: id("msg"), placeholder: "", "aria-label": "Message", "data-hint": "i" });
    const msgPh = el("div", { class: "ph", id: id("msgPh"), text: "Message" });
    const modelPicker = el("div", { class: "modelPicker", id: id("modelPicker"), role: "listbox", "aria-label": "Available Pi models" });
    modelPicker.style.display = "none";
    const imgInput = el("input", { id: id("imgInput"), type: "file", multiple: "multiple", "data-hint-excluded": "hidden-file-input", style: "display:none" });
    const attachBtn = el("button", { class: "icon-btn", id: id("attachBtn"), type: "button", title: "Attach file", "aria-label": "Attach file", "data-hint": "a", html: iconSvg("paperclip") });
    const queueBtn = el("button", { class: "icon-btn", id: id("queueBtn"), type: "button", title: "Queued messages", "aria-label": "Queued messages", "data-hint": "q", html: iconSvg("queue") });
    const sendBtn = el("button", { class: "icon-btn primary", id: id("sendBtn"), type: "submit", title: "Send", "aria-label": "Send", "data-hint": "e", html: iconSvg("send") });
    const form = el("form", {}, [
      stagedTray,
      el("div", { class: "composerInputRow" }, [attachBtn, el("div", { class: "inputWrap" }, [textarea, msgPh]), queueBtn, sendBtn]),
      modelPicker,
      imgInput,
    ]);
    composer.appendChild(form);
    return Object.freeze({ composer, form, textarea, msgPh, modelPicker, imgInput, attachBtn, queueBtn, sendBtn, stagedTray });
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

    const titleLabel = el("div", { id: "threadTitle", "data-hint": "t", text: "" });
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
    const topMeta = el("div", { class: "topMeta" });
    const titleRow = el("div", { class: "titleRow" }, [titleLabel, topMeta]);
    const titleWrap = el("div", { class: "titleWrap" }, [titleRow]);
    const chatMessageNavControls = el("div", { class: "chatMessageNavControls", role: "group", "aria-label": "User message navigation" }, [prevUserBtn, nextUserBtn]);
    const chatNavRail = el("div", { class: "chatNavRail", id: "chatNavRail", "aria-label": "Loaded chat navigation" }, [chatSearchBtn, chatMessageNavControls]);
    const chatHeader = el("div", { class: "chatHeader", id: "chatHeader" }, [chatTimeChip, chatNavRail]);
    chatWrap.append(chatHeader, chat, chatEmptyState, jumpBtn, chatSearchBar);
    const topActions = el("div", { class: "actions topActions" }, [fileBtn, diagBtn, unattendedBtn]);
    const topbar = el("div", { class: "topbar" }, [el("div", { class: "pill" }, [toggleSidebarBtn, titleWrap]), topActions]);
    const composerDOM = createComposerDOM({ el, iconSvg });
    const { composer, form, textarea, msgPh, modelPicker, imgInput, attachBtn, queueBtn, sendBtn, stagedTray } = composerDOM;
    sidebar.appendChild(el("header", {}, [el("div", { class: "title", html: `<svg class="sidebarLogo" viewBox="0 0 24 24" data-logo-motif="dog-ear-terminal" aria-hidden="true" focusable="false"><path d="M4 2h10l6 6v14H4z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="miter"/><path d="M14 2v6h6" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="miter"/><path d="m7 13 3 3-3 3M12 19h4" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="square" stroke-linejoin="miter"/></svg>Codoxear` }), sidebarHeaderActions]));
    sidebar.append(sessionsWrap, sidebarFooter);
    main.append(topbar, networkBanner, toast, chatWrap, composer);
    app.append(sidebar, main, backdrop);
    root.append(app);

    const elements = Object.freeze({ root, app, backdrop, sidebar, sessionsWrap, sidebarEmptyHint, main, chatWrap, chatHeader, chatEmptyState, chat, chatInner, olderWrap, olderBtn, olderRetryBtn, olderError, olderErrorText, bottomSentinel, jumpBtn, chatTimeChip, chatSearchInput, chatSearchPrevBtn, chatSearchNextBtn, chatSearchCloseBtn, chatSearchStatus, chatSearchAllHintEl, chatSearchBar, chatNavRail, titleLabel, topMeta, topActions, toast, networkBanner, toggleSidebarBtn, unattendedBtn, sidebarHeaderActions, voiceHost, diagBtn, prevUserBtn, nextUserBtn, chatSearchBtn, fileBtn, composer, form, textarea, msgPh, modelPicker, imgInput, attachBtn, queueBtn, sendBtn, stagedTray });
    return Object.freeze({ elements, cleanup() { root.innerHTML = ""; } });
  }

  function createApplicationModalDOM(options = {}) {
    const { root, el, iconSvg, windowTarget: window, codoxearVoice, voiceHost, chatMarkdownHtmlCached } = options;
    if (!root || typeof root.appendChild !== "function") throw new TypeError("shell dependency missing: root");
    if (typeof el !== "function" || typeof iconSvg !== "function") throw new TypeError("shell dependency missing: DOM helpers");
    if (!window || !codoxearVoice || typeof codoxearVoice.createVoiceDom !== "function") throw new TypeError("shell dependency missing: modal runtime");
    const fileBackdrop = el("div", { class: "modalBackdrop", id: "fileBackdrop" });
    const fileCloseBtn = el("button", {
      id: "fileCloseBtn",
      class: "icon-btn",
      title: "Close",
      "aria-label": "Close",
      type: "button",
      html: iconSvg("x"),
    });
    const fileStatus = el("div", { class: "muted fileStatus", id: "fileStatus", role: "status", "aria-live": "polite", text: "" });
    const filePickerInput = el("input", {
      id: "filePickerInput",
      class: "filePickerInput",
      type: "text",
      placeholder: "Choose or search files",
      autocomplete: "off",
      spellcheck: "false",
      role: "combobox",
      "aria-autocomplete": "list",
      "aria-controls": "filePickerMenu",
      "aria-expanded": "false",
    });
    const filePickerMenu = el("div", { id: "filePickerMenu", class: "filePickerMenu", role: "listbox" });
    const filePickerField = el("div", { class: "pickerField filePickerField", id: "filePickerField" }, [
      el("span", { class: "filePickerIcon", html: iconSvg("chevronDown"), "aria-hidden": "true" }),
      filePickerInput,
      filePickerMenu,
    ]);
    const fileModeDiffBtn = el("button", {
      id: "fileModeDiffBtn",
      class: "icon-btn",
      type: "button",
      title: "Toggle diff",
      "aria-label": "Toggle diff",
      html: iconSvg("diff"),
    });
    const fileModePreviewBtn = el("button", {
      id: "fileModePreviewBtn",
      class: "icon-btn",
      type: "button",
      title: "Toggle markdown preview",
      "aria-label": "Toggle markdown preview",
      html: iconSvg("preview"),
    });
    const fileEditBtn = el("button", {
      id: "fileEditBtn",
      class: "icon-btn",
      type: "button",
      title: "Edit file",
      "aria-label": "Edit file",
      html: iconSvg("edit"),
    });
    const fileVideoPreviewBtn = el("button", {
      id: "fileVideoPreviewBtn",
      class: "icon-btn",
      type: "button",
      title: "Use compatible MP4 preview",
      "aria-label": "Use compatible MP4 preview",
      html: iconSvg("play"),
    });
    fileVideoPreviewBtn.style.display = "none";
    const fileDownloadBtn = el("button", {
      id: "fileDownloadBtn",
      class: "icon-btn",
      type: "button",
      title: "Download file",
      "aria-label": "Download file",
      html: iconSvg("download"),
    });
    const fileTouchSelectBtn = el("button", {
      id: "fileTouchSelectBtn",
      class: "icon-btn fileTouchBtn",
      type: "button",
      title: "Select",
      "aria-label": "Select",
      html: iconSvg("select"),
    });
    const fileTouchCopyBtn = el("button", {
      id: "fileTouchCopyBtn",
      class: "icon-btn fileTouchBtn",
      type: "button",
      title: "Copy selection",
      "aria-label": "Copy selection",
      html: iconSvg("copy"),
    });
    const fileTouchPasteBtn = el("button", {
      id: "fileTouchPasteBtn",
      class: "icon-btn fileTouchBtn",
      type: "button",
      title: "Paste",
      "aria-label": "Paste",
      html: iconSvg("paste"),
    });
    const fileTouchUpBtn = el("button", {
      id: "fileTouchUpBtn",
      class: "icon-btn fileTouchBtn",
      type: "button",
      title: "Select up",
      "aria-label": "Select up",
      html: iconSvg("up"),
    });
    const fileTouchLeftBtn = el("button", {
      id: "fileTouchLeftBtn",
      class: "icon-btn fileTouchBtn",
      type: "button",
      title: "Select left",
      "aria-label": "Select left",
      html: iconSvg("left"),
    });
    const fileTouchDownBtn = el("button", {
      id: "fileTouchDownBtn",
      class: "icon-btn fileTouchBtn",
      type: "button",
      title: "Select down",
      "aria-label": "Select down",
      html: iconSvg("down"),
    });
    const fileTouchRightBtn = el("button", {
      id: "fileTouchRightBtn",
      class: "icon-btn fileTouchBtn",
      type: "button",
      title: "Select right",
      "aria-label": "Select right",
      html: iconSvg("right"),
    });
    const fileTouchDpad = el("div", { id: "fileTouchDpad", class: "fileTouchDpad" }, [
      el("span", { class: "fileTouchSpacer", "aria-hidden": "true" }),
      fileTouchUpBtn,
      el("span", { class: "fileTouchSpacer", "aria-hidden": "true" }),
      fileTouchLeftBtn,
      fileTouchDownBtn,
      fileTouchRightBtn,
    ]);
    const fileTouchActions = el("div", { id: "fileTouchActions", class: "fileTouchActions" }, [
      fileTouchSelectBtn,
      fileTouchCopyBtn,
      fileTouchPasteBtn,
    ]);
    const fileTouchToolbar = el("div", { id: "fileTouchToolbar", class: "fileTouchToolbar" }, [
      fileTouchDpad,
      fileTouchActions,
    ]);
    const fileDiff = el("div", { class: "fileDiff", id: "fileDiff" });
    const fileImage = el("img", { id: "fileImage", class: "fileImage", alt: "" });
    const fileVideo = el("video", { id: "fileVideo", class: "fileVideo", controls: true, preload: "metadata" });
    const fileViewer = el("div", { class: "fileViewer", id: "fileViewer", role: "dialog", "aria-modal": "true", "aria-label": "File viewer" }, [
      el("div", { class: "fileViewerHeader" }, [
        el("div", { class: "title", text: "View file" }),
        el("div", { class: "actions" }, [fileModeDiffBtn, fileModePreviewBtn, fileEditBtn, fileVideoPreviewBtn, fileDownloadBtn, fileCloseBtn]),
      ]),
      el("div", { class: "fileCandRow", id: "fileCandRow" }, [filePickerField]),
      fileStatus,
      fileDiff,
      fileImage,
      fileVideo,
      fileTouchToolbar,
    ]);
    root.appendChild(fileBackdrop);
    root.appendChild(fileViewer);

    const fileUnsavedBackdrop = el("div", { class: "modalBackdrop", id: "fileUnsavedBackdrop" });
    const fileUnsavedDialog = el("div", { class: "sendChoice fileUnsavedDialog", id: "fileUnsavedDialog", role: "dialog", "aria-modal": "true", "aria-label": "Unsaved file changes" }, [
      el("div", { class: "title", text: "Unsaved changes" }),
      el("div", { class: "muted", text: "Save this file before leaving the editor?" }),
      el("div", { class: "sendChoiceActions" }, [
        el("button", { class: "primary", id: "fileUnsavedSaveBtn", type: "button", text: "Save" }),
        el("button", { id: "fileUnsavedDiscardBtn", type: "button", text: "Discard" }),
        el("button", { id: "fileUnsavedCancelBtn", type: "button", text: "Cancel" }),
      ]),
    ]);
    root.appendChild(fileUnsavedBackdrop);
    root.appendChild(fileUnsavedDialog);
    const filePasteBackdrop = el("div", { class: "modalBackdrop", id: "filePasteBackdrop" });
    const filePasteInput = el("textarea", {
      id: "filePasteInput",
      class: "filePasteInput",
      placeholder: "Paste text here",
      spellcheck: "false",
      autocapitalize: "off",
      autocomplete: "off",
      autocorrect: "off",
    });
    const filePasteDialog = el("div", { class: "sendChoice filePasteDialog", id: "filePasteDialog", role: "dialog", "aria-modal": "true", "aria-label": "Paste into file" }, [
      el("div", { class: "title", text: "Paste into file" }),
      el("div", { class: "muted", text: "Long-press in this box to use the browser paste menu, then insert into the editor." }),
      filePasteInput,
      el("div", { class: "sendChoiceActions" }, [
        el("button", { class: "primary", id: "filePasteInsertBtn", type: "button", text: "Insert" }),
        el("button", { id: "filePasteCancelBtn", type: "button", text: "Cancel" }),
      ]),
    ]);
    root.appendChild(filePasteBackdrop);
    root.appendChild(filePasteDialog);

    const sendChoiceBackdrop = el("div", { class: "modalBackdrop", id: "sendChoiceBackdrop" });
    const sendChoice = el("div", { class: "sendChoice", id: "sendChoice", role: "dialog", "aria-modal": "true", "aria-label": "Send options" }, [
      el("div", { class: "title", text: "Current response is running" }),
      el("div", { class: "muted", text: "Choose how to handle your next message." }),
      el("div", { class: "sendChoiceActions" }, [
        el("button", { class: "primary", id: "sendChoiceNow", type: "button", text: "Send now" }),
        el("button", { id: "sendChoiceLater", type: "button", text: "Send after current" }),
        el("button", { id: "sendChoiceCancel", type: "button", text: "Cancel" }),
      ]),
    ]);
    root.appendChild(sendChoiceBackdrop);
    root.appendChild(sendChoice);

    const appConfirmBackdrop = el("div", { class: "modalBackdrop appConfirmBackdrop", id: "appConfirmBackdrop" });
    const appConfirmTitle = el("div", { class: "title", id: "appConfirmTitle", text: "Confirm action" });
    const appConfirmMessage = el("div", { class: "muted appConfirmMessage", id: "appConfirmMessage", text: "" });
    const appConfirmConfirmBtn = el("button", { class: "primary", id: "appConfirmConfirmBtn", type: "button", text: "Confirm" });
    const appConfirmCancelBtn = el("button", { id: "appConfirmCancelBtn", type: "button", text: "Cancel" });
    const appConfirm = el("div", {
      class: "sendChoice appConfirm",
      id: "appConfirm",
      role: "dialog",
      "aria-modal": "true",
      "aria-labelledby": "appConfirmTitle",
      "aria-describedby": "appConfirmMessage",
    }, [
      appConfirmTitle,
      appConfirmMessage,
      el("div", { class: "sendChoiceActions appConfirmActions" }, [appConfirmConfirmBtn, appConfirmCancelBtn]),
    ]);
    root.appendChild(appConfirmBackdrop);
    root.appendChild(appConfirm);

    const {
      queueBackdrop,
      queueCloseBtn,
      queueList,
      queueEmpty,
      queueViewer,
    } = CodoxearQueue.createQueueDom({ root, el, iconSvg });

    const helpBackdrop = el("div", { class: "modalBackdrop", id: "helpBackdrop" });
    const helpCloseBtn = el("button", {
      id: "helpCloseBtn",
      class: "icon-btn",
      title: "Close",
      "aria-label": "Close",
      type: "button",
      html: iconSvg("x"),
    });
    const helpViewer = el("div", { class: "helpViewer", id: "helpViewer", role: "dialog", "aria-modal": "true", "aria-label": "Help" }, [
      el("div", { class: "queueHeader" }, [
        el("div", { class: "title", text: "Help" }),
        el("div", { class: "actions" }, [helpCloseBtn]),
      ]),
      el("div", {
        class: "helpBody",
        html: `<div class="muted">Sessions</div>
<ul class="md">
  <li>Choose a conversation from the sidebar. On desktop, hover a row to reveal <b>Edit</b>, <b>Duplicate</b>, and <b>Delete</b>. On touch, swipe left for <b>Edit</b>/<b>Duplicate</b> and right for <b>Delete</b>.</li>
  <li>The dot on the title row shows state: <b>filled + pulsing</b> = busy, <b>hollow</b> = idle, <b>filled (no pulse)</b> = snoozed or blocked, <b>filled amber + pulsing</b> = starting.</li>
  <li>The metadata line shows the agent-backend icon first, then the session-type icon, followed by recency, model and reasoning suffix (for example <b>·hi</b>), folder, and branch.</li>
  <li>Click the conversation title to rename or reprioritize it. <b>Details</b> in the session utilities bar shows the exact backend, provider, model, reasoning level, queue state, and token usage.</li>
</ul>
<div class="muted">New session</div>
<ul class="md">
  <li><b>New session</b> can start fresh or resume a matching conversation for the currently selected backend in the current working directory.</li>
  <li>The backend tabs choose between the supported agent backends. Right now that is <b>Codex</b>, <b>Pi</b>, and <b>Claude</b>.</li>
  <li>You can choose working directory, a combined provider/model pair, reasoning level, and whether the session should start in tmux. If the directory is a Git repo, you can also start in a new worktree branch.</li>
  <li>For Pi, the reasoning level is set when the session launches. To change it later on a live session, type <b>/effort</b> in the composer; <b>/thinking</b> remains an alias (see Messages and queue).</li>
  <li>Codoxear remembers the last backend you used and the last provider/model pair for each backend.</li>
</ul>
<div class="muted">Messages and queue</div>
<ul class="md">
  <li><b>Send</b> submits immediately when the session is idle. When it is busy, a dialog offers <b>Send now</b> (sends right away, steering the running turn) or <b>Send after current</b> (queues the prompt for when the session becomes idle).</li>
  <li>The queue is stored per session and drains automatically when that session becomes idle. Use <b>Queued messages</b> to review or edit queued prompts.</li>
  <li><b>Load older messages</b> fetches more scrollback. <b>Jump to latest</b> returns to the newest turn when you are reading history.</li>
  <li>The <b>Search</b> button and <b>Previous</b>/<b>Next</b> message controls remain available in the compact navigation group at the top of the conversation on phone and desktop. Use <b>/</b> to search the conversation. The search bar shows a position such as <b>2 of 5</b>; at the oldest visible match, it tells you when <b>Previous</b> can load older matches.</li>
  <li>Busy rows show tool activity plus a reasoning-token count only when the backend reports positive token usage. Tool/thinking counts never decrease during an open turn; <b>▸N</b> is a separate live subagent gauge and can decrease when a worker finishes.</li>
  <li>On a <b>Pi</b> session, type <b>/model</b> in the composer to switch models live, or <b>/effort</b> to switch the reasoning level; <b>/thinking</b> remains an alias. Start typing to filter the list, then choose an entry. The model picker lists configured providers and models; the effort picker lists the levels the current model supports.</li>
  <li>Press <b>f</b> to show keyboard hints over each activatable control in the current viewport. Hidden, off-screen, disabled, and hit-test-covered controls are excluded. Stable shell labels include <b>1</b>–<b>9</b> sessions; <b>s</b> sidebar; <b>t</b> edit conversation; <b>b</b> files; <b>d</b> details; <b>u</b> unattended; <b>z</b> interrupt; <b>/</b> search; <b>p</b>/<b>n</b> previous/next user message; <b>o</b> older messages; <b>g</b> latest; <b>a</b> attach; <b>q</b> queued messages; <b>e</b> send; <b>i</b> message box; <b>c</b> new session; <b>h</b> help; <b>w</b> settings; and <b>l</b> log out. Extra visible controls receive their displayed dynamic label. Press <b>Escape</b> or <b>Backspace</b> to cancel.</li>
  <li>In an open dialog, press a visible button's first distinctive letter to activate it. When buttons share their first letter, use a later distinctive letter. <b>Esc</b> closes the dialog.</li>
  <li>Direct shortcuts (no leader): <b>i</b> focus message box; <b>j</b>/<b>k</b> scroll down/up; <b>d</b>/<b>u</b> scroll half-page down/up; <b>G</b> go to bottom; <b>D</b> delete current session (confirm); <b>/</b> search; <b>Esc</b> exit message box or close dialog.</li>
</ul>
<div class="muted">Unattended mode</div>
<ul class="md">
  <li>Unattended mode is a per-session idle nudge. Open the Unattended button in the session utilities bar, turn it on, and optionally add an extra request to append to the built-in unattended-work prompt.</li>
  <li><b>Cooldown time</b> is how many idle minutes must pass after the assistant finishes before the next unattended prompt is injected.</li>
  <li><b>Number of injections</b> is the remaining auto-injection budget for that session. Each unattended prompt decrements it, and unattended mode turns itself off when it reaches zero.</li>
  <li>Unattended mode runs in the server process, so it keeps working even if you close the browser tab. Enabled sessions show an <b>unattended</b> badge in the sidebar.</li>
</ul>
<div class="muted">Files</div>
<ul class="md">
  <li><b>View file</b> opens recent or changed files from the selected session, with diff, file, and preview modes where available.</li>
  <li>File paths mentioned in assistant messages become clickable when the server can resolve them.</li>
  <li><b>Attach file</b> adds local files or images to the current prompt.</li>
</ul>
<div class="muted">Announcements and notifications</div>
<ul class="md">
  <li><b>Announcement</b> is a per-browser toggle. It plays the shared server audio stream and announces every end-of-turn response. Narration announcements are optional in Settings.</li>
  <li><b>Notification</b> is a per-browser toggle. On desktop it enables live browser notifications for final responses. On iPhone/iPad it can also enable Web Push when you use the installed Home Screen app over HTTPS.</li>
  <li>If Announcement cannot be enabled yet, open <b>Settings</b> and fill in the OpenAI-compatible API base URL and API key used for summarization and speech.</li>
</ul>`,
      }),
    ]);
    root.appendChild(helpBackdrop);
    root.appendChild(helpViewer);

    const diagBackdrop = el("div", { class: "modalBackdrop", id: "diagBackdrop" });
    const diagCopyConversationBtn = el("button", {
      id: "diagCopyConversationBtn",
      class: "icon-btn",
      title: "Copy conversation",
      "aria-label": "Copy conversation",
      type: "button",
      html: iconSvg("copy-all"),
    });
    const diagCopyBtn = el("button", {
      id: "diagCopyBtn",
      class: "icon-btn",
      title: "Copy details",
      "aria-label": "Copy details",
      type: "button",
      html: iconSvg("copy"),
    });
    const diagCloseBtn = el("button", {
      id: "diagCloseBtn",
      class: "icon-btn",
      title: "Close",
      "aria-label": "Close",
      type: "button",
      html: iconSvg("x"),
    });
    // Detail actions start disabled until the controller loads the selected
    // session's details and enables their corresponding payloads.
    diagCopyConversationBtn.disabled = true;
    diagCopyBtn.disabled = true;
    const diagStatus = el("div", { class: "muted", id: "diagStatus", text: "" });
    const diagContent = el("div", { class: "detailsGrid", id: "diagContent" });
    const diagViewer = el("div", { class: "diagViewer", id: "diagViewer", role: "dialog", "aria-modal": "true", "aria-label": "Details" }, [
      el("div", { class: "queueHeader" }, [
        el("div", { class: "title", text: "Details" }),
        el("div", { class: "actions" }, [diagCopyConversationBtn, diagCopyBtn, diagCloseBtn]),
      ]),
      diagStatus,
      diagContent,
    ]);
    root.appendChild(diagBackdrop);
    root.appendChild(diagViewer);

    const editCloseBtn = el("button", {
      id: "editCloseBtn",
      class: "icon-btn",
      title: "Close",
      "aria-label": "Close",
      type: "button",
      html: iconSvg("x"),
    });
    const editStatus = el("div", { class: "muted", id: "editStatus", text: "" });
    const editNameInput = el("input", {
      id: "editNameInput",
      type: "text",
      placeholder: "Conversation title",
      maxlength: "80",
      autocomplete: "off",
    });
    const editPriorityRange = el("input", {
      id: "editPriorityRange",
      type: "range",
      min: "-1",
      max: "1",
      step: "0.05",
      value: "0",
    });
    const editPriorityValue = el("span", { class: "rangeValue", id: "editPriorityValue", text: "+0.00" });
    const editPriorityResetBtn = el("button", {
      id: "editPriorityResetBtn",
      class: "icon-btn text-btn subtleBtn",
      type: "button",
      text: "Reset",
    });
    const editSnoozeModeButtons = new Map();
    let editSnoozeMode = "none";
    const editSnoozeButtons = el("div", { class: "choiceChips", id: "editSnoozeButtons" });
    for (const [value, label] of [
      ["none", "No snooze"],
      ["4h", "4 hours"],
      ["tomorrow", "Tomorrow"],
      ["custom", "Custom"],
    ]) {
      const btn = el("button", {
        type: "button",
        class: "choiceChip",
        "data-snooze-mode": value,
        text: label,
      });
      editSnoozeModeButtons.set(value, btn);
      editSnoozeButtons.appendChild(btn);
    }
    const editSnoozeCustomDate = el("input", { id: "editSnoozeCustomDate", type: "date" });
    const editSnoozeCustomTime = el("input", { id: "editSnoozeCustomTime", type: "time", step: "60" });
    const editSnoozeCustomRow = el("div", { class: "customSnoozeRow", id: "editSnoozeCustomRow" }, [
      editSnoozeCustomDate,
      editSnoozeCustomTime,
    ]);
    const editDependencyBtn = el("button", {
      id: "editDependencyBtn",
      class: "filePickerBtn dialogPickerBtn",
      type: "button",
      "aria-label": "Choose dependency",
    });
    const editDependencyMenu = el("div", { id: "editDependencyMenu", class: "filePickerMenu dialogPickerMenu" });
    const editDependencyField = el("div", { class: "pickerField" }, [editDependencyBtn]);
    const editSaveBtn = el("button", { class: "primary", id: "editSaveBtn", type: "button", text: "Save" });
    const editViewer = el("dialog", { class: "formViewer formDialog", id: "editViewer", "aria-label": "Edit conversation" }, [
      el("div", { class: "queueHeader" }, [
        el("div", { class: "title", text: "Edit conversation" }),
        el("div", { class: "actions" }, [editCloseBtn]),
      ]),
      editStatus,
      el("div", { class: "formBody" }, [
        el("label", { class: "field" }, [
          el("span", { class: "fieldLabel", text: "Conversation name" }),
          editNameInput,
        ]),
        el("label", { class: "field editPriorityField" }, [
          el("span", { class: "fieldLabel", text: "Priority offset" }),
          el("div", { class: "sliderRow" }, [editPriorityRange, editPriorityValue, editPriorityResetBtn]),
        ]),
        el("label", { class: "field" }, [
          el("span", { class: "fieldLabel", text: "Snooze" }),
          editSnoozeButtons,
          editSnoozeCustomRow,
        ]),
        el("label", { class: "field" }, [
          el("span", { class: "fieldLabel", text: "Depends on" }),
          editDependencyField,
        ]),
      ]),
      el("div", { class: "formActions" }, [
        el("button", { id: "editCancelBtn", type: "button", text: "Cancel" }),
        editSaveBtn,
      ]),
    ]);
    root.appendChild(editViewer);
    editViewer.appendChild(editDependencyMenu);
    const voiceDom = codoxearVoice.createVoiceDom({ root, el, iconSvg, voiceHost, chatMarkdownHtmlCached });
    const {
      announceBtn,
      liveAudio,
      voiceSettingsBackdrop,
      voiceSettingsCloseBtn,
      voiceSettingsStatus,
      voiceBaseUrlInput,
      voiceApiKeyInput,
      voiceClearApiKeyToggle,
      narrationSettingToggle,
      unattendedPromptInput,
      unattendedPromptResetBtn,
      appearancePreview,
      voiceSettingsViewer,
      voiceSettingsCancelBtn,
      voiceSettingsSaveBtn,
    } = voiceDom;

    return Object.freeze({
      fileBackdrop, fileCloseBtn, fileStatus, filePickerInput, filePickerMenu, filePickerField,
      fileModeDiffBtn, fileModePreviewBtn, fileEditBtn, fileVideoPreviewBtn, fileDownloadBtn,
      fileTouchSelectBtn, fileTouchCopyBtn, fileTouchPasteBtn, fileTouchUpBtn, fileTouchLeftBtn,
      fileTouchDownBtn, fileTouchRightBtn, fileTouchDpad, fileTouchActions, fileTouchToolbar,
      fileDiff, fileImage, fileVideo, fileViewer, fileUnsavedBackdrop, fileUnsavedDialog,
      filePasteBackdrop, filePasteInput, filePasteDialog, sendChoiceBackdrop, sendChoice,
      appConfirmBackdrop, appConfirmTitle, appConfirmMessage, appConfirmConfirmBtn,
      appConfirmCancelBtn, appConfirm, queueBackdrop, queueCloseBtn, queueList, queueEmpty,
      queueViewer, helpBackdrop, helpCloseBtn, helpViewer, diagBackdrop, diagCopyConversationBtn,
      diagCopyBtn, diagCloseBtn, diagStatus, diagContent, diagViewer, editCloseBtn, editStatus,
      editNameInput, editPriorityRange, editPriorityValue, editPriorityResetBtn,
      editSnoozeModeButtons, editSnoozeButtons, editSnoozeCustomDate, editSnoozeCustomTime,
      editSnoozeCustomRow, editDependencyBtn, editDependencyMenu, editDependencyField,
      editSaveBtn, editViewer, announceBtn, liveAudio, voiceSettingsBackdrop,
      voiceSettingsCloseBtn, voiceSettingsStatus, voiceBaseUrlInput, voiceApiKeyInput,
      voiceClearApiKeyToggle, narrationSettingToggle, unattendedPromptInput,
      unattendedPromptResetBtn, appearancePreview, voiceSettingsViewer, voiceSettingsCancelBtn, voiceSettingsSaveBtn
    });
  }

export { createComposerDOM, createShellDOM, createApplicationModalDOM };
