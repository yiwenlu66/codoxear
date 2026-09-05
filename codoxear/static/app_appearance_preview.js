import * as CodoxearMessageRows from "./app_message_rows.js";
import * as CodoxearSessionState from "./app_session_state.js";
import * as CodoxearSessions from "./app_sessions.js";
import * as CodoxearShell from "./app_shell.js";


// The settings preview deliberately owns a fixture-only state graph. It uses
// the application card, message-row, and composer factories, while its theme
// token scope stops experimentation here from changing the running app.

function requireFunction(value, name) {
  if (typeof value !== "function") throw new TypeError(`appearance preview dependency missing: ${name}`);
  return value;
}

function previewSession(id, overrides = {}) {
  return {
    session_id: id,
    cwd: "/work/codoxear",
    updated_ts: 1_725_000_000,
    start_ts: 1_724_999_000,
    agent_backend: "pi",
    model: "openai/gpt-5",
    reasoning_effort: "high",
    owned: true,
    busy: false,
    ...overrides,
  };
}

function previewSessionEntries(sessions) {
  return sessions.map((session) => ({ type: "session", session }));
}

function createAppearancePreview(options = {}) {
  const el = requireFunction(options.el, "el");
  const iconSvg = requireFunction(options.iconSvg, "iconSvg");
  const chatMarkdownHtmlCached = requireFunction(options.chatMarkdownHtmlCached, "chatMarkdownHtmlCached");
  const shellComposer = CodoxearShell.createComposerDOM({ el, iconSvg, idPrefix: "appearance-preview-" });
  shellComposer.form.onsubmit = (event) => event.preventDefault();
  const sessionState = CodoxearSessionState.createSessionState({ consoleError: () => {} });
  const sessionsWrap = el("div", { class: "sessions appearancePreviewSessions", "aria-label": "Preview sessions" });
  const sidebarEmptyHint = el("div", { class: "sidebarEmptyHint muted", text: "No sessions yet" });
  const activeSession = previewSession("appearance-preview-active", { busy: true, subagents_running: 2 });
  const idleSession = previewSession("appearance-preview-idle", {
    cwd: "/work/notes",
    updated_ts: 1_724_999_400,
    model: "claude-sonnet-4-5",
    agent_backend: "cc",
    reasoning_effort: "medium",
  });
  const fixtures = Object.freeze([activeSession, idleSession]);
  sessionState.set("selected", activeSession.session_id);

  const sessionsController = CodoxearSessions.createSessionsController({
    sessionsWrap,
    sidebarEmptyHint,
    sessionState,
    el,
    iconSvg,
    sidebarRenderSignature: (entries, state) => JSON.stringify({ entries, state }),
    sidebarSessionEntries: previewSessionEntries,
    sessionDisplayName: (session) => session.session_id === activeSession.session_id ? "Implement preview" : "Review notes",
    sessionLaunchFailed: () => false,
    sessionLaunchPending: () => false,
    redactedLaunchErrorText: () => "",
    fmtRelativeAge: (seconds) => seconds < 120 ? "now" : "10m",
    sidebarEffortCode: (effort) => ({ high: "hi", medium: "med" }[effort] || ""),
    sidebarModelText: (session) => session.model,
    baseName: (path) => String(path).split("/").filter(Boolean).pop() || "",
    sessionIsFast: () => false,
    agentBackendLogoPath: (backend) => `static/logos/${backend === "cc" ? "cc" : "pi"}.svg`,
    agentBackendDisplayName: (backend) => backend === "cc" ? "Claude Code" : "Pi",
    sessionAgentBackend: (session) => session.agent_backend,
    sessionLaunchIcon: () => "terminal",
    sessionLaunchLabel: () => "Terminal session",
    confirmAction: async () => false,
    api: async () => ({}),
    clearDeletedSessionClientState: () => {},
    refreshSessions: async () => {},
    setToast: () => {},
    openEditSession: () => {},
    duplicateSession: async () => {},
    selectSession: (sessionId) => sessionState.set("selected", sessionId),
    setSidebarOpen: () => {},
    now: () => 1_725_000_000_000,
    performanceNow: () => 0,
    consoleError: () => {},
  });
  sessionsController.renderSessions(fixtures, { swipeActions: false });

  const transcript = el("div", { class: "chatInner appearancePreviewTranscript", "aria-label": "Preview conversation" });
  const messageDeps = {
    el,
    chatMarkdownHtmlCached: (text) => chatMarkdownHtmlCached(text, activeSession.session_id),
    upgradeCandidateFileRefs: () => null,
    time24: () => "10:24",
    iconSvg,
    copyToClipboard: async () => {},
    setToast: () => {},
    chatAssistantDedupeKey: () => "",
    setTimeout: () => 0,
    selectedSessionId: activeSession.session_id,
  };
  const assistantRow = CodoxearMessageRows.makeRow(
    { role: "assistant", text: "The preview uses the same cards and message rows as your workspace." },
    { ts: 1_725_000_000, pending: false },
    messageDeps,
  );
  const userRow = CodoxearMessageRows.makeRow(
    { role: "user", text: "Show the current appearance." },
    { ts: 1_725_000_060, pending: false },
    messageDeps,
  );
  transcript.append(assistantRow.row, userRow.row);

  const stateExamples = el("div", { class: "appearancePreviewStates", "aria-label": "Session state examples" }, [
    el("span", { class: "appearancePreviewState" }, [el("span", { class: "stateDot idle" }), el("span", { text: "idle" })]),
    el("span", { class: "appearancePreviewState" }, [el("span", { class: "stateDot busy" }), el("span", { text: "working" })]),
    el("span", { class: "appearancePreviewState" }, [el("span", { class: "stateDot pending" }), el("span", { text: "starting" })]),
  ]);
  const sidebar = el("aside", { class: "sidebar appearancePreviewSidebar" }, [
    el("header", {}, [el("div", { class: "title", text: "Codoxear" })]),
    sessionsWrap,
    stateExamples,
  ]);
  const topbar = el("div", { class: "topbar appearancePreviewTopbar" }, [
    el("div", { class: "pill" }, [el("div", { class: "appearancePreviewThreadTitle", text: "Implement preview" })]),
    el("div", { class: "actions topActions" }, [
      el("button", { class: "icon-btn", type: "button", title: "Files", "aria-label": "Files", html: iconSvg("file") }),
      el("button", { class: "icon-btn", type: "button", title: "Details", "aria-label": "Details", html: iconSvg("info") }),
    ]),
  ]);
  const main = el("div", { class: "main appearancePreviewMain" }, [
    topbar,
    el("div", { class: "chatWrap appearancePreviewChatWrap" }, [el("div", { class: "chat appearancePreviewChat" }, [transcript])]),
    shellComposer.composer,
  ]);
  const canvas = el("div", { class: "appearancePreviewCanvas", "data-appearance": "paper" }, [
    el("div", { class: "appearancePreviewApp" }, [sidebar, main]),
  ]);
  const appearanceChoices = Object.freeze([
    { id: "paper", label: "Paper" },
    { id: "soft-light", label: "Soft Light" },
    { id: "warm-editorial", label: "Warm Editorial" },
    { id: "terminal-night", label: "Terminal Night" },
  ]);
  const appearanceButtons = new Map(appearanceChoices.map(({ id, label }) => [
    id,
    el("button", { class: `choiceChip${id === "paper" ? " active" : ""}`, type: "button", text: label, "aria-pressed": id === "paper" ? "true" : "false" }),
  ]));
  const preview = el("section", { class: "appearancePreview", "aria-label": "Appearance preview" }, [
    el("div", { class: "appearancePreviewHeading" }, [
      el("span", { class: "fieldLabel", text: "Appearance preview" }),
      el("span", { class: "fieldHint", text: "Changes stay in this preview." }),
    ]),
    el("div", { class: "choiceChips appearancePreviewChoices", role: "group", "aria-label": "Preview appearance" }, Array.from(appearanceButtons.values())),
    canvas,
  ]);

  function setAppearance(appearance) {
    const next = appearanceChoices.some((choice) => choice.id === appearance) ? appearance : "paper";
    canvas.dataset.appearance = next;
    for (const [id, button] of appearanceButtons) {
      const active = id === next;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", active ? "true" : "false");
    }
    return next;
  }

  for (const [id, button] of appearanceButtons) button.onclick = () => setAppearance(id);

  return Object.freeze({
    element: preview,
    setAppearance,
    dispose() {
      for (const button of appearanceButtons.values()) button.onclick = null;
      sessionsController.dispose();
    },
  });
}

export { createAppearancePreview };
