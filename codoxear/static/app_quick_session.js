import * as CodoxearLaunch from "./app_launch.js";

// Owns the quick-launch controls and transient startup state. The composer
// subscribes to this state; it remains the sole writer of its enabled controls.
function createQuickSessionController(options = {}) {
  const { el, iconSvg, sidebarHeaderActions, emptyActions, emptyCopy, textarea,
    sessionState, sessionCatalog, eventBindings, deselectSession, selectSession,
    spawnSession, restoreDraft, openDialog, closeSidebar, shortcutBlocked, document } = options;
  const quickNewBtn = el("button", { id: "quickNewBtn", type: "button", class: "icon-btn", "data-hint": "r", html: iconSvg("compose") });
  sidebarHeaderActions.insertBefore(quickNewBtn, sidebarHeaderActions.firstChild);
  const quickStartBtn = el("button", { id: "chatEmptyQuickBtn", type: "button", class: "icon-btn text-btn primary", "data-hint": "r", text: "Quick start" });
  emptyActions.insertBefore(quickStartBtn, emptyActions.firstChild);
  const listeners = new Set();
  const idleCopy = emptyCopy.textContent;
  let pending = null;
  let selecting = false;
  let disposed = false;

  const launchChoice = () => CodoxearLaunch.loadLastLaunch(sessionCatalog.get("newSessionDefaults"), sessionCatalog.get("tmuxAvailable"));
  function description(record) {
    return `${CodoxearLaunch.agentBackendDisplayName(record.backend)} · ${record.model} · ${record.reasoningEffort}`;
  }
  function render() {
    const label = `Quick session — ~ · ${description(launchChoice())}`;
    for (const button of [quickNewBtn, quickStartBtn]) {
      button.title = label;
      button.setAttribute("aria-label", label);
      button.disabled = !!pending;
    }
    emptyCopy.textContent = pending ? `Starting ${description(pending.record)} in ~…` : idleCopy;
  }
  function changed() {
    render();
    for (const listener of listeners) listener();
  }
  function fail(run, error) {
    if (pending !== run || disposed) return;
    pending = null;
    changed();
    openDialog({ cwd: "~", statusText: CodoxearLaunch.redactedLaunchErrorText(error || "Start failed."), launchPreset: run.record });
  }
  async function ready(run, sessionId) {
    if (pending !== run || disposed || sessionState.get("selected")) return;
    // Capture before openSession's synchronous loadSessionDraft replaces the
    // textarea. Restore immediately after that local load, before tail I/O.
    const text = String(textarea.value || "");
    selecting = true;
    const selection = selectSession(sessionId);
    selecting = false;
    if (text) restoreDraft(sessionId, text);
    pending = null;
    changed();
    await selection;
  }
  function inspectCatalog() {
    const run = pending;
    if (!run || !run.identity) return;
    const row = (sessionCatalog.get("latestSessions") || []).find((s) =>
      typeof run.identity === "number" ? Number(s.broker_pid) === run.identity : s.launch_id === run.identity);
    if (!row) return;
    if (row.launch_state === "failed") fail(run, row.launch_error || "Start failed.");
    else if (!row.launch_state && !row.lost) void ready(run, row.session_id);
  }
  async function start() {
    if (pending || disposed) return;
    const run = { record: launchChoice(), identity: null };
    pending = run;
    deselectSession();
    changed();
    closeSidebar();
    // All of this stays within the click/keyboard gesture, including enabling
    // the textarea via subscribers, so iOS can open its keyboard.
    textarea.focus({ preventScroll: true });
    let errorText = "Start failed.";
    const r = run.record;
    try {
      const identity = await spawnSession("~", null, null, "", r.providerChoice, r.model,
        r.reasoningEffort, r.fast, r.tmux, (error) => { errorText = error && error.message || errorText; },
        r.backend, (id) => ready(run, id));
      if (pending !== run || disposed) return;
      if (!identity) { fail(run, errorText); return; }
      run.identity = identity;
      inspectCatalog();
    } catch (error) { fail(run, error && error.message); }
  }

  for (const button of [quickNewBtn, quickStartBtn]) {
    eventBindings.on(button, "click", start);
    eventBindings.on(button, "mouseenter", render);
    eventBindings.on(button, "focus", render);
  }
  eventBindings.on(document, "keydown", (event) => {
    if (event.defaultPrevented || event.isComposing || event.ctrlKey || event.metaKey || event.altKey
        || event.key !== "N" || shortcutBlocked()) return;
    event.preventDefault();
    void start();
  });
  const unsubscribers = [
    sessionState.subscribe("selected", (id) => {
      if (id && pending && !selecting) { pending = null; changed(); }
    }),
    sessionCatalog.subscribe("newSessionDefaults", render),
    sessionCatalog.subscribe("tmuxAvailable", render),
    sessionCatalog.subscribe("latestSessions", inspectCatalog),
  ];
  render();
  return Object.freeze({ start, isStarting: () => !!pending,
    subscribe(listener) { listeners.add(listener); return () => listeners.delete(listener); },
    dispose() { disposed = true; pending = null; listeners.clear(); while (unsubscribers.length) unsubscribers.pop()(); },
  });
}

export { createQuickSessionController };
