/* Topbar widget authority: elements, store projections, and interactions. */

function requireFunction(value, name) {
  if (typeof value !== "function") throw new TypeError(`topbar dependency missing: ${name}`);
  return value;
}

function requireNode(value, name) {
  if (!value || typeof value.appendChild !== "function") {
    throw new TypeError(`topbar dependency missing: ${name}`);
  }
  return value;
}

function requireSessionState(value) {
  if (!value || typeof value.get !== "function" || typeof value.subscribe !== "function") {
    throw new TypeError("topbar dependency missing: sessionState");
  }
  return value;
}

function createTopbarController(options = {}) {
  const el = requireFunction(options.el, "el");
  const iconSvg = requireFunction(options.iconSvg, "iconSvg");
  const setToast = requireFunction(options.setToast, "setToast");
  const onInterrupt = requireFunction(options.onInterrupt, "onInterrupt");
  const sessionState = requireSessionState(options.sessionState);
  const topMeta = requireNode(options.topMeta, "topMeta");
  const topActions = requireNode(options.topActions, "topActions");
  const eventBindings = options.eventBindings;
  if (!eventBindings || typeof eventBindings.on !== "function") {
    throw new TypeError("topbar dependency missing: eventBindings");
  }

  // The shell owns the title row and action layout. This controller owns every
  // topbar status widget that occupies those layout slots.
  const ctxChip = el("button", {
    class: "status-chip",
    id: "ctxChip",
    text: "",
    type: "button",
    "aria-label": "Context usage details",
    "data-hint": "y",
  });
  ctxChip.style.display = "none";
  ctxChip.disabled = true;
  const interruptBtn = el("button", {
    id: "interruptBtn",
    class: "icon-btn",
    title: "Interrupt (Esc)",
    "aria-label": "Interrupt (Esc)",
    "data-hint": "z",
    type: "button",
    html: iconSvg("stop"),
  });
  interruptBtn.style.display = "none";
  topMeta.appendChild(ctxChip);
  topActions.appendChild(interruptBtn);

  let lastToken = null;

  function renderInterrupt() {
    // The topbar carries no status chip. The queued-message count is owned by
    // the composer queue button badge (count attached to the control that acts
    // on it) and the sidebar session card badge; busy/idle is owned by the
    // sidebar state dot and this interrupt button; the subagent gauge lives in
    // the sidebar meta line and the transcript idle activity row.
    const canInterrupt = Boolean(sessionState.get("running") && sessionState.get("selected"));
    interruptBtn.style.display = canInterrupt ? "inline-flex" : "none";
    interruptBtn.disabled = !canInterrupt;
  }

  function renderContext() {
    const token = sessionState.get("token");
    if (!token || typeof token !== "object") {
      lastToken = null;
      ctxChip.style.display = "none";
      ctxChip.disabled = true;
      ctxChip.textContent = "";
      ctxChip.title = "";
      return;
    }
    const contextWindow = Number(token.context_window);
    const used = Number(token.tokens_in_context);
    const percentRemaining = Number(token.percent_remaining);
    if (!Number.isFinite(contextWindow) || !Number.isFinite(used) || contextWindow <= 0 || used < 0) {
      lastToken = null;
      ctxChip.style.display = "none";
      ctxChip.disabled = true;
      ctxChip.textContent = "";
      ctxChip.title = "";
      return;
    }
    const percent = Number.isFinite(percentRemaining) ? Math.max(0, Math.min(100, Math.round(percentRemaining))) : null;
    const maxInput = Number(token.max_input_tokens);
    const reserved = Number(token.reserved_tokens);
    const effectiveMaxInput = Number.isFinite(maxInput) && maxInput >= 0 ? maxInput : contextWindow;
    const effectiveReserved = Number.isFinite(reserved) && reserved >= 0 ? reserved : Math.max(contextWindow - effectiveMaxInput, 0);
    lastToken = { contextWindow, used, percent, remaining: Math.max(effectiveMaxInput - used, 0), maxInput: effectiveMaxInput, reserved: effectiveReserved };
    ctxChip.style.display = "inline-flex";
    ctxChip.disabled = false;
    ctxChip.textContent = percent === null ? "Ctx" : `Ctx ${percent}%`;
    ctxChip.title = `Context input: ${used}/${effectiveMaxInput} tokens (${effectiveReserved} reserved; window ${contextWindow}).`;
  }

  const unsubscribers = [
    sessionState.subscribe("selected", renderInterrupt),
    sessionState.subscribe("running", renderInterrupt),
    sessionState.subscribe("token", renderContext),
  ];
  renderInterrupt();
  renderContext();

  eventBindings.on(ctxChip, "click", () => {
    if (!lastToken) return;
    setToast(`ctx ${lastToken.used}/${lastToken.contextWindow} (${lastToken.percent ?? "?"}% left)`);
  });
  eventBindings.on(interruptBtn, "click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    void onInterrupt();
  });

  function dispose() {
    while (unsubscribers.length) unsubscribers.pop()();
  }

  return Object.freeze({
    elements: Object.freeze({ ctxChip, interruptBtn }),
    dispose,
  });
}

export { createTopbarController };
