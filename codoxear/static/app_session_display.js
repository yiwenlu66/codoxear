/* Session display authority: status, context pressure, and interrupt visibility. */

function requireFunction(value, name) {
  if (typeof value !== "function") throw new TypeError(`session display dependency missing: ${name}`);
  return value;
}

function requireSessionState(value) {
  if (!value || typeof value.get !== "function" || typeof value.subscribe !== "function") {
    throw new TypeError("session display dependency missing: sessionState");
  }
  return value;
}

function createSessionDisplayController(options = {}) {
  const getSelected = requireFunction(options.getSelected, "getSelected");
  const setToast = requireFunction(options.setToast, "setToast");
  const sessionState = requireSessionState(options.sessionState);
  const { statusChip, interruptBtn, ctxChip, eventBindings } = options;
  if (!statusChip || !interruptBtn || !ctxChip || !eventBindings) throw new TypeError("session display dependency missing: status DOM");
  let lastToken = null;

  function renderStatus() {
    // The topbar chip only carries payload not shown elsewhere: the queued
    // message count. Busy/idle is owned by the sidebar state dot and the
    // interrupt button; the ▸N subagent gauge lives in the sidebar meta line
    // and the transcript idle activity row.
    const queueLen = Math.max(0, Number(sessionState.get("queueLen")) || 0);
    if (queueLen > 0) {
      statusChip.style.display = "inline-flex";
      statusChip.textContent = `Queue ${queueLen}`;
    } else {
      statusChip.style.display = "none";
      statusChip.textContent = "";
    }
    const canInterrupt = Boolean(sessionState.get("running") && getSelected());
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
    sessionState.subscribe("running", renderStatus),
    sessionState.subscribe("queueLen", renderStatus),
    sessionState.subscribe("token", renderContext),
  ];
  renderStatus();
  renderContext();

  eventBindings.on(ctxChip, "click", () => {
    if (!lastToken) return;
    setToast(`ctx ${lastToken.used}/${lastToken.contextWindow} (${lastToken.percent ?? "?"}% left)`);
  });

  function dispose() {
    while (unsubscribers.length) unsubscribers.pop()();
  }

  return Object.freeze({ dispose });
}

export { createSessionDisplayController };
