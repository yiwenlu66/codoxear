/* Update settings section: check, apply, auto-update toggle.
 *
 * Owns a <section> mounted into the Settings dialog body. It follows the
 * widget rule: this module creates the section DOM, renders it, and owns its
 * own state — the dialog only mounts the node and calls activate/deactivate.
 *
 * A check runs once per activation (settings opened); the section never polls
 * for updates while it is merely visible. An applied update is different: its
 * restart poll belongs to the update transaction, not to this section's
 * visibility, because the reload it waits for is what picks up the new assets.
 * Closing the dialog therefore does not cancel it — a canceled poll would
 * leave the page running old JavaScript against a new server.
 *
 * States: idle | checking | uptodate | available | not-configured | updating |
 * failed | check-failed. The last one is the honest answer to a check that
 * could not reach the server: unknown is not the same as up to date.
 */

const UPDATE_POLL_INTERVAL_MS = 2000;
// A deploy script has to finish an update command, a reinstall, and a restart
// before the new process answers; a 60-second budget would call a healthy slow
// update failed. Two minutes of grace, then the section reports the failure it
// can actually observe.
const UPDATE_POLL_TIMEOUT_MS = 2 * 60 * 1000;

const HINT_IDLE = "Live sessions are not affected. The page reloads when ready.";
const HINT_UPDATING = "The server is restarting. This page will reload automatically.";
const AUTO_HINT = "Check for updates when this page loads. Apply automatically if an update command is configured.";
const AUTO_HINT_NO_COMMAND = "Requires an update command. Follow the setup above.";
const SETUP_PROMPT = "Set up Codoxear auto-updates";
const FAILURE_PROMPT = "Codoxear update failed. Diagnose and fix \u2014 check ~/.local/share/codoxear/update.log";

function requireFunction(value, name) {
  if (typeof value !== "function") throw new TypeError(`update section dependency missing: ${name}`);
  return value;
}

function createUpdateSection(options = {}) {
  if (!options || typeof options !== "object") throw new TypeError("update section dependency missing: options");
  const el = requireFunction(options.el, "el");
  const api = requireFunction(options.api, "api");
  const setToast = requireFunction(options.setToast, "setToast");
  const copyToClipboard = requireFunction(options.copyToClipboard, "copyToClipboard");
  const setTimeoutFn = requireFunction(options.setTimeout, "setTimeout");
  const clearTimeoutFn = requireFunction(options.clearTimeout, "clearTimeout");
  const windowTarget = options.windowTarget;
  if (!windowTarget || typeof windowTarget !== "object") throw new TypeError("update section dependency missing: windowTarget");
  const onUpdateAvailable = typeof options.onUpdateAvailable === "function" ? options.onUpdateAvailable : () => {};

  // --- DOM ---
  const statusLine = el("div", { class: "updateStatusLine", role: "status", "aria-live": "polite" });
  const errorLine = el("div", { class: "updateErrorLine", role: "alert", style: "display:none" });

  const updateBtn = el("button", { class: "updateApplyBtn primary", type: "button", text: "Update now", style: "display:none" });
  const retryBtn = el("button", { class: "updateApplyBtn primary", type: "button", text: "Retry", style: "display:none" });
  const btnHint = el("div", { class: "fieldHint updateActionHint", text: HINT_IDLE, style: "display:none" });

  // Setup guidance: the prompt is a real button, not a clickable div, so
  // keyboard users and hint mode can reach it (a div with a click listener is
  // invisible to both). The label and the copied text are the same string.
  function promptButton(prompt) {
    const box = el("button", { class: "updatePromptBox", type: "button", title: "Click to copy" }, [
      el("span", { class: "updatePromptText", text: prompt }),
      el("span", { class: "updateCopyIcon", "aria-hidden": "true", text: "\u2398" }),
    ]);
    box.addEventListener("click", () => copyPrompt(prompt));
    return box;
  }

  function stepRow(number, text) {
    // The ordinal stays readable to assistive tech: it is part of the
    // instruction ("1", "2"), and the leading space keeps the text nodes from
    // running together. The circle is the visual form of the same number.
    return el("p", { class: "updateStep" }, [
      el("span", { class: "updateStepNum", text: String(number) }),
      el("span", { class: "updateStepText", text: ` ${text}` }),
    ]);
  }

  const setupPromptBox = promptButton(SETUP_PROMPT);
  const setupBox = el("div", { class: "updateSetupBox", style: "display:none" }, [
    el("p", { class: "updateSetupLead", text: "Updates require a configured update command for your environment." }),
    stepRow(1, "Open a session in the Codoxear installation directory"),
    stepRow(2, "Send this prompt:"),
    setupPromptBox,
  ]);

  const failPromptBox = promptButton(FAILURE_PROMPT);
  const failBox = el("div", { class: "updateSetupBox", style: "display:none" }, [
    el("p", { class: "updateSetupLead", text: "The update command exited with an error. To fix this:" }),
    stepRow(1, "Open a session in the Codoxear installation directory"),
    stepRow(2, "Send this prompt:"),
    failPromptBox,
  ]);

  // Auto-update checkbox: the app's one checkbox component (.checkField); the
  // glyph is its CSS mask, never a second box built here.
  const autoCheck = el("input", { type: "checkbox", id: "updateAutoCheck" });
  const autoLabel = el("label", { class: "checkField updateAutoField" }, [autoCheck, el("span", { text: "Update automatically" })]);
  const autoHint = el("div", { class: "fieldHint updateAutoHint", text: AUTO_HINT });

  const section = el("section", { class: "settingsSection", id: "updateSettingsSection", "aria-labelledby": "updateSettingsHeading" }, [
    el("h3", { class: "settingsSectionTitle", id: "updateSettingsHeading", text: "Updates" }),
    statusLine,
    errorLine,
    updateBtn,
    retryBtn,
    btnHint,
    setupBox,
    failBox,
    autoLabel,
    autoHint,
  ]);

  // --- State ---
  let currentState = "idle"; // idle | checking | uptodate | available | not-configured | updating | failed | check-failed
  let lastCheck = null;
  let pollTimer = null;
  let active = false;

  function behindCount(data) {
    const raw = Number(data && data.behind_count);
    return Number.isFinite(raw) && raw > 0 ? Math.floor(raw) : 0;
  }

  function updateCountText(count) {
    // A check can report an update without a usable count (a backend that only
    // knows "behind"); never render "0 updates available" for that.
    if (!(count > 0)) return "Update available";
    return `${count} update${count === 1 ? "" : "s"} available`;
  }

  function lastStatusState(data) {
    return data && data.last_status && typeof data.last_status === "object" ? data.last_status.state : null;
  }

  function render(state, data, errorText = "") {
    currentState = state;
    // Every state starts from one clean baseline: a previous state's class,
    // hint text, or hidden control must never leak into the next one.
    statusLine.style.display = "none";
    statusLine.className = "updateStatusLine";
    statusLine.textContent = "";
    errorLine.style.display = "none";
    errorLine.textContent = "";
    updateBtn.style.display = "none";
    retryBtn.style.display = "none";
    btnHint.style.display = "none";
    btnHint.textContent = HINT_IDLE;
    setupBox.style.display = "none";
    failBox.style.display = "none";
    autoLabel.style.display = "";
    autoHint.style.display = "";
    autoHint.textContent = AUTO_HINT;
    autoCheck.disabled = false;

    if (state === "checking") {
      statusLine.innerHTML = '<span class="updateSpinner" aria-hidden="true"></span>Checking for updates\u2026';
      statusLine.style.display = "";
    } else if (state === "uptodate") {
      statusLine.textContent = "Up to date";
      statusLine.className = "updateStatusLine muted";
      statusLine.style.display = "";
    } else if (state === "check-failed") {
      statusLine.textContent = "Could not check for updates";
      statusLine.className = "updateStatusLine muted";
      statusLine.style.display = "";
    } else if (state === "available") {
      statusLine.innerHTML = `<span class="accent">${updateCountText(behindCount(data))}</span>`;
      statusLine.style.display = "";
      updateBtn.style.display = "";
      btnHint.style.display = "";
    } else if (state === "not-configured") {
      statusLine.innerHTML = `<span class="accent">${updateCountText(behindCount(data))}</span>`;
      statusLine.style.display = "";
      setupBox.style.display = "";
      autoCheck.disabled = true;
      autoHint.textContent = AUTO_HINT_NO_COMMAND;
    } else if (state === "updating") {
      statusLine.innerHTML = '<span class="updateSpinner" aria-hidden="true"></span>Updating\u2026';
      statusLine.style.display = "";
      btnHint.textContent = HINT_UPDATING;
      btnHint.style.display = "";
      autoLabel.style.display = "none";
      autoHint.style.display = "none";
    } else if (state === "failed") {
      // A failure that outlived the check that found the update has no count
      // to report; the error line alone then carries the state.
      if (behindCount(data) > 0) {
        statusLine.innerHTML = `<span class="accent">${updateCountText(behindCount(data))}</span>`;
        statusLine.style.display = "";
      }
      errorLine.textContent = errorText || "Update failed \u2014 the previous version is still running.";
      errorLine.style.display = "";
      failBox.style.display = "";
      retryBtn.style.display = "";
    }
  }

  // --- API ---
  async function checkForUpdates() {
    render("checking", lastCheck);
    let resp;
    try {
      resp = await api("/api/update/check");
    } catch (e) {
      // Unknown is not up to date: say so instead of claiming a result the
      // server never gave.
      render("check-failed", null);
      return;
    }
    lastCheck = resp;
    autoCheck.checked = !!(resp && resp.auto_update);
    const status = lastStatusState(resp);
    if (status === "running") {
      render("updating", resp);
      startUpdatePoll(resp && resp.latest_commit);
      return;
    }
    if (status === "failed") {
      render("failed", resp);
      onUpdateAvailable(true);
      return;
    }
    // The check itself can fail (no remote branch, unreadable commits). That is
    // an unknown position, not "up to date" — most often it means this checkout
    // has no upstream to compare against.
    if (!resp || resp.error) {
      render("check-failed", null);
      return;
    }
    if (!resp.available) {
      render("uptodate", resp);
      onUpdateAvailable(false);
      return;
    }
    onUpdateAvailable(true);
    if (!resp.update_command_configured) {
      render("not-configured", resp);
      return;
    }
    render("available", resp);
    if (resp.auto_update) {
      await applyUpdate();
    }
  }

  async function applyUpdate() {
    if (currentState === "updating") return;
    render("updating", lastCheck);
    let resp;
    try {
      resp = await api("/api/update/apply", { method: "POST", body: {} });
    } catch (e) {
      // 409 means an update is already in flight (another client, or a second
      // click): this browser is joining that transaction, not failing one.
      if (e && e.status === 409) {
        render("updating", lastCheck);
        startUpdatePoll(lastCheck && lastCheck.latest_commit);
        return;
      }
      // The apply request was refused: the old version is still serving.
      render("failed", lastCheck);
      onUpdateAvailable(true);
      const detail = e && e.message ? `: ${e.message}` : "";
      setToast(`Update failed to start${detail}`);
      return;
    }
    startUpdatePoll(resp && resp.target_commit);
  }

  function updateLanded(resp, targetCommit) {
    if (!resp) return false;
    // A status of "running" is the transaction saying it is still in flight.
    // The checkout can already show no pending update while the update command
    // is still installing or restarting, and a reload in that window would load
    // one version's assets against the other version's server. The runner
    // writes "done" only after the command (and its restart) finishes.
    if (lastStatusState(resp) === "running") return false;
    // Otherwise either proof is enough: the exact commit the update targeted,
    // or a server with nothing left to apply.
    if (targetCommit && resp.current_commit === targetCommit) return true;
    return !resp.available;
  }

  // One in-flight request at a time: a fixed interval would overlap checks
  // while the restarting server stalls a request, and the answers would then
  // arrive out of order. The deadline starts when the poll starts, so a long
  // hang cannot extend the budget.
  function startUpdatePoll(targetCommit) {
    clearPollTimer();
    const deadline = Date.now() + UPDATE_POLL_TIMEOUT_MS;
    const tick = async () => {
      pollTimer = null;
      let resp = null;
      try {
        resp = await api("/api/update/check");
      } catch (e) {
        resp = null;
      }
      if (resp) {
        if (lastStatusState(resp) === "failed") {
          render("failed", resp);
          onUpdateAvailable(true);
          return;
        }
        if (updateLanded(resp, targetCommit)) {
          windowTarget.location.reload();
          return;
        }
      }
      if (Date.now() >= deadline) {
        // The restart never became observable. That is not proof the update
        // failed, so say what was observed instead of asserting a rollback.
        render("failed", lastCheck, "The update did not finish in time \u2014 check ~/.local/share/codoxear/update.log.");
        onUpdateAvailable(true);
        setToast("Update did not finish in time");
        return;
      }
      pollTimer = setTimeoutFn(tick, UPDATE_POLL_INTERVAL_MS);
    };
    pollTimer = setTimeoutFn(tick, UPDATE_POLL_INTERVAL_MS);
  }

  function clearPollTimer() {
    if (pollTimer !== null) {
      clearTimeoutFn(pollTimer);
      pollTimer = null;
    }
  }

  async function saveAutoUpdate(enabled) {
    try {
      await api("/api/update/settings", { method: "POST", body: { auto_update: enabled } });
    } catch (e) {
      setToast("Failed to save setting");
      autoCheck.checked = !enabled;
    }
  }

  async function copyPrompt(prompt) {
    try {
      await copyToClipboard(prompt);
      setToast("Copied to clipboard");
    } catch (e) {
      setToast("Failed to copy");
    }
  }

  // --- Events ---
  updateBtn.addEventListener("click", () => applyUpdate());
  retryBtn.addEventListener("click", () => applyUpdate());
  autoCheck.addEventListener("change", () => saveAutoUpdate(autoCheck.checked));

  // Initial state goes through the same writer every later state uses, so the
  // section is consistent before its first check (nothing claimed, checkbox
  // alone).
  render("idle", null);

  // --- Lifecycle ---
  function activate() {
    if (active) return;
    active = true;
    checkForUpdates();
  }

  // Visibility only: an in-flight update poll keeps running, because the page
  // still has to reload once the server is back on the new commit.
  function deactivate() {
    active = false;
  }

  function dispose() {
    active = false;
    clearPollTimer();
  }

  // Page-load check: raises the badge only. Applying stays tied to opening
  // Settings, so a reload never restarts the server underneath the reader.
  let pageLoadCheckDone = false;
  async function pageLoadCheck() {
    if (pageLoadCheckDone) return;
    pageLoadCheckDone = true;
    try {
      const resp = await api("/api/update/check");
      lastCheck = resp;
      if (resp && resp.available) onUpdateAvailable(true);
    } catch (e) {
      // Silent: the badge is an affordance, not a status channel. Opening
      // Settings surfaces the check result it can actually observe.
    }
  }

  return Object.freeze({
    section,
    activate,
    deactivate,
    dispose,
    pageLoadCheck,
  });
}

export { createUpdateSection };
