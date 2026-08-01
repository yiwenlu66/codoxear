(function () {
  "use strict";

  // Unattended mode popover authority. Owns every piece of unattended-menu
  // state that used to live as app.js locals (menu open/token/session-id,
  // return-focus element, cfg cache, number-input drafts/dirty flags,
  // per-session save timers/in-flight/pending maps) plus the button projection
  // for the unattended control, the async /unattended load with stale
  // open-token/session guards, the debounced per-session save orchestration
  // (snapshot coercion/merge/debounce, in-flight blocking, pending drain, 401
  // auth-loss), the number-input draft preservation + invalid-blur restore, the
  // menu show/hide/toggle positioning + focus behavior, and the button/menu/
  // input + document Escape/click/window resize event handling.
  //
  // Pure helpers (sessionLaunchFailed) come from window.CodoxearSessionHelpers;
  // modal focus helper (restoreModalFocus) comes from window.CodoxearModal.
  // Everything that touches app-level runtime state (selected session, session
  // index, app disposed flag, API, session refresh, auth loss, toasts, event
  // registration, animation frame, timers, document/window targets, optional
  // shell projection callback) is injected through createUnattendedController
  // (options) so the controller has no hidden coupling to app.js globals and
  // can be exercised in a VM with fakes.

  const codoxearSessionHelpers = window.CodoxearSessionHelpers;
  if (
    !codoxearSessionHelpers ||
    typeof codoxearSessionHelpers.sessionLaunchFailed !== "function"
  )
    throw new Error("Codoxear session helpers failed to load");

  const codoxearModal = window.CodoxearModal;
  if (
    !codoxearModal ||
    typeof codoxearModal.restoreModalFocus !== "function"
  )
    throw new Error("Codoxear modal helpers failed to load");

  const sessionLaunchFailed = codoxearSessionHelpers.sessionLaunchFailed;
  const restoreModalFocus = codoxearModal.restoreModalFocus;

  const UNATTENDED_SAVE_DEBOUNCE_MS = 450;

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`unattended controller dependency missing: ${name}`);
    return value;
  }

  function requireNode(value, name) {
    if (!value || typeof value !== "object" || !value.style) throw new TypeError(`unattended controller dependency missing: ${name}`);
    return value;
  }

  function createUnattendedController(options = {}) {
    if (!options || typeof options !== "object") throw new TypeError("unattended controller dependency missing: options");

    // DOM nodes (created and owned by app.js).
    const unattendedBtn = requireNode(options.unattendedBtn, "unattendedBtn");
    const unattendedMenu = requireNode(options.unattendedMenu, "unattendedMenu");
    const enabledEl = options.enabledEl == null ? null : options.enabledEl;
    const cooldownEl = options.cooldownEl == null ? null : options.cooldownEl;
    const remainingEl = options.remainingEl == null ? null : options.remainingEl;
    const requestEl = options.requestEl == null ? null : options.requestEl;

    // App-level runtime state accessors and effects.
    const getSelected = requireFunction(options.getSelected, "getSelected");
    const getSessionInfo = requireFunction(options.getSessionInfo, "getSessionInfo");
    const isAppDisposed = requireFunction(options.isAppDisposed, "isAppDisposed");
    const api = requireFunction(options.api, "api");
    const refreshSessions = requireFunction(options.refreshSessions, "refreshSessions");
    const handleAppAuthLoss = requireFunction(options.handleAppAuthLoss, "handleAppAuthLoss");
    const setToast = requireFunction(options.setToast, "setToast");
    const addAppEvent = requireFunction(options.addAppEvent, "addAppEvent");
    const documentTarget = options.documentTarget || document;
    const windowTarget = options.windowTarget || window;

    const requestFrame = typeof options.requestFrame === "function" ? options.requestFrame : requestAnimationFrame;
    const setTimeoutFn = typeof options.setTimeout === "function" ? options.setTimeout : setTimeout;
    const clearTimeoutFn = typeof options.clearTimeout === "function" ? options.clearTimeout : clearTimeout;
    // Optional callback app.js wires to its full shell button projection
    // (updateUnattendedBtnState). Invoked after an input handler mutates cfg /
    // session state so the app-shell projection (attach/file/send/queue/diag
    // buttons, context bar, etc.) re-runs exactly as it did before extraction.
    const requestShellProjection = typeof options.requestShellProjection === "function" ? options.requestShellProjection : null;

    // Unattended state owned by this controller.
    let unattendedMenuOpen = false;
    let unattendedMenuToken = 0;
    let unattendedMenuSessionId = null;
    let unattendedReturnFocusEl = null;
    let unattendedCfg = { enabled: false, request: "", cooldown_minutes: 5, remaining_injections: 10 };
    let unattendedNumberDraft = { cooldown_minutes: "5", remaining_injections: "10" };
    let unattendedNumberDirty = { cooldown_minutes: false, remaining_injections: false };
    const unattendedSaveTimers = new Map();
    const unattendedSaveInFlight = new Map();
    const unattendedSavePending = new Map();

    function selectedSessionLaunchFailed() {
      const selected = getSelected();
      return sessionLaunchFailed(selected ? getSessionInfo(selected) : null);
    }

    function parseUnattendedDraftInt(name) {
      const raw = String(unattendedNumberDraft[name] ?? "").trim();
      if (!raw) return null;
      const minValue = name === "cooldown_minutes" ? 1 : 0;
      const value = Number.parseInt(raw, 10);
      if (!Number.isInteger(value) || value < minValue) return null;
      return value;
    }

    function syncUnattendedNumberDraftsFromCfg() {
      if (!unattendedNumberDirty.cooldown_minutes) unattendedNumberDraft.cooldown_minutes = String(unattendedCfg.cooldown_minutes);
      if (!unattendedNumberDirty.remaining_injections) unattendedNumberDraft.remaining_injections = String(unattendedCfg.remaining_injections);
    }

    function syncUnattendedNumberInputs() {
      if (cooldownEl) {
        cooldownEl.value = unattendedNumberDirty.cooldown_minutes
          ? unattendedNumberDraft.cooldown_minutes
          : String(unattendedCfg.cooldown_minutes);
      }
      if (remainingEl) {
        remainingEl.value = unattendedNumberDirty.remaining_injections
          ? unattendedNumberDraft.remaining_injections
          : String(unattendedCfg.remaining_injections);
      }
    }

    function setUnattendedControlsDisabled(disabled) {
      const value = Boolean(disabled);
      [enabledEl, cooldownEl, remainingEl, requestEl].forEach((node) => {
        if (node) node.disabled = value;
      });
    }

    function restoreUnattendedNumberDraft(name) {
      unattendedNumberDirty[name] = false;
      unattendedNumberDraft[name] = String(unattendedCfg[name]);
      syncUnattendedNumberInputs();
    }

    function finalizeUnattendedNumberDraft(name) {
      const value = parseUnattendedDraftInt(name);
      if (value === null || value !== unattendedCfg[name]) return;
      unattendedNumberDirty[name] = false;
      unattendedNumberDraft[name] = String(unattendedCfg[name]);
    }

    function validateUnattendedPayload(data) {
      if (!data || typeof data !== "object") throw new Error("invalid unattended response");
      if (typeof data.enabled !== "boolean") throw new Error("invalid unattended.enabled");
      if (typeof data.request !== "string") throw new Error("invalid unattended.request");
      if (!Number.isInteger(data.cooldown_minutes) || data.cooldown_minutes < 1) throw new Error("invalid unattended.cooldown_minutes");
      if (!Number.isInteger(data.remaining_injections) || data.remaining_injections < 0) throw new Error("invalid unattended.remaining_injections");
    }

    async function loadUnattendedCfgForSelected({ sid = getSelected(), openToken = null } = {}) {
      if (!sid) return;
      sid = String(sid);
      const d = await api(`/api/sessions/${sid}/unattended`);
      if (getSelected() !== sid) return;
      if (openToken !== null && (unattendedMenuToken !== openToken || unattendedMenuSessionId !== sid || !unattendedMenuOpen)) return;
      validateUnattendedPayload(d);
      const reconciled = reconcileUnattendedServerPayload(d, sid);
      unattendedCfg = {
        enabled: reconciled.enabled,
        request: reconciled.request,
        cooldown_minutes: reconciled.cooldown_minutes,
        remaining_injections: reconciled.remaining_injections,
      };
      unattendedNumberDirty.cooldown_minutes = false;
      unattendedNumberDirty.remaining_injections = false;
      syncUnattendedNumberDraftsFromCfg();
      if (enabledEl) enabledEl.checked = unattendedCfg.enabled;
      syncUnattendedNumberInputs();
      if (requestEl) requestEl.value = unattendedCfg.request;
    }

    function unattendedSaveSnapshot(patch = {}) {
      const out = {};
      const has = (name) => Object.prototype.hasOwnProperty.call(patch, name);
      if (has("request")) out.request = String(patch.request || "");
      if (has("cooldown_minutes")) out.cooldown_minutes = patch.cooldown_minutes;
      if (has("remaining_injections")) {
        const remaining = Number(patch.remaining_injections);
        out.remaining_injections = remaining;
        if (Number.isFinite(remaining) && remaining <= 0) out.enabled = false;
      }
      if (has("enabled")) {
        const remaining = has("remaining_injections") ? Number(out.remaining_injections) : Number(unattendedCfg.remaining_injections);
        out.enabled = Boolean(patch.enabled) && Number.isFinite(remaining) && remaining > 0;
      }
      return out;
    }

    function unattendedPatchIsLocallyAuthoritative(sid, name) {
      const has = (value) => Boolean(value && typeof value === "object" && Object.prototype.hasOwnProperty.call(value, name));
      return has(unattendedSavePending.get(sid)) || has(unattendedSaveInFlight.get(sid));
    }

    function reconcileUnattendedServerPayload(serverPayload, sid) {
      return {
        ...serverPayload,
        ...(unattendedSaveInFlight.get(sid) || {}),
        ...(unattendedSavePending.get(sid) || {}),
      };
    }

    function applySavedUnattendedCfg(saved, sid) {
      if (getSelected() !== sid) return;
      if (unattendedMenuOpen && unattendedMenuSessionId !== sid) return;
      unattendedCfg = {
        enabled: saved.enabled,
        request: saved.request,
        cooldown_minutes: saved.cooldown_minutes,
        remaining_injections: saved.remaining_injections,
      };
      const s = getSessionInfo(sid);
      if (s) {
        s.unattended_enabled = Boolean(saved.enabled);
        s.unattended_cooldown_minutes = saved.cooldown_minutes;
        s.unattended_remaining_injections = saved.remaining_injections;
      }
      finalizeUnattendedNumberDraft("cooldown_minutes");
      finalizeUnattendedNumberDraft("remaining_injections");
      syncUnattendedNumberDraftsFromCfg();
      syncUnattendedNumberInputs();
      if (enabledEl) enabledEl.checked = Boolean(saved.enabled);
      if (requestEl) requestEl.value = String(saved.request || "");
    }

    async function flushUnattendedSave(sid) {
      if (!sid || isAppDisposed() || unattendedSaveInFlight.get(sid)) return;
      const snapshot = unattendedSavePending.get(sid);
      if (!snapshot) return;
      unattendedSavePending.delete(sid);
      unattendedSaveInFlight.set(sid, snapshot);
      try {
        const saved = await api(`/api/sessions/${sid}/unattended`, {
          method: "POST",
          body: snapshot,
        });
        validateUnattendedPayload(saved);
        if (isAppDisposed()) return;
        if (!unattendedSavePending.has(sid)) applySavedUnattendedCfg(saved, sid);
        await refreshSessions();
      } catch (e) {
        if (e && e.status === 401) {
          handleAppAuthLoss();
          return;
        }
        console.error("save unattended mode failed", e);
        if (!isAppDisposed() && getSelected() === sid) setToast(`unattended save error: ${e && e.message ? e.message : "unknown error"}`);
      } finally {
        unattendedSaveInFlight.delete(sid);
        if (!isAppDisposed() && unattendedSavePending.has(sid)) void flushUnattendedSave(sid);
        else if (!isAppDisposed() && getSelected() === sid) {
          // Mirror the pre-extraction finally, which called app.js
          // updateUnattendedBtnState (full shell projection). When app.js wires
          // requestShellProjection that re-runs the whole shell projection
          // (including syncButtonState); otherwise project the unattended
          // control directly so the button reflects the just-applied cfg.
          if (requestShellProjection) requestShellProjection();
          else projectButtonState();
        }
      }
    }

    function scheduleUnattendedSave(patch = {}) {
      const sid = getSelected();
      if (!sid) return;
      const snapshot = unattendedSaveSnapshot(patch);
      if (!Object.keys(snapshot).length) return;
      unattendedSavePending.set(sid, { ...(unattendedSavePending.get(sid) || {}), ...snapshot });
      const existing = unattendedSaveTimers.get(sid);
      if (existing) clearTimeoutFn(existing);
      const timer = setTimeoutFn(() => {
        unattendedSaveTimers.delete(sid);
        void flushUnattendedSave(sid);
      }, UNATTENDED_SAVE_DEBOUNCE_MS);
      unattendedSaveTimers.set(sid, timer);
    }

    // Unattended-specific button + cfg/input projection. This is the body that
    // used to live inside app.js updateUnattendedBtnState for the unattended
    // control only; the app-shell projection (attach/file/send/queue/diag,
    // context bar, chat nav) stays in app.js and calls syncButtonState().
    function projectButtonState() {
      const selected = getSelected();
      const s = selected ? getSessionInfo(selected) : null;
      // The session-list poll is server truth, except for fields whose user
      // edit has not crossed the debounced-save commit boundary yet.  Keep the
      // local enabled edit authoritative while queued/in flight so an older
      // poll cannot visibly undo it and then re-apply it after the POST.
      const enabledLocallyAuthoritative = Boolean(selected && unattendedPatchIsLocallyAuthoritative(selected, "enabled"));
      const on = enabledLocallyAuthoritative ? Boolean(unattendedCfg.enabled) : Boolean(s && s.unattended_enabled);
      const unattendedBlocked = Boolean(selected && sessionLaunchFailed(s));
      const unattendedLabel = !selected ? "Select a session for unattended mode" : unattendedBlocked ? "Failed launch has no unattended mode" : "Unattended mode";
      unattendedBtn.disabled = !selected || unattendedBlocked;
      unattendedBtn.title = unattendedLabel;
      unattendedBtn.setAttribute("aria-label", unattendedLabel);
      unattendedBtn.classList.toggle("active", Boolean(selected && on));
      if (
        selected &&
        s &&
        !unattendedNumberDirty.cooldown_minutes &&
        !unattendedPatchIsLocallyAuthoritative(selected, "cooldown_minutes") &&
        Number.isInteger(s.unattended_cooldown_minutes) &&
        s.unattended_cooldown_minutes >= 1
      ) {
        unattendedCfg.cooldown_minutes = s.unattended_cooldown_minutes;
      }
      if (
        selected &&
        s &&
        !unattendedNumberDirty.remaining_injections &&
        !unattendedPatchIsLocallyAuthoritative(selected, "remaining_injections") &&
        Number.isInteger(s.unattended_remaining_injections) &&
        s.unattended_remaining_injections >= 0
      ) {
        unattendedCfg.remaining_injections = s.unattended_remaining_injections;
      }
      if (
        selected &&
        s &&
        typeof s.unattended_request === "string" &&
        !unattendedPatchIsLocallyAuthoritative(selected, "request") &&
        (!unattendedMenuOpen || unattendedMenuSessionId !== selected)
      ) {
        unattendedCfg.request = s.unattended_request;
      }
      syncUnattendedNumberDraftsFromCfg();
      if (unattendedMenuOpen) {
        syncUnattendedNumberInputs();
        if (enabledEl) enabledEl.checked = Boolean(selected && on);
      }
      if (unattendedMenuOpen && (!selected || unattendedMenuSessionId !== selected)) hideUnattendedMenu();
    }

    function setUnattendedMenuExpanded(open) {
      unattendedMenuOpen = Boolean(open);
      unattendedMenu.style.display = unattendedMenuOpen ? "block" : "none";
      unattendedBtn.setAttribute("aria-expanded", unattendedMenuOpen ? "true" : "false");
    }

    function restoreUnattendedFocus() {
      const target = unattendedReturnFocusEl;
      unattendedReturnFocusEl = null;
      restoreModalFocus(target, () => unattendedMenuOpen, requestFrame);
    }

    function focusUnattendedInitialControl() {
      requestFrame(() => {
        if (!unattendedMenuOpen) return;
        const target = enabledEl || unattendedMenu;
        try {
          target.focus({ preventScroll: true });
        } catch {}
      });
    }

    function hideUnattendedMenu({ restoreFocus = false } = {}) {
      const wasOpen = unattendedMenuOpen;
      unattendedMenuToken += 1;
      unattendedMenuSessionId = null;
      setUnattendedMenuExpanded(false);
      if (restoreFocus && wasOpen) restoreUnattendedFocus();
      else unattendedReturnFocusEl = null;
    }

    async function showUnattendedMenu({ opener = null } = {}) {
      const selected = getSelected();
      if (!selected) return;
      if (selectedSessionLaunchFailed()) {
        setToast("failed launch has no unattended mode");
        return;
      }
      const sid = selected;
      const openToken = unattendedMenuToken + 1;
      unattendedMenuToken = openToken;
      unattendedMenuSessionId = sid;
      unattendedReturnFocusEl = opener instanceof HTMLElement ? opener : documentTarget.activeElement instanceof HTMLElement ? documentTarget.activeElement : null;
      setUnattendedControlsDisabled(true);
      setUnattendedMenuExpanded(true);
      const rect = unattendedBtn.getBoundingClientRect();
      const winHeight = windowTarget.innerHeight;
      const winWidth = windowTarget.innerWidth;
      const top = Math.min(winHeight - 12, rect.bottom + 8);
      unattendedMenu.style.top = `${top}px`;
      unattendedMenu.style.left = "12px";
      unattendedMenu.style.right = "auto";
      const w = unattendedMenu.offsetWidth || 320;
      const left = Math.max(12, Math.min(winWidth - 12 - w, rect.right - w));
      unattendedMenu.style.left = `${left}px`;
      try {
        await loadUnattendedCfgForSelected({ sid, openToken });
        if (unattendedMenuOpen && unattendedMenuToken === openToken && unattendedMenuSessionId === sid && getSelected() === sid) {
          setUnattendedControlsDisabled(false);
          focusUnattendedInitialControl();
        }
      } catch (e) {
        if (unattendedMenuToken !== openToken || unattendedMenuSessionId !== sid || getSelected() !== sid) return;
        console.error("load unattended mode failed", e);
        setToast(`unattended load error: ${e && e.message ? e.message : "unknown error"}`);
        setUnattendedControlsDisabled(false);
        hideUnattendedMenu({ restoreFocus: true });
      }
    }

    function toggleUnattendedMenu({ opener = null } = {}) {
      if (unattendedMenuOpen) hideUnattendedMenu({ restoreFocus: true });
      else showUnattendedMenu({ opener });
    }

    // --- event handlers (button/menu/input/document/window) ---

    unattendedBtn.onclick = (e) => {
      e.preventDefault();
      e.stopPropagation();
      toggleUnattendedMenu({ opener: e.currentTarget });
    };
    unattendedMenu.onclick = (e) => e.stopPropagation();
    const onUnattendedKeydown = (e) => {
      if (e.key !== "Escape" || !unattendedMenuOpen) return;
      e.preventDefault();
      e.stopPropagation();
      hideUnattendedMenu({ restoreFocus: true });
    };
    const onDocClick = () => {
      if (unattendedMenuOpen) hideUnattendedMenu();
    };
    const onResize = () => {
      if (unattendedMenuOpen) hideUnattendedMenu();
    };
    addAppEvent(documentTarget, "keydown", onUnattendedKeydown, true);
    addAppEvent(documentTarget, "click", onDocClick);
    addAppEvent(windowTarget, "resize", onResize);

    if (enabledEl) {
      enabledEl.onchange = (e) => {
        const selected = getSelected();
        if (!selected) return;
        const requested = Boolean(e.target.checked);
        unattendedCfg.enabled = requested && Number(unattendedCfg.remaining_injections) > 0;
        if (requested && !unattendedCfg.enabled) setToast("increase injections before enabling unattended mode");
        e.target.checked = unattendedCfg.enabled;
        const s = getSessionInfo(selected);
        if (s) {
          s.unattended_enabled = unattendedCfg.enabled;
        }
        if (requestShellProjection) requestShellProjection();
        else projectButtonState();
        scheduleUnattendedSave({ enabled: unattendedCfg.enabled });
      };
    }
    if (cooldownEl) {
      cooldownEl.oninput = (e) => {
        const selected = getSelected();
        if (!selected) return;
        unattendedNumberDraft.cooldown_minutes = String(e.target.value ?? "");
        unattendedNumberDirty.cooldown_minutes = true;
        const value = parseUnattendedDraftInt("cooldown_minutes");
        if (value === null) return;
        unattendedCfg.cooldown_minutes = value;
        scheduleUnattendedSave({ cooldown_minutes: value });
      };
      cooldownEl.onblur = () => {
        if (parseUnattendedDraftInt("cooldown_minutes") !== null) return;
        restoreUnattendedNumberDraft("cooldown_minutes");
      };
    }
    if (remainingEl) {
      remainingEl.oninput = (e) => {
        const selected = getSelected();
        if (!selected) return;
        unattendedNumberDraft.remaining_injections = String(e.target.value ?? "");
        unattendedNumberDirty.remaining_injections = true;
        const value = parseUnattendedDraftInt("remaining_injections");
        if (value === null) return;
        unattendedCfg.remaining_injections = value;
        const s = getSessionInfo(selected);
        if (s) {
          s.unattended_remaining_injections = value;
          if (value <= 0) {
            unattendedCfg.enabled = false;
            if (enabledEl) enabledEl.checked = false;
            s.unattended_enabled = false;
          }
        }
        if (requestShellProjection) requestShellProjection();
        else projectButtonState();
        scheduleUnattendedSave({ remaining_injections: value, ...(value <= 0 ? { enabled: false } : {}) });
      };
      remainingEl.onblur = () => {
        if (parseUnattendedDraftInt("remaining_injections") !== null) return;
        restoreUnattendedNumberDraft("remaining_injections");
      };
    }
    if (requestEl) {
      requestEl.oninput = (e) => {
        const selected = getSelected();
        if (!selected) return;
        unattendedCfg.request = String(e.target.value ?? "");
        scheduleUnattendedSave({ request: unattendedCfg.request });
      };
    }

    function syncButtonState() {
      projectButtonState();
    }

    function isOpen() {
      return unattendedMenuOpen;
    }

    function menuSessionId() {
      return unattendedMenuSessionId;
    }

    function dispose() {
      unattendedSaveTimers.forEach((timer) => clearTimeoutFn(timer));
      unattendedSaveTimers.clear();
      unattendedSavePending.clear();
      unattendedSaveInFlight.clear();
      unattendedMenuToken += 1;
      unattendedMenuSessionId = null;
      unattendedMenuOpen = false;
      unattendedReturnFocusEl = null;
      if (unattendedMenu.style) unattendedMenu.style.display = "none";
      unattendedBtn.setAttribute("aria-expanded", "false");
    }

    return Object.freeze({
      syncButtonState,
      show: showUnattendedMenu,
      hide: hideUnattendedMenu,
      toggle: toggleUnattendedMenu,
      isOpen,
      menuSessionId,
      dispose,
    });
  }

  window.CodoxearUnattended = Object.freeze({ createUnattendedController });
})();
