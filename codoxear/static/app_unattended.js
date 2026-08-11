import * as CodoxearModal from "./app_modal.js";
import * as CodoxearSessionHelpers from "./app_session_helpers.js";

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
  // Pure helpers (sessionLaunchFailed) come from CodoxearSessionHelpers;
  // modal focus helper (restoreModalFocus) comes from CodoxearModal.
  // Everything that touches app-level runtime state (selected session, session
  // index, app disposed flag, API, session refresh, auth loss, toasts, event
  // registration, animation frame, timers, document/window targets, optional
  // shell projection callback) is injected through createUnattendedController
  // (options) so the controller has no hidden coupling to app.js globals and
  // can be exercised in a VM with fakes.

  const codoxearSessionHelpers = CodoxearSessionHelpers;
  if (
    !codoxearSessionHelpers ||
    typeof codoxearSessionHelpers.sessionLaunchFailed !== "function"
  )
    throw new Error("Codoxear session helpers failed to load");

  const codoxearModal = CodoxearModal;
  if (
    !codoxearModal ||
    typeof codoxearModal.restoreModalFocus !== "function"
  )
    throw new Error("Codoxear modal helpers failed to load");

  const sessionLaunchFailed = codoxearSessionHelpers.sessionLaunchFailed;
  const restoreModalFocus = codoxearModal.restoreModalFocus;

  const UNATTENDED_SAVE_DEBOUNCE_MS = 450;
  const UNATTENDED_SAVE_RETRY_INITIAL_MS = 1500;
  const UNATTENDED_SAVE_RETRY_MAX_MS = 30000;
  const UNATTENDED_PENDING_STORAGE_KEY = "codexweb.unattended.pending.v1";

  function createUnattendedDom(options = {}) {
    if (!options || typeof options !== "object") throw new TypeError("unattended DOM dependency missing: options");
    const el = requireFunction(options.el, "el");
    const iconSvg = requireFunction(options.iconSvg, "iconSvg");
    const unattendedBtn = options.unattendedBtn || el("button", {
      id: "unattendedBtn",
      class: "icon-btn",
      title: "Unattended mode",
      "aria-label": "Unattended mode",
      "aria-controls": "unattendedMenu",
      "aria-expanded": "false",
      "aria-haspopup": "dialog",
      type: "button",
      html: iconSvg("unattended"),
    });
    unattendedBtn.disabled = true;
    const enabledEl = el("input", { type: "checkbox", id: "unattendedEnabled" });
    const cooldownEl = el("input", {
      id: "unattendedCooldownMinutes",
      type: "number",
      min: "1",
      step: "1",
      inputmode: "numeric",
      "aria-label": "Unattended cooldown time in minutes",
    });
    const remainingEl = el("input", {
      id: "unattendedRemainingInjections",
      type: "number",
      min: "0",
      step: "1",
      inputmode: "numeric",
      "aria-label": "Unattended remaining injections",
    });
    const requestEl = el("textarea", {
      id: "unattendedRequest",
      "aria-label": "Additional request for unattended prompt",
    });
    const unattendedMenu = el("div", {
      id: "unattendedMenu",
      class: "unattendedMenu",
      role: "dialog",
      "aria-label": "Unattended mode settings",
    }, [
      el("div", { class: "row" }, [el("label", {}, [enabledEl, el("span", { text: "Unattended mode" })])]),
      el("div", { class: "unattendedGrid" }, [
        el("div", {}, [el("div", { class: "label", text: "Cooldown time (minutes)" }), cooldownEl]),
        el("div", {}, [el("div", { class: "label", text: "Number of injections" }), remainingEl]),
      ]),
      el("div", { class: "label", text: "Additional request to append (optional; per session)" }),
      requestEl,
    ]);
    return Object.freeze({ unattendedBtn, unattendedMenu, enabledEl, cooldownEl, remainingEl, requestEl });
  }

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

    // DOM nodes (created and owned by app.js through createUnattendedDom).
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
    // The pending patch is browser-owned until the server has acknowledged it.
    // app.js injects its guarded localStorage facade so private-mode/storage
    // failures follow the rest of the browser persistence contract.
    const storageGetItem = requireFunction(options.storageGetItem, "storageGetItem");
    const storageSetItem = requireFunction(options.storageSetItem, "storageSetItem");
    const storageRemoveItem = requireFunction(options.storageRemoveItem, "storageRemoveItem");
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
    const unattendedPersistedPending = new Map();
    const unattendedSaveRetryCounts = new Map();
    const unattendedSaveRetryPaused = new Set();

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

    function validateUnattendedPatch(patch) {
      if (!patch || typeof patch !== "object" || Array.isArray(patch)) throw new Error("invalid persisted unattended patch");
      const names = Object.keys(patch);
      if (names.length !== 1 || names[0] !== "request" || typeof patch.request !== "string") throw new Error("invalid persisted unattended request draft");
    }

    function readPersistedUnattendedPatches() {
      let raw;
      try {
        raw = storageGetItem(UNATTENDED_PENDING_STORAGE_KEY);
      } catch (error) {
        console.error("read unattended pending drafts failed", error);
        return;
      }
      if (!raw) return;
      try {
        const parsed = JSON.parse(raw);
        if (!parsed || typeof parsed !== "object" || parsed.version !== 1 || !parsed.patches || typeof parsed.patches !== "object" || Array.isArray(parsed.patches)) throw new Error("invalid persisted unattended draft envelope");
        for (const [sid, record] of Object.entries(parsed.patches)) {
          if (!sid || !record || typeof record !== "object" || !Number.isInteger(record.revision) || record.revision < 1) throw new Error("invalid persisted unattended draft record");
          validateUnattendedPatch(record.patch);
          const snapshot = { revision: record.revision, patch: { request: record.patch.request } };
          unattendedPersistedPending.set(sid, snapshot);
          unattendedSavePending.set(sid, snapshot);
        }
      } catch (error) {
        // A malformed local entry must be visible in diagnostics but cannot
        // strand every later controller construction behind the same bad JSON.
        console.error("invalid persisted unattended drafts", error);
        try { storageRemoveItem(UNATTENDED_PENDING_STORAGE_KEY); }
        catch (removeError) { console.error("clear invalid unattended pending drafts failed", removeError); }
        unattendedPersistedPending.clear();
        unattendedSavePending.clear();
      }
    }

    function storedUnattendedRequestPatch(sid) {
      try {
        const raw = storageGetItem(UNATTENDED_PENDING_STORAGE_KEY);
        if (!raw) return null;
        const parsed = JSON.parse(raw);
        if (!parsed || typeof parsed !== "object" || parsed.version !== 1 || !parsed.patches || typeof parsed.patches !== "object" || Array.isArray(parsed.patches)) throw new Error("invalid persisted unattended draft envelope");
        const record = parsed.patches[sid];
        if (!record) return null;
        if (!Number.isInteger(record.revision) || record.revision < 1) throw new Error("invalid persisted unattended draft record");
        validateUnattendedPatch(record.patch);
        return { revision: record.revision, patch: { request: record.patch.request }, envelope: parsed };
      } catch (error) {
        console.error("read unattended pending draft acknowledgement failed", error);
        return null;
      }
    }

    function clearAcknowledgedUnattendedRequest(sid, snapshot) {
      const stored = storedUnattendedRequestPatch(sid);
      if (
        !stored
        || stored.revision !== snapshot.revision
        || !Object.prototype.hasOwnProperty.call(snapshot.patch, "request")
        || stored.patch.request !== snapshot.patch.request
      ) return false;
      try {
        delete stored.envelope.patches[sid];
        if (Object.keys(stored.envelope.patches).length) storageSetItem(UNATTENDED_PENDING_STORAGE_KEY, JSON.stringify(stored.envelope));
        else storageRemoveItem(UNATTENDED_PENDING_STORAGE_KEY);
        unattendedPersistedPending.delete(sid);
        return true;
      } catch (error) {
        console.error("clear acknowledged unattended pending draft failed", error);
        if (!isAppDisposed()) setToast(`unattended draft persistence error: ${error && error.message ? error.message : "unknown error"}`);
        return false;
      }
    }

    function persistUnattendedPatches() {
      const patches = {};
      try {
        const raw = storageGetItem(UNATTENDED_PENDING_STORAGE_KEY);
        if (raw) {
          const parsed = JSON.parse(raw);
          if (!parsed || typeof parsed !== "object" || parsed.version !== 1 || !parsed.patches || typeof parsed.patches !== "object" || Array.isArray(parsed.patches)) throw new Error("invalid persisted unattended draft envelope");
          for (const [sid, record] of Object.entries(parsed.patches)) {
            if (!sid || !record || typeof record !== "object" || !Number.isInteger(record.revision) || record.revision < 1) throw new Error("invalid persisted unattended draft record");
            validateUnattendedPatch(record.patch);
            patches[sid] = { revision: record.revision, patch: { request: record.patch.request } };
          }
        }
      } catch (error) {
        console.error("read unattended pending drafts before persist failed", error);
      }
      unattendedPersistedPending.forEach((record, sid) => {
        const existing = patches[sid];
        if (!existing || record.revision >= existing.revision) patches[sid] = { revision: record.revision, patch: { request: record.patch.request } };
      });
      try {
        if (Object.keys(patches).length) storageSetItem(UNATTENDED_PENDING_STORAGE_KEY, JSON.stringify({ version: 1, patches }));
        else storageRemoveItem(UNATTENDED_PENDING_STORAGE_KEY);
      } catch (error) {
        console.error("persist unattended pending drafts failed", error);
        if (!isAppDisposed()) setToast(`unattended draft persistence error: ${error && error.message ? error.message : "unknown error"}`);
      }
    }

    function persistedUnattendedPatch(sid) {
      return unattendedPersistedPending.get(sid) || null;
    }

    function latestUnattendedPatch(sid) {
      return unattendedSavePending.get(sid) || unattendedSaveInFlight.get(sid) || null;
    }

    function applyPersistedUnattendedPatch(sid) {
      const record = persistedUnattendedPatch(sid);
      if (record) unattendedCfg = { ...unattendedCfg, ...record.patch };
    }

    function unattendedPatchIsLocallyAuthoritative(sid, name) {
      const latest = latestUnattendedPatch(sid);
      const persisted = persistedUnattendedPatch(sid);
      return Boolean(
        (latest && Object.prototype.hasOwnProperty.call(latest.patch, name))
        || (persisted && Object.prototype.hasOwnProperty.call(persisted.patch, name)),
      );
    }

    function reconcileUnattendedServerPayload(serverPayload, sid) {
      const inFlight = unattendedSaveInFlight.get(sid);
      const pending = unattendedSavePending.get(sid);
      const persisted = persistedUnattendedPatch(sid);
      return {
        ...serverPayload,
        ...(inFlight ? inFlight.patch : {}),
        ...(pending ? pending.patch : {}),
        ...(persisted ? persisted.patch : {}),
      };
    }

    function retryDelayForUnattendedSave(sid) {
      const failures = (unattendedSaveRetryCounts.get(sid) || 0) + 1;
      unattendedSaveRetryCounts.set(sid, failures);
      return Math.min(UNATTENDED_SAVE_RETRY_INITIAL_MS * (2 ** Math.min(failures - 1, 5)), UNATTENDED_SAVE_RETRY_MAX_MS);
    }

    function scheduleUnattendedFlush(sid, delay) {
      if (!sid || unattendedSaveRetryPaused.has(sid) || unattendedSaveInFlight.has(sid) || !unattendedSavePending.has(sid)) return;
      const existing = unattendedSaveTimers.get(sid);
      if (existing) clearTimeoutFn(existing);
      const timer = setTimeoutFn(() => {
        unattendedSaveTimers.delete(sid);
        void flushUnattendedSave(sid);
      }, delay);
      unattendedSaveTimers.set(sid, timer);
    }

    function resumePersistedUnattendedSave(sid) {
      if (!sid || unattendedSaveRetryPaused.has(sid) || unattendedSaveInFlight.has(sid) || unattendedSaveTimers.has(sid)) return;
      const record = persistedUnattendedPatch(sid);
      if (!record) return;
      unattendedSavePending.set(sid, record);
      scheduleUnattendedFlush(sid, UNATTENDED_SAVE_DEBOUNCE_MS);
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
      if (!sid || isAppDisposed() || unattendedSaveInFlight.has(sid) || unattendedSaveRetryPaused.has(sid)) return;
      const snapshot = unattendedSavePending.get(sid);
      if (!snapshot) return;
      unattendedSavePending.delete(sid);
      unattendedSaveInFlight.set(sid, snapshot);
      let outcome = "success";
      try {
        const saved = await api(`/api/sessions/${sid}/unattended`, {
          method: "POST",
          body: snapshot.patch,
        });
        validateUnattendedPayload(saved);
        if (isAppDisposed()) return;
        // Only the user-authored request is durable. Numeric budget/config is
        // server authority because another tab or unattended injection can
        // change it while this request is in flight. An old acknowledgement
        // can clear a WAL record only when it carried that exact request and
        // no newer request revision has replaced it.
        const persisted = persistedUnattendedPatch(sid);
        if (persisted) clearAcknowledgedUnattendedRequest(sid, snapshot);
        unattendedSaveRetryCounts.delete(sid);
        if (!unattendedSavePending.has(sid) && !persistedUnattendedPatch(sid)) applySavedUnattendedCfg(saved, sid);
        await refreshSessions();
      } catch (e) {
        if (isAppDisposed()) return;
        if (e && e.status === 401) {
          outcome = "auth";
          unattendedSaveRetryPaused.add(sid);
          handleAppAuthLoss();
          return;
        }
        outcome = "retry";
        console.error("save unattended mode failed", e);
        // Keep the in-memory full patch for this tab and the browser-owned
        // request WAL for a replacement controller. A retry is delayed and
        // bounded exponentially rather than recursively spinning on outage.
        const newer = unattendedSavePending.get(sid);
        unattendedSavePending.set(sid, newer || snapshot);
        if (getSelected() === sid) setToast(`unattended save error: ${e && e.message ? e.message : "unknown error"}`);
      } finally {
        unattendedSaveInFlight.delete(sid);
        if (!isAppDisposed()) {
          if (outcome === "retry" && getSelected() === sid) {
            scheduleUnattendedFlush(sid, retryDelayForUnattendedSave(sid));
          } else if (outcome === "success" && unattendedSavePending.has(sid)) {
            // This is a newer local edit that arrived while the prior POST
            // crossed the network boundary; drain it once, without waiting
            // for a stale response to rewrite the controls.
            void flushUnattendedSave(sid);
          }
          if (getSelected() === sid) {
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
    }

    function scheduleUnattendedSave(patch = {}) {
      const sid = getSelected();
      if (!sid) return;
      const patchSnapshot = unattendedSaveSnapshot(patch);
      if (!Object.keys(patchSnapshot).length) return;
      const prior = latestUnattendedPatch(sid);
      const stored = storedUnattendedRequestPatch(sid);
      const priorRevision = Math.max(
        prior ? prior.revision : 0,
        persistedUnattendedPatch(sid) ? persistedUnattendedPatch(sid).revision : 0,
        stored ? stored.revision : 0,
      );
      const snapshot = {
        revision: priorRevision + 1,
        patch: { ...(prior ? prior.patch : {}), ...patchSnapshot },
      };
      unattendedSaveRetryPaused.delete(sid);
      unattendedSavePending.set(sid, snapshot);
      if (Object.prototype.hasOwnProperty.call(snapshot.patch, "request")) {
        unattendedPersistedPending.set(sid, { revision: snapshot.revision, patch: { request: snapshot.patch.request } });
        persistUnattendedPatches();
      }
      scheduleUnattendedFlush(sid, UNATTENDED_SAVE_DEBOUNCE_MS);
    }

    // Browser persistence deliberately contains only the user-authored request
    // text. The server remains sole authority for enabled state and injection
    // budget, which can change in another tab or through an injection.
    readPersistedUnattendedPatches();

    // Unattended-specific button + cfg/input projection. This is the body that
    // used to live inside app.js updateUnattendedBtnState for the unattended
    // control only; the app-shell projection (attach/file/send/queue/diag,
    // context bar, chat nav) stays in app.js and calls syncButtonState().
    function projectButtonState() {
      const selected = getSelected();
      if (selected) {
        applyPersistedUnattendedPatch(selected);
        resumePersistedUnattendedSave(selected);
      }
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
      // Pending request text remains in browser storage until its matching
      // server acknowledgement. Disposal only releases this controller's
      // volatile retry/in-flight bookkeeping.
      unattendedSavePending.clear();
      unattendedSaveInFlight.clear();
      unattendedSaveRetryCounts.clear();
      unattendedSaveRetryPaused.clear();
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

  const unattendedApi = { createUnattendedController };
  Object.defineProperty(unattendedApi, "createUnattendedDom", { value: createUnattendedDom });

export { createUnattendedController, createUnattendedDom };
