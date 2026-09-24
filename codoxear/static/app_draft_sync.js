/* Server-synced composer drafts.
 *
 * localStorage remains the write-through cache (instant UX, offline-safe);
 * the server draft is the cross-device authority. Conflicts resolve by
 * last-writer-wins on the server timestamp — no merging, and simultaneous
 * multi-client editing is explicitly out of scope.
 *
 * Clearing a draft is itself a write: an empty POST stores a server
 * tombstone whose updated_ts takes part in the same last-writer-wins
 * comparison, so a deletion propagates to other clients exactly like an
 * edit. An updated_ts of 0 means the server holds no entry at all (never
 * drafted, or its state was lost) — it is never a deletion signal.
 *
 * The companion key `codexweb.draft.<sid>.server_ts` records the server
 * updated_ts this client's local draft corresponds to; comparing it with the
 * server's updated_ts (directly, or via the session row's draft_updated_ts)
 * decides which side wins.
 *
 * This controller mutates only the composer textarea value (through the
 * injected applyServerDraft seam), the localStorage draft cache, and the
 * companion key. No other rendered state.
 */

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`draft sync controller dependency missing: ${name}`);
    return value;
  }

  function createDraftSyncController(options = {}) {
    if (!options || typeof options !== "object") throw new TypeError("draft sync controller dependency missing: options");
    const sessionState = options.sessionState;
    if (!sessionState || typeof sessionState.get !== "function" || typeof sessionState.subscribe !== "function") throw new TypeError("draft sync dependency missing: sessionState");
    const sessionCatalog = options.sessionCatalog;
    if (!sessionCatalog || typeof sessionCatalog.get !== "function" || typeof sessionCatalog.subscribe !== "function") throw new TypeError("draft sync dependency missing: sessionCatalog");
    const api = requireFunction(options.api, "api");
    const storageGetItem = requireFunction(options.storageGetItem, "storageGetItem");
    const storageSetItem = requireFunction(options.storageSetItem, "storageSetItem");
    const storageRemoveItem = requireFunction(options.storageRemoveItem, "storageRemoveItem");
    const getComposerText = requireFunction(options.getComposerText, "getComposerText");
    const applyServerDraft = requireFunction(options.applyServerDraft, "applyServerDraft");
    const resolveAppUrl = requireFunction(options.resolveAppUrl, "resolveAppUrl");
    const windowTarget = options.windowTarget && typeof options.windowTarget.addEventListener === "function" ? options.windowTarget : null;
    const setTimeoutFn = typeof options.setTimeout === "function" ? options.setTimeout : (callback, delay) => setTimeout(callback, delay);
    const clearTimeoutFn = typeof options.clearTimeout === "function" ? options.clearTimeout : (handle) => clearTimeout(handle);
    const fetchFn = typeof options.fetch === "function" ? options.fetch : typeof fetch === "function" ? fetch : null;
    const consoleError = typeof options.consoleError === "function" ? options.consoleError : () => {};
    const debounceMs = Number.isFinite(options.debounceMs) ? options.debounceMs : 800;

    const draftKey = (sessionId) => `codexweb.draft.${sessionId}`;
    const companionKey = (sessionId) => `codexweb.draft.${sessionId}.server_ts`;
    const draftPath = (sessionId) => `/api/sessions/${encodeURIComponent(sessionId)}/draft`;
    const DEBOUNCE_MS = debounceMs;

    let disposed = false;
    // The edit awaiting its debounced upload: { sessionId, text }. Cleared on
    // upload success, send-clear, or pagehide flush; kept on failure so a
    // later flush retries it.
    let pending = null;
    let debounceHandle = null;
    // Identical in-flight upload, so a session-switch flush does not duplicate
    // a POST the debounce already started.
    let inFlightWrite = null;
    // Per-session text at the last sync point (upload, server replace, or
    // send-clear). Pull-if-clean compares the live composer text against it.
    const lastSyncedText = new Map();
    const cleanups = [];

    function companionValue(sessionId) {
      const raw = storageGetItem(companionKey(sessionId));
      if (raw === null || raw === undefined || raw === "") return null;
      const value = Number(raw);
      return Number.isFinite(value) && value > 0 ? value : null;
    }

    function writeCompanion(sessionId, updatedTs) {
      if (updatedTs > 0) storageSetItem(companionKey(sessionId), String(updatedTs));
      else storageRemoveItem(companionKey(sessionId));
    }

    function readLocalDraft(sessionId) {
      return String(storageGetItem(draftKey(sessionId)) || "");
    }

    function selectedStill(sessionId) {
      return !disposed && sessionState.get("selected") === sessionId;
    }

    function cancelDebounce() {
      if (debounceHandle === null) return;
      clearTimeoutFn(debounceHandle);
      debounceHandle = null;
    }

    function applyServerDraftState(sessionId, text, updatedTs) {
      const value = String(text || "");
      // The applied server state supersedes any not-yet-uploaded edit echo
      // for this session. The pull path only runs with a clean composer, so
      // a surviving pending write would necessarily repeat the old sync
      // point's text — re-pushing content the server state just replaced
      // (resurrecting a tombstone-cleared draft when its debounce fires).
      if (pending && pending.sessionId === sessionId) {
        pending = null;
        cancelDebounce();
      }
      applyServerDraft(sessionId, value);
      writeCompanion(sessionId, updatedTs);
      lastSyncedText.set(sessionId, value);
    }

    async function pushDraft(sessionId, text) {
      const body = String(text || "");
      if (inFlightWrite && inFlightWrite.sessionId === sessionId && inFlightWrite.text === body) {
        return inFlightWrite.promise;
      }
      const promise = (async () => {
        try {
          const response = await api(draftPath(sessionId), { method: "POST", body: { text: body } });
          if (!disposed) {
            writeCompanion(sessionId, Number(response && response.updated_ts) || 0);
            lastSyncedText.set(sessionId, body);
            if (pending && pending.sessionId === sessionId && pending.text === body) pending = null;
          }
          return true;
        } catch (error) {
          consoleError("draft upload failed", error);
          return false;
        } finally {
          if (inFlightWrite && inFlightWrite.promise === promise) inFlightWrite = null;
        }
      })();
      inFlightWrite = { sessionId, text: body, promise };
      return promise;
    }

    async function flushPendingWrite() {
      debounceHandle = null;
      if (!pending || disposed) return;
      await pushDraft(pending.sessionId, pending.text);
    }

    function noteDraftEdited(text) {
      if (disposed) return;
      const sessionId = sessionState.get("selected");
      if (!sessionId) return;
      pending = { sessionId, text: String(text || "") };
      cancelDebounce();
      debounceHandle = setTimeoutFn(flushPendingWrite, DEBOUNCE_MS);
    }

    function handleSendCleared(sessionId) {
      if (disposed || !sessionId) return;
      if (!pending || pending.sessionId === sessionId) pending = null;
      cancelDebounce();
      // The empty POST writes a server tombstone and returns its ts;
      // storing that ts as the companion makes this client's own clear a
      // synced state (baseline text ""), so a later reconcile cannot
      // resurrect the sent draft. clearComposer has already removed the
      // localStorage draft key. If the POST fails, the companion survives
      // at the old ts and the next reconcile sees the divergence (local ""
      // vs server text at the same ts) and pushes the deletion again.
      lastSyncedText.set(sessionId, "");
      void pushDraft(sessionId, "");
    }

    async function reconcileSession(sessionId) {
      let response;
      try {
        response = await api(draftPath(sessionId), { method: "GET" });
      } catch (error) {
        // Unknown session or transport failure: the local cache stands.
        return;
      }
      if (!selectedStill(sessionId)) return;
      const serverText = String(response && response.text || "");
      const serverTs = Number(response && response.updated_ts) || 0;
      const localText = readLocalDraft(sessionId);
      const companion = companionValue(sessionId);
      if (serverTs > (companion || 0)) {
        // Another client wrote a newer draft — or cleared it (a tombstone:
        // newer ts, empty text). The server wins in both cases; empty server
        // text is not gated out.
        applyServerDraftState(sessionId, serverText, serverTs);
        return;
      }
      if (companion === null) {
        // Legacy local draft with no companion timestamp.
        if (serverTs > 0) {
          // The server already holds a draft: it is the cross-device authority.
          applyServerDraftState(sessionId, serverText, serverTs);
          return;
        }
        if (localText) {
          await pushDraft(sessionId, localText);
          return;
        }
        lastSyncedText.set(sessionId, "");
        return;
      }
      if (serverTs === 0) {
        // The server holds no entry at all. A clear is a tombstone with a
        // newer ts, so ts 0 is never a deletion signal: a local draft (or a
        // local clear, when the cached draft is empty) is newer than the
        // server state — push it up. This also re-pushes after a server-side
        // state reset while a companion ts survives.
        if (localText) await pushDraft(sessionId, localText);
        else {
          writeCompanion(sessionId, 0);
          lastSyncedText.set(sessionId, "");
        }
        return;
      }
      if (serverTs === companion && localText !== serverText) {
        // Offline divergence at the same timestamp: the local edit wins.
        await pushDraft(sessionId, localText);
        return;
      }
      lastSyncedText.set(sessionId, serverText);
    }

    function onSelectedChanged(sessionId) {
      if (disposed) return;
      // Runs inside the store's synchronous notify, before the lifecycle's
      // saveComposerDraft/loadComposerDraft pair: the outgoing session's
      // pending write must be on its way before the new draft loads.
      if (pending && pending.sessionId && pending.sessionId !== sessionId) {
        cancelDebounce();
        void pushDraft(pending.sessionId, pending.text);
      }
      if (sessionId) void reconcileSession(sessionId);
    }

    function onCatalogChanged() {
      if (disposed) return;
      const sessionId = sessionState.get("selected");
      if (!sessionId || !lastSyncedText.has(sessionId)) return;
      const row = sessionCatalog.get("sessionIndex").get(sessionId);
      const rowTs = Number(row && row.draft_updated_ts) || 0;
      const companion = companionValue(sessionId) || 0;
      if (!(rowTs > companion)) return;
      // Never clobber an actively edited composer: only pull when the text is
      // unmodified since this client's last sync point.
      if (getComposerText() !== lastSyncedText.get(sessionId)) return;
      void pullServerDraft(sessionId);
    }

    async function pullServerDraft(sessionId) {
      let response;
      try {
        response = await api(draftPath(sessionId), { method: "GET" });
      } catch (error) {
        return;
      }
      if (!selectedStill(sessionId)) return;
      const serverTs = Number(response && response.updated_ts) || 0;
      if (serverTs > (companionValue(sessionId) || 0)) {
        applyServerDraftState(sessionId, String(response && response.text || ""), serverTs);
      }
    }

    function flushPendingForUnload() {
      if (!pending) return;
      const write = pending;
      pending = null;
      cancelDebounce();
      if (!fetchFn) return;
      // beforeunload is unconditional full teardown in this app (it fires
      // before pagehide on navigation close); pagehide plus a keepalive fetch
      // is the flush hook, and dispose() performs the same flush so the
      // beforeunload-teardown order cannot drop the pending edit. Cookies
      // ride same-origin fetches.
      try {
        const request = fetchFn(resolveAppUrl(draftPath(write.sessionId)), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: write.text }),
          keepalive: true,
          credentials: "same-origin",
        });
        if (request && typeof request.catch === "function") request.catch(() => {});
      } catch (error) {
        consoleError("draft unload flush failed", error);
      }
    }

    function flushOnPageHide() {
      if (disposed) return;
      flushPendingForUnload();
    }

    cleanups.push(sessionState.subscribe("selected", onSelectedChanged));
    cleanups.push(sessionCatalog.subscribe("sessionIndex", onCatalogChanged));
    if (windowTarget) {
      windowTarget.addEventListener("pagehide", flushOnPageHide);
      cleanups.push(() => windowTarget.removeEventListener("pagehide", flushOnPageHide));
    }

    return Object.freeze({
      noteDraftEdited,
      handleSendCleared,
      dispose() {
        if (disposed) return;
        // App teardown can be triggered by beforeunload, which fires before
        // pagehide on close: flush the pending edit here too (exactly-once —
        // both paths clear `pending` first).
        flushPendingForUnload();
        disposed = true;
        cancelDebounce();
        pending = null;
        while (cleanups.length) cleanups.pop()();
      },
    });
  }

export { createDraftSyncController };
