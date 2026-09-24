"""Behavioral coverage for the server-synced composer draft controller.

The harness executes the shipped controller source (plus the real session
state/catalog stores) in a Node VM with stubbed api/localStorage/timers, and
asserts the sync decisions: debounced upload, last-writer-wins reconcile in
both directions, tombstone (deletion) propagation through reconcile and
pull-if-clean, send-clear, dispose, and the pagehide keepalive flush. A
second section exercises the composer's draft-sync seams (``onDraftEdited``
notification and ``setDraftFromServer`` replacement).
"""

from __future__ import annotations
from frontend_module_loader import module_path

import json
import subprocess
import textwrap
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    name: (module_path(name)).read_text(encoding="utf-8")
    for name in ("app_session_state.js", "app_session_catalog.js", "app_draft_sync.js", "app_composer.js")
}


def run_draft_sync_harness() -> dict[str, Any]:
    script = "(async () => {\n" + textwrap.dedent(
        f"""
        const vm = require("vm");
        const sources = {json.dumps(SOURCES)};
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        for (const name of ["app_session_state.js", "app_session_catalog.js", "app_draft_sync.js", "app_composer.js"]) {{
          vm.runInContext(sources[name], ctx, {{ filename: name }});
        }}
        const settle = async () => {{
          for (let i = 0; i < 6; i++) await new Promise((resolve) => setImmediate(resolve));
        }};

        function makeWorld() {{
          const storage = new Map();
          const storageGetItem = (key) => (storage.has(key) ? storage.get(key) : null);
          const storageSetItem = (key, value) => storage.set(key, String(value));
          const storageRemoveItem = (key) => storage.delete(key);
          let clock = 0;
          let timerSeq = 1;
          const timers = new Map();
          const setTimeoutFn = (fn, ms) => {{
            const id = timerSeq++;
            timers.set(id, {{ fn, at: clock + Number(ms || 0) }});
            return id;
          }};
          const clearTimeoutFn = (id) => timers.delete(id);
          const advance = (ms) => {{
            clock += ms;
            const due = [...timers.entries()].filter(([, t]) => t.at <= clock).sort((a, b) => a[1].at - b[1].at);
            for (const [id, timer] of due) {{ timers.delete(id); timer.fn(); }}
          }};
          const apiCalls = [];
          let apiHandler = null;
          const api = async (path, opts) => {{
            apiCalls.push({{ path, method: (opts && opts.method) || "GET", body: opts && opts.body }});
            if (!apiHandler) throw new Error("unexpected api call");
            return apiHandler(path, opts);
          }};
          const fetchCalls = [];
          const fetchFn = async (url, opts) => {{ fetchCalls.push({{ url, opts }}); return {{}}; }};
          const listeners = {{}};
          const windowTarget = {{
            addEventListener: (type, fn) => {{ listeners[type] = fn; }},
            removeEventListener: (type) => {{ delete listeners[type]; }},
          }};
          const sessionState = ctx.window.CodoxearSessionState.createSessionState({{ consoleError: () => {{}} }});
          const sessionCatalog = ctx.window.CodoxearSessionCatalog.createSessionCatalog({{ consoleError: () => {{}} }});
          const composer = {{ text: "" }};
          const applied = [];
          const controller = ctx.window.CodoxearDraftSync.createDraftSyncController({{
            sessionState, sessionCatalog, api,
            storageGetItem, storageSetItem, storageRemoveItem,
            getComposerText: () => composer.text,
            applyServerDraft: (sessionId, text) => {{
              applied.push({{ sessionId, text }});
              composer.text = text;
              // Mirrors the real seam (composer setDraftFromServer →
              // saveSessionDraft): cache non-empty drafts, drop the key on
              // empty text.
              if (text) storage.set(`codexweb.draft.${{sessionId}}`, text);
              else storage.delete(`codexweb.draft.${{sessionId}}`);
            }},
            resolveAppUrl: (path) => `https://app.test${{path}}`,
            windowTarget, setTimeout: setTimeoutFn, clearTimeout: clearTimeoutFn,
            consoleError: () => {{}}, fetch: fetchFn, debounceMs: 800,
          }});
          return {{
            storage, controller, sessionState, sessionCatalog, apiCalls, applied, fetchCalls, listeners, composer,
            advance,
            posts: () => apiCalls.filter((call) => call.method === "POST"),
            gets: () => apiCalls.filter((call) => call.method === "GET"),
            setApiHandler: (fn) => {{ apiHandler = fn; }},
          }};
        }}

        const out = {{}};

        // (a) rapid input coalesces into one debounced POST
        const w1 = makeWorld();
        w1.setApiHandler((path, opts) => (opts && opts.method === "POST"
          ? {{ ok: true, updated_ts: 42 }}
          : {{ ok: true, text: "", updated_ts: 0 }}));
        w1.sessionState.set("selected", "s1");
        await settle();
        w1.controller.noteDraftEdited("hel");
        w1.advance(300);
        w1.controller.noteDraftEdited("hell");
        w1.advance(300);
        w1.controller.noteDraftEdited("hello");
        out.coalescedPostsBeforeDebounce = w1.posts().length;
        w1.advance(800);
        await settle();
        out.debouncedPosts = w1.posts();
        out.debouncedCompanion = w1.storage.get("codexweb.draft.s1.server_ts");

        // switch flush: pending write leaves before the debounce window, for
        // the outgoing session, and the new session reconciles
        const w2 = makeWorld();
        w2.setApiHandler((path, opts) => (opts && opts.method === "POST"
          ? {{ ok: true, updated_ts: 77 }}
          : {{ ok: true, text: "", updated_ts: 0 }}));
        w2.sessionState.set("selected", "s1");
        await settle();
        w2.controller.noteDraftEdited("switch draft");
        w2.sessionState.set("selected", "s2");
        await settle();
        out.switchFlushPosts = w2.posts();
        out.switchFlushReconciledNewSession = w2.gets().some((call) => call.path === "/api/sessions/s2/draft");

        // (b) reconcile: server-newer replaces local
        const w3 = makeWorld();
        w3.storage.set("codexweb.draft.s1", "old local");
        w3.storage.set("codexweb.draft.s1.server_ts", "5");
        w3.setApiHandler(() => ({{ ok: true, text: "from server", updated_ts: 10 }}));
        w3.sessionState.set("selected", "s1");
        await settle();
        out.serverNewerApplied = w3.applied;
        out.serverNewerCompanion = w3.storage.get("codexweb.draft.s1.server_ts");
        out.serverNewerPosts = w3.posts();

        // (c) reconcile: offline divergence pushes local up
        const w4 = makeWorld();
        w4.storage.set("codexweb.draft.s1", "local edit");
        w4.storage.set("codexweb.draft.s1.server_ts", "9");
        w4.setApiHandler((path, opts) => (opts && opts.method === "POST"
          ? {{ ok: true, updated_ts: 11 }}
          : {{ ok: true, text: "server copy", updated_ts: 9 }}));
        w4.sessionState.set("selected", "s1");
        await settle();
        out.divergencePosts = w4.posts();
        out.divergenceCompanion = w4.storage.get("codexweb.draft.s1.server_ts");
        out.divergenceApplied = w4.applied;

        // (d) legacy local draft (no companion ts)
        const w5a = makeWorld();
        w5a.storage.set("codexweb.draft.s1", "legacy text");
        w5a.setApiHandler((path, opts) => (opts && opts.method === "POST"
          ? {{ ok: true, updated_ts: 3 }}
          : {{ ok: true, text: "", updated_ts: 0 }}));
        w5a.sessionState.set("selected", "s1");
        await settle();
        out.legacyEmptyServerPosts = w5a.posts();
        out.legacyEmptyServerCompanion = w5a.storage.get("codexweb.draft.s1.server_ts");

        const w5b = makeWorld();
        w5b.storage.set("codexweb.draft.s1", "legacy text");
        w5b.setApiHandler(() => ({{ ok: true, text: "server wins", updated_ts: 7 }}));
        w5b.sessionState.set("selected", "s1");
        await settle();
        out.legacyServerHasDraftApplied = w5b.applied;
        out.legacyServerHasDraftPosts = w5b.posts();
        out.legacyServerHasDraftCompanion = w5b.storage.get("codexweb.draft.s1.server_ts");

        // legacy local draft (no companion ts) vs a server tombstone: the
        // server entry wins and the local draft is dropped
        const w5c = makeWorld();
        w5c.storage.set("codexweb.draft.s1", "legacy text");
        w5c.composer.text = "legacy text";
        w5c.setApiHandler(() => ({{ ok: true, text: "", updated_ts: 7 }}));
        w5c.sessionState.set("selected", "s1");
        await settle();
        out.legacyTombstoneApplied = w5c.applied;
        out.legacyTombstonePosts = w5c.posts();
        out.legacyTombstoneDraftKey = w5c.storage.has("codexweb.draft.s1");
        out.legacyTombstoneCompanion = w5c.storage.get("codexweb.draft.s1.server_ts");

        // (e) pull-if-clean via session-row draft_updated_ts
        const w6 = makeWorld();
        w6.storage.set("codexweb.draft.s1", "base");
        w6.storage.set("codexweb.draft.s1.server_ts", "10");
        w6.composer.text = "base";
        w6.setApiHandler(() => ({{ ok: true, text: "base", updated_ts: 10 }}));
        w6.sessionState.set("selected", "s1");
        await settle();
        w6.setApiHandler(() => ({{ ok: true, text: "refreshed", updated_ts: 20 }}));
        w6.sessionCatalog.applySnapshot({{ latestSessions: [{{ session_id: "s1", draft_updated_ts: 20 }}] }});
        await settle();
        out.pullCleanApplied = w6.applied;
        out.pullCleanCompanion = w6.storage.get("codexweb.draft.s1.server_ts");
        out.pullCleanDraftNotifications = w6.gets().length;

        const w7 = makeWorld();
        w7.storage.set("codexweb.draft.s1", "base");
        w7.storage.set("codexweb.draft.s1.server_ts", "10");
        w7.composer.text = "base";
        w7.setApiHandler(() => ({{ ok: true, text: "base", updated_ts: 10 }}));
        w7.sessionState.set("selected", "s1");
        await settle();
        const dirtyGetsBefore = w7.gets().length;
        w7.composer.text = "user typing since sync";
        w7.setApiHandler(() => ({{ ok: true, text: "refreshed", updated_ts: 20 }}));
        w7.sessionCatalog.applySnapshot({{ latestSessions: [{{ session_id: "s1", draft_updated_ts: 20 }}] }});
        await settle();
        out.pullDirtyApplied = w7.applied;
        out.pullDirtyGets = w7.gets().length - dirtyGetsBefore;
        out.pullDirtyCompanion = w7.storage.get("codexweb.draft.s1.server_ts");

        // (f) send success clears the server draft (tombstone) and cancels
        // the pending upload; the returned tombstone ts becomes the companion
        const w8 = makeWorld();
        w8.setApiHandler((path, opts) => (opts && opts.method === "POST"
          ? {{ ok: true, updated_ts: 21 }}
          : {{ ok: true, text: "", updated_ts: 0 }}));
        w8.sessionState.set("selected", "s1");
        await settle();
        w8.controller.noteDraftEdited("draft being sent");
        w8.controller.handleSendCleared("s1");
        await settle();
        out.sendClearPosts = w8.posts();
        out.sendClearCompanionStored = w8.storage.get("codexweb.draft.s1.server_ts");
        w8.advance(2000);
        await settle();
        out.sendClearPostsAfterDebounceWindow = w8.posts().length;

        // (g) dispose cancels pending timers and subscriptions
        const w9 = makeWorld();
        w9.setApiHandler((path, opts) => (opts && opts.method === "POST"
          ? {{ ok: true, updated_ts: 1 }}
          : {{ ok: true, text: "", updated_ts: 0 }}));
        w9.sessionState.set("selected", "s1");
        await settle();
        w9.controller.noteDraftEdited("never sent");
        const disposeGetsBefore = w9.gets().length;
        w9.controller.dispose();
        out.disposePageHideListenerRemoved = !("pagehide" in w9.listeners);
        // Teardown (beforeunload order) still flushes the pending edit, via a
        // keepalive fetch rather than the api wrapper.
        out.disposeFlushFetch = w9.fetchCalls.map((call) => ({{
          url: call.url, method: call.opts.method, keepalive: call.opts.keepalive, body: call.opts.body,
        }}));
        w9.advance(2000);
        await settle();
        out.disposeNoUpload = w9.posts().length;
        w9.sessionState.set("selected", "s2");
        await settle();
        out.disposeNoReconcile = w9.gets().length - disposeGetsBefore;

        // send-clear whose empty POST fails: companion survives, and the next
        // reconcile observes the divergence (local empty vs server text) and
        // pushes the deletion again
        const w11 = makeWorld();
        w11.storage.set("codexweb.draft.s1", "sent text");
        w11.storage.set("codexweb.draft.s1.server_ts", "9");
        let sendClearPostFailed = false;
        w11.setApiHandler((path, opts) => {{
          if (opts && opts.method === "POST") {{
            if (!sendClearPostFailed) {{
              sendClearPostFailed = true;
              throw new Error("offline");
            }}
            return {{ ok: true, updated_ts: 12 }};
          }}
          return {{ ok: true, text: "sent text", updated_ts: 9 }};
        }});
        w11.sessionState.set("selected", "s1");
        await settle();
        w11.controller.handleSendCleared("s1");
        w11.storage.delete("codexweb.draft.s1"); // composer clearComposer removes the cached draft
        await settle();
        out.sendClearFailedCompanionKept = w11.storage.get("codexweb.draft.s1.server_ts");
        w11.sessionState.set("selected", "s2");
        await settle();
        w11.sessionState.set("selected", "s1");
        await settle();
        out.sendClearFailedRetryPosts = w11.posts().map((call) => call.body);
        out.sendClearFailedCompanionAfterConverge = w11.storage.get("codexweb.draft.s1.server_ts");

        // failed debounced upload keeps the pending edit; the switch flush retries it
        const w12 = makeWorld();
        let uploadFailed = false;
        w12.setApiHandler((path, opts) => {{
          if (opts && opts.method === "POST") {{
            if (!uploadFailed) {{
              uploadFailed = true;
              throw new Error("offline");
            }}
            return {{ ok: true, updated_ts: 5 }};
          }}
          return {{ ok: true, text: "", updated_ts: 0 }};
        }});
        w12.sessionState.set("selected", "s1");
        await settle();
        w12.controller.noteDraftEdited("retry me");
        w12.advance(800);
        await settle();
        out.failedUploadFirstAttempt = w12.posts().map((call) => call.body);
        out.failedUploadCompanionAbsent = !w12.storage.has("codexweb.draft.s1.server_ts");
        w12.sessionState.set("selected", "s2");
        await settle();
        out.failedUploadRetryPosts = w12.posts().map((call) => call.body);
        out.failedUploadCompanionAfterRetry = w12.storage.get("codexweb.draft.s1.server_ts");

        // tombstone reconcile: a server tombstone newer than the companion
        // replaces the local draft with "" and stores the tombstone ts
        const w13 = makeWorld();
        w13.storage.set("codexweb.draft.s1", "stale local");
        w13.storage.set("codexweb.draft.s1.server_ts", "5");
        w13.composer.text = "stale local";
        w13.setApiHandler(() => ({{ ok: true, text: "", updated_ts: 12 }}));
        w13.sessionState.set("selected", "s1");
        await settle();
        out.tombstoneReconcileApplied = w13.applied;
        out.tombstoneReconcileComposer = w13.composer.text;
        out.tombstoneReconcileDraftKey = w13.storage.has("codexweb.draft.s1");
        out.tombstoneReconcileCompanion = w13.storage.get("codexweb.draft.s1.server_ts");
        out.tombstoneReconcilePosts = w13.posts();

        // tombstone row clears a clean open composer via pull-if-clean
        const w14 = makeWorld();
        w14.storage.set("codexweb.draft.s1", "open draft");
        w14.storage.set("codexweb.draft.s1.server_ts", "10");
        w14.composer.text = "open draft";
        w14.setApiHandler(() => ({{ ok: true, text: "open draft", updated_ts: 10 }}));
        w14.sessionState.set("selected", "s1");
        await settle();
        w14.setApiHandler(() => ({{ ok: true, text: "", updated_ts: 25 }}));
        w14.sessionCatalog.applySnapshot({{ latestSessions: [{{ session_id: "s1", draft_updated_ts: 25 }}] }});
        await settle();
        out.tombstonePullApplied = w14.applied;
        out.tombstonePullComposer = w14.composer.text;
        out.tombstonePullDraftKey = w14.storage.has("codexweb.draft.s1");
        out.tombstonePullCompanion = w14.storage.get("codexweb.draft.s1.server_ts");
        out.tombstonePullPosts = w14.posts();

        // dirty composer survives a tombstone row: newer typing wins via its
        // next upload
        const w15 = makeWorld();
        w15.storage.set("codexweb.draft.s1", "open draft");
        w15.storage.set("codexweb.draft.s1.server_ts", "10");
        w15.composer.text = "open draft";
        w15.setApiHandler(() => ({{ ok: true, text: "open draft", updated_ts: 10 }}));
        w15.sessionState.set("selected", "s1");
        await settle();
        const w15GetsBefore = w15.gets().length;
        w15.composer.text = "typing newer content";
        w15.setApiHandler(() => ({{ ok: true, text: "", updated_ts: 25 }}));
        w15.sessionCatalog.applySnapshot({{ latestSessions: [{{ session_id: "s1", draft_updated_ts: 25 }}] }});
        await settle();
        out.tombstonePullDirtyApplied = w15.applied.length;
        out.tombstonePullDirtyGets = w15.gets().length - w15GetsBefore;
        out.tombstonePullDirtyComposer = w15.composer.text;
        out.tombstonePullDirtyCompanion = w15.storage.get("codexweb.draft.s1.server_ts");

        // send-clear followed by reconcile must not re-push the sent draft
        // (the resurrection regression under the old DELETE semantics)
        const w16 = makeWorld();
        let w16ts = 0;
        w16.setApiHandler((path, opts) => {{
          if (opts && opts.method === "POST") {{
            w16ts += 1;
            return {{ ok: true, updated_ts: w16ts }};
          }}
          return {{ ok: true, text: "", updated_ts: w16ts }};
        }});
        w16.sessionState.set("selected", "s1");
        await settle();
        w16.composer.text = "message";
        w16.storage.set("codexweb.draft.s1", "message");
        w16.controller.noteDraftEdited("message");
        w16.advance(800);
        await settle();
        w16.controller.handleSendCleared("s1");
        w16.composer.text = ""; // composer clearComposer clears the textarea...
        w16.storage.delete("codexweb.draft.s1"); // ...and the cached draft
        await settle();
        w16.sessionState.set("selected", "s2");
        await settle();
        w16.sessionState.set("selected", "s1");
        await settle();
        out.sendClearThenReconcilePosts = w16.posts().map((call) => call.body);
        out.sendClearThenReconcileComposer = w16.composer.text;
        out.sendClearThenReconcileCompanion = w16.storage.get("codexweb.draft.s1.server_ts");

        // server state lost (ts 0) while a companion ts survives: ts 0 is no
        // deletion signal under tombstones, so the local draft is re-pushed
        const w17 = makeWorld();
        w17.storage.set("codexweb.draft.s1", "local after reset");
        w17.storage.set("codexweb.draft.s1.server_ts", "9");
        w17.composer.text = "local after reset";
        w17.setApiHandler((path, opts) => (opts && opts.method === "POST"
          ? {{ ok: true, updated_ts: 14 }}
          : {{ ok: true, text: "", updated_ts: 0 }}));
        w17.sessionState.set("selected", "s1");
        await settle();
        out.serverResetPosts = w17.posts();
        out.serverResetCompanion = w17.storage.get("codexweb.draft.s1.server_ts");

        // a debounced echo of already-synced text must not resurrect a
        // tombstone: the user reverts to the synced text inside the debounce
        // window, the tombstone pull applies "", and the pending echo is
        // discarded instead of uploaded when the debounce window elapses
        const w18 = makeWorld();
        w18.storage.set("codexweb.draft.s1", "base");
        w18.storage.set("codexweb.draft.s1.server_ts", "10");
        w18.composer.text = "base";
        w18.setApiHandler(() => ({{ ok: true, text: "base", updated_ts: 10 }}));
        w18.sessionState.set("selected", "s1");
        await settle();
        w18.composer.text = "base tweaked";
        w18.controller.noteDraftEdited("base tweaked");
        w18.composer.text = "base";
        w18.controller.noteDraftEdited("base");
        w18.setApiHandler(() => ({{ ok: true, text: "", updated_ts: 30 }}));
        w18.sessionCatalog.applySnapshot({{ latestSessions: [{{ session_id: "s1", draft_updated_ts: 30 }}] }});
        await settle();
        w18.advance(800);
        await settle();
        out.tombstonePullDiscardsEchoPosts = w18.posts();
        out.tombstonePullDiscardsEchoComposer = w18.composer.text;
        out.tombstonePullDiscardsEchoCompanion = w18.storage.get("codexweb.draft.s1.server_ts");

        // pagehide flush rides a keepalive fetch, not api()
        const w10 = makeWorld();
        w10.setApiHandler((path, opts) => (opts && opts.method === "POST"
          ? {{ ok: true, updated_ts: 5 }}
          : {{ ok: true, text: "", updated_ts: 0 }}));
        w10.sessionState.set("selected", "s1");
        await settle();
        w10.controller.noteDraftEdited("last words");
        w10.listeners.pagehide();
        out.pagehideFetch = w10.fetchCalls.map((call) => ({{
          url: call.url,
          method: call.opts.method,
          keepalive: call.opts.keepalive,
          credentials: call.opts.credentials,
          contentType: call.opts.headers["Content-Type"],
          body: call.opts.body,
        }}));
        out.pagehideApiPosts = w10.posts().length;
        w10.advance(2000);
        await settle();
        out.pagehideNoDelayedUpload = w10.posts().length;

        // composer seams: input notification and server-side replacement
        function composerNode() {{
          const listeners = new Map();
          return {{
            style: {{}}, value: "", disabled: false, scrollHeight: 32,
            classList: {{ toggle: () => {{}}, add: () => {{}}, remove: () => {{}} }},
            addEventListener: (type, handler) => listeners.set(type, handler),
            removeEventListener: (type) => listeners.delete(type),
            emit: (type, event = {{}}) => listeners.get(type)?.(event),
            setAttribute: () => {{}}, focus: () => {{}}, blur: () => {{}},
          }};
        }}
        const draftNotifications = [];
        let autoGrowCalls = 0;
        const composerStorage = new Map();
        const composerState = ctx.window.CodoxearSessionState.createSessionState({{ consoleError: () => {{}} }});
        composerState.set("selected", "sid");
        const composerCatalog = ctx.window.CodoxearSessionCatalog.createSessionCatalog({{ consoleError: () => {{}} }});
        composerCatalog.set("latestSessions", [{{ session_id: "sid", launch_state: "ready" }}]);
        const composerTextarea = composerNode();
        const composerController = ctx.window.CodoxearComposer.createComposerController({{
          form: Object.assign(composerNode(), {{ requestSubmit: () => {{}} }}),
          textarea: composerTextarea,
          msgPh: composerNode(), sendBtn: composerNode(),
          sendChoice: composerNode(), sendChoiceBackdrop: composerNode(),
          sendChoiceNowBtn: composerNode(), sendChoiceLaterBtn: composerNode(), sendChoiceCancelBtn: composerNode(),
          sessionCatalog: composerCatalog, sessionLaunchFailed: () => false, sessionState: composerState,
          getStagedAttachments: () => [], api: async () => ({{}}), setToast: () => {{}},
          setPollFastUntilMs: () => {{}}, kickPoll: () => {{}},
          sendText: async () => true, enqueueComposerText: async () => true,
          prepareModalOpen: () => {{}}, afterModalVisibilityChanged: () => {{}}, restoreModalFocus: () => {{}},
          storageGetItem: (key) => (composerStorage.has(key) ? composerStorage.get(key) : null),
          storageSetItem: (key, value) => composerStorage.set(key, String(value)),
          storageRemoveItem: (key) => composerStorage.delete(key),
          onDraftEdited: (text) => draftNotifications.push(text),
          onAutoGrow: () => {{ autoGrowCalls += 1; }},
          requestFrame: (callback) => callback(),
          getComputedStyle: () => ({{ minHeight: "32px" }}),
          activeElement: () => null, isHTMLElement: () => false,
        }});
        composerTextarea.value = "typed draft";
        composerTextarea.emit("input");
        out.composerInputNotification = draftNotifications.slice();
        out.composerInputDraftCached = composerStorage.get("codexweb.draft.sid");
        const growBeforeServerReplace = autoGrowCalls;
        composerController.setDraftFromServer("sid", "from server");
        out.composerServerReplaceValue = composerTextarea.value;
        out.composerServerReplaceCached = composerStorage.get("codexweb.draft.sid");
        out.composerServerReplaceNotifications = draftNotifications.length;
        out.composerServerReplaceAutoGrow = autoGrowCalls > growBeforeServerReplace;
        composerController.setDraftFromServer("sid", "");
        out.composerServerClearValue = composerTextarea.value;
        out.composerServerClearCached = composerStorage.has("codexweb.draft.sid");

        process.stdout.write(JSON.stringify(out));
        """
    ) + "\n})().catch((error) => { console.error(error); process.exit(1); });\n"
    completed = subprocess.run(
        ["node", "-"],
        check=True,
        input=script,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_rapid_input_coalesces_into_one_debounced_upload() -> None:
    result = run_draft_sync_harness()
    assert result["coalescedPostsBeforeDebounce"] == 0
    assert result["debouncedPosts"] == [
        {"path": "/api/sessions/s1/draft", "method": "POST", "body": {"text": "hello"}}
    ]
    assert result["debouncedCompanion"] == "42"


def test_session_switch_flushes_pending_write_before_reconciling_new_session() -> None:
    result = run_draft_sync_harness()
    assert result["switchFlushPosts"] == [
        {"path": "/api/sessions/s1/draft", "method": "POST", "body": {"text": "switch draft"}}
    ]
    assert result["switchFlushReconciledNewSession"] is True


def test_reconcile_server_newer_replaces_local() -> None:
    result = run_draft_sync_harness()
    assert result["serverNewerApplied"] == [{"sessionId": "s1", "text": "from server"}]
    assert result["serverNewerCompanion"] == "10"
    assert result["serverNewerPosts"] == []


def test_reconcile_offline_divergence_pushes_local_up() -> None:
    result = run_draft_sync_harness()
    assert result["divergencePosts"] == [
        {"path": "/api/sessions/s1/draft", "method": "POST", "body": {"text": "local edit"}}
    ]
    assert result["divergenceCompanion"] == "11"
    assert result["divergenceApplied"] == []


def test_reconcile_applies_server_tombstone_newer_than_companion() -> None:
    result = run_draft_sync_harness()
    assert result["tombstoneReconcileApplied"] == [{"sessionId": "s1", "text": ""}]
    assert result["tombstoneReconcileComposer"] == ""
    assert result["tombstoneReconcileDraftKey"] is False
    assert result["tombstoneReconcileCompanion"] == "12"
    assert result["tombstoneReconcilePosts"] == []


def test_legacy_local_draft_pushes_only_when_server_is_empty() -> None:
    result = run_draft_sync_harness()
    assert result["legacyEmptyServerPosts"] == [
        {"path": "/api/sessions/s1/draft", "method": "POST", "body": {"text": "legacy text"}}
    ]
    assert result["legacyEmptyServerCompanion"] == "3"
    assert result["legacyServerHasDraftApplied"] == [{"sessionId": "s1", "text": "server wins"}]
    assert result["legacyServerHasDraftPosts"] == []
    assert result["legacyServerHasDraftCompanion"] == "7"


def test_legacy_local_draft_drops_against_server_tombstone() -> None:
    result = run_draft_sync_harness()
    assert result["legacyTombstoneApplied"] == [{"sessionId": "s1", "text": ""}]
    assert result["legacyTombstonePosts"] == []
    assert result["legacyTombstoneDraftKey"] is False
    assert result["legacyTombstoneCompanion"] == "7"


def test_pull_if_clean_pulls_newer_row_only_with_unmodified_composer() -> None:
    result = run_draft_sync_harness()
    assert result["pullCleanApplied"] == [{"sessionId": "s1", "text": "refreshed"}]
    assert result["pullCleanCompanion"] == "20"
    assert result["pullCleanDraftNotifications"] == 2  # initial reconcile + pull
    assert result["pullDirtyApplied"] == []
    assert result["pullDirtyGets"] == 0
    assert result["pullDirtyCompanion"] == "10"


def test_tombstone_row_clears_clean_open_composer_via_pull() -> None:
    result = run_draft_sync_harness()
    assert result["tombstonePullApplied"] == [{"sessionId": "s1", "text": ""}]
    assert result["tombstonePullComposer"] == ""
    assert result["tombstonePullDraftKey"] is False
    assert result["tombstonePullCompanion"] == "25"
    assert result["tombstonePullPosts"] == []


def test_tombstone_row_does_not_clobber_dirty_composer() -> None:
    result = run_draft_sync_harness()
    assert result["tombstonePullDirtyApplied"] == 0
    assert result["tombstonePullDirtyGets"] == 0
    assert result["tombstonePullDirtyComposer"] == "typing newer content"
    assert result["tombstonePullDirtyCompanion"] == "10"


def test_send_clear_stores_tombstone_ts_and_cancels_pending_upload() -> None:
    result = run_draft_sync_harness()
    assert result["sendClearPosts"] == [
        {"path": "/api/sessions/s1/draft", "method": "POST", "body": {"text": ""}}
    ]
    # The clear is a synced state: the returned tombstone ts becomes the
    # companion, so this client never re-pushes the sent draft.
    assert result["sendClearCompanionStored"] == "21"
    assert result["sendClearPostsAfterDebounceWindow"] == 1


def test_send_clear_then_reconcile_does_not_repush_sent_draft() -> None:
    result = run_draft_sync_harness()
    assert result["sendClearThenReconcilePosts"] == [{"text": "message"}, {"text": ""}]
    assert result["sendClearThenReconcileComposer"] == ""
    assert result["sendClearThenReconcileCompanion"] == "2"


def test_server_reset_zero_ts_still_repushes_local_draft() -> None:
    result = run_draft_sync_harness()
    assert result["serverResetPosts"] == [
        {"path": "/api/sessions/s1/draft", "method": "POST", "body": {"text": "local after reset"}}
    ]
    assert result["serverResetCompanion"] == "14"


def test_dispose_cancels_timers_and_subscriptions() -> None:
    result = run_draft_sync_harness()
    assert result["disposePageHideListenerRemoved"] is True
    # The beforeunload-order teardown flushes the pending edit once, keepalive:
    assert result["disposeFlushFetch"] == [{
        "url": "https://app.test/api/sessions/s1/draft",
        "method": "POST",
        "keepalive": True,
        "body": '{"text":"never sent"}',
    }]
    assert result["disposeNoUpload"] == 0
    assert result["disposeNoReconcile"] == 0


def test_send_clear_failure_converges_on_next_reconcile() -> None:
    result = run_draft_sync_harness()
    assert result["sendClearFailedCompanionKept"] == "9"
    assert result["sendClearFailedRetryPosts"] == [{"text": ""}, {"text": ""}]
    # The retried deletion converges to the tombstone: its ts is stored as
    # the companion, not dropped.
    assert result["sendClearFailedCompanionAfterConverge"] == "12"


def test_tombstone_pull_discards_debounced_echo_of_synced_text() -> None:
    result = run_draft_sync_harness()
    assert result["tombstonePullDiscardsEchoPosts"] == []
    assert result["tombstonePullDiscardsEchoComposer"] == ""
    assert result["tombstonePullDiscardsEchoCompanion"] == "30"


def test_failed_upload_is_retried_on_session_switch() -> None:
    result = run_draft_sync_harness()
    assert result["failedUploadFirstAttempt"] == [{"text": "retry me"}]
    assert result["failedUploadCompanionAbsent"] is True
    assert result["failedUploadRetryPosts"] == [{"text": "retry me"}, {"text": "retry me"}]
    assert result["failedUploadCompanionAfterRetry"] == "5"


def test_pagehide_flush_uses_keepalive_fetch() -> None:
    result = run_draft_sync_harness()
    assert result["pagehideFetch"] == [{
        "url": "https://app.test/api/sessions/s1/draft",
        "method": "POST",
        "keepalive": True,
        "credentials": "same-origin",
        "contentType": "application/json",
        "body": '{"text":"last words"}',
    }]
    assert result["pagehideApiPosts"] == 0
    assert result["pagehideNoDelayedUpload"] == 0


def test_composer_notifies_draft_edits_and_replaces_server_drafts_without_reupload() -> None:
    result = run_draft_sync_harness()
    assert result["composerInputNotification"] == ["typed draft"]
    assert result["composerInputDraftCached"] == "typed draft"
    assert result["composerServerReplaceValue"] == "from server"
    assert result["composerServerReplaceCached"] == "from server"
    assert result["composerServerReplaceNotifications"] == 1  # no re-notification on programmatic replace
    assert result["composerServerReplaceAutoGrow"] is True
    assert result["composerServerClearValue"] == ""
    assert result["composerServerClearCached"] is False
