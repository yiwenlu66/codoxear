import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_TRANSCRIPT_JS = ROOT / "codoxear" / "static" / "app_transcript.js"
APP_MESSAGE_IDENTITY_JS = ROOT / "codoxear" / "static" / "app_message_identity.js"


def _run_node(js: str) -> dict:
    proc = subprocess.run(
        ["node", "-e", js],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(proc.stdout)


def _source_between(start: str, end: str) -> str:
    source = APP_JS.read_text(encoding="utf-8")
    i = source.index(start)
    j = source.index(end, i)
    return source[i:j]


class TestChatTranscriptRuntime(unittest.TestCase):
    def test_transcript_scroll_runtime_owns_bottom_lock_and_input_policy(self) -> None:
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const ctx = {{ window: {{}} }};
            const vm = require("vm");
            vm.createContext(ctx);
            vm.runInContext({json.dumps(transcript_source)}, ctx);
            const rafs = [];
            const chat = {{ scrollTop: 0, scrollHeight: 500, clientHeight: 100 }};
            const jumpButton = {{ style: {{ display: "" }} }};
            const timeChip = {{ style: {{ display: "" }}, textContent: "" }};
            let selected = true;
            let searchOpen = false;
            let cancelOlder = 0;
            let autoLoadOlder = 0;
            let cancelEligible = false;
            const firstRow = {{ dataset: {{ ts: "1000" }} }};
            const runtime = ctx.window.CodoxearTranscript.createTranscriptScrollRuntime({{
              chat,
              jumpButton,
              timeChip,
              requestAnimationFrame: (fn) => rafs.push(fn),
              hasSelection: () => selected,
              isSearchOpen: () => searchOpen,
              firstVisibleMessageRow: () => firstRow,
              dayLabel: () => "Day",
              time24: () => "12:34",
              shouldCancelOlderLoad: () => cancelEligible,
              cancelOlderLoad: () => {{ cancelOlder += 1; }},
              autoLoadOlder: () => {{ autoLoadOlder += 1; }},
              bottomThresholdPx: 80,
              olderTopTriggerPx: 1,
              olderCancelPx: 48,
            }});
            runtime.syncJumpButton();
            const initialProjection = {{ jump: jumpButton.style.display, time: timeChip.style.display, text: timeChip.textContent }};
            runtime.scrollToBottom();
            const afterBottom = {{ top: chat.scrollTop, snapshot: runtime.snapshot() }};
            chat.scrollTop = 100;
            const upScroll = runtime.handleScroll();
            runtime.markDetachedWindow();
            const detachedTime = runtime.syncVisibleTimeIndicator();
            searchOpen = true;
            const searchHidden = runtime.syncVisibleTimeIndicator();
            searchOpen = false;
            cancelEligible = true;
            chat.scrollTop = 50;
            runtime.handleScroll();
            chat.scrollTop = 0;
            runtime.handleScroll();
            const afterThresholds = {{ cancelOlder, autoLoadOlder, snapshot: runtime.snapshot(), jump: jumpButton.style.display }};
            chat.scrollTop = 400;
            runtime.handleScroll();
            const afterNearBottom = runtime.snapshot();
            chat.scrollTop = 10;
            runtime.handleWheel({{ deltaY: -1 }});
            const afterWheelAwayFromTop = {{ autoLoadOlder, snapshot: runtime.snapshot() }};
            chat.scrollTop = 0;
            runtime.handleWheel({{ deltaY: -1 }});
            runtime.handleTouchStart({{ touches: [{{ clientY: 10 }}] }});
            runtime.handleTouchMove({{ touches: [{{ clientY: 20 }}] }});
            const afterWheelTouchAtTop = {{ autoLoadOlder, snapshot: runtime.snapshot() }};
            chat.scrollTop = 5;
            chat.scrollHeight = 800;
            runtime.markLiveTail();
            runtime.enableAutoScroll();
            runtime.scheduleScrollToBottom({{ double: true, syncJump: true }});
            const scheduledBeforeRun = rafs.length;
            rafs.shift()();
            const afterFirstRaf = {{ top: chat.scrollTop, queued: rafs.length }};
            rafs.shift()();
            const afterSecondRaf = {{ top: chat.scrollTop, jump: jumpButton.style.display }};
            runtime.reset({{ scrollTop: 0 }});
            const reset = {{ snapshot: runtime.snapshot(), top: chat.scrollTop, jump: jumpButton.style.display, time: timeChip.style.display }};
            let missingError = "";
            try {{ ctx.window.CodoxearTranscript.createTranscriptScrollRuntime({{ chat, jumpButton, timeChip }}); }} catch (err) {{ missingError = err && err.message ? err.message : String(err); }}
            process.stdout.write(JSON.stringify({{
              initialProjection,
              afterBottom,
              upScroll,
              detachedTime,
              searchHidden,
              afterThresholds,
              afterNearBottom,
              afterWheelAwayFromTop,
              afterWheelTouchAtTop,
              scheduledBeforeRun,
              afterFirstRaf,
              afterSecondRaf,
              reset,
              missingError,
              frozen: Object.isFrozen(runtime),
            }}));
            """
        )
        out = _run_node(js)
        self.assertEqual(out["initialProjection"], {"jump": "none", "time": "none", "text": ""})
        self.assertEqual(out["afterBottom"]["top"], 500)
        self.assertTrue(out["afterBottom"]["snapshot"]["autoScroll"])
        self.assertLess(out["upScroll"]["delta"], 0)
        self.assertFalse(out["upScroll"]["autoScroll"])
        self.assertEqual(out["detachedTime"], {"visible": True, "text": "Day · 12:34"})
        self.assertEqual(out["searchHidden"], {"visible": False, "text": ""})
        self.assertEqual(out["afterThresholds"]["cancelOlder"], 1)
        self.assertEqual(out["afterThresholds"]["autoLoadOlder"], 1)
        self.assertEqual(out["afterThresholds"]["jump"], "inline-flex")
        self.assertTrue(out["afterNearBottom"]["autoScroll"])
        self.assertEqual(out["afterWheelAwayFromTop"]["autoLoadOlder"], 1)
        self.assertFalse(out["afterWheelAwayFromTop"]["snapshot"]["autoScroll"])
        self.assertEqual(out["afterWheelTouchAtTop"]["autoLoadOlder"], 3)
        self.assertFalse(out["afterWheelTouchAtTop"]["snapshot"]["autoScroll"])
        self.assertEqual(out["scheduledBeforeRun"], 1)
        self.assertEqual(out["afterFirstRaf"], {"top": 800, "queued": 1})
        self.assertEqual(out["afterSecondRaf"], {"top": 800, "jump": "none"})
        self.assertEqual(out["reset"]["snapshot"], {"autoScroll": True, "renderedAtLiveTail": True, "lastScrollTop": 0})
        self.assertEqual(out["reset"]["top"], 0)
        self.assertEqual(out["reset"]["jump"], "none")
        self.assertIn("transcript dependency missing: requestAnimationFrame", out["missingError"])
        self.assertTrue(out["frozen"])

    def test_older_load_runtime_owns_state_currentness_and_ui_projection(self) -> None:
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const ctx = {{ window: {{}}, AbortController }};
            const vm = require("vm");
            vm.createContext(ctx);
            vm.runInContext({json.dumps(transcript_source)}, ctx);
            let now = 1000;
            const olderWrap = {{ style: {{ display: "" }} }};
            const olderButton = {{ disabled: false, textContent: "" }};
            const olderError = {{ style: {{ display: "" }} }};
            const olderErrorText = {{ textContent: "" }};
            const runtime = ctx.window.CodoxearTranscript.createOlderLoadRuntime({{
              olderWrap,
              olderButton,
              olderError,
              olderErrorText,
              AbortControllerCtor: AbortController,
              nowMs: () => now,
              autoCooldownMs: 450,
            }});
            const initial = runtime.snapshot();
            runtime.setState({{ hasMore: true, isLoading: false }});
            const visible = {{ wrap: olderWrap.style.display, disabled: olderButton.disabled, text: olderButton.textContent }};
            const firstAuto = runtime.markAutoTrigger();
            const secondAuto = runtime.markAutoTrigger();
            now += 500;
            const thirdAuto = runtime.markAutoTrigger();
            const load = runtime.beginLoad({{ cancelOnScroll: true }});
            const loading = {{ snapshot: runtime.snapshot(), wrap: olderWrap.style.display, disabled: olderButton.disabled, text: olderButton.textContent }};
            const currentBeforeInvalidate = runtime.isCurrent(load);
            const cancelBeforeInvalidate = runtime.shouldCancelOnScroll();
            runtime.invalidate();
            const currentAfterInvalidate = runtime.isCurrent(load);
            const afterInvalidate = runtime.snapshot();
            runtime.showError();
            const errorShown = {{ display: olderError.style.display, text: olderErrorText.textContent }};
            runtime.setState({{ hasMore: false, isLoading: false }});
            const afterHide = {{ snapshot: runtime.snapshot(), wrap: olderWrap.style.display, error: olderError.style.display, errorText: olderErrorText.textContent }};
            let missingError = "";
            try {{ ctx.window.CodoxearTranscript.createOlderLoadRuntime({{ olderWrap }}); }} catch (err) {{ missingError = err && err.message ? err.message : String(err); }}
            process.stdout.write(JSON.stringify({{
              initial,
              visible,
              firstAuto,
              secondAuto,
              thirdAuto,
              loading,
              currentBeforeInvalidate,
              cancelBeforeInvalidate,
              currentAfterInvalidate,
              afterInvalidate,
              errorShown,
              afterHide,
              missingError,
              frozen: Object.isFrozen(runtime),
            }}));
            """
        )
        out = _run_node(js)
        self.assertEqual(out["initial"], {"hasMore": False, "isLoading": False, "requestId": 0, "cancelOnScroll": True, "hasController": False})
        self.assertEqual(out["visible"], {"wrap": "flex", "disabled": False, "text": "Load older messages"})
        self.assertTrue(out["firstAuto"])
        self.assertFalse(out["secondAuto"])
        self.assertTrue(out["thirdAuto"])
        self.assertTrue(out["loading"]["snapshot"]["isLoading"])
        self.assertTrue(out["loading"]["snapshot"]["hasController"])
        self.assertEqual(out["loading"]["text"], "Loading...")
        self.assertTrue(out["currentBeforeInvalidate"])
        self.assertTrue(out["cancelBeforeInvalidate"])
        self.assertFalse(out["currentAfterInvalidate"])
        self.assertFalse(out["afterInvalidate"]["isLoading"])
        self.assertFalse(out["afterInvalidate"]["hasController"])
        self.assertEqual(out["errorShown"], {"display": "flex", "text": "Couldn’t load older messages."})
        self.assertEqual(out["afterHide"]["wrap"], "none")
        self.assertEqual(out["afterHide"]["error"], "none")
        self.assertEqual(out["afterHide"]["errorText"], "")
        self.assertIn("transcript dependency missing: olderButton", out["missingError"])
        self.assertTrue(out["frozen"])

    def test_loaded_chat_search_runtime_owns_open_query_matches_and_index(self) -> None:
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const ctx = {{ window: {{}} }};
            const vm = require("vm");
            vm.createContext(ctx);
            vm.runInContext({json.dumps(transcript_source)}, ctx);
            const runtime = ctx.window.CodoxearTranscript.createLoadedChatSearchRuntime();
            const rowA = {{ dataset: {{}}, id: "a" }};
            const rowB = {{ dataset: {{}}, id: "b" }};
            const rowC = {{ dataset: {{}}, id: "c" }};
            const initial = runtime.snapshot();
            runtime.setOpen(true);
            const query = runtime.setQuery("  Needle ");
            const first = runtime.setMatches([rowA, rowB], {{ preserveCurrent: false }});
            const focused = runtime.focusIndex(1);
            const preserved = runtime.setMatches([rowB, rowC], {{ preserveCurrent: true }});
            const targetIndex = runtime.ensureTargetRow(rowA, "Needle", (x, y) => x.id.localeCompare(y.id));
            runtime.setLoadingOlder(true);
            const loading = runtime.snapshot();
            runtime.reset();
            const reset = runtime.snapshot();
            process.stdout.write(JSON.stringify({{
              initial,
              query,
              first: {{ index: first.index, ids: first.matches.map((r) => r.id) }},
              focused: {{ index: focused.index, row: focused.row.id, ids: focused.matches.map((r) => r.id) }},
              preserved: {{ index: preserved.index, ids: preserved.matches.map((r) => r.id) }},
              targetIndex,
              rowAForcedQuery: rowA.dataset.searchForcedQuery,
              loading: {{ open: loading.open, query: loading.query, index: loading.index, ids: loading.matches.map((r) => r.id), loadingOlder: loading.loadingOlder }},
              reset,
              frozen: Object.isFrozen(runtime),
            }}));
            """
        )
        out = _run_node(js)
        self.assertEqual(out["initial"], {"open": False, "query": "", "matches": [], "index": -1, "loadingOlder": False})
        self.assertEqual(out["query"], "needle")
        self.assertEqual(out["first"], {"index": 0, "ids": ["a", "b"]})
        self.assertEqual(out["focused"], {"index": 1, "row": "b", "ids": ["a", "b"]})
        self.assertEqual(out["preserved"], {"index": 0, "ids": ["b", "c"]})
        self.assertEqual(out["targetIndex"], 0)
        self.assertEqual(out["rowAForcedQuery"], "needle")
        self.assertEqual(out["loading"], {"open": True, "query": "needle", "index": 0, "ids": ["a", "b", "c"], "loadingOlder": True})
        self.assertEqual(out["reset"], {"open": False, "query": "", "matches": [], "index": -1, "loadingOlder": False})
        self.assertTrue(out["frozen"])

    def test_chat_search_all_runtime_owns_debounce_currentness_and_result_state(self) -> None:
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const ctx = {{ window: {{}}, AbortController }};
            const vm = require("vm");
            vm.createContext(ctx);
            vm.runInContext({json.dumps(transcript_source)}, ctx);
            const timers = [];
            const cleared = [];
            const runtime = ctx.window.CodoxearTranscript.createChatSearchAllRuntime({{
              setTimeout: (fn, ms) => {{ const timer = {{ fn, ms, id: timers.length + 1 }}; timers.push(timer); return timer; }},
              clearTimeout: (timer) => cleared.push(timer && timer.id),
              AbortControllerCtor: AbortController,
              debounceMs: 300,
            }});
            const empty = runtime.schedule("", () => {{ throw new Error("empty should not run"); }});
            const scheduled = runtime.schedule(" query ", (q) => {{ ctx.ranQuery = q; }});
            const scheduledSnapshot = runtime.snapshot();
            timers[timers.length - 1].fn();
            const request1 = runtime.beginRequest();
            const current1 = runtime.isCurrent(request1);
            const request2 = runtime.beginRequest();
            const oldCurrentAfterSecond = runtime.isCurrent(request1);
            const completedOld = runtime.completeRequest(request1, {{ count: 99, truncated: true, hint: "stale" }});
            const completedNew = runtime.completeRequest(request2, {{ count: "5", truncated: true, hint: "first match" }});
            const afterComplete = runtime.snapshot();
            runtime.finishRequest(request2);
            const afterFinish = runtime.snapshot();
            const request3 = runtime.beginRequest();
            runtime.failRequest(request3);
            const afterFail = runtime.snapshot();
            runtime.schedule("later", () => {{ ctx.laterRan = true; }});
            const beforeDispose = runtime.snapshot();
            runtime.dispose();
            const afterDispose = runtime.snapshot();
            let missingError = "";
            try {{ ctx.window.CodoxearTranscript.createChatSearchAllRuntime({{ setTimeout: () => {{}} }}); }} catch (err) {{ missingError = err && err.message ? err.message : String(err); }}
            process.stdout.write(JSON.stringify({{
              empty,
              scheduled,
              scheduledSnapshot,
              ranQuery: ctx.ranQuery,
              current1,
              oldCurrentAfterSecond,
              completedOld,
              completedNew,
              afterComplete,
              afterFinish,
              afterFail,
              beforeDispose,
              afterDispose,
              cleared,
              missingError,
              frozen: Object.isFrozen(runtime),
            }}));
            """
        )
        out = _run_node(js)
        self.assertFalse(out["empty"]["scheduled"])
        self.assertTrue(out["scheduled"]["scheduled"])
        self.assertTrue(out["scheduledSnapshot"]["hasTimer"])
        self.assertEqual(out["ranQuery"], "query")
        self.assertTrue(out["current1"])
        self.assertFalse(out["oldCurrentAfterSecond"])
        self.assertFalse(out["completedOld"])
        self.assertTrue(out["completedNew"])
        self.assertEqual(out["afterComplete"]["count"], 5)
        self.assertTrue(out["afterComplete"]["truncated"])
        self.assertEqual(out["afterComplete"]["hint"], "first match")
        self.assertFalse(out["afterFinish"]["hasAbort"])
        self.assertIsNone(out["afterFail"]["count"])
        self.assertFalse(out["afterFail"]["truncated"])
        self.assertEqual(out["afterFail"]["hint"], "")
        self.assertTrue(out["beforeDispose"]["hasTimer"])
        self.assertIsNone(out["afterDispose"]["count"])
        self.assertFalse(out["afterDispose"]["hasAbort"])
        self.assertFalse(out["afterDispose"]["hasTimer"])
        self.assertIn(2, out["cleared"])
        self.assertIn("transcript dependency missing: clearTimeout", out["missingError"])
        self.assertTrue(out["frozen"])

    def test_transcript_module_normalizes_and_trims_tail_cache(self) -> None:
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const ctx = {{ window: {{}} }};
            const vm = require("vm");
            vm.createContext(ctx);
            vm.runInContext({json.dumps(transcript_source)}, ctx);
            const tx = ctx.window.CodoxearTranscript;
            const tailCache = new Map();
            const sessionIndex = new Map([["sid", {{ thread_id: "session-thread", log_path: "/session.jsonl" }}]]);
            tx.rememberTailSnapshot(tailCache, "sid", {{ thread_id: "fallback-thread", log_path: "/fallback.jsonl" }}, {{
              transcript_state: "bound",
              thread_id: "tail-thread",
              log_path: "/tail.jsonl",
              live_cursor: "c1",
              has_older: true,
              busy: true,
              queue_len: "2",
              token: {{ pct: 10 }},
              events: [
                {{ role: "system", text: "skip" }},
                {{ role: "user", text: "one", ts: 1, history_cursor: "h1" }},
                {{ role: "assistant", text: "two", message_id: "m2" }},
                {{ role: "assistant", text: "three" }},
              ],
            }}, 2);
            tx.appendTailSnapshotEvents(tailCache, sessionIndex, "sid", [{{ role: "user", text: "four" }}, {{ role: "assistant", text: "" }}], {{ maxEvents: 2, identityData: {{}} }});
            const afterAppend = tailCache.get("sid");
            tx.rememberTailSnapshot(tailCache, "sid", {{ thread_id: "fallback-thread", log_path: "/fallback.jsonl" }}, {{ transcript_state: "pending_bind" }}, 2);
            process.stdout.write(JSON.stringify({{
              afterAppend,
              deleted: !tailCache.has("sid"),
              key: tx.transcriptKey("thread", "/log"),
              failedState: tx.normalizeTranscriptState({{ transcript_state: "failed" }}),
              frozen: Object.isFrozen(tx),
            }}));
            """
        )
        out = _run_node(js)
        self.assertEqual(out["afterAppend"]["threadId"], "session-thread")
        self.assertEqual(out["afterAppend"]["logPath"], "/session.jsonl")
        self.assertEqual([ev["text"] for ev in out["afterAppend"]["events"]], ["three", "four"])
        self.assertEqual(out["afterAppend"]["queueLen"], 2)
        self.assertTrue(out["afterAppend"]["busy"])
        self.assertTrue(out["deleted"])
        self.assertEqual(out["key"], "thread\n/log")
        self.assertEqual(out["failedState"], "failed")
        self.assertTrue(out["frozen"])

    def test_transcript_renewal_ignores_old_bound_identity_until_new_log_arrives(self) -> None:
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        snippet = _source_between("function normalizeTranscriptState(data) {", "function tailCacheMatchesSession(")
        js = textwrap.dedent(
            f"""
            const ctx = {{
              selected: "sid",
              activeTranscriptState: "pending_bind",
              activeThreadId: null,
              activeLogPath: null,
              sessionTranscriptSlots: new Map(),
              pendingUser: [],
              chatInner: {{ querySelector: () => null }},
              window: {{}},
            }};
            vm = require("vm");
            vm.createContext(ctx);
            vm.runInContext({json.dumps(transcript_source)}, ctx);
            ctx.codoxearTranscript = ctx.window.CodoxearTranscript;
            vm.runInContext({json.dumps(snippet)}, ctx);
            ctx.updateSessionTranscriptSlot("sid", {{
              transcript_state: "bound",
              thread_id: "old-thread",
              log_path: "/old.jsonl",
            }});
            const before = ctx.getSessionTranscriptSlot("sid");
            ctx.beginTranscriptRenewal("sid");
            const pending = ctx.getSessionTranscriptSlot("sid");
            const stale = ctx.updateSessionTranscriptSlot("sid", {{
              transcript_state: "bound",
              thread_id: "old-thread",
              log_path: "/old.jsonl",
            }});
            const afterStale = ctx.getSessionTranscriptSlot("sid");
            const fresh = ctx.updateSessionTranscriptSlot("sid", {{
              transcript_state: "bound",
              thread_id: "new-thread",
              log_path: "/new.jsonl",
            }});
            const afterFresh = ctx.getSessionTranscriptSlot("sid");
            process.stdout.write(JSON.stringify({{
              before,
              pending,
              staleIgnored: stale.ignoredStaleBound,
              afterStale,
              freshIgnored: fresh.ignoredStaleBound,
              afterFresh,
              activeTranscriptState: ctx.activeTranscriptState,
              activeThreadId: ctx.activeThreadId,
              activeLogPath: ctx.activeLogPath,
            }}));
            """
        )
        out = _run_node(js)

        self.assertEqual(out["before"]["state"], "bound")
        self.assertEqual(out["before"]["key"], "old-thread\n/old.jsonl")
        self.assertEqual(out["pending"]["state"], "pending_bind")
        self.assertEqual(out["pending"]["ignoredKey"], "old-thread\n/old.jsonl")
        self.assertTrue(out["staleIgnored"])
        self.assertEqual(out["afterStale"]["state"], "pending_bind")
        self.assertEqual(out["afterStale"]["ignoredKey"], "old-thread\n/old.jsonl")
        self.assertFalse(out["freshIgnored"])
        self.assertEqual(out["afterFresh"]["state"], "bound")
        self.assertEqual(out["afterFresh"]["key"], "new-thread\n/new.jsonl")
        self.assertEqual(out["afterFresh"]["ignoredKey"], None)
        self.assertEqual(out["activeTranscriptState"], "bound")
        self.assertEqual(out["activeThreadId"], "new-thread")
        self.assertEqual(out["activeLogPath"], "/new.jsonl")

    def test_history_request_cursor_comes_from_oldest_rendered_row(self) -> None:
        snippet = _source_between("async function loadOlderMessages({ auto = false, cancelOnScroll = true } = {}) {", "function maybeAutoLoadOlder()")
        js = textwrap.dedent(
            f"""
            const ctx = {{
              selected: "sid",
              hasOlder: true,
              loadingOlder: false,
              pollGen: 7,
              olderLoadRequestId: 0,
              olderAutoTriggerAt: 0,
              OLDER_AUTO_COOLDOWN_MS: 450,
              olderLoadController: null,
              performance: {{ now: () => 1000 }},
              AbortController,
              encodeURIComponent,
              olderPageLimit: () => 60,
              oldestRenderedHistoryCursor: () => "cursor-oldest-row",
              setOlderState: (state) => {{ ctx.lastOlderState = state; ctx.hasOlder = Boolean(state.hasMore); ctx.loadingOlder = Boolean(state.isLoading); }},
              clearOlderLoadError: () => {{ ctx.clearedOlderError = true; }},
              showOlderLoadError: () => {{ ctx.showedOlderError = true; }},
              prependOlderEvents: (events, opts) => {{ ctx.prepended = {{ events, opts }}; }},
              openSession: async () => {{ throw new Error("should not reopen"); }},
              api: async (url) => {{
                ctx.requestUrl = url;
                return {{ events: [{{ role: "assistant", text: "older" }}], has_older: false }};
              }},
            }};
            ctx.olderLoadSnapshot = () => ctx.olderLoadRuntime.snapshot();
            ctx.hasOlderMessages = () => ctx.olderLoadSnapshot().hasMore;
            ctx.isLoadingOlderMessages = () => ctx.olderLoadSnapshot().isLoading;
            ctx.olderLoadRuntime = {{
              snapshot: () => ({{ hasMore: Boolean(ctx.hasOlder), isLoading: Boolean(ctx.loadingOlder), requestId: ctx.olderLoadRequestId, cancelOnScroll: ctx.olderLoadCancelOnScroll !== false, hasController: Boolean(ctx.olderLoadController) }}),
              markAutoTrigger: () => {{
                const now = ctx.performance.now();
                if (now - ctx.olderAutoTriggerAt < ctx.OLDER_AUTO_COOLDOWN_MS) return false;
                ctx.olderAutoTriggerAt = now;
                return true;
              }},
              beginLoad: ({{ cancelOnScroll = true }} = {{}}) => {{
                ctx.olderLoadRequestId += 1;
                const ctl = new AbortController();
                ctx.olderLoadController = ctl;
                ctx.olderLoadCancelOnScroll = Boolean(cancelOnScroll);
                ctx.setOlderState({{ hasMore: ctx.hasOlder, isLoading: true }});
                return {{ requestId: ctx.olderLoadRequestId, controller: ctl, signal: ctl.signal }};
              }},
              isCurrent: (load) => load && load.requestId === ctx.olderLoadRequestId,
              finishLoad: (load) => {{ if (load && ctx.olderLoadController === load.controller) ctx.olderLoadController = null; ctx.olderLoadCancelOnScroll = true; }},
              invalidate: () => {{ ctx.olderLoadRequestId += 1; ctx.olderLoadController = null; ctx.olderLoadCancelOnScroll = true; if (ctx.loadingOlder) ctx.setOlderState({{ hasMore: ctx.hasOlder, isLoading: false }}); }},
            }};
            vm = require("vm");
            vm.createContext(ctx);
            vm.runInContext({json.dumps(snippet)}, ctx);
            ctx.loadOlderMessages({{ auto: false }}).then(() => {{
              process.stdout.write(JSON.stringify({{
                requestUrl: ctx.requestUrl,
                lastOlderState: ctx.lastOlderState,
                prepended: ctx.prepended,
              }}));
            }});
            """
        )
        out = _run_node(js)

        self.assertIn("cursor=cursor-oldest-row", out["requestUrl"])
        self.assertEqual(out["lastOlderState"], {"hasMore": False, "isLoading": False})
        self.assertEqual(out["prepended"]["events"][0]["text"], "older")

    def test_history_failure_preserves_has_older_and_shows_retry_error(self) -> None:
        snippet = _source_between("async function loadOlderMessages({ auto = false, cancelOnScroll = true } = {}) {", "function maybeAutoLoadOlder()")
        js = textwrap.dedent(
            f"""
            const ctx = {{
              selected: "sid",
              hasOlder: true,
              loadingOlder: false,
              pollGen: 7,
              olderLoadRequestId: 0,
              olderAutoTriggerAt: 0,
              OLDER_AUTO_COOLDOWN_MS: 450,
              olderLoadController: null,
              performance: {{ now: () => 1000 }},
              AbortController,
              encodeURIComponent,
              olderPageLimit: () => 60,
              oldestRenderedHistoryCursor: () => "cursor-oldest-row",
              setOlderState: (state) => {{ ctx.lastOlderState = state; ctx.hasOlder = Boolean(state.hasMore); ctx.loadingOlder = Boolean(state.isLoading); }},
              clearOlderLoadError: () => {{ ctx.clearedOlderError = true; }},
              showOlderLoadError: () => {{ ctx.showedOlderError = true; }},
              prependOlderEvents: () => {{ ctx.prepended = true; }},
              openSession: async () => {{ throw new Error("should not reopen"); }},
              api: async () => {{ const err = new Error("unavailable"); err.status = 503; throw err; }},
            }};
            ctx.olderLoadSnapshot = () => ctx.olderLoadRuntime.snapshot();
            ctx.hasOlderMessages = () => ctx.olderLoadSnapshot().hasMore;
            ctx.isLoadingOlderMessages = () => ctx.olderLoadSnapshot().isLoading;
            ctx.olderLoadRuntime = {{
              snapshot: () => ({{ hasMore: Boolean(ctx.hasOlder), isLoading: Boolean(ctx.loadingOlder), requestId: ctx.olderLoadRequestId, cancelOnScroll: ctx.olderLoadCancelOnScroll !== false, hasController: Boolean(ctx.olderLoadController) }}),
              markAutoTrigger: () => {{
                const now = ctx.performance.now();
                if (now - ctx.olderAutoTriggerAt < ctx.OLDER_AUTO_COOLDOWN_MS) return false;
                ctx.olderAutoTriggerAt = now;
                return true;
              }},
              beginLoad: ({{ cancelOnScroll = true }} = {{}}) => {{
                ctx.olderLoadRequestId += 1;
                const ctl = new AbortController();
                ctx.olderLoadController = ctl;
                ctx.olderLoadCancelOnScroll = Boolean(cancelOnScroll);
                ctx.setOlderState({{ hasMore: ctx.hasOlder, isLoading: true }});
                return {{ requestId: ctx.olderLoadRequestId, controller: ctl, signal: ctl.signal }};
              }},
              isCurrent: (load) => load && load.requestId === ctx.olderLoadRequestId,
              finishLoad: (load) => {{ if (load && ctx.olderLoadController === load.controller) ctx.olderLoadController = null; ctx.olderLoadCancelOnScroll = true; }},
              invalidate: () => {{ ctx.olderLoadRequestId += 1; ctx.olderLoadController = null; ctx.olderLoadCancelOnScroll = true; if (ctx.loadingOlder) ctx.setOlderState({{ hasMore: ctx.hasOlder, isLoading: false }}); }},
            }};
            vm = require("vm");
            vm.createContext(ctx);
            vm.runInContext({json.dumps(snippet)}, ctx);
            ctx.loadOlderMessages({{ auto: false }}).then(() => {{
              process.stdout.write(JSON.stringify({{
                lastOlderState: ctx.lastOlderState,
                showedOlderError: Boolean(ctx.showedOlderError),
                prepended: Boolean(ctx.prepended),
              }}));
            }});
            """
        )
        out = _run_node(js)

        self.assertEqual(out["lastOlderState"], {"hasMore": True, "isLoading": False})
        self.assertTrue(out["showedOlderError"])
        self.assertFalse(out["prepended"])

    def test_history_401_triggers_auth_loss_without_retry_error(self) -> None:
        snippet = _source_between("async function loadOlderMessages({ auto = false, cancelOnScroll = true } = {}) {", "function maybeAutoLoadOlder()")
        js = textwrap.dedent(
            f"""
            const ctx = {{
              selected: "sid",
              hasOlder: true,
              loadingOlder: false,
              pollGen: 7,
              olderLoadRequestId: 0,
              olderAutoTriggerAt: 0,
              OLDER_AUTO_COOLDOWN_MS: 450,
              olderLoadController: null,
              performance: {{ now: () => 1000 }},
              AbortController,
              encodeURIComponent,
              olderPageLimit: () => 60,
              oldestRenderedHistoryCursor: () => "cursor-oldest-row",
              setOlderState: (state) => {{ ctx.lastOlderState = state; ctx.hasOlder = Boolean(state.hasMore); ctx.loadingOlder = Boolean(state.isLoading); }},
              clearOlderLoadError: () => {{ ctx.clearedOlderError = true; }},
              showOlderLoadError: () => {{ ctx.showedOlderError = true; }},
              handleAppAuthLoss: () => {{ ctx.authLoss = true; }},
              prependOlderEvents: () => {{ ctx.prepended = true; }},
              openSession: async () => {{ throw new Error("should not reopen"); }},
              api: async () => {{ const err = new Error("unauthorized"); err.status = 401; throw err; }},
            }};
            ctx.olderLoadSnapshot = () => ctx.olderLoadRuntime.snapshot();
            ctx.hasOlderMessages = () => ctx.olderLoadSnapshot().hasMore;
            ctx.isLoadingOlderMessages = () => ctx.olderLoadSnapshot().isLoading;
            ctx.olderLoadRuntime = {{
              snapshot: () => ({{ hasMore: Boolean(ctx.hasOlder), isLoading: Boolean(ctx.loadingOlder), requestId: ctx.olderLoadRequestId, cancelOnScroll: ctx.olderLoadCancelOnScroll !== false, hasController: Boolean(ctx.olderLoadController) }}),
              markAutoTrigger: () => {{
                const now = ctx.performance.now();
                if (now - ctx.olderAutoTriggerAt < ctx.OLDER_AUTO_COOLDOWN_MS) return false;
                ctx.olderAutoTriggerAt = now;
                return true;
              }},
              beginLoad: ({{ cancelOnScroll = true }} = {{}}) => {{
                ctx.olderLoadRequestId += 1;
                const ctl = new AbortController();
                ctx.olderLoadController = ctl;
                ctx.olderLoadCancelOnScroll = Boolean(cancelOnScroll);
                ctx.setOlderState({{ hasMore: ctx.hasOlder, isLoading: true }});
                return {{ requestId: ctx.olderLoadRequestId, controller: ctl, signal: ctl.signal }};
              }},
              isCurrent: (load) => load && load.requestId === ctx.olderLoadRequestId,
              finishLoad: (load) => {{ if (load && ctx.olderLoadController === load.controller) ctx.olderLoadController = null; ctx.olderLoadCancelOnScroll = true; }},
              invalidate: () => {{ ctx.olderLoadRequestId += 1; ctx.olderLoadController = null; ctx.olderLoadCancelOnScroll = true; if (ctx.loadingOlder) ctx.setOlderState({{ hasMore: ctx.hasOlder, isLoading: false }}); }},
            }};
            vm = require("vm");
            vm.createContext(ctx);
            vm.runInContext({json.dumps(snippet)}, ctx);
            ctx.loadOlderMessages({{ auto: false }}).then(() => {{
              process.stdout.write(JSON.stringify({{
                authLoss: Boolean(ctx.authLoss),
                showedOlderError: Boolean(ctx.showedOlderError),
                prepended: Boolean(ctx.prepended),
              }}));
            }});
            """
        )
        out = _run_node(js)

        self.assertTrue(out["authLoss"])
        self.assertFalse(out["showedOlderError"])
        self.assertFalse(out["prepended"])

    def test_stale_history_401_still_triggers_auth_loss(self) -> None:
        snippet = _source_between("async function loadOlderMessages({ auto = false, cancelOnScroll = true } = {}) {", "function maybeAutoLoadOlder()")
        js = textwrap.dedent(
            f"""
            const ctx = {{
              selected: "sid",
              hasOlder: true,
              loadingOlder: false,
              pollGen: 7,
              olderLoadRequestId: 0,
              olderAutoTriggerAt: 0,
              OLDER_AUTO_COOLDOWN_MS: 450,
              olderLoadController: null,
              performance: {{ now: () => 1000 }},
              AbortController,
              encodeURIComponent,
              olderPageLimit: () => 60,
              oldestRenderedHistoryCursor: () => "cursor-oldest-row",
              setOlderState: (state) => {{ ctx.lastOlderState = state; ctx.hasOlder = Boolean(state.hasMore); ctx.loadingOlder = Boolean(state.isLoading); }},
              clearOlderLoadError: () => {{ ctx.clearedOlderError = true; }},
              showOlderLoadError: () => {{ ctx.showedOlderError = true; }},
              handleAppAuthLoss: () => {{ ctx.authLoss = true; }},
              prependOlderEvents: () => {{ ctx.prepended = true; }},
              openSession: async () => {{ throw new Error("should not reopen"); }},
              api: async () => {{ ctx.olderLoadRequestId += 1; const err = new Error("unauthorized"); err.status = 401; throw err; }},
            }};
            ctx.olderLoadSnapshot = () => ctx.olderLoadRuntime.snapshot();
            ctx.hasOlderMessages = () => ctx.olderLoadSnapshot().hasMore;
            ctx.isLoadingOlderMessages = () => ctx.olderLoadSnapshot().isLoading;
            ctx.olderLoadRuntime = {{
              snapshot: () => ({{ hasMore: Boolean(ctx.hasOlder), isLoading: Boolean(ctx.loadingOlder), requestId: ctx.olderLoadRequestId, cancelOnScroll: ctx.olderLoadCancelOnScroll !== false, hasController: Boolean(ctx.olderLoadController) }}),
              markAutoTrigger: () => {{
                const now = ctx.performance.now();
                if (now - ctx.olderAutoTriggerAt < ctx.OLDER_AUTO_COOLDOWN_MS) return false;
                ctx.olderAutoTriggerAt = now;
                return true;
              }},
              beginLoad: ({{ cancelOnScroll = true }} = {{}}) => {{
                ctx.olderLoadRequestId += 1;
                const ctl = new AbortController();
                ctx.olderLoadController = ctl;
                ctx.olderLoadCancelOnScroll = Boolean(cancelOnScroll);
                ctx.setOlderState({{ hasMore: ctx.hasOlder, isLoading: true }});
                return {{ requestId: ctx.olderLoadRequestId, controller: ctl, signal: ctl.signal }};
              }},
              isCurrent: (load) => load && load.requestId === ctx.olderLoadRequestId,
              finishLoad: (load) => {{ if (load && ctx.olderLoadController === load.controller) ctx.olderLoadController = null; ctx.olderLoadCancelOnScroll = true; }},
              invalidate: () => {{ ctx.olderLoadRequestId += 1; ctx.olderLoadController = null; ctx.olderLoadCancelOnScroll = true; if (ctx.loadingOlder) ctx.setOlderState({{ hasMore: ctx.hasOlder, isLoading: false }}); }},
            }};
            vm = require("vm");
            vm.createContext(ctx);
            vm.runInContext({json.dumps(snippet)}, ctx);
            ctx.loadOlderMessages({{ auto: false }}).then(() => {{
              process.stdout.write(JSON.stringify({{
                authLoss: Boolean(ctx.authLoss),
                showedOlderError: Boolean(ctx.showedOlderError),
                prepended: Boolean(ctx.prepended),
              }}));
            }});
            """
        )
        out = _run_node(js)

        self.assertTrue(out["authLoss"])
        self.assertFalse(out["showedOlderError"])
        self.assertFalse(out["prepended"])

    def test_live_delta_does_not_splice_into_history_window(self) -> None:
        snippet = _source_between("function appendEvent(ev) {", "function renderTranscript(")
        js = textwrap.dedent(
            f"""
            const ctx = {{
              transcriptScrollRuntime: {{
                snapshot: () => ({{ renderedAtLiveTail: false }}),
                shouldStickToBottom: () => false,
                syncJumpButton: () => {{ ctx.jumps += 1; }},
                scheduleScrollToBottom: () => {{ ctx.scrolled = true; }},
              }},
              seen: 0,
              jumps: 0,
              made: 0,
              consumePendingUserIfMatches: () => false,
              isDuplicateEvent: () => false,
              isAdjacentAssistantDuplicateEvent: () => false,
              isNearBottom: () => false,
              markEventSeen: () => {{ ctx.seen += 1; }},
              syncJumpButton: () => {{ ctx.jumps += 1; }},
              safeMakeRow: () => {{
                ctx.made += 1;
                return {{ row: {{}}, bubble: {{}} }};
              }},
              Date,
              typingRow: null,
              bottomSentinel: {{}},
              chatInner: {{ insertBefore: () => {{ ctx.inserted = true; }} }},
              trimRenderedRows: () => {{ ctx.trimmed = true; }},
              rebuildDecorations: () => {{ ctx.rebuilt = true; }},
              markClickFirstPaint: () => {{ ctx.painted = true; }},
              requestAnimationFrame: (fn) => fn(),
              scrollToBottom: () => {{ ctx.scrolled = true; }},
            }};
            vm = require("vm");
            vm.createContext(ctx);
            vm.runInContext({json.dumps(snippet)}, ctx);
            ctx.appendEvent({{ role: "assistant", text: "new tail", ts: 2 }});
            process.stdout.write(JSON.stringify({{
              made: ctx.made,
              seen: ctx.seen,
              jumps: ctx.jumps,
              inserted: Boolean(ctx.inserted),
              trimmed: Boolean(ctx.trimmed),
              rebuilt: Boolean(ctx.rebuilt),
              painted: Boolean(ctx.painted),
            }}));
            """
        )
        out = _run_node(js)

        self.assertEqual(out["made"], 0)
        self.assertEqual(out["seen"], 1)
        self.assertEqual(out["jumps"], 1)
        self.assertFalse(out["inserted"])
        self.assertFalse(out["trimmed"])
        self.assertFalse(out["rebuilt"])
        self.assertFalse(out["painted"])

    def test_live_delta_dedupes_adjacent_assistant_text_across_polls(self) -> None:
        identity_source = APP_MESSAGE_IDENTITY_JS.read_text(encoding="utf-8")
        helper_snippet = _source_between("function eventKey(ev) {", "function isTranscriptRenewalCommand(")
        append_snippet = _source_between("function appendEvent(ev) {", "function renderTranscript(")
        js = textwrap.dedent(
            f"""
            const ctx = {{
              transcriptScrollRuntime: {{
                snapshot: () => ({{ renderedAtLiveTail: true }}),
                shouldStickToBottom: () => true,
                syncJumpButton: () => {{ ctx.jumped = true; }},
                scheduleScrollToBottom: () => {{ ctx.scrolled = true; }},
              }},
              recentEventKeys: [],
              recentEventKeySet: new Set(),
              RECENT_EVENT_KEYS_MAX: 320,
              made: 0,
              inserted: 0,
              rows: [{{ dataset: {{ role: "assistant", assistantDedupeKey: "final_response|same final text" }} }}],
              window: {{}},
              renderedMessageRows: () => ctx.rows,
              consumePendingUserIfMatches: () => false,
              isNearBottom: () => true,
              safeMakeRow: () => {{
                ctx.made += 1;
                return {{ row: {{}}, bubble: {{}} }};
              }},
              typingRow: null,
              bottomSentinel: {{}},
              chatInner: {{ insertBefore: () => {{ ctx.inserted += 1; }} }},
              trimRenderedRows: () => {{ ctx.trimmed = true; }},
              rebuildDecorations: () => {{ ctx.rebuilt = true; }},
              markClickFirstPaint: () => {{ ctx.painted = true; }},
              requestAnimationFrame: (fn) => fn(),
              scrollToBottom: () => {{ ctx.scrolled = true; }},
              syncJumpButton: () => {{ ctx.jumped = true; }},
            }};
            vm = require("vm");
            vm.createContext(ctx);
            vm.runInContext({json.dumps(identity_source)}, ctx);
            ctx.codoxearMessageIdentity = ctx.window.CodoxearMessageIdentity;
            vm.runInContext({json.dumps(helper_snippet)}, ctx);
            vm.runInContext({json.dumps(append_snippet)}, ctx);
            ctx.appendEvent({{ role: "assistant", text: "same final text", message_class: "final_response", ts: 2.4 }});
            const afterDuplicate = {{ made: ctx.made, inserted: ctx.inserted, seen: ctx.recentEventKeys.slice() }};
            ctx.rows = [{{ dataset: {{ role: "user" }} }}];
            ctx.appendEvent({{ role: "assistant", text: "same final text", message_class: "final_response", ts: 3.0 }});
            process.stdout.write(JSON.stringify({{
              afterDuplicate,
              final: {{ made: ctx.made, inserted: ctx.inserted, seen: ctx.recentEventKeys.slice() }},
            }}));
            """
        )
        out = _run_node(js)

        self.assertEqual(out["afterDuplicate"]["made"], 0)
        self.assertEqual(out["afterDuplicate"]["inserted"], 0)
        self.assertEqual(out["afterDuplicate"]["seen"], ["assistant|2400|same final text"])
        self.assertEqual(out["final"]["made"], 1)
        self.assertEqual(out["final"]["inserted"], 1)
        self.assertEqual(out["final"]["seen"], ["assistant|2400|same final text", "assistant|3000|same final text"])

    def test_new_command_send_failure_does_not_detach_current_transcript(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("async function sendText(")
        end = source.index("form.onsubmit = async", start)
        snippet = source[start:end]
        js = textwrap.dedent(
            f"""
            const ctx = {{
              selected: "sid",
              sending: false,
              localEchoSeq: 0,
              transcriptScrollRuntime: {{
                snapshot: () => ({{ renderedAtLiveTail: true }}),
              }},
              pendingUser: [],
              sessionIndex: new Map([["sid", {{ agent_backend: "codex" }}]]),
              detached: 0,
              renderedPending: 0,
              sessionTailCache: {{ delete: () => {{ ctx.deletedCache = true; }} }},
              beginTranscriptRenewal: () => {{ ctx.detached += 1; }},
              clearRenderedTranscriptRange: () => {{ ctx.clearedRange = true; }},
              invalidateOlderLoad: () => {{ ctx.invalidated = true; }},
              renderPendingTranscriptSlot: () => {{ ctx.renderedPending += 1; }},
              sessionAgentBackend: (s) => s.agent_backend || "codex",
              sessionLaunchFailed: (s) => Boolean(s && String(s.launch_state || "").toLowerCase() === "failed"),
              isTranscriptRenewalCommand: () => true,
              setToast: (text) => {{ ctx.toast = text; }},
              $: () => ({{ disabled: false }}),
              chatInner: {{ querySelector: () => null }},
              api: async () => {{ throw new Error("broker down"); }},
              setAttachCount: () => {{}},
              syncSendButtonState: () => {{}},
              syncAttachButtonState: () => {{}},
              kickPoll: () => {{}},
              refreshSessions: async () => {{}},
              console,
              Date,
            }};
            vm = require("vm");
            vm.createContext(ctx);
            vm.runInContext({json.dumps(snippet)}, ctx);
            ctx.sendText("/new").then((ok) => {{
              process.stdout.write(JSON.stringify({{
                ok,
                detached: ctx.detached,
                renderedPending: ctx.renderedPending,
                deletedCache: Boolean(ctx.deletedCache),
                toast: ctx.toast,
              }}));
            }});
            """
        )
        out = _run_node(js)

        self.assertFalse(out["ok"])
        self.assertEqual(out["detached"], 0)
        self.assertEqual(out["renderedPending"], 0)
        self.assertFalse(out["deletedCache"])
        self.assertEqual(out["toast"], "send error: broker down")


if __name__ == "__main__":
    unittest.main()
