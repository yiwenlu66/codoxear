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

    def test_transcript_event_runtime_owns_recent_events_and_pending_echoes(self) -> None:
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        identity_source = APP_MESSAGE_IDENTITY_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const ctx = {{ window: {{}} }};
            const vm = require("vm");
            vm.createContext(ctx);
            vm.runInContext({json.dumps(identity_source)}, ctx);
            vm.runInContext({json.dumps(transcript_source)}, ctx);
            const identity = ctx.window.CodoxearMessageIdentity;
            const runtime = ctx.window.CodoxearTranscript.createTranscriptEventRuntime({{
              eventKey: identity.eventKey,
              pendingMatchKey: identity.pendingMatchKey,
              normalizePendingText: identity.normalizeTextForPendingMatch,
              assistantDedupeKey: identity.chatAssistantDedupeKey,
              maxRecentEventKeys: 2,
            }});
            const firstSeen = runtime.markEventSeen({{ role: "assistant", text: "one", ts: 1 }});
            const firstDuplicate = runtime.isDuplicateEvent({{ role: "assistant", text: "one", ts: 1 }});
            runtime.markEventSeen({{ role: "assistant", text: "two", ts: 2 }});
            runtime.markEventSeen({{ role: "assistant", text: "three", ts: 3 }});
            const afterEvict = {{ keys: runtime.snapshot().recentEventKeys, evictedDuplicate: runtime.isDuplicateEvent({{ role: "assistant", text: "one", ts: 1 }}) }};
            const adjacentTrue = runtime.isAdjacentAssistantDuplicateEvent(
              {{ role: "assistant", text: "same final text", message_class: "final_response", ts: 4 }},
              {{ renderedAtLiveTail: true, rows: [{{ dataset: {{ role: "assistant", assistantDedupeKey: "final_response|same final text" }} }}] }}
            );
            const adjacentFalseOffTail = runtime.isAdjacentAssistantDuplicateEvent(
              {{ role: "assistant", text: "same final text", message_class: "final_response", ts: 5 }},
              {{ renderedAtLiveTail: false, rows: [{{ dataset: {{ role: "assistant", assistantDedupeKey: "final_response|same final text" }} }}] }}
            );
            const id1 = runtime.nextLocalEchoId();
            runtime.addPendingUser({{ id: id1, sessionId: "sid", epoch: 1, text: "hello  ", t0: 10 }});
            runtime.addPendingUser({{ sessionId: "sid", epoch: 1, text: "later", t0: 12 }});
            runtime.addPendingUser({{ sessionId: "sid", epoch: 2, text: "other epoch", t0: 8 }});
            const pendingEpoch1 = runtime.pendingUsersForSession("sid", 1).map((item) => [item.id, item.text, item.epoch]);
            const exactMatch = runtime.takePendingUserMatch({{ role: "user", text: "hello", ts: 10.2 }}, "sid", 1);
            const hasAfterExact = runtime.hasPendingForSession("sid");
            const noUntimed = runtime.takePendingUserMatch({{ role: "user", text: "unrelated" }}, "sid", 1, {{ allowUntimedCommit: false }});
            const fallbackTimed = runtime.takePendingUserMatch({{ role: "user", text: "unrelated", ts: 20 }}, "sid", 1);
            const dropped = runtime.dropPendingUsers("sid", (item) => item.epoch === 2);
            const finalSnapshot = runtime.snapshot();
            let missingError = "";
            try {{ ctx.window.CodoxearTranscript.createTranscriptEventRuntime({{ eventKey: () => "" }}); }} catch (err) {{ missingError = err && err.message ? err.message : String(err); }}
            process.stdout.write(JSON.stringify({{
              firstSeen,
              firstDuplicate,
              afterEvict,
              adjacentTrue,
              adjacentFalseOffTail,
              id1,
              pendingEpoch1,
              exactMatch: exactMatch && {{ id: exactMatch.id, text: exactMatch.text, epoch: exactMatch.epoch }},
              hasAfterExact,
              noUntimed,
              fallbackTimed: fallbackTimed && {{ text: fallbackTimed.text, epoch: fallbackTimed.epoch }},
              dropped: dropped.map((item) => item.text),
              finalSnapshot,
              missingError,
              frozen: Object.isFrozen(runtime),
            }}));
            """
        )
        out = _run_node(js)
        self.assertTrue(out["firstSeen"])
        self.assertTrue(out["firstDuplicate"])
        self.assertEqual(out["afterEvict"], {"keys": ["assistant|2000|two", "assistant|3000|three"], "evictedDuplicate": False})
        self.assertTrue(out["adjacentTrue"])
        self.assertFalse(out["adjacentFalseOffTail"])
        self.assertEqual(out["id1"], 1)
        self.assertEqual(out["pendingEpoch1"], [[1, "hello  ", 1], [2, "later", 1]])
        self.assertEqual(out["exactMatch"], {"id": 1, "text": "hello  ", "epoch": 1})
        self.assertTrue(out["hasAfterExact"])
        self.assertFalse(out["noUntimed"])
        self.assertEqual(out["fallbackTimed"], {"text": "later", "epoch": 1})
        self.assertEqual(out["dropped"], ["other epoch"])
        self.assertEqual(out["finalSnapshot"]["pendingCount"], 0)
        self.assertIn("transcript dependency missing: pendingMatchKey", out["missingError"])
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
        self.assertEqual(out["afterAppend"]["historyCursor"], "h1")
        self.assertTrue(out["afterAppend"]["hasOlder"])
        self.assertTrue(out["afterAppend"]["busy"])
        self.assertTrue(out["deleted"])
        self.assertEqual(out["key"], "thread\n/log")
        self.assertEqual(out["failedState"], "failed")
        self.assertTrue(out["frozen"])

    def test_recovered_failed_tail_can_page_when_history_cursor_exists(self) -> None:
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const ctx = {{ window: {{}} }};
            const vm = require("vm");
            vm.createContext(ctx);
            vm.runInContext({json.dumps(transcript_source)}, ctx);
            const tx = ctx.window.CodoxearTranscript;
            const recoveredTopLevel = {{ transcript_state: "failed", has_older: true, history_cursor: "top-cursor", events: [] }};
            const recoveredEventCursor = {{ transcript_state: "failed", has_older: true, events: [{{ role: "assistant", text: "backend stopped", history_cursor: "row-cursor" }}] }};
            const preLogFailed = {{ transcript_state: "failed", has_older: true, events: [{{ role: "assistant", text: "launch failed" }}] }};
            const noOlder = {{ transcript_state: "failed", has_older: false, history_cursor: "top-cursor", events: [] }};
            process.stdout.write(JSON.stringify({{
              topCursor: tx.historyCursorFromPayload(recoveredTopLevel),
              eventCursor: tx.historyCursorFromPayload(recoveredEventCursor),
              topUsable: tx.hasUsableOlderHistory(recoveredTopLevel),
              eventUsable: tx.hasUsableOlderHistory(recoveredEventCursor),
              preLogUsable: tx.hasUsableOlderHistory(preLogFailed),
              noOlderUsable: tx.hasUsableOlderHistory(noOlder),
              frozen: Object.isFrozen(tx),
            }}));
            """
        )
        out = _run_node(js)

        self.assertEqual(out["topCursor"], "top-cursor")
        self.assertEqual(out["eventCursor"], "row-cursor")
        self.assertTrue(out["topUsable"])
        self.assertTrue(out["eventUsable"])
        self.assertFalse(out["preLogUsable"])
        self.assertFalse(out["noOlderUsable"])
        self.assertTrue(out["frozen"])

    def test_app_runtime_uses_cursor_not_bound_state_for_older_affordance(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        block = _source_between("function applySessionRuntimeFromTail(sessionId, data) {", "function renderSessionTail(events)")
        self.assertIn("activeTailHistoryCursor = usableOlderHistoryCursor(data);", block)
        self.assertIn("setOlderState({ hasMore: Boolean(activeTailHistoryCursor), isLoading: false });", block)
        self.assertNotIn("slot.state === \"bound\" && Boolean(data && data.has_older)", block)

    def test_transcript_renewal_ignores_old_bound_identity_until_new_log_arrives(self) -> None:
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const ctx = {{ window: {{}} }};
            const vm = require("vm");
            vm.createContext(ctx);
            vm.runInContext({json.dumps(transcript_source)}, ctx);
            const tx = ctx.window.CodoxearTranscript;
            const sessionIndex = new Map([["sid", {{ thread_id: "new-thread", log_path: "/new.jsonl" }}]]);
            const runtime = tx.createTranscriptSlotRuntime({{ sessionIndex, maxTailEvents: 2 }});
            runtime.updateSlot("sid", {{ transcript_state: "bound", thread_id: "old-thread", log_path: "/old.jsonl" }});
            runtime.syncActiveSlot("sid");
            const before = runtime.getSlot("sid");
            const renewal = runtime.beginRenewal("sid");
            runtime.syncActiveSlot("sid");
            const pending = runtime.getSlot("sid");
            const stale = runtime.updateSlot("sid", {{ transcript_state: "bound", thread_id: "old-thread", log_path: "/old.jsonl" }});
            runtime.syncActiveSlot("sid");
            const afterStale = runtime.getSlot("sid");
            const fresh = runtime.updateSlot("sid", {{ transcript_state: "bound", thread_id: "new-thread", log_path: "/new.jsonl" }});
            runtime.syncActiveSlot("sid");
            const afterFresh = runtime.getSlot("sid");
            runtime.setLiveCursor("cursor-1");
            const activeWithCursor = runtime.activeSnapshot();
            runtime.rememberTail("sid", {{ thread_id: "new-thread", log_path: "/new.jsonl" }}, {{
              transcript_state: "bound",
              thread_id: "new-thread",
              log_path: "/new.jsonl",
              live_cursor: "tail-cursor",
              events: [
                {{ role: "user", text: "one" }},
                {{ role: "assistant", text: "two" }},
                {{ role: "assistant", text: "three" }},
              ],
              busy: true,
              queue_len: 1,
              token: {{ pct: 50 }},
            }});
            const cached = runtime.getTailCache("sid");
            runtime.appendTailEvents("sid", [{{ role: "user", text: "four" }}], {{ liveCursor: "live-2", busy: false, queueLen: 2, token: {{ pct: 49 }} }});
            const afterAppend = runtime.getTailCache("sid");
            const matchesSession = runtime.tailCacheMatchesSession(afterAppend, {{ thread_id: "new-thread", log_path: "/new.jsonl" }});
            const beforeDelete = runtime.snapshot();
            runtime.deleteSession("sid");
            const afterDelete = runtime.snapshot();
            runtime.setActiveFailed();
            const failed = runtime.activeSnapshot();
            process.stdout.write(JSON.stringify({{
              before,
              renewal,
              pending,
              staleIgnored: stale.ignoredStaleBound,
              afterStale,
              freshIgnored: fresh.ignoredStaleBound,
              afterFresh,
              active: activeWithCursor,
              cachedTexts: cached.events.map((ev) => ev.text),
              afterAppendTexts: afterAppend.events.map((ev) => ev.text),
              afterAppendCursor: afterAppend.liveCursor,
              afterAppendQueue: afterAppend.queueLen,
              matchesSession,
              beforeDelete,
              afterDelete,
              failed,
              frozen: Object.isFrozen(runtime),
            }}));
            """
        )
        out = _run_node(js)

        self.assertEqual(out["before"]["state"], "bound")
        self.assertEqual(out["before"]["key"], "old-thread\n/old.jsonl")
        self.assertEqual(out["renewal"]["current"]["ignoredKey"], "old-thread\n/old.jsonl")
        self.assertEqual(out["pending"]["state"], "pending_bind")
        self.assertTrue(out["staleIgnored"])
        self.assertEqual(out["afterStale"]["state"], "pending_bind")
        self.assertFalse(out["freshIgnored"])
        self.assertEqual(out["afterFresh"]["state"], "bound")
        self.assertEqual(out["afterFresh"]["key"], "new-thread\n/new.jsonl")
        self.assertEqual(out["active"]["liveCursor"], "cursor-1")
        self.assertEqual(out["cachedTexts"], ["two", "three"])
        self.assertEqual(out["afterAppendTexts"], ["three", "four"])
        self.assertEqual(out["afterAppendCursor"], "live-2")
        self.assertEqual(out["afterAppendQueue"], 2)
        self.assertTrue(out["matchesSession"])
        self.assertEqual(out["beforeDelete"]["slotCount"], 1)
        self.assertEqual(out["beforeDelete"]["tailCacheCount"], 1)
        self.assertEqual(out["afterDelete"]["slotCount"], 0)
        self.assertEqual(out["afterDelete"]["tailCacheCount"], 0)
        self.assertEqual(out["failed"]["state"], "failed")
        self.assertTrue(out["frozen"])

    def test_transient_pending_snapshot_can_rebind_current_log(self) -> None:
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const ctx = {{ window: {{}} }};
            const vm = require("vm");
            vm.createContext(ctx);
            vm.runInContext({json.dumps(transcript_source)}, ctx);
            const runtime = ctx.window.CodoxearTranscript.createTranscriptSlotRuntime();
            runtime.updateSlot("sid", {{ transcript_state: "bound", thread_id: "thread-a", log_path: "/log-a.jsonl" }});
            const pending = runtime.updateSlot("sid", {{ transcript_state: "pending_bind" }});
            const rebound = runtime.updateSlot("sid", {{ transcript_state: "bound", thread_id: "thread-a", log_path: "/log-a.jsonl" }});
            process.stdout.write(JSON.stringify({{ pending, rebound, slot: runtime.getSlot("sid") }}));
            """
        )
        out = _run_node(js)

        self.assertTrue(out["pending"]["resetPending"])
        self.assertIsNone(out["pending"]["current"]["ignoredKey"])
        self.assertFalse(out["rebound"]["ignoredStaleBound"])
        self.assertEqual(out["slot"]["state"], "bound")
        self.assertEqual(out["slot"]["key"], "thread-a\n/log-a.jsonl")

    def test_normalized_transcript_events_filters_dedupes_and_consumes_pending(self) -> None:
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const ctx = {{ window: {{}} }};
            const vm = require("vm");
            vm.createContext(ctx);
            vm.runInContext({json.dumps(transcript_source)}, ctx);
            const tx = ctx.window.CodoxearTranscript;
            const consumed = [];
            const events = [
              null,
              {{ role: "system", text: "skip" }},
              {{ role: "user", text: "one", id: "u1" }},
              {{ role: "user", text: "one duplicate", id: "u1" }},
              {{ role: "assistant", text: "two", id: "a2" }},
              {{ role: "assistant", text: "no key" }},
              {{ role: "assistant", text: "no key again" }},
            ];
            const normalized = tx.normalizedTranscriptEvents(events, {{
              consumePending: true,
              selectedSessionId: "sid",
              eventKey: (ev) => ev.id || "",
              takePendingMatch: (ev, sid, opts) => consumed.push([ev.text, sid, opts.allowUntimedCommit]),
            }});
            const withoutConsume = tx.normalizedTranscriptEvents(events, {{
              eventKey: (ev) => ev.id || "",
            }});
            let missingKey = false;
            try {{ tx.normalizedTranscriptEvents(events, {{}}); }} catch (err) {{ missingKey = /eventKey/.test(String(err && err.message || err)); }}
            let missingTake = false;
            try {{ tx.normalizedTranscriptEvents(events, {{ consumePending: true, eventKey: () => "" }}); }} catch (err) {{ missingTake = /takePendingMatch/.test(String(err && err.message || err)); }}
            process.stdout.write(JSON.stringify({{
              normalizedTexts: normalized.map((ev) => ev.text),
              withoutConsumeTexts: withoutConsume.map((ev) => ev.text),
              consumed,
              missingKey,
              missingTake,
            }}));
            """
        )
        out = _run_node(js)

        self.assertEqual(out["normalizedTexts"], ["one", "two", "no key", "no key again"])
        self.assertEqual(out["withoutConsumeTexts"], ["one", "two", "no key", "no key again"])
        self.assertEqual(out["consumed"], [
            ["one", "sid", False],
            ["one duplicate", "sid", False],
            ["two", "sid", False],
            ["no key", "sid", False],
            ["no key again", "sid", False],
        ])
        self.assertTrue(out["missingKey"])
        self.assertTrue(out["missingTake"])

    def test_transcript_render_runtime_owns_window_render_and_history_prepend(self) -> None:
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const ctx = {{ window: {{}} }};
            const vm = require("vm");
            vm.createContext(ctx);
            vm.runInContext({json.dumps(transcript_source)}, ctx);
            const tx = ctx.window.CodoxearTranscript;
            const calls = [];
            const bottom = {{ name: "bottom" }};
            const firstMsg = {{ name: "first", offsetTop: 20, isConnected: true }};
            const root = {{
              insertBefore(node, before) {{ calls.push(["insert", node.children ? node.children.map((row) => row.name) : node.name || "node", before && before.name]); }},
              querySelector: (selector) => selector === ".msg-row:not(.typing-row)" ? firstMsg : null,
            }};
            const scrollRuntime = {{
              shouldStickToBottom: () => true,
              snapshot: () => ({{ renderedAtLiveTail: ctx.renderedAtLiveTail }}),
              syncJumpButton: () => calls.push(["jump"]),
              scheduleScrollToBottom: () => calls.push(["scroll"]),
              markLiveTail: () => calls.push(["markLiveTail"]),
              disableAutoScroll: () => calls.push(["disableAuto"]),
              setRenderedAtLiveTail: (value) => {{ ctx.renderedAtLiveTail = value; calls.push(["liveTail", value]); }},
              setScrollTop: (value) => calls.push(["scrollTop", value]),
            }};
            const domRuntime = {{
              clear: () => calls.push(["clear"]),
              rebuildDecorations: (opts) => calls.push(["rebuild", Boolean(opts.preserveScroll)]),
              trimRenderedRows: (opts) => {{ calls.push(["trim", opts.fromTop, opts.maxRows || null]); ctx.renderedAtLiveTail = Boolean(opts.fromTop); return 1; }},
            }};
            function makeRuntime(eventsForNormalize) {{
              return tx.createTranscriptRenderRuntime({{
                root,
                bottomSentinel: bottom,
                document: {{ createDocumentFragment: () => ({{ children: [], appendChild(row) {{ this.children.push(row); }} }}) }},
                safeMakeRow: (ev) => (calls.push(["make", ev.text]), {{ row: {{ name: ev.text }}, bubble: {{}} }}),
                normalizeEvents: (events, opts) => (calls.push(["normalize", Boolean(opts.consumePending)]), eventsForNormalize || events),
                consumePendingUserIfMatches: () => false,
                isDuplicateEvent: () => false,
                isAdjacentAssistantDuplicateEvent: () => false,
                markEventSeen: (ev) => calls.push(["seen", ev.text]),
                markFirstPaint: () => calls.push(["paint"]),
                renderRecoveryPanel: (sid) => calls.push(["recovery", sid]),
                restorePendingRows: (sid) => calls.push(["restore", sid]),
                resetRecentEvents: () => calls.push(["resetRecent"]),
                setOlderState: (state) => calls.push(["older", state.hasMore, state.isLoading]),
                firstVisibleMessageRow: () => firstMsg,
                getScrollTop: () => 5,
                getSelectedSessionId: () => "sid",
                domRuntime,
                scrollRuntime,
                typingRowRuntime: {{ anchor: () => bottom }},
                historySlackRows: 99,
              }});
            }}
            ctx.renderedAtLiveTail = true;
            const runtime = makeRuntime([{{ role: "user", text: "one", ts: 1 }}, {{ role: "assistant", text: "two", ts: 2 }}]);
            const full = runtime.renderTranscript([{{ role: "user", text: "ignored" }}], {{ preserveScroll: true }});
            const afterFull = calls.slice();
            calls.length = 0;
            const detached = runtime.renderDetachedTranscriptWindow([{{ role: "assistant", text: "det" }}], {{ hasMore: true }});
            const afterDetached = calls.slice();
            calls.length = 0;
            ctx.renderedAtLiveTail = true;
            const prepended = runtime.prependOlderEvents([{{ role: "system", text: "skip" }}, {{ role: "user", text: "old" }}], {{ preserveViewport: true }});
            const afterPrepend = calls.slice();
            calls.length = 0;
            const empty = makeRuntime([]).renderTranscript([], {{ preserveScroll: false }});
            process.stdout.write(JSON.stringify({{ full, detached, prepended, empty, afterFull, afterDetached, afterPrepend, frozen: Object.isFrozen(runtime) }}));
            """
        )
        out = _run_node(js)

        self.assertTrue(out["full"])
        self.assertTrue(out["detached"])
        self.assertTrue(out["prepended"])
        self.assertFalse(out["empty"])
        self.assertEqual(out["afterFull"], [
            ["normalize", True], ["markLiveTail"], ["clear"], ["resetRecent"],
            ["seen", "one"], ["make", "one"], ["seen", "two"], ["make", "two"],
            ["insert", ["one", "two"], "bottom"], ["rebuild", True], ["restore", "sid"],
        ])
        self.assertEqual(out["afterDetached"], [
            ["normalize", False], ["disableAuto"], ["liveTail", False], ["clear"], ["older", True, False], ["resetRecent"],
            ["seen", "one"], ["make", "one"], ["seen", "two"], ["make", "two"],
            ["insert", ["one", "two"], "bottom"], ["rebuild", False], ["scrollTop", 1], ["jump"],
        ])
        self.assertEqual(out["afterPrepend"], [
            ["disableAuto"], ["make", "old"], ["insert", ["old"], "first"], ["trim", False, 99], ["disableAuto"],
            ["rebuild", False], ["scrollTop", 5], ["jump"],
        ])
        self.assertTrue(out["frozen"])

    def test_transcript_dom_runtime_owns_clear_decorate_and_trim_window(self) -> None:
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const ctx = {{ window: {{}} }};
            const vm = require("vm");
            vm.createContext(ctx);
            vm.runInContext({json.dumps(transcript_source)}, ctx);
            const tx = ctx.window.CodoxearTranscript;
            const calls = [];
            function classList(initial) {{
              const values = new Set(String(initial || "").split(/\\s+/).filter(Boolean));
              return {{
                values,
                add: (...names) => names.forEach((name) => values.add(name)),
                remove: (...names) => names.forEach((name) => values.delete(name)),
                contains: (name) => values.has(name),
              }};
            }}
            function makeNode(name, cls = "") {{
              const node = {{
                name,
                attrs: {{ class: cls }},
                children: [],
                dataset: {{}},
                classList: classList(cls),
                isConnected: true,
                appendChild(child) {{ this.children.push(child); return child; }},
                remove() {{
                  const idx = root.children.indexOf(this);
                  if (idx >= 0) root.children.splice(idx, 1);
                  this.isConnected = false;
                }},
              }};
              return node;
            }}
            const older = makeNode("older");
            const bottom = makeNode("bottom");
            const oldSep = makeNode("old-sep", "day-sep");
            const row1 = makeNode("row1", "msg-row user");
            row1.dataset.ts = "86400";
            const row2 = makeNode("row2", "msg-row user");
            row2.dataset.ts = "86520";
            const row3 = makeNode("row3", "msg-row assistant");
            row3.dataset.ts = "172800";
            const root = {{
              children: [older, oldSep, row1, row2, row3, bottom],
              appendChild(node) {{ this.children.push(node); node.isConnected = true; return node; }},
              insertBefore(node, before) {{
                const existing = this.children.indexOf(node);
                if (existing >= 0) this.children.splice(existing, 1);
                const idx = this.children.indexOf(before);
                this.children.splice(idx >= 0 ? idx : this.children.length, 0, node);
                node.isConnected = true;
                return node;
              }},
              querySelectorAll(selector) {{ return selector === ".day-sep" ? this.children.filter((node) => node.classList && node.classList.contains("day-sep")) : []; }},
            }};
            Object.defineProperty(root, "innerHTML", {{ set() {{ this.children = []; }} }});
            function fakeEl(tag, attrs = {{}}) {{
              const node = makeNode(attrs.text || attrs.class || tag, attrs.class || "");
              node.tag = tag;
              node.textContent = attrs.text || "";
              return node;
            }}
            const scrollRuntime = {{
              captureScrollPosition: () => (calls.push(["capture"]), {{ top: 10 }}),
              preserveScrollFrom: (pos) => calls.push(["preserve", pos.top]),
              snapshot: () => ({{ autoScroll: true }}),
              scheduleScrollToBottom: () => calls.push(["scroll"]),
              syncJumpButton: () => calls.push(["jump"]),
              setRenderedAtLiveTail: (value) => calls.push(["liveTail", value]),
            }};
            const runtime = tx.createTranscriptDomRuntime({{
              root,
              olderWrap: older,
              bottomSentinel: bottom,
              el: fakeEl,
              ymd: (date) => `day-${{date.getUTCDate()}}`,
              dayLabel: (date) => `Day ${{date.getUTCDate()}}`,
              getRenderedRows: () => [row1, row2, row3].filter((row) => row.isConnected),
              trimRenderedRowTargets: (rows, fromTop, maxRows, defaultRows) => {{ calls.push(["trim", fromTop, maxRows, defaultRows]); return rows.slice(0, 1); }},
              trimRowsBeforeViewportTargets: (rows, maxRows, defaultRows, viewportTop) => {{ calls.push(["trimViewport", maxRows, defaultRows, viewportTop]); return rows.slice(0, 1); }},
              scrollRuntime,
              defaultWindowRows: 4,
              afterDecorate: () => calls.push(["after"]),
            }});
            runtime.rebuildDecorations({{ preserveScroll: true }});
            const afterDecorate = {{
              children: root.children.map((node) => node.name),
              row2Grouped: row2.classList.contains("grouped"),
              row3Grouped: row3.classList.contains("grouped"),
              row1Connected: row1.isConnected,
              oldSepConnected: oldSep.isConnected,
              sepDays: root.children.filter((node) => node.classList && node.classList.contains("day-sep")).map((node) => node.dataset.day),
              calls: calls.slice(),
            }};
            calls.length = 0;
            const trimmed = runtime.trimRenderedRows({{ fromTop: true, maxRows: 2 }});
            const trimmedViewport = runtime.trimRowsBeforeViewport({{ maxRows: 3, viewportTop: 42 }});
            const afterTrim = {{ trimmed, trimmedViewport, row1Connected: row1.isConnected, row2Connected: row2.isConnected, calls: calls.slice() }};
            runtime.clear();
            let missingRoot = false;
            try {{ tx.createTranscriptDomRuntime({{ olderWrap: older, bottomSentinel: bottom, el: fakeEl, ymd: () => "", dayLabel: () => "", getRenderedRows: () => [], trimRenderedRowTargets: () => [], trimRowsBeforeViewportTargets: () => [], scrollRuntime, afterDecorate: () => {{}} }}); }} catch (err) {{ missingRoot = /root/.test(String(err && err.message || err)); }}
            process.stdout.write(JSON.stringify({{
              afterDecorate,
              afterTrim,
              afterClear: root.children.map((node) => node.name),
              missingRoot,
              frozen: Object.isFrozen(runtime),
            }}));
            """
        )
        out = _run_node(js)

        self.assertFalse(out["afterDecorate"]["oldSepConnected"])
        self.assertTrue(out["afterDecorate"]["row2Grouped"])
        self.assertFalse(out["afterDecorate"]["row3Grouped"])
        self.assertEqual(out["afterDecorate"]["sepDays"], ["day-2", "day-3"])
        self.assertEqual(out["afterDecorate"]["calls"], [["capture"], ["preserve", 10], ["scroll"], ["jump"], ["after"]])
        self.assertEqual(out["afterTrim"]["trimmed"], 1)
        self.assertEqual(out["afterTrim"]["trimmedViewport"], 1)
        self.assertFalse(out["afterTrim"]["row1Connected"])
        self.assertFalse(out["afterTrim"]["row2Connected"])
        self.assertEqual(out["afterTrim"]["calls"], [["trim", True, 2, 4], ["liveTail", True], ["trimViewport", 3, 4, 42]])
        self.assertEqual(out["afterClear"], ["older", "bottom"])
        self.assertTrue(out["missingRoot"])
        self.assertTrue(out["frozen"])

    def test_typing_row_runtime_owns_row_anchor_and_scroll_projection(self) -> None:
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const ctx = {{ window: {{}} }};
            const vm = require("vm");
            vm.createContext(ctx);
            vm.runInContext({json.dumps(transcript_source)}, ctx);
            const tx = ctx.window.CodoxearTranscript;
            const calls = [];
            const bottom = {{ name: "bottom", isConnected: true }};
            const root = {{
              children: [bottom],
              insertBefore(node, before) {{
                const existing = this.children.indexOf(node);
                if (existing >= 0) this.children.splice(existing, 1);
                const idx = this.children.indexOf(before);
                this.children.splice(idx >= 0 ? idx : this.children.length, 0, node);
                node.isConnected = true;
                refreshSiblings();
              }},
            }};
            function refreshSiblings() {{
              for (let i = 0; i < root.children.length; i += 1) root.children[i].nextSibling = root.children[i + 1] || null;
            }}
            function fakeEl(tag, attrs = {{}}, children = []) {{
              const node = {{
                tag,
                attrs,
                children: [],
                dataset: {{}},
                isConnected: false,
                appendChild(child) {{ this.children.push(child); return child; }},
                remove() {{
                  const idx = root.children.indexOf(this);
                  if (idx >= 0) root.children.splice(idx, 1);
                  this.isConnected = false;
                  refreshSiblings();
                }},
              }};
              for (const child of children || []) node.appendChild(child);
              return node;
            }}
            let autoScroll = true;
            const runtime = tx.createTypingRowRuntime({{
              root,
              bottomSentinel: bottom,
              el: fakeEl,
              shouldAutoScroll: () => autoScroll,
              scheduleScrollToBottom: () => calls.push(["scroll"]),
            }});
            const initialAnchor = runtime.anchor().name;
            const shown = runtime.setVisible(true);
            const row = root.children[0];
            const afterShowAnchorIsRow = runtime.anchor() === row;
            runtime.setVisible(true);
            autoScroll = false;
            runtime.setVisible(true);
            const beforeHide = {{ childNames: root.children.map((node) => node.name || node.attrs.class), scrollCalls: calls.slice(), rowNextIsBottom: row.nextSibling === bottom }};
            runtime.setVisible(false);
            const afterHide = {{ snapshot: runtime.snapshot(), anchor: runtime.anchor().name, childNames: root.children.map((node) => node.name || node.attrs.class) }};
            runtime.setVisible(true);
            runtime.reset();
            let missingRoot = false;
            try {{ tx.createTypingRowRuntime({{ bottomSentinel: bottom, el: fakeEl, shouldAutoScroll: () => false, scheduleScrollToBottom: () => {{}} }}); }} catch (err) {{ missingRoot = /root/.test(String(err && err.message || err)); }}
            process.stdout.write(JSON.stringify({{
              initialAnchor,
              shown,
              rowClass: row.attrs.class,
              rowRole: row.dataset.role,
              bubbleClass: row.children[0].attrs.class,
              dotsClass: row.children[0].children[0].attrs.class,
              dotCount: row.children[0].children[0].children.length,
              afterShowAnchorIsRow,
              beforeHide,
              afterHide,
              afterReset: runtime.snapshot(),
              missingRoot,
              frozen: Object.isFrozen(runtime),
            }}));
            """
        )
        out = _run_node(js)

        self.assertEqual(out["initialAnchor"], "bottom")
        self.assertTrue(out["shown"]["connected"])
        self.assertEqual(out["rowClass"], "msg-row assistant typing-row")
        self.assertEqual(out["rowRole"], "assistant")
        self.assertEqual(out["bubbleClass"], "msg assistant typing")
        self.assertEqual(out["dotsClass"], "typingDots")
        self.assertEqual(out["dotCount"], 3)
        self.assertTrue(out["afterShowAnchorIsRow"])
        self.assertEqual(out["beforeHide"]["childNames"], ["msg-row assistant typing-row", "bottom"])
        self.assertEqual(out["beforeHide"]["scrollCalls"], [["scroll"], ["scroll"]])
        self.assertTrue(out["beforeHide"]["rowNextIsBottom"])
        self.assertFalse(out["afterHide"]["snapshot"]["connected"])
        self.assertEqual(out["afterHide"]["anchor"], "bottom")
        self.assertEqual(out["afterHide"]["childNames"], ["bottom"])
        self.assertFalse(out["afterReset"]["connected"])
        self.assertTrue(out["missingRoot"])
        self.assertTrue(out["frozen"])

    def test_transcript_slot_runtime_uses_current_session_lookup_for_tail_append(self) -> None:
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const ctx = {{ window: {{}} }};
            const vm = require("vm");
            vm.createContext(ctx);
            vm.runInContext({json.dumps(transcript_source)}, ctx);
            const tx = ctx.window.CodoxearTranscript;
            let sessions = new Map([["sid", {{ thread_id: "old-thread", log_path: "/old.jsonl" }}]]);
            const runtime = tx.createTranscriptSlotRuntime({{
              getSession: (sessionId) => sessions.get(sessionId) || null,
              maxTailEvents: 4,
            }});
            runtime.rememberTail("sid", sessions.get("sid"), {{
              transcript_state: "bound",
              thread_id: "old-thread",
              log_path: "/old.jsonl",
              events: [{{ role: "assistant", text: "old" }}],
              live_cursor: "old-cursor",
            }});
            sessions = new Map([["sid", {{ thread_id: "new-thread", log_path: "/new.jsonl" }}]]);
            runtime.appendTailEvents("sid", [{{ role: "assistant", text: "new" }}], {{ liveCursor: "new-cursor" }});
            const cache = runtime.getTailCache("sid");
            process.stdout.write(JSON.stringify({{
              threadId: cache.threadId,
              logPath: cache.logPath,
              texts: cache.events.map((ev) => ev.text),
              matchesNew: runtime.tailCacheMatchesSession(cache, sessions.get("sid")),
              matchesOld: runtime.tailCacheMatchesSession(cache, {{ thread_id: "old-thread", log_path: "/old.jsonl" }}),
            }}));
            """
        )
        out = _run_node(js)

        self.assertEqual(out["threadId"], "new-thread")
        self.assertEqual(out["logPath"], "/new.jsonl")
        self.assertEqual(out["texts"], ["old", "new"])
        self.assertTrue(out["matchesNew"])
        self.assertFalse(out["matchesOld"])

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
              usableOlderHistoryCursor: (data) => (data && data.has_older ? (data.history_cursor || (Array.isArray(data.events) && data.events.find((ev) => ev && ev.history_cursor)?.history_cursor) || null) : null),
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
              usableOlderHistoryCursor: (data) => (data && data.has_older ? (data.history_cursor || (Array.isArray(data.events) && data.events.find((ev) => ev && ev.history_cursor)?.history_cursor) || null) : null),
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
              usableOlderHistoryCursor: (data) => (data && data.has_older ? (data.history_cursor || (Array.isArray(data.events) && data.events.find((ev) => ev && ev.history_cursor)?.history_cursor) || null) : null),
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
              usableOlderHistoryCursor: (data) => (data && data.has_older ? (data.history_cursor || (Array.isArray(data.events) && data.events.find((ev) => ev && ev.history_cursor)?.history_cursor) || null) : null),
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
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const ctx = {{ window: {{}} }};
            const vm = require("vm");
            vm.createContext(ctx);
            vm.runInContext({json.dumps(transcript_source)}, ctx);
            const calls = [];
            const bottom = {{}};
            const scrollRuntime = {{
              snapshot: () => ({{ renderedAtLiveTail: false }}),
              shouldStickToBottom: () => false,
              syncJumpButton: () => calls.push(["jump"]),
              scheduleScrollToBottom: () => calls.push(["scroll"]),
              markLiveTail: () => calls.push(["markLiveTail"]),
              disableAutoScroll: () => calls.push(["disableAuto"]),
              setRenderedAtLiveTail: (value) => calls.push(["liveTail", value]),
              setScrollTop: (value) => calls.push(["scrollTop", value]),
            }};
            const runtime = ctx.window.CodoxearTranscript.createTranscriptRenderRuntime({{
              root: {{ insertBefore: () => calls.push(["insert"]), querySelector: () => null }},
              bottomSentinel: bottom,
              document: {{ createDocumentFragment: () => ({{ appendChild: () => {{}} }}) }},
              safeMakeRow: () => (calls.push(["make"]), {{ row: {{}}, bubble: {{}} }}),
              normalizeEvents: () => [],
              consumePendingUserIfMatches: () => false,
              isDuplicateEvent: () => false,
              isAdjacentAssistantDuplicateEvent: () => false,
              markEventSeen: () => calls.push(["seen"]),
              markFirstPaint: () => calls.push(["paint"]),
              renderRecoveryPanel: () => calls.push(["recovery"]),
              restorePendingRows: () => calls.push(["restore"]),
              resetRecentEvents: () => calls.push(["resetRecent"]),
              setOlderState: () => calls.push(["older"]),
              firstVisibleMessageRow: () => null,
              getScrollTop: () => 0,
              getSelectedSessionId: () => "sid",
              domRuntime: {{ clear: () => calls.push(["clear"]), rebuildDecorations: () => calls.push(["rebuild"]), trimRenderedRows: () => calls.push(["trim"]) }},
              scrollRuntime,
              typingRowRuntime: {{ anchor: () => bottom }},
              historySlackRows: 3,
            }});
            runtime.appendEvent({{ role: "assistant", text: "new tail", ts: 2 }});
            process.stdout.write(JSON.stringify({{ calls }}));
            """
        )
        out = _run_node(js)

        self.assertEqual(out["calls"], [["seen"], ["jump"]])

    def test_live_delta_dedupes_adjacent_assistant_text_across_polls(self) -> None:
        identity_source = APP_MESSAGE_IDENTITY_JS.read_text(encoding="utf-8")
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const ctx = {{ window: {{}} }};
            const vm = require("vm");
            vm.createContext(ctx);
            vm.runInContext({json.dumps(identity_source)}, ctx);
            vm.runInContext({json.dumps(transcript_source)}, ctx);
            const id = ctx.window.CodoxearMessageIdentity;
            const tx = ctx.window.CodoxearTranscript;
            const eventRuntime = tx.createTranscriptEventRuntime({{
              eventKey: id.eventKey,
              pendingMatchKey: id.pendingMatchKey,
              normalizePendingText: id.normalizeTextForPendingMatch,
              assistantDedupeKey: id.chatAssistantDedupeKey,
              maxRecentEventKeys: 320,
            }});
            const calls = [];
            let rows = [{{ dataset: {{ role: "assistant", assistantDedupeKey: "final_response|same final text" }} }}];
            const bottom = {{}};
            const scrollRuntime = {{
              snapshot: () => ({{ renderedAtLiveTail: true }}),
              shouldStickToBottom: () => true,
              syncJumpButton: () => calls.push(["jump"]),
              scheduleScrollToBottom: () => calls.push(["scroll"]),
              markLiveTail: () => calls.push(["markLiveTail"]),
              disableAutoScroll: () => calls.push(["disableAuto"]),
              setRenderedAtLiveTail: (value) => calls.push(["liveTail", value]),
              setScrollTop: (value) => calls.push(["scrollTop", value]),
            }};
            const runtime = tx.createTranscriptRenderRuntime({{
              root: {{ insertBefore: () => calls.push(["insert"]), querySelector: () => null }},
              bottomSentinel: bottom,
              document: {{ createDocumentFragment: () => ({{ appendChild: () => {{}} }}) }},
              safeMakeRow: () => (calls.push(["make"]), {{ row: {{}}, bubble: {{}} }}),
              normalizeEvents: () => [],
              consumePendingUserIfMatches: () => false,
              isDuplicateEvent: (ev) => eventRuntime.isDuplicateEvent(ev),
              isAdjacentAssistantDuplicateEvent: (ev) => eventRuntime.isAdjacentAssistantDuplicateEvent(ev, {{ renderedAtLiveTail: true, rows }}),
              markEventSeen: (ev) => eventRuntime.markEventSeen(ev),
              markFirstPaint: () => calls.push(["paint"]),
              renderRecoveryPanel: () => calls.push(["recovery"]),
              restorePendingRows: () => calls.push(["restore"]),
              resetRecentEvents: () => eventRuntime.resetRecentEvents(),
              setOlderState: () => calls.push(["older"]),
              firstVisibleMessageRow: () => null,
              getScrollTop: () => 0,
              getSelectedSessionId: () => "sid",
              domRuntime: {{ clear: () => calls.push(["clear"]), rebuildDecorations: () => calls.push(["rebuild"]), trimRenderedRows: () => calls.push(["trim"]) }},
              scrollRuntime,
              typingRowRuntime: {{ anchor: () => bottom }},
              historySlackRows: 3,
            }});
            runtime.appendEvent({{ role: "assistant", text: "same final text", message_class: "final_response", ts: 2.4 }});
            const afterDuplicate = {{ calls: calls.slice(), seen: eventRuntime.snapshot().recentEventKeys }};
            calls.length = 0;
            rows = [{{ dataset: {{ role: "user" }} }}];
            runtime.appendEvent({{ role: "assistant", text: "same final text", message_class: "final_response", ts: 3.0 }});
            process.stdout.write(JSON.stringify({{
              afterDuplicate,
              final: {{ calls: calls.slice(), seen: eventRuntime.snapshot().recentEventKeys }},
            }}));
            """
        )
        out = _run_node(js)

        self.assertEqual(out["afterDuplicate"]["calls"], [])
        self.assertEqual(out["afterDuplicate"]["seen"], ["assistant|2400|same final text"])
        self.assertEqual(out["final"]["calls"], [["make"], ["insert"], ["trim"], ["rebuild"], ["paint"], ["scroll"], ["jump"]])
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
              transcriptEventRuntime: {{
                nextLocalEchoId: () => 1,
                addPendingUser: () => ({{ id: 1 }}),
                dropPendingUsers: () => [],
                hasPendingForSession: () => false,
              }},
              transcriptScrollRuntime: {{
                snapshot: () => ({{ renderedAtLiveTail: true }}),
              }},
              sessionIndex: new Map([["sid", {{ agent_backend: "codex" }}]]),
              stagedAttachments: [],
              normalizedStagedAttachments: (list) => Array.isArray(list) ? list : [],
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
