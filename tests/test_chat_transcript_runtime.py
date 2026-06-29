import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_TRANSCRIPT_JS = ROOT / "codoxear" / "static" / "app_transcript.js"


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
              renderedAtLiveTail: false,
              autoScroll: false,
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
        helper_snippet = _source_between("function eventKey(ev) {", "function isTranscriptRenewalCommand(")
        append_snippet = _source_between("function appendEvent(ev) {", "function renderTranscript(")
        js = textwrap.dedent(
            f"""
            const ctx = {{
              renderedAtLiveTail: true,
              autoScroll: true,
              recentEventKeys: [],
              recentEventKeySet: new Set(),
              RECENT_EVENT_KEYS_MAX: 320,
              made: 0,
              inserted: 0,
              rows: [{{ dataset: {{ role: "assistant", assistantDedupeKey: "final_response|same final text" }} }}],
              normalizeTextForPendingMatch: (s) => String(s || ""),
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
              renderedAtLiveTail: true,
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
