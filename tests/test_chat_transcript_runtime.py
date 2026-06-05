import json
import subprocess
import textwrap
import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


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
    def test_transcript_renewal_ignores_old_bound_identity_until_new_log_arrives(self) -> None:
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
            }};
            vm = require("vm");
            vm.createContext(ctx);
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
        snippet = _source_between("async function loadOlderMessages({ auto = false } = {}) {", "function maybeAutoLoadOlder()")
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
              isTranscriptRenewalCommand: () => true,
              setToast: (text) => {{ ctx.toast = text; }},
              $: () => ({{ disabled: false }}),
              chatInner: {{ querySelector: () => null }},
              api: async () => {{ throw new Error("broker down"); }},
              setAttachCount: () => {{}},
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
