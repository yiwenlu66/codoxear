import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPT_JS = ROOT / "codoxear" / "static" / "app_transcript.js"


def run_vm(script: str) -> dict:
    try:
        proc = subprocess.run(
            ["node", "-e", script],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        raise AssertionError(error.stderr) from error
    return json.loads(proc.stdout)


class TestOlderLoad(unittest.TestCase):
    def test_top_scroll_load_preserves_anchor_deduplicates_ids_and_hides_terminal_boundary(self) -> None:
        source = TRANSCRIPT_JS.read_text(encoding="utf-8")
        script = textwrap.dedent(
            f"""
            const vm = require("vm");
            const ctx = {{ window: {{}}, AbortController }};
            vm.createContext(ctx);
            vm.runInContext({json.dumps(source)}, ctx);
            const tx = ctx.window.CodoxearTranscript;

            const chat = {{ scrollTop: 100, scrollHeight: 1000, clientHeight: 200 }};
            const jumpButton = {{ style: {{ display: "" }} }};
            const timeChip = {{ style: {{ display: "" }}, textContent: "" }};
            const olderWrap = {{ style: {{ display: "" }} }};
            const olderButton = {{ disabled: false, textContent: "" }};
            const olderError = {{ style: {{ display: "" }} }};
            const olderErrorText = {{ textContent: "" }};
            let now = 1000;
            const older = tx.createOlderLoadRuntime({{
              olderWrap,
              olderButton,
              olderError,
              olderErrorText,
              AbortControllerCtor: AbortController,
              nowMs: () => now,
              autoCooldownMs: 450,
            }});
            older.setState({{ hasMore: true, isLoading: false }});
            let starts = 0;
            let load = null;
            const scroll = tx.createTranscriptScrollRuntime({{
              chat,
              jumpButton,
              timeChip,
              requestAnimationFrame: (fn) => fn(),
              hasSelection: () => true,
              isSearchOpen: () => false,
              firstVisibleMessageRow: () => null,
              dayLabel: () => "",
              time24: () => "",
              shouldCancelOlderLoad: () => older.shouldCancelOnScroll(),
              cancelOlderLoad: () => older.invalidate(),
              autoLoadOlder: () => {{
                if (!older.snapshot().hasMore || older.snapshot().isLoading || !older.markAutoTrigger()) return;
                starts += 1;
                load = older.beginLoad({{ cancelOnScroll: true }});
              }},
              bottomThresholdPx: 80,
              olderTopTriggerPx: 1,
              olderCancelPx: 48,
            }});
            chat.scrollTop = 1;
            scroll.handleScroll();
            chat.scrollTop = 0;
            scroll.handleScroll();
            const whileLoading = older.snapshot();
            older.finishLoad(load);
            older.setState({{ hasMore: false, isLoading: false }});
            const terminalBoundary = {{ state: older.snapshot(), wrap: olderWrap.style.display }};

            function node(name, messageId = "") {{
              return {{
                name,
                isMessage: Boolean(messageId),
                isConnected: true,
                dataset: messageId ? {{ messageId }} : {{}},
                offsetTop: 0,
                offsetHeight: messageId ? 100 : 0,
              }};
            }}
            const currentA = node("current-a", "current-a");
            const currentTail = node("current-tail", "current-tail");
            const bottom = node("bottom");
            const root = {{
              children: [currentA, currentTail, bottom],
              querySelectorAll(selector) {{
                if (selector === ".msg-row[data-message-id]") return this.children.filter((child) => child.isMessage && child.dataset.messageId);
                return [];
              }},
              querySelector(selector) {{
                if (selector === ".msg-row:not(.typing-row)") return this.children.find((child) => child.isMessage) || null;
                return null;
              }},
              insertBefore(value, before) {{
                const insert = value.fragment ? value.children : [value];
                for (const child of insert) {{
                  const existing = this.children.indexOf(child);
                  if (existing >= 0) this.children.splice(existing, 1);
                }}
                const index = this.children.indexOf(before);
                this.children.splice(index >= 0 ? index : this.children.length, 0, ...insert);
                for (const child of insert) child.isConnected = true;
                relayout();
                return value;
              }},
            }};
            function relayout() {{
              let top = 0;
              for (const child of root.children) {{
                child.offsetTop = top;
                if (child.isMessage) top += child.offsetHeight;
              }}
            }}
            relayout();
            const viewport = {{ top: 50 }};
            const renderScroll = {{
              shouldStickToBottom: () => false,
              scheduleScrollToBottom: () => {{}},
              markLiveTail: () => {{}},
              disableAutoScroll: () => {{}},
              snapshot: () => ({{ renderedAtLiveTail: false }}),
              setScrollTop: (top) => {{ viewport.top = top; }},
              setRenderedAtLiveTail: () => {{}},
              syncJumpButton: () => {{}},
            }};
            const domRuntime = {{
              trimRenderedRows: () => 0,
              rebuildDecorations: () => {{}},
              clear: () => {{}},
            }};
            const render = tx.createTranscriptRenderRuntime({{
              root,
              bottomSentinel: bottom,
              document: {{
                createDocumentFragment: () => ({{
                  fragment: true,
                  children: [],
                  appendChild(child) {{ this.children.push(child); }},
                }}),
              }},
              safeMakeRow: (event) => ({{ row: node(event.message_id, event.message_id) }}),
              normalizeEvents: (events) => events,
              consumePendingUserIfMatches: () => false,
              isDuplicateEvent: () => false,
              isAdjacentAssistantDuplicateEvent: () => false,
              markEventSeen: () => {{}},
              markFirstPaint: () => {{}},
              restorePendingRows: () => {{}},
              resetRecentEvents: () => {{}},
              setOlderState: () => {{}},
              firstVisibleMessageRow: () => currentTail,
              getScrollTop: () => viewport.top,
              getSelectedSessionId: () => "session",
              domRuntime,
              scrollRuntime: renderScroll,
              typingRowRuntime: {{ anchor: () => bottom }},
              historySlackRows: 20,
            }});
            const initialAnchorOffset = currentTail.offsetTop - viewport.top;
            const prepended = render.prependOlderEvents([
              {{ role: "assistant", text: "older A", message_id: "old-a" }},
              {{ role: "assistant", text: "duplicate old A", message_id: "old-a" }},
              {{ role: "assistant", text: "duplicate tail", message_id: "current-tail" }},
              {{ role: "user", text: "older B", message_id: "old-b" }},
            ], {{ preserveViewport: true }});
            const ids = root.children.filter((child) => child.isMessage).map((child) => child.dataset.messageId);
            process.stdout.write(JSON.stringify({{
              autoStarts: starts,
              whileLoading,
              terminalBoundary,
              prepended,
              ids,
              uniqueIds: [...new Set(ids)],
              initialAnchorOffset,
              finalAnchorOffset: currentTail.offsetTop - viewport.top,
            }}));
            """
        )
        result = run_vm(script)

        self.assertEqual(result["autoStarts"], 1)
        self.assertTrue(result["whileLoading"]["isLoading"])
        self.assertEqual(result["terminalBoundary"]["state"]["hasMore"], False)
        self.assertEqual(result["terminalBoundary"]["wrap"], "none")
        self.assertTrue(result["prepended"])
        self.assertEqual(result["ids"], ["old-a", "old-b", "current-a", "current-tail"])
        self.assertEqual(result["ids"], result["uniqueIds"])
        self.assertEqual(result["finalAnchorOffset"], result["initialAnchorOffset"])


if __name__ == "__main__":
    unittest.main()
