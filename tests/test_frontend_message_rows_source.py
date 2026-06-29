import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_MESSAGE_ROWS_JS = ROOT / "codoxear" / "static" / "app_message_rows.js"
INDEX_HTML = ROOT / "codoxear" / "static" / "index.html"


def eval_message_rows() -> dict:
    source = APP_MESSAGE_ROWS_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx);
        const rows = ctx.window.CodoxearMessageRows;
        function fakeEl(tag, attrs = {{}}) {{
          const node = {{
            tag,
            attrs,
            children: [],
            dataset: {{}},
            style: {{}},
            textContent: attrs.text || "",
            innerHTML: attrs.html || "",
            onclick: null,
            classList: {{
              values: new Set(String(attrs.class || "").split(/\\s+/).filter(Boolean)),
              add(...names) {{ for (const name of names) this.values.add(name); }},
              remove(...names) {{ for (const name of names) this.values.delete(name); }},
              has(name) {{ return this.values.has(name); }},
              contains(name) {{ return this.values.has(name); }},
            }},
            appendChild(child) {{ this.children.push(child); return child; }},
            setAttribute(name, value) {{ this.attrs[name] = String(value); }},
          }};
          return node;
        }}
        const calls = [];
        const deps = {{
          el: fakeEl,
          chatMarkdownHtmlCached: (text, sessionId) => `<p>${{sessionId}}:${{text}}</p>`,
          selectedSessionId: "s1",
          upgradeCandidateFileRefs: (node) => calls.push(["upgrade", node.innerHTML]),
          time24: (date) => `T${{date.getUTCMinutes()}}`,
          iconSvg: (name) => `<svg>${{name}}</svg>`,
          copyToClipboard: async (text) => calls.push(["copy", text]),
          setToast: (text) => calls.push(["toast", text]),
          chatAssistantDedupeKey: (ev) => ev.role === "assistant" ? `assistant|${{ev.text}}` : "",
          setTimeout: (fn, ms) => {{ calls.push(["timeout", ms]); fn(); }},
          consoleError: (...args) => calls.push(["error", args[0]]),
        }};
        const made = rows.makeRow({{ role: "assistant", text: "hello", ts: 7, history_cursor: "h1", message_class: "warning" }}, {{ ts: 12, pending: false }}, deps);
        const copyBtn = made.row.children[0].children[1];
        copyBtn.onclick({{ preventDefault() {{}}, stopPropagation() {{}} }}).then(() => {{
          const fallbackDeps = {{ ...deps, chatMarkdownHtmlCached: () => {{ throw new Error("markdown boom"); }} }};
          const fallback = rows.safeMakeRow({{ role: "assistant", text: "raw", history_cursor: "h2", message_class: "error" }}, {{ ts: 14, pending: true }}, fallbackDeps);
          const userRow = fakeEl("div", {{ class: "msg-row user" }});
          userRow.dataset.role = "user";
          userRow.querySelector = (selector) => selector === ".msg-copy-btn" ? fakeEl("button", {{ class: "msg-copy-btn" }}) : null;
          const assistantRow = fakeEl("div", {{ class: "msg-row assistant" }});
          assistantRow.dataset.role = "assistant";
          assistantRow.querySelector = () => null;
          const typingRow = fakeEl("div", {{ class: "msg-row assistant typing-row" }});
          const recoveryRow = fakeEl("div", {{ class: "msg-row assistant recovery-panel-row" }});
          const container = {{ querySelectorAll: () => [userRow, assistantRow, typingRow, recoveryRow] }};
          const mdNode = fakeEl("div", {{ class: "md" }});
          mdNode.textContent = " searchable body ";
          const searchRow = fakeEl("div", {{ class: "msg-row assistant" }});
          searchRow.textContent = "fallback body";
          searchRow.querySelector = (selector) => selector === ".md" ? mdNode : null;
          const nodeConstants = {{ DOCUMENT_POSITION_FOLLOWING: 4, DOCUMENT_POSITION_PRECEDING: 2 }};
          const first = {{ compareDocumentPosition: (other) => other === second ? 4 : 0 }};
          const second = {{ compareDocumentPosition: (other) => other === first ? 2 : 0 }};
          const navRows = [
            {{ name: "r1", offsetTop: 10, isConnected: true }},
            {{ name: "r2", offsetTop: 30, isConnected: true }},
            {{ name: "r3", offsetTop: 50, isConnected: true }},
          ];
          const disconnectedRow = {{ name: "old", offsetTop: 30, isConnected: false }};
          const markA = fakeEl("div", {{ class: "msg-row chat-search-hit chat-search-current" }});
          const markB = fakeEl("div", {{ class: "msg-row chat-search-hit chat-search-current" }});
          const cursorRows = [
            {{ dataset: {{ historyCursor: "" }} }},
            {{ dataset: {{ historyCursor: "cursor-2" }} }},
            {{ dataset: {{ historyCursor: "cursor-3" }} }},
          ];
          const visibleRows = [
            {{ name: "above", offsetTop: 0, offsetHeight: 5 }},
            {{ name: "visible", offsetTop: 10, offsetHeight: 10 }},
            {{ name: "below", offsetTop: 30, offsetHeight: 10 }},
          ];
          rows.clearChatSearchMarks([markA, markB]);
          const marksCleared = !markA.classList.has("chat-search-hit") && !markA.classList.has("chat-search-current") && !markB.classList.has("chat-search-hit") && !markB.classList.has("chat-search-current");
          rows.applyChatSearchMarks([markA, markB], markB);
          process.stdout.write(JSON.stringify({{
            frozen: Object.isFrozen(rows),
            role: made.row.dataset.role,
            ts: made.row.dataset.ts,
            historyCursor: made.row.dataset.historyCursor,
            dedupeKey: made.row.dataset.assistantDedupeKey,
            bubbleClasses: Array.from(made.bubble.classList.values).sort(),
            markdownHtml: made.bubble.children[0].innerHTML,
            timestampText: made.bubble.children[1].textContent,
            copyAttrs: copyBtn.attrs,
            copiedClassAfterTimer: copyBtn.classList.has("copied"),
            calls,
            fallbackRole: fallback.row.dataset.role,
            fallbackTs: fallback.row.dataset.ts,
            fallbackHistoryCursor: fallback.row.dataset.historyCursor || "",
            fallbackDedupeKey: fallback.row.dataset.assistantDedupeKey,
            fallbackPending: fallback.bubble.attrs["data-pending"],
            fallbackText: fallback.bubble.children[0].textContent,
            fallbackClasses: Array.from(fallback.bubble.classList.values).sort(),
            renderedCount: rows.renderedMessageRows(container).length,
            userCount: rows.loadedUserMessageRows(container).length,
            copyCount: rows.loadedCopyMessageRows(container).length,
            copyButtonFound: Boolean(rows.messageCopyButtonForRow(userRow)),
            activeCopy: rows.activeElementIsMessageCopyButton({{ activeElement: {{ classList: {{ contains: (name) => name === "msg-copy-btn" }} }} }}),
            rowSearchText: rows.rowSearchText(searchRow),
            compareForward: rows.compareRowsInDomOrder(first, second, nodeConstants),
            compareBackward: rows.compareRowsInDomOrder(second, first, nodeConstants),
            userPrevTarget: rows.loadedUserJumpTarget(navRows, -1, 35).target.name,
            userPrevBoundary: rows.loadedUserJumpTarget(navRows, -1, 5).reason,
            userNextTarget: rows.loadedUserJumpTarget(navRows, 1, 35).target.name,
            userNextBoundary: rows.loadedUserJumpTarget(navRows, 1, 55).reason,
            copyActiveNextTarget: rows.loadedCopyJumpTarget(navRows, navRows[1], 1, 0).target.name,
            copyActivePrevTarget: rows.loadedCopyJumpTarget(navRows, navRows[1], -1, 0).target.name,
            copyThresholdPrevTarget: rows.loadedCopyJumpTarget(navRows, disconnectedRow, -1, 35).target.name,
            copyThresholdNextTarget: rows.loadedCopyJumpTarget(navRows, null, 1, 35).target.name,
            copyFirstBoundary: rows.loadedCopyJumpTarget(navRows, navRows[0], -1, 0).reason,
            copyLastBoundary: rows.loadedCopyJumpTarget(navRows, navRows[2], 1, 0).reason,
            copyEmptyBoundary: rows.loadedCopyJumpTarget([], null, 1, 0).reason,
            marksCleared,
            markAHit: markA.classList.has("chat-search-hit"),
            markACurrent: markA.classList.has("chat-search-current"),
            markBHit: markB.classList.has("chat-search-hit"),
            markBCurrent: markB.classList.has("chat-search-current"),
            oldestCursor: rows.oldestRenderedHistoryCursor(cursorRows),
            emptyCursor: rows.oldestRenderedHistoryCursor([{{ dataset: {{ historyCursor: "" }} }}]),
            firstVisible: rows.firstVisibleMessageRow(visibleRows, 6).name,
            fallbackVisible: rows.firstVisibleMessageRow(visibleRows, 99).name,
            emptyVisible: rows.firstVisibleMessageRow([], 1),
          }}));
        }}).catch((err) => {{ process.stderr.write(String(err && err.stack || err)); process.exit(1); }});
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


class TestFrontendMessageRowsSource(unittest.TestCase):
    def test_index_loads_message_rows_between_identity_and_app(self) -> None:
        source = INDEX_HTML.read_text(encoding="utf-8")
        self.assertIn('app_message_rows.js?v=__CODOXEAR_ASSET_VERSION__', source)
        self.assertLess(source.index('app_message_identity.js?v=__CODOXEAR_ASSET_VERSION__'), source.index('app_message_rows.js?v=__CODOXEAR_ASSET_VERSION__'))
        self.assertLess(source.index('app_message_rows.js?v=__CODOXEAR_ASSET_VERSION__'), source.index('app.js?v=__CODOXEAR_ASSET_VERSION__'))

    def test_app_js_requires_message_rows_without_fallback(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        helper_source = APP_MESSAGE_ROWS_JS.read_text(encoding="utf-8")
        self.assertIn("const codoxearMessageRows = window.CodoxearMessageRows;", source)
        self.assertIn('throw new Error("Codoxear message row helpers failed to load")', source)
        self.assertIn("return codoxearMessageRows.makeRow(ev, { ts, pending }, messageRowDeps());", source)
        self.assertIn("return codoxearMessageRows.safeMakeRow(ev, opts, messageRowDeps());", source)
        self.assertIn("typeof codoxearMessageRows.renderedMessageRows !== \"function\"", source)
        self.assertIn("typeof codoxearMessageRows.clearChatSearchMarks !== \"function\"", source)
        self.assertIn("typeof codoxearMessageRows.applyChatSearchMarks !== \"function\"", source)
        self.assertIn("typeof codoxearMessageRows.oldestRenderedHistoryCursor !== \"function\"", source)
        self.assertIn("typeof codoxearMessageRows.firstVisibleMessageRow !== \"function\"", source)
        self.assertIn("return codoxearMessageRows.loadedUserMessageRows(chatInner);", source)
        self.assertIn("return codoxearMessageRows.messageCopyButtonForRow(row);", source)
        self.assertIn("return codoxearMessageRows.activeElementIsMessageCopyButton(document);", source)
        self.assertIn("return codoxearMessageRows.rowSearchText(row);", source)
        self.assertIn("return codoxearMessageRows.compareRowsInDomOrder(a, b, Node);", source)
        self.assertIn("return codoxearMessageRows.loadedUserJumpTarget(rows, direction, threshold);", source)
        self.assertIn("return codoxearMessageRows.loadedCopyJumpTarget(rows, activeRow, direction, threshold);", source)
        self.assertIn("codoxearMessageRows.clearChatSearchMarks(renderedMessageRows());", source)
        self.assertIn("codoxearMessageRows.applyChatSearchMarks(matches, currentRow);", source)
        self.assertIn("return codoxearMessageRows.oldestRenderedHistoryCursor(renderedMessageRows());", source)
        self.assertIn("return codoxearMessageRows.firstVisibleMessageRow(renderedMessageRows(), chat.scrollTop + 1);", source)
        self.assertIn("chatAssistantDedupeKey,", source)
        self.assertIn("consoleError: console.error.bind(console)", source)
        self.assertNotIn("function makeRow(ev, { ts, pending }) {\n          const role = ev.role", source)
        self.assertNotIn("console.error(\"makeRow failed\", err);", source)
        self.assertIn("function makeRow(ev, { ts, pending }, deps)", helper_source)
        self.assertIn("function safeMakeRow(ev, opts, deps)", helper_source)
        self.assertIn("function renderedMessageRows(chatInner)", helper_source)
        self.assertIn("function loadedUserMessageRows(chatInner)", helper_source)
        self.assertIn("function messageCopyButtonForRow(row)", helper_source)
        self.assertIn("function rowSearchText(row)", helper_source)
        self.assertIn("function compareRowsInDomOrder(a, b, nodeLike)", helper_source)
        self.assertIn("function loadedUserJumpTarget(rows, direction, threshold)", helper_source)
        self.assertIn("function loadedCopyJumpTarget(rows, activeRow, direction, threshold)", helper_source)
        self.assertIn("function clearChatSearchMarks(rows)", helper_source)
        self.assertIn("function applyChatSearchMarks(matches, currentRow)", helper_source)
        self.assertIn("function oldestRenderedHistoryCursor(rows)", helper_source)
        self.assertIn("function firstVisibleMessageRow(rows, viewportTop)", helper_source)
        self.assertIn("consoleError(\"makeRow failed\", err);", helper_source)

    def test_message_rows_builds_copyable_rows_and_safe_fallback(self) -> None:
        result = eval_message_rows()
        self.assertTrue(result["frozen"])
        self.assertEqual(result["role"], "assistant")
        self.assertEqual(result["ts"], "12")
        self.assertEqual(result["historyCursor"], "h1")
        self.assertEqual(result["dedupeKey"], "assistant|hello")
        self.assertIn("warning", result["bubbleClasses"])
        self.assertEqual(result["markdownHtml"], "<p>s1:hello</p>")
        self.assertEqual(result["timestampText"], "T0")
        self.assertEqual(result["copyAttrs"]["title"], "Copy raw markdown")
        self.assertFalse(result["copiedClassAfterTimer"])
        self.assertIn(["upgrade", "<p>s1:hello</p>"], result["calls"])
        self.assertIn(["copy", "hello"], result["calls"])
        self.assertIn(["timeout", 1200], result["calls"])
        self.assertIn(["toast", "Copied markdown"], result["calls"])
        self.assertIn(["error", "makeRow failed"], result["calls"])
        self.assertEqual(result["fallbackRole"], "assistant")
        self.assertEqual(result["fallbackTs"], "14")
        self.assertEqual(result["fallbackHistoryCursor"], "")
        self.assertEqual(result["fallbackDedupeKey"], "assistant|raw")
        self.assertEqual(result["fallbackPending"], "1")
        self.assertEqual(result["fallbackText"], "raw")
        self.assertIn("error", result["fallbackClasses"])
        self.assertEqual(result["renderedCount"], 2)
        self.assertEqual(result["userCount"], 1)
        self.assertEqual(result["copyCount"], 1)
        self.assertTrue(result["copyButtonFound"])
        self.assertTrue(result["activeCopy"])
        self.assertEqual(result["rowSearchText"], " searchable body ")
        self.assertEqual(result["compareForward"], -1)
        self.assertEqual(result["compareBackward"], 1)
        self.assertEqual(result["userPrevTarget"], "r2")
        self.assertEqual(result["userPrevBoundary"], "first")
        self.assertEqual(result["userNextTarget"], "r3")
        self.assertEqual(result["userNextBoundary"], "last")
        self.assertEqual(result["copyActiveNextTarget"], "r3")
        self.assertEqual(result["copyActivePrevTarget"], "r1")
        self.assertEqual(result["copyThresholdPrevTarget"], "r2")
        self.assertEqual(result["copyThresholdNextTarget"], "r3")
        self.assertEqual(result["copyFirstBoundary"], "first")
        self.assertEqual(result["copyLastBoundary"], "last")
        self.assertEqual(result["copyEmptyBoundary"], "none")
        self.assertTrue(result["marksCleared"])
        self.assertTrue(result["markAHit"])
        self.assertFalse(result["markACurrent"])
        self.assertTrue(result["markBHit"])
        self.assertTrue(result["markBCurrent"])
        self.assertEqual(result["oldestCursor"], "cursor-2")
        self.assertIsNone(result["emptyCursor"])
        self.assertEqual(result["firstVisible"], "visible")
        self.assertEqual(result["fallbackVisible"], "below")
        self.assertIsNone(result["emptyVisible"])


if __name__ == "__main__":
    unittest.main()
