import json
import subprocess
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPT = ROOT / "codoxear" / "static" / "app_transcript.js"
CHAT_SEARCH = ROOT / "codoxear" / "static" / "app_chat_search.js"


def run_chat(body: str) -> dict:
    transcript = json.dumps(TRANSCRIPT.read_text(encoding="utf-8"))
    module = json.dumps(CHAT_SEARCH.read_text(encoding="utf-8"))
    script = f"""
const vm = require('vm'); const ctx = {{ window: {{ setTimeout, clearTimeout }}, AbortController, HTMLElement: function HTMLElement() {{}}, document: {{}} }}; vm.createContext(ctx);
vm.runInContext({transcript}, ctx); vm.runInContext({module}, ctx);
{body}
"""
    proc = subprocess.run(["node", "-e", textwrap.dedent(script)], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return json.loads(proc.stdout)


class TestFrontendChatSearchModuleBehavior(unittest.TestCase):
    def test_controller_fails_loudly_when_dom_dependencies_are_missing(self) -> None:
        result = run_chat(
            """
let error = ''; try { ctx.window.CodoxearChatSearch.createChatSearchController({}); } catch (e) { error = e.message; }
process.stdout.write(JSON.stringify({ frozen: Object.isFrozen(ctx.window.CodoxearChatSearch), error }));
"""
        )
        self.assertTrue(result["frozen"])
        self.assertEqual(result["error"], "chat search controller dependency missing: chatSearchBtn")

    def test_open_close_match_and_step_execute_through_the_controller(self) -> None:
        result = run_chat(
            r'''
const makeNode = () => ({ style: {}, disabled: false, textContent: '', title: '', value: '', dataset: {}, focus() {}, select() {} });
const nodes = { chatSearchBtn: makeNode(), chatSearchInput: makeNode(), chatSearchPrevBtn: makeNode(), chatSearchNextBtn: makeNode(), chatSearchCloseBtn: makeNode(), chatSearchStatus: makeNode(), chatSearchAllHintEl: makeNode(), chatSearchBar: makeNode() };
const events = []; const rows = [
 { dataset: {}, text: 'first needle', scrollIntoView() { events.push('first'); } },
 { dataset: {}, text: 'second needle', scrollIntoView() { events.push('second'); } },
];
const transcript = ctx.window.CodoxearTranscript;
const controller = ctx.window.CodoxearChatSearch.createChatSearchController({ ...nodes, createLoadedChatSearchRuntime: transcript.createLoadedChatSearchRuntime, createChatSearchAllRuntime: transcript.createChatSearchAllRuntime, getSelected: () => 'sid-1', getPollGen: () => 1, api: async () => ({ match_count: 2, matches: [] }), setToast(message) { events.push(message); }, openSession: async () => {}, handleAppAuthLoss() {}, chatSearchTranscriptHint: () => '', syncVisibleTimeIndicator() {}, renderedMessageRows: () => rows, rowSearchText: (row) => row.text, compareRowsInDomOrder: (a, b) => rows.indexOf(a) - rows.indexOf(b), clearChatSearchMarks() {}, applyChatSearchMarks(matches, current) { events.push(`mark:${matches.length}:${rows.indexOf(current)}`); }, pulseNavigatedRow(row) { events.push(`pulse:${rows.indexOf(row)}`); }, prefersReducedMotion: () => true, oldestRenderedHistoryCursor: () => '', renderDetachedTranscriptWindow: () => false, invalidateOlderLoad() {}, setOlderState() {}, showOlderLoadError() {}, hasOlderMessages: () => false, isLoadingOlderMessages: () => false, olderPageLimit: () => 50, loadOlderMessages: async () => false, olderLoadRuntime: { beginLoad: () => ({ signal: {} }), isCurrent: () => true, finishLoad() {} } });
nodes.chatSearchInput.value = 'needle'; controller.open(); const opened = { open: controller.isOpen(), display: nodes.chatSearchBar.style.display, status: nodes.chatSearchStatus.textContent, matches: controller.currentMatches().length }; controller.step(1).then(() => { const stepped = controller.snapshot(); controller.close(); process.stdout.write(JSON.stringify({ opened, stepped, closed: { open: controller.isOpen(), display: nodes.chatSearchBar.style.display }, events })); });
'''
        )
        self.assertEqual(result["opened"], {"open": True, "display": "flex", "status": "1/2 loaded", "matches": 2})
        self.assertEqual(result["stepped"]["index"], 1)
        self.assertEqual(result["closed"], {"open": False, "display": "none"})
        self.assertIn("pulse:1", result["events"])


if __name__ == "__main__":
    unittest.main()
