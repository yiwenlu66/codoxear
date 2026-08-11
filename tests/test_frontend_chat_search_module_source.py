from frontend_module_loader import module_path
import json
import subprocess
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPT = module_path("app_transcript.js")
CHAT_SEARCH = module_path("app_chat_search.js")


def run_chat(body: str) -> dict:
    transcript = json.dumps(TRANSCRIPT.read_text(encoding="utf-8"))
    module = json.dumps(CHAT_SEARCH.read_text(encoding="utf-8"))
    script = f"""
const vm = require('vm');
const ctx = {{ window: {{ setTimeout, clearTimeout }}, AbortController, document: {{}} }};
vm.createContext(ctx); vm.runInContext({transcript}, ctx); vm.runInContext({module}, ctx);
{body}
"""
    proc = subprocess.run(["node", "-e", textwrap.dedent(script)], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return json.loads(proc.stdout)


HARNESS = r'''
const events = [];
const makeNode = () => ({ style: {}, disabled: false, textContent: '', title: '', value: '', clientWidth: 390, focus() {}, select() {} });
const nodes = { chatSearchBtn: makeNode(), chatSearchInput: makeNode(), chatSearchPrevBtn: makeNode(), chatSearchNextBtn: makeNode(), chatSearchCloseBtn: makeNode(), chatSearchStatus: makeNode(), chatSearchAllHintEl: makeNode(), chatSearchBar: makeNode() };
let rows = [{ dataset: { messageId: 'm2', historyCursor: 'c2' }, text: 'second needle', scrollIntoView() { events.push('scroll:m2'); } }];
const matches = [
  { message_id: 'm1', before_byte: 'c1', role: 'assistant', snippet: 'first needle' },
  { message_id: 'm2', before_byte: 'c2', role: 'assistant', snippet: 'second needle' },
];
let windowLoads = 0;
const transcript = ctx.window.CodoxearTranscript;
const controller = ctx.window.CodoxearChatSearch.createChatSearchController({
  ...nodes,
  createLoadedChatSearchRuntime: transcript.createLoadedChatSearchRuntime,
  createChatSearchAllRuntime: transcript.createChatSearchAllRuntime,
  getSelected: () => 'sid-1', getPollGen: () => 1,
  api: async (url) => { events.push(`api:${url}`); return { total: 5, matches, truncated: true }; },
  loadTranscriptWindowAtCursor: async (cursor) => { windowLoads += 1; events.push(`window:${cursor}`); rows = [{ dataset: { messageId: 'm1', historyCursor: 'c1' }, text: 'first needle', scrollIntoView() { events.push('scroll:m1'); } }]; return { jumped_window: true }; },
  handleAppAuthLoss() {}, syncVisibleTimeIndicator() {}, renderedMessageRows: () => rows,
  rowSearchText: (row) => row.text, clearChatSearchMarks() { events.push('clear'); },
  applyChatSearchMarks(found, current, query) { events.push(`marks:${found.length}:${current ? current.dataset.messageId : '-'}:${query}`); },
  pulseNavigatedRow(row) { events.push(`pulse:${row.dataset.messageId}`); }, prefersReducedMotion: () => true,
});
'''


class TestFrontendChatSearchModuleBehavior(unittest.TestCase):
    def test_controller_fails_loudly_when_dom_dependencies_are_missing(self) -> None:
        result = run_chat("""
let error = ''; try { ctx.window.CodoxearChatSearch.createChatSearchController({}); } catch (e) { error = e.message; }
process.stdout.write(JSON.stringify({ frozen: Object.isFrozen(ctx.window.CodoxearChatSearch), error }));
""")
        self.assertTrue(result["frozen"])
        self.assertEqual(result["error"], "chat search controller dependency missing: chatSearchBtn")

    def test_exact_whole_transcript_status_and_cross_window_jump(self) -> None:
        result = run_chat(HARNESS + r'''
(async () => {
  nodes.chatSearchInput.value = 'needle'; controller.open();
  await controller.step(-1);
  process.stdout.write(JSON.stringify({ status: nodes.chatSearchStatus.textContent, windowLoads, events, marks: events.filter(x => x.startsWith('marks:')) }));
})();
''')
        self.assertEqual(result["status"], "4/5")
        self.assertEqual(result["windowLoads"], 1)
        self.assertIn("window:c1", result["events"])
        self.assertIn("scroll:m1", result["events"])
        self.assertTrue(any(item == "marks:1:m1:needle" for item in result["marks"]))

    def test_status_uses_spaced_format_when_roomy(self) -> None:
        result = run_chat(HARNESS + r'''
(async () => {
  nodes.chatSearchBar.clientWidth = 600; nodes.chatSearchInput.value = 'needle'; controller.open();
  await controller.step(1);
  process.stdout.write(JSON.stringify({ status: nodes.chatSearchStatus.textContent }));
})();
''')
        self.assertRegex(result["status"], r"^[1-5] of 5$")

    def test_escape_closes_and_clears_marks(self) -> None:
        result = run_chat(HARNESS + r'''
nodes.chatSearchInput.value = 'needle'; controller.open();
nodes.chatSearchInput.onkeydown({ key: 'Escape', preventDefault() {} });
process.stdout.write(JSON.stringify({ open: controller.isOpen(), display: nodes.chatSearchBar.style.display, cleared: events.filter(x => x === 'clear').length }));
''')
        self.assertFalse(result["open"])
        self.assertEqual(result["display"], "none")
        self.assertGreaterEqual(result["cleared"], 1)


if __name__ == "__main__":
    unittest.main()
