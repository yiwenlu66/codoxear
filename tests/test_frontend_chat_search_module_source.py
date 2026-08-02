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
        self.assertEqual(result["opened"], {"open": True, "display": "flex", "status": "1 of 2", "matches": 2})
        self.assertEqual(result["stepped"]["index"], 1)
        self.assertEqual(result["closed"], {"open": False, "display": "none"})
        self.assertContains("pulse:1", result["events"])

    def test_status_explains_how_to_reach_older_matches(self) -> None:
        result = run_chat(
            r'''
const makeNode = () => ({ style: {}, disabled: false, textContent: '', title: '', value: '', dataset: {}, focus() {}, select() {} });
const nodes = { chatSearchBtn: makeNode(), chatSearchInput: makeNode(), chatSearchPrevBtn: makeNode(), chatSearchNextBtn: makeNode(), chatSearchCloseBtn: makeNode(), chatSearchStatus: makeNode(), chatSearchAllHintEl: makeNode(), chatSearchBar: makeNode() };
const rows = [{ dataset: {}, text: 'needle', scrollIntoView() {} }];
const transcript = ctx.window.CodoxearTranscript;
const controller = ctx.window.CodoxearChatSearch.createChatSearchController({ ...nodes, createLoadedChatSearchRuntime: transcript.createLoadedChatSearchRuntime, createChatSearchAllRuntime: transcript.createChatSearchAllRuntime, getSelected: () => 'sid-1', getPollGen: () => 1, api: async () => ({ match_count: 2, matches: [] }), setToast() {}, openSession: async () => {}, handleAppAuthLoss() {}, chatSearchTranscriptHint: () => '', syncVisibleTimeIndicator() {}, renderedMessageRows: () => rows, rowSearchText: (row) => row.text, compareRowsInDomOrder: () => 0, clearChatSearchMarks() {}, applyChatSearchMarks() {}, pulseNavigatedRow() {}, prefersReducedMotion: () => true, oldestRenderedHistoryCursor: () => 'cursor', renderDetachedTranscriptWindow: () => false, invalidateOlderLoad() {}, setOlderState() {}, showOlderLoadError() {}, hasOlderMessages: () => true, isLoadingOlderMessages: () => false, olderPageLimit: () => 50, loadOlderMessages: async () => false, olderLoadRuntime: { beginLoad: () => ({ signal: {} }), isCurrent: () => true, finishLoad() {} } });
nodes.chatSearchInput.value = 'needle'; controller.open(); process.stdout.write(JSON.stringify({ status: nodes.chatSearchStatus.textContent, prevDisabled: nodes.chatSearchPrevBtn.disabled }));
'''
        )
        self.assertEqual(result, {"status": "1 of 1 · Older matches may exist; Previous loads them", "prevDisabled": False})
    def test_query_change_rehighlights_after_the_short_debounce(self) -> None:
        result = run_chat(
            r'''
const timers = [];
ctx.window.setTimeout = (fn, ms) => { const timer = { fn, ms, cancelled: false }; timers.push(timer); return timer; };
ctx.window.clearTimeout = (timer) => { if (timer) timer.cancelled = true; };
const events = [];
const makeNode = () => ({ style: {}, disabled: false, textContent: '', title: '', value: '', dataset: {}, focus() {}, select() {} });
const nodes = { chatSearchBtn: makeNode(), chatSearchInput: makeNode(), chatSearchPrevBtn: makeNode(), chatSearchNextBtn: makeNode(), chatSearchCloseBtn: makeNode(), chatSearchStatus: makeNode(), chatSearchAllHintEl: makeNode(), chatSearchBar: makeNode() };
const rows = [{ dataset: {}, text: 'needle then other', scrollIntoView() {} }];
const transcript = ctx.window.CodoxearTranscript;
const controller = ctx.window.CodoxearChatSearch.createChatSearchController({ ...nodes, createLoadedChatSearchRuntime: transcript.createLoadedChatSearchRuntime, createChatSearchAllRuntime: transcript.createChatSearchAllRuntime, getSelected: () => 'sid-1', getPollGen: () => 1, api: async () => ({ match_count: 1, matches: [] }), setToast() {}, openSession: async () => {}, handleAppAuthLoss() {}, chatSearchTranscriptHint: () => '', syncVisibleTimeIndicator() {}, renderedMessageRows: () => rows, rowSearchText: (row) => row.text, compareRowsInDomOrder: () => 0, clearChatSearchMarks() { events.push('clear'); }, applyChatSearchMarks(matches, current, query) { events.push(`mark:${query}`); }, pulseNavigatedRow() {}, prefersReducedMotion: () => true, oldestRenderedHistoryCursor: () => '', renderDetachedTranscriptWindow: () => false, invalidateOlderLoad() {}, setOlderState() {}, showOlderLoadError() {}, hasOlderMessages: () => false, isLoadingOlderMessages: () => false, olderPageLimit: () => 50, loadOlderMessages: async () => false, olderLoadRuntime: { beginLoad: () => ({ signal: {} }), isCurrent: () => true, finishLoad() {} } });
nodes.chatSearchInput.value = 'needle'; controller.open(); events.length = 0;
nodes.chatSearchInput.value = 'other'; nodes.chatSearchInput.oninput();
const immediate = [...events];
const timer = timers.find((item) => item.ms === 150 && !item.cancelled); timer.fn();
process.stdout.write(JSON.stringify({ immediate, afterDebounce: events, status: nodes.chatSearchStatus.textContent }));
'''
        )
        self.assertEqual(result["immediate"], [])
        self.assertEqual(result["afterDebounce"], ["clear", "mark:other"])
        self.assertEqual(result["status"], "1 of 1")

    def test_status_says_no_matches_for_a_nonempty_query(self) -> None:
        result = run_chat(
            r'''
const makeNode = () => ({ style: {}, disabled: false, textContent: '', title: '', value: '', dataset: {}, focus() {}, select() {} });
const nodes = { chatSearchBtn: makeNode(), chatSearchInput: makeNode(), chatSearchPrevBtn: makeNode(), chatSearchNextBtn: makeNode(), chatSearchCloseBtn: makeNode(), chatSearchStatus: makeNode(), chatSearchAllHintEl: makeNode(), chatSearchBar: makeNode() };
const transcript = ctx.window.CodoxearTranscript;
const controller = ctx.window.CodoxearChatSearch.createChatSearchController({ ...nodes, createLoadedChatSearchRuntime: transcript.createLoadedChatSearchRuntime, createChatSearchAllRuntime: transcript.createChatSearchAllRuntime, getSelected: () => 'sid-1', getPollGen: () => 1, api: async () => ({ match_count: 0, matches: [] }), setToast() {}, openSession: async () => {}, handleAppAuthLoss() {}, chatSearchTranscriptHint: () => '', syncVisibleTimeIndicator() {}, renderedMessageRows: () => [], rowSearchText: () => '', compareRowsInDomOrder: () => 0, clearChatSearchMarks() {}, applyChatSearchMarks() {}, pulseNavigatedRow() {}, prefersReducedMotion: () => true, oldestRenderedHistoryCursor: () => '', renderDetachedTranscriptWindow: () => false, invalidateOlderLoad() {}, setOlderState() {}, showOlderLoadError() {}, hasOlderMessages: () => false, isLoadingOlderMessages: () => false, olderPageLimit: () => 50, loadOlderMessages: async () => false, olderLoadRuntime: { beginLoad: () => ({ signal: {} }), isCurrent: () => true, finishLoad() {} } });
nodes.chatSearchInput.value = 'absent'; controller.open(); process.stdout.write(JSON.stringify({ status: nodes.chatSearchStatus.textContent, prevDisabled: nodes.chatSearchPrevBtn.disabled, nextDisabled: nodes.chatSearchNextBtn.disabled }));
'''
        )
        self.assertEqual(result, {"status": "no matches", "prevDisabled": True, "nextDisabled": True})

    def test_navigation_controls_keep_focus_and_pass_the_query_to_marking(self) -> None:
        result = run_chat(
            r'''
const events = [];
const makeNode = (name) => ({ style: {}, disabled: false, textContent: '', title: '', value: '', dataset: {}, focus() { events.push(`focus:${name}`); }, select() {} });
const nodes = { chatSearchBtn: makeNode('button'), chatSearchInput: makeNode('input'), chatSearchPrevBtn: makeNode('previous'), chatSearchNextBtn: makeNode('next'), chatSearchCloseBtn: makeNode('close'), chatSearchStatus: makeNode('status'), chatSearchAllHintEl: makeNode('hint'), chatSearchBar: makeNode('bar') };
const rows = [
  { dataset: {}, text: 'first needle', scrollIntoView() {} },
  { dataset: {}, text: 'second needle', scrollIntoView() {} },
];
const transcript = ctx.window.CodoxearTranscript;
const controller = ctx.window.CodoxearChatSearch.createChatSearchController({ ...nodes, createLoadedChatSearchRuntime: transcript.createLoadedChatSearchRuntime, createChatSearchAllRuntime: transcript.createChatSearchAllRuntime, getSelected: () => 'sid-1', getPollGen: () => 1, api: async () => ({ match_count: 2, matches: [] }), setToast() {}, openSession: async () => {}, handleAppAuthLoss() {}, chatSearchTranscriptHint: () => '', syncVisibleTimeIndicator() {}, renderedMessageRows: () => rows, rowSearchText: (row) => row.text, compareRowsInDomOrder: (a, b) => rows.indexOf(a) - rows.indexOf(b), clearChatSearchMarks() { events.push('clear'); }, applyChatSearchMarks(matches, current, query) { events.push(`mark:${rows.indexOf(current)}:${query}`); }, pulseNavigatedRow() {}, prefersReducedMotion: () => true, oldestRenderedHistoryCursor: () => '', renderDetachedTranscriptWindow: () => false, invalidateOlderLoad() {}, setOlderState() {}, showOlderLoadError() {}, hasOlderMessages: () => false, isLoadingOlderMessages: () => false, olderPageLimit: () => 50, loadOlderMessages: async () => false, olderLoadRuntime: { beginLoad: () => ({ signal: {} }), isCurrent: () => true, finishLoad() {} } });
const event = { preventDefault() {}, stopPropagation() {} };
nodes.chatSearchInput.value = 'needle'; controller.open();
nodes.chatSearchNextBtn.onclick(event);
const afterNext = controller.snapshot().index;
nodes.chatSearchInput.onkeydown({ key: 'Enter', shiftKey: false, preventDefault() {} });
const afterEnter = controller.snapshot().index;
nodes.chatSearchInput.onkeydown({ key: 'Enter', shiftKey: true, preventDefault() {} });
const afterShiftEnter = controller.snapshot().index;
nodes.chatSearchInput.onkeydown({ key: 'Escape', preventDefault() {} });
process.stdout.write(JSON.stringify({ afterNext, afterEnter, afterShiftEnter, open: controller.isOpen(), marks: events.filter((event) => event.startsWith('mark:')), focusedInput: events.filter((event) => event === 'focus:input').length, clears: events.filter((event) => event === 'clear').length }));
'''
        )
        self.assertEqual(result["afterNext"], 1)
        self.assertEqual(result["afterEnter"], 0)
        self.assertEqual(result["afterShiftEnter"], 1)
        self.assertFalse(result["open"])
        self.assertEqual(result["focusedInput"], 2)  # opening search and using Next both focus the input
        self.assertGreaterEqual(result["clears"], 5)
        self.assertEqual(result["marks"], ["mark:0:needle", "mark:0:needle", "mark:1:needle", "mark:1:needle", "mark:0:needle", "mark:0:needle", "mark:1:needle"])


if __name__ == "__main__":
    unittest.main()
