from frontend_module_loader import module_path
import json
import subprocess
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = module_path("app_chat_navigation.js")


def run_node(body: str) -> dict:
    source = json.dumps(MODULE.read_text(encoding="utf-8"))
    script = f"""
const vm = require('vm');
const ctx = {{ window: {{}}, document: {{}}, console }}; vm.createContext(ctx); vm.runInContext({source}, ctx);
{body}
"""
    proc = subprocess.run(["node", "-e", textwrap.dedent(script)], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return json.loads(proc.stdout)


HARNESS = r'''
const events = []; let selected = 'sid-1'; let rows = []; let apiMode = 'total'; let windowMode = 'success'; let olderMode = 'success'; let keyHandler = null;
function row(id, cursor, top) { return { dataset: { messageId: id, historyCursor: cursor }, offsetTop: top, scrollIntoView() { events.push(`scroll:${id}`); } }; }
const prev = { style: {}, disabled: false }, next = { style: {}, disabled: false };
const sessionState = { get: () => selected };
const deps = {
  prevUserBtn: prev, nextUserBtn: next, sessionState, pollingRuntime: { currentGeneration: () => 1 },
  api: async (url) => {
    events.push(`api:${url}`);
    if (apiMode === 'error') throw Object.assign(new Error('failed'), { status: 500 });
    if (apiMode === 'stale' && url.includes('/messages/neighbor')) { selected = 'sid-other'; return { neighbor: { message_id: 'u0', history_cursor: 'c0', before_byte: 'c0', same_log: true } }; }
    if (url.includes('/messages/neighbor')) {
      if (apiMode === 'boundary') return { neighbor: { message_id: 'u0', history_cursor: 'c0', before_byte: 'c0', same_log: true } };
      if (apiMode === 'cross') return { neighbor: { message_id: 'u0', history_cursor: 'c0', before_byte: 'c0', same_log: false } };
      return { neighbor: null, same_log: false };
    }
    if (apiMode === 'zero') return { total: 0, matches: [] };
    return { total: rows.length, matches: [] };
  },
  loadTranscriptWindowAtCursor: async (cursor) => { events.push(`window:${cursor}`); if (windowMode === 'failure') return null; rows = [row('u0', 'c0', 0)]; return { jumped_window: true }; },
  loadOlderMessages: async () => { events.push('older'); if (olderMode === 'failure') return false; if (olderMode === 'success') rows = [row('u0', 'c0', 0), ...rows]; return true; },
  loadedUserMessageRows: () => rows, loadedCopyMessageRows: () => rows,
  loadedUserJumpTarget: (items, direction) => direction > 0 && items.length > 1 ? { target: items[1], reason: 'target' } : { target: null, reason: direction < 0 ? 'first' : 'last' },
  loadedCopyJumpTarget: () => ({ target: null, reason: 'last' }), getScrollTop: () => 0,
  pulseNavigatedRow: (item) => events.push(`pulse:${item.dataset.messageId}`), setToast: (text) => events.push(`toast:${text}`),
  openChatSearch: () => events.push('open-search'), handleAppAuthLoss() {}, isTextEntryElement: () => false,
  modalIsolationTargets: [], isModalTargetOpen: () => false,
  addAppEvent: (_target, type, handler) => { if (type === 'keydown') keyHandler = handler; },
  documentTarget: { body: { classList: { contains: () => false } } }, isSidebarOpen: () => false,
};
const controller = ctx.window.CodoxearChatNavigation.createChatNavigationController(deps);
'''


class TestFrontendChatNavigationModuleSource(unittest.TestCase):
    def test_buttons_disable_only_after_server_reports_zero_users(self) -> None:
        result = run_node(HARNESS + r'''
(async () => {
  apiMode = 'zero'; controller.syncButtons(); const immediate = [prev.disabled, next.disabled];
  await new Promise((resolve) => setImmediate(resolve));
  process.stdout.write(JSON.stringify({ immediate, final: [prev.disabled, next.disabled] }));
})();
''')
        self.assertEqual(result["immediate"], [False, False])
        self.assertEqual(result["final"], [True, True])

    def test_local_user_target_scrolls_without_window_fetch(self) -> None:
        result = run_node(HARNESS + r'''
(async () => {
  rows = [row('u1', 'c1', 0), row('u2', 'c2', 100)];
  await controller.jumpToLoadedUserMessage(1);
  process.stdout.write(JSON.stringify({ events }));
})();
''')
        self.assertIn("scroll:u2", result["events"])
        self.assertIn("pulse:u2", result["events"])
        self.assertFalse(any(item.startswith("window:") for item in result["events"]))

    def test_boundary_fetches_server_target_and_loads_its_window(self) -> None:
        result = run_node(HARNESS + r'''
(async () => {
  rows = [row('u1', 'c1', 0)]; apiMode = 'boundary';
  await controller.jumpToLoadedUserMessage(-1);
  process.stdout.write(JSON.stringify({ events }));
})();
''')
        self.assertTrue(any("/messages/neighbor" in item and "direction=previous" in item and "cursor=c1" in item for item in result["events"]))
        self.assertIn("older", result["events"])
        self.assertNotIn("window:c0", result["events"])
        self.assertIn("scroll:u0", result["events"])

    def test_backward_boundary_progressively_prepends_until_target_is_loaded(self) -> None:
        result = run_node(HARNESS + r'''
(async () => {
  rows = [row('u2', 'c2', 100)]; apiMode = 'boundary';
  await controller.jumpToLoadedUserMessage(-1);
  process.stdout.write(JSON.stringify({ events, ids: rows.map((item) => item.dataset.messageId) }));
})();
''')
        self.assertEqual(result["ids"], ["u0", "u2"])
        self.assertEqual(result["events"].count("older"), 1)
        self.assertNotIn("window:c0", result["events"])
        self.assertIn("scroll:u0", result["events"])

    def test_cross_log_previous_uses_detached_window_without_prepend(self) -> None:
        result = run_node(HARNESS + r'''
(async () => {
  rows = [row('u1', 'c1', 0)]; apiMode = 'cross';
  await controller.jumpToLoadedUserMessage(-1);
  process.stdout.write(JSON.stringify({ events }));
})();
''')
        self.assertIn("window:c0", result["events"])
        self.assertNotIn("older", result["events"])
        self.assertIn("scroll:u0", result["events"])

    def test_next_boundary_uses_detached_window(self) -> None:
        result = run_node(HARNESS + r'''
(async () => {
  rows = [row('u1', 'c1', 0)]; apiMode = 'cross';
  await controller.jumpToLoadedUserMessage(1);
  process.stdout.write(JSON.stringify({ events }));
})();
''')
        self.assertIn("window:c0", result["events"])
        self.assertNotIn("older", result["events"])
        self.assertIn("scroll:u0", result["events"])

    def test_same_log_prepend_exhaustion_falls_back_to_window(self) -> None:
        result = run_node(HARNESS + r'''
(async () => {
  rows = [row('u2', 'c2', 100)]; apiMode = 'boundary'; olderMode = 'failure';
  await controller.jumpToLoadedUserMessage(-1);
  process.stdout.write(JSON.stringify({ events }));
})();
''')
        self.assertEqual(result["events"].count("older"), 1)
        self.assertIn("window:c0", result["events"])
        self.assertIn("scroll:u0", result["events"])

    def test_navigation_failure_toast_is_distinct_from_boundary(self) -> None:
        result = run_node(HARNESS + r'''
(async () => {
  rows = [row('u1', 'c1', 0)]; apiMode = 'cross'; windowMode = 'failure';
  await controller.jumpToLoadedUserMessage(1);
  apiMode = 'error';
  await controller.jumpToLoadedUserMessage(-1);
  process.stdout.write(JSON.stringify({ events }));
})();
''')
        self.assertEqual(result["events"].count("toast:Could not reach that message"), 2)
        self.assertNotIn("toast:At first user message", result["events"])
        self.assertNotIn("toast:At last user message", result["events"])

    def test_stale_session_switch_is_silent_not_a_failure_toast(self) -> None:
        result = run_node(HARNESS + r'''
(async () => {
  rows = [row('u1', 'c1', 0)]; apiMode = 'stale';
  await controller.jumpToLoadedUserMessage(-1);
  process.stdout.write(JSON.stringify({ events }));
})();
''')
        self.assertFalse(any(item.startswith("toast:") for item in result["events"]))
        self.assertNotIn("older", result["events"])
        self.assertFalse(any(item.startswith("window:") for item in result["events"]))
        self.assertFalse(any(item.startswith("scroll:") for item in result["events"]))

    def test_server_boundary_toasts_have_no_window_qualifier(self) -> None:
        result = run_node(HARNESS + r'''
(async () => {
  rows = [row('u1', 'c1', 0)]; apiMode = 'total';
  await controller.jumpToLoadedUserMessage(-1);
  await controller.jumpToLoadedUserMessage(1);
  process.stdout.write(JSON.stringify({ events }));
})();
''')
        self.assertIn("toast:At first user message", result["events"])
        self.assertIn("toast:At last user message", result["events"])

    def test_slash_opens_search_when_unblocked(self) -> None:
        result = run_node(HARNESS + r'''
const event = { defaultPrevented: false, key: '/', ctrlKey: false, metaKey: false, altKey: false, target: null, preventDefault() { events.push('prevent'); } };
keyHandler(event); process.stdout.write(JSON.stringify({ events }));
''')
        self.assertIn("open-search", result["events"])
        self.assertIn("prevent", result["events"])


if __name__ == "__main__":
    unittest.main()
