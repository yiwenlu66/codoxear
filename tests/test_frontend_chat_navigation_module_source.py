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
const events = []; let selected = 'sid-1'; let rows = []; let allRows = []; let apiMode = 'total'; let windowMode = 'success'; let olderMode = 'success'; let keyHandler = null; let liveTail = false;
function row(id, cursor, top) { return { dataset: { messageId: id, historyCursor: cursor }, offsetTop: top, scrollIntoView() { events.push(`scroll:${id}`); } }; }
function arow(id, top) { const item = row(id, `c-${id}`, top); item.dataset.role = 'assistant'; return item; }
const prev = { style: {}, disabled: false }, next = { style: {}, disabled: false };
const chatNavRail = { style: {} }, chatEmptyState = { style: {} };
const sessionState = { get: () => selected, subscribe: () => () => {} };
const deps = {
  prevUserBtn: prev, nextUserBtn: next, chatNavRail, chatEmptyState, sessionState, currentGeneration: () => 1,
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
  renderedMessageRows: () => allRows,
  loadedUserJumpTarget: (items, direction) => direction > 0 && items.length > 1 ? { target: items[1], reason: 'target' } : { target: null, reason: direction < 0 ? 'first' : 'last' },
  loadedCopyJumpTarget: () => ({ target: null, reason: 'last' }), getScrollTop: () => 0,
  pulseNavigatedRow: (item) => events.push(`pulse:${item.dataset.messageId}`), setToast: (text) => events.push(`toast:${text}`),
  openChatSearch: () => events.push('open-search'), handleAppAuthLoss() {}, isTextEntryElement: () => false,
  modalIsolationTargets: [], isModalTargetOpen: () => false,
  addAppEvent: (_target, type, handler) => { if (type === 'keydown') keyHandler = handler; },
  documentTarget: { body: { classList: { contains: () => false } } }, isSidebarOpen: () => false,
  renderedAtLiveTail: () => liveTail,
};
const controller = ctx.window.CodoxearChatNavigation.createChatNavigationController(deps);
'''


class TestFrontendChatNavigationModuleSource(unittest.TestCase):
    def test_buttons_follow_selection_without_any_total_fetch(self) -> None:
        result = run_node(HARNESS + r'''
(async () => {
  controller.syncButtons();
  await new Promise((resolve) => setImmediate(resolve));
  const enabled = [prev.disabled, next.disabled];
  selected = null; controller.syncButtons();
  process.stdout.write(JSON.stringify({ enabled, disabled: [prev.disabled, next.disabled], apiCalls: events.filter((item) => item.startsWith('api:')) }));
})();
''')
        self.assertEqual(result["enabled"], [False, False])
        self.assertEqual(result["disabled"], [True, True])
        self.assertEqual(result["apiCalls"], [])

    def test_next_mode_swaps_to_assistant_top_past_last_user_message(self) -> None:
        result = run_node(HARNESS + r'''
(async () => {
  liveTail = true;
  rows = [row('u1', 'c1', 0)];
  allRows = [rows[0], arow('a1', -500)];
  controller.syncButtons();
  const title = next.title;
  next.onclick({ preventDefault() {}, stopPropagation() {} });
  process.stdout.write(JSON.stringify({ title, events }));
})();
''')
        self.assertEqual(result["title"], "Go to top of current assistant message")
        self.assertIn("scroll:a1", result["events"])
        self.assertIn("pulse:a1", result["events"])
        self.assertFalse(any(item.startswith("api:") for item in result["events"]))

    def test_assistant_top_walks_to_previous_message_when_already_at_top(self) -> None:
        result = run_node(HARNESS + r'''
(async () => {
  liveTail = true;
  rows = [row('u1', 'c1', 0)];
  allRows = [rows[0], arow('a1', -300), arow('a2', 24)];
  controller.syncButtons();
  next.onclick({ preventDefault() {}, stopPropagation() {} });
  process.stdout.write(JSON.stringify({ events }));
})();
''')
        self.assertIn("scroll:a1", result["events"])
        self.assertNotIn("scroll:a2", result["events"])

    def test_assistant_top_targets_reply_below_viewport(self) -> None:
        result = run_node(HARNESS + r'''
(async () => {
  liveTail = true;
  rows = [row('u1', 'c1', 0)];
  allRows = [rows[0], arow('a1', 500)];
  controller.syncButtons();
  next.onclick({ preventDefault() {}, stopPropagation() {} });
  process.stdout.write(JSON.stringify({ events }));
})();
''')
        self.assertIn("scroll:a1", result["events"])

    def test_assistant_top_mode_reports_first_assistant_boundary(self) -> None:
        result = run_node(HARNESS + r'''
(async () => {
  liveTail = true;
  rows = [];
  allRows = [arow('a1', 24)];
  controller.syncButtons();
  next.onclick({ preventDefault() {}, stopPropagation() {} });
  process.stdout.write(JSON.stringify({ events }));
})();
''')
        self.assertIn("toast:At first assistant message", result["events"])
        self.assertFalse(any(item.startswith("scroll:") for item in result["events"]))

    def test_assistant_top_mode_falls_back_to_user_navigation_without_assistant_rows(self) -> None:
        result = run_node(HARNESS + r'''
(async () => {
  liveTail = true;
  rows = [row('u1', 'c1', 0)];
  allRows = rows;
  controller.syncButtons();
  next.onclick({ preventDefault() {}, stopPropagation() {} });
  await new Promise((resolve) => setImmediate(resolve));
  process.stdout.write(JSON.stringify({ events }));
})();
''')
        self.assertIn("toast:At last user message", result["events"])

    def test_next_mode_stays_user_when_user_row_below_viewport(self) -> None:
        result = run_node(HARNESS + r'''
(async () => {
  liveTail = true;
  rows = [row('u1', 'c1', 0), row('u2', 'c2', 300)];
  allRows = rows;
  controller.syncButtons();
  process.stdout.write(JSON.stringify({ title: next.title }));
})();
''')
        self.assertEqual(result["title"], "Next user message")

    def test_next_mode_stays_user_in_detached_window(self) -> None:
        result = run_node(HARNESS + r'''
(async () => {
  liveTail = false;
  rows = [row('u1', 'c1', 0)];
  allRows = rows;
  controller.syncButtons();
  process.stdout.write(JSON.stringify({ title: next.title }));
})();
''')
        self.assertEqual(result["title"], "Next user message")

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
  rows = [row('u1', 'c1', 0)]; allRows = rows; apiMode = 'boundary';
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
  rows = [row('u2', 'c2', 100)]; allRows = rows; apiMode = 'boundary';
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
  rows = [row('u1', 'c1', 0)]; allRows = rows; apiMode = 'cross';
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
  rows = [row('u1', 'c1', 0)]; allRows = rows; apiMode = 'cross';
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
  rows = [row('u2', 'c2', 100)]; allRows = rows; apiMode = 'boundary'; olderMode = 'failure';
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
  rows = [row('u1', 'c1', 0)]; allRows = rows; apiMode = 'cross'; windowMode = 'failure';
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
  rows = [row('u1', 'c1', 0)]; allRows = rows; apiMode = 'stale';
  await controller.jumpToLoadedUserMessage(-1);
  process.stdout.write(JSON.stringify({ events }));
})();
''')
        self.assertFalse(any(item.startswith("toast:") for item in result["events"]))
        self.assertNotIn("older", result["events"])
        self.assertFalse(any(item.startswith("window:") for item in result["events"]))
        self.assertFalse(any(item.startswith("scroll:") for item in result["events"]))

    def test_prev_without_user_rows_anchors_on_window_start(self) -> None:
        result = run_node(HARNESS + r'''
(async () => {
  rows = []; allRows = [row('a1', 'ca1', 0), row('a2', 'ca2', 100)]; apiMode = 'boundary';
  await controller.jumpToLoadedUserMessage(-1);
  process.stdout.write(JSON.stringify({ events }));
})();
''')
        self.assertTrue(any("/messages/neighbor" in item and "direction=previous" in item and "cursor=ca1" in item for item in result["events"]))
        self.assertIn("older", result["events"])
        self.assertIn("scroll:u0", result["events"])
        self.assertNotIn("toast:Could not reach that message", result["events"])

    def test_next_without_user_rows_anchors_on_window_end(self) -> None:
        result = run_node(HARNESS + r'''
(async () => {
  rows = []; allRows = [row('a1', 'ca1', 0), row('a2', 'ca2', 100)]; apiMode = 'cross';
  await controller.jumpToLoadedUserMessage(1);
  process.stdout.write(JSON.stringify({ events }));
})();
''')
        self.assertTrue(any("/messages/neighbor" in item and "direction=next" in item and "cursor=ca2" in item for item in result["events"]))
        self.assertIn("window:c0", result["events"])
        self.assertIn("scroll:u0", result["events"])
        self.assertNotIn("toast:Could not reach that message", result["events"])

    def test_next_anchor_is_window_boundary_not_last_user_row(self) -> None:
        result = run_node(HARNESS + r'''
(async () => {
  rows = [row('u1', 'c1', 0)]; allRows = [row('u1', 'c1', 0), row('a1', 'ca1', 100)]; apiMode = 'cross';
  await controller.jumpToLoadedUserMessage(1);
  process.stdout.write(JSON.stringify({ events }));
})();
''')
        self.assertTrue(any("/messages/neighbor" in item and "direction=next" in item and "cursor=ca1" in item for item in result["events"]))
        self.assertIn("scroll:u0", result["events"])

    def test_no_cursor_bearing_rows_keeps_quiet_error(self) -> None:
        result = run_node(HARNESS + r'''
(async () => {
  rows = []; allRows = [{ dataset: {} }];
  await controller.jumpToLoadedUserMessage(-1);
  await controller.jumpToLoadedUserMessage(1);
  process.stdout.write(JSON.stringify({ events }));
})();
''')
        self.assertEqual(result["events"].count("toast:Could not reach that message"), 2)
        self.assertFalse(any("/messages/neighbor" in item for item in result["events"]))

    def test_server_boundary_toasts_have_no_window_qualifier(self) -> None:
        result = run_node(HARNESS + r'''
(async () => {
  rows = [row('u1', 'c1', 0)]; allRows = rows; apiMode = 'total';
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
