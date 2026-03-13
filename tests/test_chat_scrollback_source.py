import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


class TestChatScrollbackSource(unittest.TestCase):
    def test_cached_session_selection_refreshes_init_page_state(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("async function selectSession(id) {")
        end = source.index('$("#refreshBtn").onclick = refreshSessions;', start)
        block = source[start:end]
        self.assertIn("await refreshInitPageState(sid, myGen);", block)
        self.assertIn("const data = await refreshInitPageState(sid, myGen, { rerender: true });", block)

    def test_scroll_listener_triggers_older_autoload_at_top(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index('chat.addEventListener("scroll", () => {')
        end = source.index('chat.addEventListener(\n          "wheel"', start)
        block = source[start:end]
        self.assertIn("if (cur <= 1 && d <= 0) maybeAutoLoadOlder();", block)

    def test_prepending_older_events_keeps_view_near_top_boundary(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("function prependOlderEvents(allEvents) {")
        end = source.index("async function loadOlderMessages", start)
        block = source[start:end]
        self.assertIn("rebuildDecorations({ preserveScroll: false });", block)
        self.assertIn("chat.scrollTop = 1;", block)
        self.assertNotIn("chat.scrollTop = oldTop + (chat.scrollHeight - oldH);", block)


if __name__ == "__main__":
    unittest.main()
