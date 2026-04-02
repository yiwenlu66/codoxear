import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


class TestChatScrollbackSource(unittest.TestCase):
    def test_jump_button_reloads_latest_page_before_scroll(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("async function jumpToLatest() {")
        end = source.index("async function selectSession(id) {", start)
        block = source[start:end]
        self.assertIn("await refreshInitPageState(sid, gen, { rerender: true });", block)
        self.assertIn("kickPoll(0);", block)

        handler_start = source.index("jumpBtn.onclick = () => {")
        handler_end = source.index("olderBtn.onclick = () => {", handler_start)
        handler = source[handler_start:handler_end]
        self.assertIn("void jumpToLatest();", handler)

    def test_cached_session_selection_refreshes_init_page_state(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("async function selectSession(id) {")
        end = source.index("function updateHarnessBtnState() {", start)
        block = source[start:end]
        self.assertIn("await refreshInitPageState(sid, myGen);", block)
        self.assertIn("const data = await refreshInitPageState(sid, myGen, { rerender: true });", block)
        self.assertIn("if (cached && !cacheMatchesSession(cached, s0)) clearCache(sid);", block)
        self.assertIn("cacheMatchesSession(cached, s0)", block)

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
        self.assertIn("trimRenderedRowsBeforeViewport();", block)
        self.assertIn("rebuildDecorations({ preserveScroll: false });", block)
        self.assertIn("chat.scrollTop = 1;", block)
        self.assertNotIn("chat.scrollTop = oldTop + (chat.scrollHeight - oldH);", block)
        self.assertNotIn("trimRenderedRows({ fromTop: false });", block)

    def test_viewport_tail_trim_only_removes_rows_before_visible_band(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("function trimRenderedRowsBeforeViewport() {")
        end = source.index("function makeRow(ev, { ts, pending }) {", start)
        block = source[start:end]
        self.assertIn("const viewportTop = chat.scrollTop + 1;", block)
        self.assertIn("const removable = Math.min(extra, firstVisible);", block)
        self.assertIn("for (const row of rows.slice(0, removable)) row.remove();", block)

    def test_cache_metadata_tracks_thread_identity(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("return `codexweb.cache.v5.${sid}`;", source)
        start = source.index("function setCacheMeta(")
        end = source.index("function cacheMatchesSession(", start)
        block = source[start:end]
        self.assertIn("threadId", block)
        self.assertIn("cache.events = [];", block)
        self.assertIn("cache.offset = 0;", block)

    def test_refresh_init_rerenders_when_thread_or_log_identity_changes(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("async function refreshInitPageState(")
        end = source.index("async function jumpToLatest()", start)
        block = source[start:end]
        self.assertIn("const identityChanged = (activeLogPath || null) !== nextLogPath || (activeThreadId || null) !== nextThreadId;", block)
        self.assertIn("if (rerender || identityChanged) {", block)
        self.assertIn("else replaceCacheEvents(sid, []);", block)

    def test_pending_log_transition_clears_cached_messages(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("async function pollMessages(")
        end = source.index("async function pollLoop()", start)
        block = source[start:end]
        self.assertIn("setCacheMeta(sid, { threadId: activeThreadId, logPath: null, offset: 0, olderBefore: 0, hasOlder: false });", block)
        self.assertIn("replaceCacheEvents(sid, []);", block)


if __name__ == "__main__":
    unittest.main()
