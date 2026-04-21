import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


class TestChatScrollbackSource(unittest.TestCase):
    def test_jump_button_reloads_latest_tail_before_scroll(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("async function jumpToLatest() {")
        end = source.index("async function selectSession(id) {", start)
        block = source[start:end]
        self.assertIn("invalidateOlderLoad();", block)
        self.assertIn("await refreshInitPageState(sid, gen, { rerender: true });", block)
        self.assertIn("kickPoll(0);", block)

    def test_cached_session_selection_uses_byte_cursors(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("async function selectSession(id) {")
        end = source.index("function parseHarnessDraftInt(", start)
        block = source[start:end]
        self.assertIn("Number(cached.after_byte) > 0", block)
        self.assertIn("afterByte = Number(cached.after_byte) || 0;", block)
        self.assertIn("beforeByte = Number(cached.before_byte) || 0;", block)
        self.assertIn("await refreshInitPageState(sid, myGen);", block)

    def test_scroll_listener_triggers_older_autoload_at_top(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index('chat.addEventListener("scroll", () => {')
        end = source.index('chat.addEventListener(\n          "wheel"', start)
        block = source[start:end]
        self.assertIn("if (loadingOlder && cur > OLDER_CANCEL_PX) invalidateOlderLoad();", block)
        self.assertIn("if (cur <= OLDER_TOP_TRIGGER_PX && d <= 0) maybeAutoLoadOlder();", block)

    def test_prepending_older_events_keeps_view_near_top_boundary(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("function prependOlderEvents(allEvents, { preserveViewport = false } = {}) {")
        end = source.index("async function loadOlderMessages", start)
        block = source[start:end]
        self.assertIn("const anchorRow = preserveViewport ? firstVisibleMessageRow() : null;", block)
        self.assertIn("trimRenderedRowsBeforeViewport({ maxRows: CHAT_DOM_WINDOW_WITH_HISTORY_SLACK });", block)
        self.assertIn("chat.scrollTop = Math.max(0, anchorRow.offsetTop - anchorOffset);", block)
        self.assertIn("chat.scrollTop = 1;", block)

    def test_load_older_messages_uses_history_before_byte_cursor(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("async function loadOlderMessages({ auto = false } = {}) {")
        end = source.index("function maybeAutoLoadOlder()", start)
        block = source[start:end]
        self.assertIn("const reqBefore = Math.max(0, Number(beforeByte) || 0);", block)
        self.assertIn("`/api/sessions/${sid}/messages/history?before_byte=${reqBefore}&limit=${olderPageLimit()}`", block)
        self.assertIn("const nextBefore = Number.isFinite(Number(data.before_byte)) ? Number(data.before_byte) : reqBefore;", block)
        self.assertIn("beforeByte = nextBefore;", block)

    def test_cache_metadata_tracks_thread_identity_and_byte_cursors(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("return `codexweb.cache.v7.${sid}`;", source)
        start = source.index("function setCacheMeta(")
        end = source.index("function cacheMatchesSession(", start)
        block = source[start:end]
        self.assertIn("afterByte: after", block)
        self.assertIn("beforeByte: before", block)
        self.assertIn("cache.after_byte = 0;", block)
        self.assertIn("cache.before_byte = 0;", block)

    def test_polling_uses_live_after_byte_cursor(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("async function pollMessages(")
        end = source.index("async function pollLoop()", start)
        block = source[start:end]
        self.assertIn("const reqAfter = afterByte;", block)
        self.assertIn("await api(`/api/sessions/${sid}/messages/live?after_byte=${reqAfter}`);", block)
        self.assertIn("afterByte = Number(data.after_byte) || reqAfter;", block)
        self.assertNotIn("olderBefore += polledEventCount;", block)

    def test_refresh_init_uses_tail_and_preserves_existing_before_byte(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("async function refreshInitPageState(")
        end = source.index("async function jumpToLatest()", start)
        block = source[start:end]
        self.assertIn("await api(`/api/sessions/${sid}/messages/tail?limit=${initPageLimit()}`);", block)
        self.assertIn("const nextAfterByte = Number(data.after_byte) || 0;", block)
        self.assertIn("const previousAfterByte = Number(afterByte) || 0;", block)
        self.assertIn("const tailAdvanced = previousAfterByte !== nextAfterByte;", block)
        self.assertIn("const shouldRerender = rerender || identityChanged || tailAdvanced;", block)
        self.assertIn("shouldRerender || !(Number(beforeByte) > 0) ? nextBeforeByteBase : Number(beforeByte);", block)
        self.assertIn("afterByte = nextAfterByte;", block)

    def test_pending_log_transition_clears_cached_messages_and_byte_cursors(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("async function pollMessages(")
        end = source.index("async function pollLoop()", start)
        block = source[start:end]
        self.assertIn("setCacheMeta(sid, { threadId: activeThreadId, logPath: null, afterByte: 0, beforeByte: 0, hasOlder: false });", block)
        self.assertIn("replaceCacheEvents(sid, []);", block)


if __name__ == "__main__":
    unittest.main()
