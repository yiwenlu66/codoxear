import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


class TestChatScrollbackSource(unittest.TestCase):
    def test_jump_button_reloads_selected_tail(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("async function jumpToLatest() {")
        end = source.index("async function selectSession(id) {", start)
        block = source[start:end]
        self.assertIn("invalidateOlderLoad();", block)
        self.assertIn("await openSession(sid, { useCache: false });", block)
        self.assertIn("kickPoll(0);", block)

    def test_open_session_is_single_render_path(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("async function openSession(")
        end = source.index("async function pollMessages(", start)
        block = source[start:end]
        self.assertIn("const optimisticBusy = Boolean(s && s.busy);", block)
        self.assertIn("setStatus({ running: optimisticBusy, queueLen: optimisticQueueLen });", block)
        self.assertIn("setTyping(optimisticBusy);", block)
        self.assertIn("const cachedTail = s ? sessionTailCache.get(sessionId) : null;", block)
        self.assertIn("tailCacheMatchesSession(cachedTail, s)", block)
        self.assertIn("applyCachedTail(sessionId, cachedTail, s);", block)
        self.assertIn("const data = await api(`/api/sessions/${sessionId}/messages/tail?limit=${initPageLimit()}`);", block)
        self.assertIn("renderSessionTail(Array.isArray(data.events) ? data.events : []);", block)
        self.assertIn("applySessionRuntimeFromTail(sessionId, data);", block)
        self.assertLess(
            block.index("renderSessionTail(Array.isArray(data.events) ? data.events : []);"),
            block.index("applySessionRuntimeFromTail(sessionId, data);"),
        )
        self.assertNotIn("refreshInitPageState", block)

    def test_refresh_sessions_is_sidebar_only(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("async function refreshSessions() {")
        end = source.index("function appendEvent(ev) {", start)
        block = source[start:end]
        self.assertNotIn("/messages/tail", block)
        self.assertNotIn("/messages/live", block)
        self.assertNotIn("/messages/history", block)
        self.assertNotIn("await openSession(", block)

    def test_load_older_messages_uses_history_cursor_only(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("async function loadOlderMessages({ auto = false } = {}) {")
        end = source.index("function maybeAutoLoadOlder()", start)
        block = source[start:end]
        self.assertIn("if (!historyCursor) throw new Error(\"history cursor missing\");", block)
        self.assertIn("`/api/sessions/${sid}/messages/history?cursor=${encodeURIComponent(reqCursor)}&limit=${olderPageLimit()}`", block)
        self.assertIn("historyCursor = typeof data.history_cursor === \"string\" && data.history_cursor ? data.history_cursor : null;", block)
        self.assertIn("await openSession(sid, { useCache: false });", block)

    def test_poll_messages_uses_live_cursor_only(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("async function pollMessages(")
        end = source.index("async function pollLoop()", start)
        block = source[start:end]
        self.assertIn("if (!liveCursor) {", block)
        self.assertIn("await openSession(sid, { useCache: false });", block)
        self.assertIn("await api(`/api/sessions/${sid}/messages/live?cursor=${encodeURIComponent(reqCursor)}`);", block)
        self.assertIn("liveCursor = typeof data.live_cursor === \"string\" && data.live_cursor ? data.live_cursor : null;", block)
        self.assertNotIn("after_byte", block)
        self.assertNotIn("before_byte", block)

    def test_no_transcript_localstorage_cache(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertNotIn("codexweb.cache.v7", source)
        self.assertNotIn("cacheStorageKey(", source)
        self.assertNotIn("setCacheMeta(", source)
        self.assertNotIn("replaceCacheEvents(", source)
        self.assertNotIn("appendCacheEvents(", source)

    def test_send_text_keeps_session_scoped_optimistic_echo(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("async function sendText(")
        end = source.index("form.onsubmit = async", start)
        block = source[start:end]
        self.assertIn("pendingUser.push({ id: localId, sessionId, key: pendingMatchKey(raw)", block)
        self.assertIn("appendEvent({ role: \"user\", text: raw, pending: true, localId, ts: t0 });", block)
        self.assertIn("void refreshSessions().catch((e) => console.error(\"refreshSessions failed\", e));", block)


if __name__ == "__main__":
    unittest.main()
