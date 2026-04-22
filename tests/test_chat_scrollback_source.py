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
        self.assertIn('activeTranscriptState = "pending_bind";', block)
        self.assertIn("const optimisticBusy = Boolean(s && s.busy);", block)
        self.assertIn("setStatus({ running: optimisticBusy, queueLen: optimisticQueueLen });", block)
        self.assertIn("setTyping(optimisticBusy);", block)
        self.assertIn("const cachedTail = s ? sessionTailCache.get(sessionId) : null;", block)
        self.assertIn("tailCacheMatchesSession(cachedTail, s)", block)
        self.assertIn("applyCachedTail(sessionId, cachedTail, s);", block)
        self.assertIn("const data = await api(`/api/sessions/${sessionId}/messages/tail?limit=${initPageLimit()}`);", block)
        self.assertIn("const slotChange = updateSessionTranscriptSlot(sessionId, data);", block)
        self.assertIn('if (slotChange.current.state === "bound") renderSessionTail(Array.isArray(data.events) ? data.events : []);', block)
        self.assertIn("else renderPendingTranscriptSlot(sessionId);", block)
        self.assertIn("applySessionRuntimeFromTail(sessionId, data);", block)
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
        self.assertIn('if (activeTranscriptState === "pending_bind") {', block)
        self.assertIn("const slotChange = updateSessionTranscriptSlot(sid, data);", block)
        self.assertIn('if (slotChange.current.state === "bound") renderSessionTail(Array.isArray(data.events) ? data.events : []);', block)
        self.assertIn("await openSession(sid, { useCache: false });", block)
        self.assertIn("await api(`/api/sessions/${sid}/messages/live?cursor=${encodeURIComponent(reqCursor)}`);", block)
        self.assertIn("const slotInfo = transcriptSnapshotFromData(data);", block)
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

    def test_send_text_scopes_optimistic_echo_to_transcript_epoch(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("async function sendText(")
        end = source.index("form.onsubmit = async", start)
        block = source[start:end]
        self.assertIn("const slot = getSessionTranscriptSlot(sessionId);", block)
        self.assertIn("pendingUser.push({ id: localId, sessionId, epoch: slot.epoch, key: pendingMatchKey(raw)", block)
        self.assertIn("appendEvent({ role: \"user\", text: raw, pending: true, localId, ts: t0 });", block)
        self.assertIn("void refreshSessions().catch((e) => console.error(\"refreshSessions failed\", e));", block)

    def test_restore_pending_rows_is_bound_to_current_transcript_slot(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("function restorePendingUserRowsForSession(sessionId) {")
        end = source.index("function updateQueueBadge()", start)
        block = source[start:end]
        self.assertIn("const slot = getSessionTranscriptSlot(sessionId);", block)
        self.assertIn("Number(item.epoch || 0) === Number(slot.epoch || 0)", block)

    def test_render_transcript_rebuilds_authoritative_events_after_pending_match(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("function renderTranscript(events, { preserveScroll = false } = {}) {")
        end = source.index("function prependOlderEvents(", start)
        block = source[start:end]
        self.assertIn("takePendingUserMatch(ev);", block)
        self.assertIn("msgs.push(ev);", block)
        self.assertNotIn("if (consumePendingUserIfMatches(ev)) continue;", block)


if __name__ == "__main__":
    unittest.main()
