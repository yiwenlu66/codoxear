import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"
APP_CSS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.css"


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
        self.assertIn("let displayedCachedTail = false;", block)
        self.assertIn("tailCacheMatchesSession(cachedTail, s)", block)
        self.assertIn("applyCachedTail(sessionId, cachedTail, s);", block)
        self.assertIn("displayedCachedTail = true;", block)
        self.assertIn("if (!displayedCachedTail) renderTranscriptLoading(sessionId);", block)
        self.assertIn("data = await api(`/api/sessions/${sessionId}/messages/tail?limit=${initPageLimit()}`);", block)
        self.assertIn("renderTranscriptLoadError(sessionId, e);", block)
        self.assertIn("if (e && e.status === 401) {", block)
        self.assertIn("handleAppAuthLoss();", block)
        self.assertIn("if (pollGen !== myGen || selected !== sessionId) return null;", block)
        self.assertIn("const slotChange = updateSessionTranscriptSlot(sessionId, data);", block)
        self.assertIn('if (slotChange.current.state === "bound" || slotChange.current.state === "failed") renderSessionTail(Array.isArray(data.events) ? data.events : []);', block)
        self.assertIn("else renderPendingTranscriptSlot(sessionId);", block)
        self.assertIn("applySessionRuntimeFromTail(sessionId, data);", block)
        self.assertIn('if (slotChange.current.state !== "failed") kickPoll(900);', block)
        self.assertNotIn("refreshInitPageState", block)

    def test_transcript_loading_row_is_non_transcript_feedback(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        css = APP_CSS.read_text(encoding="utf-8")
        start = source.index("function renderTranscriptLoading(sessionId)")
        end = source.index("function renderTranscriptLoadError", start)
        block = source[start:end]
        self.assertIn('class: "msg-row assistant typing-row transcript-loading-row"', block)
        self.assertIn('role: "status", "aria-live": "polite", text: "Loading transcript…"', block)
        self.assertIn("chatInner.insertBefore(row, bottomSentinel);", block)
        self.assertIn(".msg.loading", css)
        self.assertIn("color: var(--muted);", css)

    def test_transcript_load_error_row_is_non_transcript_feedback(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        css = APP_CSS.read_text(encoding="utf-8")
        start = source.index("function renderTranscriptLoadError(sessionId, err)")
        end = source.index("function applyCachedTail", start)
        block = source[start:end]
        self.assertIn('class: "msg-row assistant typing-row transcript-error-row"', block)
        self.assertIn('role: "alert"', block)
        self.assertIn("Could not load transcript.", block)
        self.assertIn("Select the conversation again to retry.", block)
        self.assertIn("setTyping(false);", block)
        self.assertIn("markClickFirstPaint();", block)
        self.assertIn(".msg.transcript-error", css)

    def test_refresh_sessions_does_not_fetch_messages(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("async function refreshSessions() {")
        end = source.index("function appendEvent(ev) {", start)
        block = source[start:end]
        self.assertNotIn("/messages/tail", block)
        self.assertNotIn("/messages/live", block)
        self.assertNotIn("/messages/history", block)
        self.assertNotIn("await openSession(", block)
        self.assertIn("if (selected && !sessionIndex.has(selected)) {", block)
        self.assertIn("localStorage.removeItem(\"codexweb.selected\");", block)
        self.assertIn("titleLabel.textContent = \"No session selected\";", block)
        self.assertIn("applySessionListTranscriptIdentity(selected, sessionIndex.get(selected));", block)

    def test_session_list_pending_bind_clears_active_transcript_slot(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("function applySessionListTranscriptIdentity(")
        end = source.index("function updateQueueBadge()", start)
        block = source[start:end]
        self.assertIn("const slotChange = updateSessionTranscriptSlot(sessionId, sessionMeta);", block)
        self.assertIn("if (!slotChange.resetPending) return;", block)
        self.assertIn("sessionTailCache.delete(sessionId);", block)
        self.assertIn("liveCursor = null;", block)
        self.assertIn("clearRenderedTranscriptRange();", block)
        self.assertIn('if (slotChange.current.state === "pending_bind") {', block)
        self.assertIn("renderPendingTranscriptSlot(sessionId);", block)
        self.assertIn("kickPoll(0);", block)

    def test_load_older_messages_uses_oldest_rendered_row_cursor(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("async function loadOlderMessages({ auto = false, cancelOnScroll = true } = {}) {")
        end = source.index("function maybeAutoLoadOlder()", start)
        block = source[start:end]
        self.assertIn("const reqCursor = oldestRenderedHistoryCursor();", block)
        self.assertIn("if (!reqCursor) throw new Error(\"history cursor missing\");", block)
        self.assertIn("`/api/sessions/${sid}/messages/history?cursor=${encodeURIComponent(reqCursor)}&limit=${olderPageLimit()}`", block)
        self.assertNotIn("historyCursor", block)
        self.assertIn("await openSession(sid, { useCache: false });", block)

    def test_live_append_does_not_splice_into_history_window(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("function appendEvent(ev) {")
        end = source.index("function renderTranscript(", start)
        block = source[start:end]
        self.assertIn("const stick = pending || (renderedAtLiveTail && (autoScroll || isNearBottom()));", block)
        self.assertIn("if (!pending && !renderedAtLiveTail) {", block)
        self.assertIn("markEventSeen(ev);", block)
        self.assertIn("return;", block)
        self.assertIn("trimRenderedRows({ fromTop: stick });", block)
        self.assertNotIn("trimRenderedRows({ fromTop: true });", block)

    def test_history_request_cursor_is_derived_from_rendered_rows(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("function oldestRenderedHistoryCursor() {")
        end = source.index("function clearRenderedTranscriptRange()", start)
        block = source[start:end]
        self.assertIn("for (const row of renderedMessageRows())", block)
        self.assertIn("row.dataset.historyCursor", block)
        self.assertIn("return cursor;", block)

    def test_history_prepend_does_not_trim_newly_fetched_older_rows(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("function prependOlderEvents(allEvents")
        end = source.index("async function loadOlderMessages", start)
        block = source[start:end]
        self.assertIn("chatInner.insertBefore(frag, anchor);", block)
        self.assertIn("trimRenderedRows({ fromTop: false, maxRows: CHAT_DOM_WINDOW_WITH_HISTORY_SLACK });", block)
        self.assertNotIn("trimRenderedRowsBeforeViewport({ maxRows: CHAT_DOM_WINDOW_WITH_HISTORY_SLACK });", block)

    def test_rendered_rows_keep_server_history_cursor(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('if (typeof ev.history_cursor === "string" && ev.history_cursor) out.history_cursor = ev.history_cursor;', source)
        start = source.index("function makeRow(ev, { ts, pending }) {")
        end = source.index("function safeMakeRow(ev, opts) {", start)
        block = source[start:end]
        self.assertIn('row.dataset.historyCursor = ev.history_cursor;', block)
        self.assertNotIn("let historyCursor", source)

    def test_poll_messages_uses_live_cursor_only(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("async function pollMessages(")
        end = source.index("async function pollLoop()", start)
        block = source[start:end]
        self.assertIn("if (!liveCursor) {", block)
        self.assertIn('if (activeTranscriptState === "pending_bind") {', block)
        self.assertIn("const slotChange = updateSessionTranscriptSlot(sid, data);", block)
        self.assertIn('if (slotChange.current.state === "bound" || slotChange.current.state === "failed") renderSessionTail(Array.isArray(data.events) ? data.events : []);', block)
        self.assertIn('if (activeTranscriptState === "failed") return;', block)
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
        self.assertIn("return true;", block)
        self.assertIn("return false;", block)

    def test_submit_clears_composer_only_after_send_success(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        send_start = source.index("async function sendText(")
        start = source.index("form.onsubmit = async", send_start)
        end = source.index("(async () =>", start)
        block = source[start:end]
        self.assertNotIn("clearComposer();\n          await sendText(raw);", block)
        self.assertIn("const ok = await sendText(raw);", block)
        self.assertIn('if (ok && $("#msg").value === raw) clearComposer();', block)

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
        self.assertIn("takePendingUserMatch(ev, selected, { allowUntimedCommit: false });", block)
        self.assertIn("msgs.push(ev);", block)
        self.assertNotIn("if (consumePendingUserIfMatches(ev)) continue;", block)

    def test_pending_commit_reconciliation_does_not_require_text_equality(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("function takePendingUserMatch(")
        end = source.index("function consumePendingUserIfMatches(", start)
        block = source[start:end]
        self.assertIn("const sameSlot = [];", block)
        self.assertIn("const exactCandidates = [];", block)
        self.assertIn("sameSlot.push(candidate);", block)
        self.assertIn("exactCandidates.length", block)
        self.assertIn("evTs >= Number(x.t0 || 0) - 5", block)
        self.assertIn("allowUntimedCommit", block)

    def test_error_and_warning_message_classes_are_rendered(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("function makeRow(ev, { ts, pending }) {")
        end = source.index("function safeMakeRow(ev, opts) {", start)
        block = source[start:end]
        self.assertIn('messageClass === "error" || messageClass === "warning"', block)
        self.assertIn("bubble.classList.add(messageClass);", block)

    def test_orphan_recovery_session_does_not_fetch_transcript_tail(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("async function openSession(")
        end = source.index("const cachedTail =", start)
        block = source[start:end]
        self.assertIn("if (s && s.orphan_recovery) {", block)
        self.assertIn("activeTranscriptState = \"failed\";", block)
        self.assertIn("syncAttachButtonState();", block)
        self.assertIn("syncQueueSubmitState();", block)
        self.assertIn("syncSendButtonState();", block)
        self.assertIn("return { events: [], busy: false, queue_len: optimisticQueueLen, token: null, transcript_state: \"failed\" };", block)

    def test_new_command_begins_transcript_renewal_after_send_ack(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("async function sendText(")
        end = source.index("form.onsubmit = async", start)
        block = source[start:end]
        self.assertIn("const renewsTranscript = isTranscriptRenewalCommand(raw, sessionId);", block)
        self.assertIn("const sessionInfo = sessionIndex.get(sessionId) || null;", block)
        self.assertIn("if (sessionInfo && sessionInfo.commit_unknown_send) {", block)
        self.assertIn("void clearCommitUnknownSend(sessionId, sessionInfo.commit_unknown_send_text || \"\");", block)
        self.assertIn("const localAttachmentCount = typeof attachedFiles === \"number\" ? attachedFiles : 0;", block)
        self.assertIn("let allowPendingAttachment = Boolean(renderHere && localAttachmentCount > 0);", block)
        self.assertIn("sessionInfo && sessionInfo.pending_attachment", block)
        self.assertIn("window.confirm(\"This session has a pending file attachment. Send it with this message?\")", block)
        self.assertLess(block.index("window.confirm"), block.index("pendingUser.push"))
        self.assertIn("const res = await api(`/api/sessions/${sessionId}/send`, { method: \"POST\", body: { text: raw, allow_pending_attachment: allowPendingAttachment } });", block)
        self.assertIn("if (renderHere && renewsTranscript) {", block)
        self.assertIn("beginTranscriptRenewal(sessionId);", block)
        self.assertIn("renderPendingTranscriptSlot(sessionId);", block)
        self.assertLess(block.index("const res = await api"), block.index("beginTranscriptRenewal(sessionId);"))
        self.assertIn("if (renderHere && !renewsTranscript) {", block)
        self.assertIn("if (!renderedAtLiveTail)", block)
        self.assertIn("const commitUnknown = Boolean(e2 && e2.obj && e2.obj.commit_unknown);", block)
        self.assertIn("setToast(\"send status unknown; check transcript before retrying\");", block)
        self.assertIn("currentSessionInfo.commit_unknown_send = true;", block)
        self.assertIn("currentSessionInfo.commit_unknown_send_text = raw;", block)
        self.assertIn("syncAttachButtonState();", block)
        self.assertIn("void refreshSessions().catch((e) => console.error(\"refreshSessions failed\", e));", block)
        self.assertIn("/broker must be restarted/i.test", block)
        self.assertIn("pending_attachment/clear", block)
        self.assertIn("attachment status unknown; check before retrying", source)
        self.assertIn("pendingUser.splice(i, 1);", block)
        self.assertIn("const pendingRow = pendingEl.closest(\".msg-row\");", block)
        self.assertIn("if (pendingRow) pendingRow.remove();", block)
        self.assertIn("currentRunning = false;", block)


if __name__ == "__main__":
    unittest.main()
