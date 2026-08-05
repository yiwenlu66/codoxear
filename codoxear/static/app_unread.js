(() => {
  "use strict";

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`unread controller dependency missing: ${name}`);
    return value;
  }

  function unreadPayload(value) {
    const count = Math.max(0, Math.floor(Number(value && value.count) || 0));
    const firstUnreadEventId = typeof (value && value.first_unread_event_id) === "string" ? value.first_unread_event_id.trim() : "";
    const lastUnreadEventId = typeof (value && value.last_unread_event_id) === "string" ? value.last_unread_event_id.trim() : "";
    return Object.freeze({
      count,
      firstUnreadEventId: count ? firstUnreadEventId : "",
      lastUnreadEventId: count ? lastUnreadEventId : "",
    });
  }

  function rowForMessageId(rows, messageId) {
    const target = String(messageId || "").trim();
    if (!target) return null;
    return (Array.isArray(rows) ? rows : []).find((row) => row && row.dataset && row.dataset.messageId === target) || null;
  }

  function createUnreadController(options = {}) {
    const api = requireFunction(options.api, "api");
    const patchSessionInfo = requireFunction(options.patchSessionInfo, "patchSessionInfo");
    const pendingBySession = new Map();

    async function refreshSidebarCounts(sessions) {
      await Promise.all((Array.isArray(sessions) ? sessions : []).map(async (session) => {
        if (!session || typeof session.session_id !== "string" || !session.session_id) return;
        const unread = unreadPayload(await api(`/api/sessions/${session.session_id}/unread`));
        session.unread_count = unread.count;
      }));
    }

    async function loadForOpen(sessionId) {
      const sid = String(sessionId || "").trim();
      if (!sid) return unreadPayload(null);
      pendingBySession.delete(sid);
      const unread = unreadPayload(await api(`/api/sessions/${sid}/unread`));
      if (unread.count && unread.firstUnreadEventId && unread.lastUnreadEventId) {
        pendingBySession.set(sid, { ...unread, firstDelivered: false, marking: false });
      } else {
        pendingBySession.delete(sid);
      }
      patchSessionInfo(sid, { unread_count: unread.count });
      return unread;
    }

    function firstUnreadForInitialRender(sessionId) {
      const pending = pendingBySession.get(String(sessionId || ""));
      if (!pending || pending.firstDelivered) return "";
      pending.firstDelivered = true;
      return pending.firstUnreadEventId;
    }

    async function markReadIfScrolledPast(sessionId, rows, viewportTop) {
      const sid = String(sessionId || "").trim();
      const pending = pendingBySession.get(sid);
      if (!pending || pending.marking) return false;
      const target = rowForMessageId(rows, pending.lastUnreadEventId);
      if (!target) return false;
      const rowEnd = Math.max(0, Number(target.offsetTop) || 0) + Math.max(1, Number(target.offsetHeight) || 0);
      if ((Number(viewportTop) || 0) < rowEnd) return false;
      pending.marking = true;
      try {
        await api(`/api/sessions/${sid}/read`, { method: "POST", body: { event_id: pending.lastUnreadEventId } });
        pendingBySession.delete(sid);
        patchSessionInfo(sid, { unread_count: 0 });
        return true;
      } catch (error) {
        pending.marking = false;
        throw error;
      }
    }

    return Object.freeze({
      firstUnreadForInitialRender,
      loadForOpen,
      markReadIfScrolledPast,
      refreshSidebarCounts,
    });
  }

  window.CodoxearUnread = Object.freeze({ createUnreadController, rowForMessageId, unreadPayload });
})();
