/**
 * Session-catalog and launch-defaults authority.
 *
 * `sessionIndex` is derived atomically from `latestSessions`: callers cannot
 * set it independently and therefore cannot publish a list whose lookup index
 * describes a different refresh snapshot. Session records remain shared
 * objects so optimistic patches are visible through both catalog views; every
 * such patch crosses the catalog's observable write boundary.
 */

const INITIAL_DEFAULTS = Object.freeze({
  default_backend: "pi",
  backends: Object.freeze({ codex: null, pi: null, cc: null }),
});

const INITIAL_STATE = Object.freeze({
  latestSessions: Object.freeze([]),
  sessionIndex: new Map(),
  recentCwds: Object.freeze([]),
  newSessionDefaults: INITIAL_DEFAULTS,
  tmuxAvailable: false,
});

const ATTACHMENT_FIELDS = ["attachments_revision", "staged_attachments", "pending_attachment"];
const CHECKPOINT_KEY = "codoxear.attachment-resources.v1";

function attachmentRevision(row) {
  return row && Number.isSafeInteger(row.attachments_revision) && row.attachments_revision >= 0
    ? row.attachments_revision : null;
}

function sameIncarnation(a, b) {
  return a && b && a.start_ts != null && b.start_ts != null && a.start_ts === b.start_ts;
}

function reconcileAttachmentResource(current, incoming) {
  const revision = attachmentRevision(current);
  const nextRevision = attachmentRevision(incoming);
  if (revision === null || (nextRevision !== null && nextRevision >= revision)) return incoming;
  const result = { ...incoming };
  for (const field of ATTACHMENT_FIELDS) {
    if (Object.prototype.hasOwnProperty.call(current, field)) result[field] = current[field];
    else delete result[field];
  }
  return result;
}

function createSessionCatalog(options = {}) {
  const consoleError = options.consoleError === undefined ? console.error : options.consoleError;
  if (typeof consoleError !== "function") throw new TypeError("session catalog dependency missing: consoleError");

  // A bounded metadata checkpoint survives reload into an older HTTP cache.
  // It is an input to this reducer, never an independent rendered-state writer.
  let storage = null;
  let checkpoints = [];
  try {
    storage = options.storage === undefined ? globalThis.localStorage : options.storage;
    const text = storage && storage.getItem(CHECKPOINT_KEY);
    const saved = text && text.length <= 262144 ? JSON.parse(text) : [];
    if (Array.isArray(saved)) checkpoints = saved.filter((row) => row && row.session_id && row.start_ts != null &&
      attachmentRevision(row) !== null && Array.isArray(row.staged_attachments)).slice(-100);
  } catch (_) { /* Storage denial/corruption must not block the live catalog. */ }

  function checkpoint(row) {
    if (!row || row.start_ts == null || attachmentRevision(row) === null || !Array.isArray(row.staged_attachments)) return;
    const resource = {
      session_id: row.session_id, start_ts: row.start_ts,
      attachments_revision: row.attachments_revision, pending_attachment: Boolean(row.pending_attachment),
      staged_attachments: row.staged_attachments.filter((item) => item && typeof item.id === "string").map((item) => ({
        id: item.id, display_name: item.display_name, filename: item.filename,
        size: item.size, created_ts: item.created_ts,
      })),
    };
    const previous = checkpoints.find((item) => item.session_id === row.session_id);
    if (previous && JSON.stringify(previous) === JSON.stringify(resource)) return;
    checkpoints = checkpoints.filter((item) => item.session_id !== row.session_id);
    // Never persist a truncated resource. Oversized metadata stays memory-only.
    if (JSON.stringify(resource).length <= 32768) checkpoints.push(resource);
    checkpoints = checkpoints.slice(-100);
    while (checkpoints.length && JSON.stringify(checkpoints).length > 262144) checkpoints.shift();
    try { if (storage) storage.setItem(CHECKPOINT_KEY, JSON.stringify(checkpoints)); } catch (_) { /* Best effort. */ }
  }

  function reconcileRow(row) {
    if (!row || typeof row !== "object") return row;
    const live = values.sessionIndex.get(row.session_id);
    const saved = checkpoints.find((item) => item.session_id === row.session_id && sameIncarnation(item, row));
    const current = live && (sameIncarnation(live, row) || (live.start_ts == null && row.start_ts == null)) ? live : saved;
    const reconciled = reconcileAttachmentResource(current, row);
    checkpoint(reconciled);
    return reconciled;
  }

  const values = {
    latestSessions: [],
    sessionIndex: new Map(),
    recentCwds: [],
    newSessionDefaults: INITIAL_DEFAULTS,
    tmuxAvailable: false,
  };
  const subscribers = Object.fromEntries(Object.keys(INITIAL_STATE).map((field) => [field, new Set()]));

  function requireField(field) {
    if (!Object.prototype.hasOwnProperty.call(INITIAL_STATE, field)) {
      throw new TypeError(`unknown session catalog field: ${String(field)}`);
    }
    return field;
  }

  function deliver(callback, field) {
    try {
      callback(values[field], field);
    } catch (error) {
      try {
        consoleError(error);
      } catch (_) {
        // Reporting must not prevent remaining subscribers from running.
      }
    }
  }

  function notify(fields) {
    const notified = new Set();
    for (const field of fields) {
      for (const callback of [...subscribers[field]]) {
        if (notified.has(callback)) continue;
        notified.add(callback);
        deliver(callback, field);
      }
    }
  }

  function get(field) {
    return values[requireField(field)];
  }

  function normalizedFieldValue(field, value) {
    if (field === "latestSessions") {
      if (!Array.isArray(value)) throw new TypeError("session catalog latestSessions must be an array");
      return value.map(reconcileRow);
    }
    if (field === "recentCwds") {
      if (!Array.isArray(value)) throw new TypeError("session catalog recentCwds must be an array");
      return value.slice();
    }
    if (field === "newSessionDefaults") {
      if (!value || typeof value !== "object" || Array.isArray(value)) {
        throw new TypeError("session catalog newSessionDefaults must be an object");
      }
      return value;
    }
    if (field === "tmuxAvailable") return Boolean(value);
    return value;
  }

  function applySnapshot(patch) {
    if (!patch || typeof patch !== "object" || Array.isArray(patch)) {
      throw new TypeError("session catalog snapshot patch must be an object");
    }

    const fields = Object.keys(patch);
    fields.forEach((field) => {
      requireField(field);
      if (field === "sessionIndex") throw new TypeError("session catalog field is derived: sessionIndex");
    });
    const nextValues = Object.fromEntries(fields.map((field) => [field, normalizedFieldValue(field, patch[field])]));
    const changed = fields.filter((field) => {
      if (field === "latestSessions") return !Object.is(values.latestSessions, patch.latestSessions);
      return !Object.is(values[field], nextValues[field]);
    });
    let nextIndex = null;
    if (Object.prototype.hasOwnProperty.call(nextValues, "latestSessions")) {
      nextIndex = new Map();
      for (const session of nextValues.latestSessions) {
        const sessionId = session && session.session_id;
        if (sessionId !== undefined && sessionId !== null) nextIndex.set(sessionId, session);
      }
    }

    // A server response is one catalog generation: install every field before
    // any subscriber can render it, then deduplicate callbacks shared by fields.
    for (const field of fields) values[field] = nextValues[field];
    if (nextIndex) values.sessionIndex = nextIndex;
    const affected = changed.slice();
    if (nextIndex) affected.push("sessionIndex");
    notify(affected);
    return affected;
  }

  function set(field, value) {
    requireField(field);
    if (field === "sessionIndex") throw new TypeError("session catalog field is derived: sessionIndex");
    return applySnapshot({ [field]: value }).length > 0;
  }

  function patchSession(sessionId, patch) {
    if (!patch || typeof patch !== "object" || Array.isArray(patch)) {
      throw new TypeError("session catalog session patch must be an object");
    }
    const session = values.sessionIndex.get(sessionId);
    if (!session) return null;
    // Missing revision is not permission to rewrite a versioned resource.
    const reconciled = reconcileAttachmentResource(session, patch);
    Object.assign(session, reconciled);
    checkpoint(session);
    // Both views expose the same record. Notify both fields atomically so a
    // callback subscribed to either view observes the complete optimistic patch.
    notify(["sessionIndex", "latestSessions"]);
    return session;
  }

  function subscribe(field, callback) {
    requireField(field);
    if (typeof callback !== "function") throw new TypeError("session catalog subscriber must be a function");
    subscribers[field].add(callback);
    return () => subscribers[field].delete(callback);
  }

  return Object.freeze({ get, set, patchSession, applySnapshot, subscribe });
}

export { createSessionCatalog };
