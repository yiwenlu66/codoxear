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

function createSessionCatalog(options = {}) {
  const consoleError = options.consoleError === undefined ? console.error : options.consoleError;
  if (typeof consoleError !== "function") throw new TypeError("session catalog dependency missing: consoleError");

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
      return value.slice();
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
    Object.assign(session, patch);
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
