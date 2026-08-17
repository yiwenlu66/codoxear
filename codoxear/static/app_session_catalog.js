/**
 * Session-catalog and launch-defaults authority.
 *
 * `sessionIndex` is derived atomically from `latestSessions`: callers cannot
 * set it independently and therefore cannot publish a list whose lookup index
 * describes a different refresh snapshot.
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

  function set(field, value) {
    requireField(field);
    if (field === "sessionIndex") {
      throw new TypeError("session catalog field is derived: sessionIndex");
    }

    if (field === "latestSessions") {
      if (!Array.isArray(value)) throw new TypeError("session catalog latestSessions must be an array");
      const sessions = value.slice();
      const index = new Map();
      for (const session of sessions) {
        const sessionId = session && session.session_id;
        if (sessionId !== undefined && sessionId !== null) index.set(sessionId, session);
      }
      const changed = [];
      if (!Object.is(values.latestSessions, value)) changed.push("latestSessions");
      values.latestSessions = sessions;
      values.sessionIndex = index;
      changed.push("sessionIndex");
      notify(changed);
      return true;
    }

    let next = value;
    if (field === "recentCwds") {
      if (!Array.isArray(value)) throw new TypeError("session catalog recentCwds must be an array");
      next = value.slice();
    } else if (field === "newSessionDefaults") {
      if (!value || typeof value !== "object" || Array.isArray(value)) {
        throw new TypeError("session catalog newSessionDefaults must be an object");
      }
    } else if (field === "tmuxAvailable") {
      next = Boolean(value);
    }

    if (Object.is(values[field], next)) return false;
    values[field] = next;
    notify([field]);
    return true;
  }

  function subscribe(field, callback) {
    requireField(field);
    if (typeof callback !== "function") throw new TypeError("session catalog subscriber must be a function");
    subscribers[field].add(callback);
    return () => subscribers[field].delete(callback);
  }

  return Object.freeze({ get, set, subscribe });
}

export { createSessionCatalog };
