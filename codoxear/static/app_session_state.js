/**
 * Session-runtime authority: every rendered session-runtime value has exactly
 * one writer—the session-state store. Widgets subscribe to their fields rather
 * than being re-invoked by unrelated controllers.
 */

const INITIAL_STATE = Object.freeze({
  selected: null,
  running: false,
  queueLen: 0,
  subagentsRunning: 0,
  turnOpen: false,
  sending: false,
  token: null,
});

function createSessionState(options = {}) {
  const consoleError = options.consoleError === undefined ? console.error : options.consoleError;
  if (typeof consoleError !== "function") throw new TypeError("session state dependency missing: consoleError");

  const values = { ...INITIAL_STATE };
  const subscribers = Object.fromEntries(Object.keys(INITIAL_STATE).map((field) => [field, new Set()]));

  function requireField(field) {
    if (!Object.prototype.hasOwnProperty.call(INITIAL_STATE, field)) {
      throw new TypeError(`unknown session state field: ${String(field)}`);
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
        // Reporting must not prevent the remaining subscribers from running.
      }
    }
  }

  function notify(field) {
    for (const callback of [...subscribers[field]]) deliver(callback, field);
  }

  function get(field) {
    return values[requireField(field)];
  }

  function set(field, value) {
    requireField(field);
    if (Object.is(values[field], value)) return false;
    values[field] = value;
    notify(field);
    return true;
  }

  function applyRuntime(patch) {
    if (!patch || typeof patch !== "object" || Array.isArray(patch)) {
      throw new TypeError("session state runtime patch must be an object");
    }

    const fields = Object.keys(patch);
    fields.forEach(requireField);
    const changed = fields.filter((field) => !Object.is(values[field], patch[field]));
    changed.forEach((field) => {
      values[field] = patch[field];
    });
    // Runtime snapshots are atomic: after every changed value is present,
    // each affected widget callback receives at most one notification.
    const notified = new Set();
    for (const field of changed) {
      for (const callback of [...subscribers[field]]) {
        if (notified.has(callback)) continue;
        notified.add(callback);
        deliver(callback, field);
      }
    }
    return changed;
  }

  function subscribe(field, callback) {
    requireField(field);
    if (typeof callback !== "function") throw new TypeError("session state subscriber must be a function");
    subscribers[field].add(callback);
    return () => subscribers[field].delete(callback);
  }

  return Object.freeze({ get, set, applyRuntime, subscribe });
}

export { createSessionState };
