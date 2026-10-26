const MENTION_LOCAL_PART_RE = /[A-Za-z0-9._%+-]/;

// A mention is an `@` that is not glued to an email local part. The renderer
// encodes the same rule (app_markdown.js: the file-ref boundary class, with an
// email-shaped token matched whole from its own start); the composer may not
// offer to complete a token the transcript would refuse to link back.
function mentionBoundaryAt(value, at) {
  return at <= 0 || !MENTION_LOCAL_PART_RE.test(value[at - 1]);
}

function hasPathPrefix(token, start) {
  return token.startsWith("./") || token.startsWith("../") || token.startsWith("~/")
    || (start > 0 && token.startsWith("/"))
    || token.startsWith("@");
}

function tokenEnd(value, caret) {
  let end = caret;
  while (end < value.length && !/[ \t\n\r]/.test(value[end])) end += 1;
  return end;
}

function pathTokenAt(value, caret) {
  let start = caret;
  while (start > 0 && !/[ \t\n\r]/.test(value[start - 1])) start -= 1;
  const end = tokenEnd(value, caret);
  const token = value.slice(start, end);
  if (hasPathPrefix(token, start)) return { start, end, token };
  // A mention anchors at its own `@`, which need not be the whitespace
  // boundary: CJK prose has no spaces (`看看@src/x`). The last qualifying `@`
  // before the caret wins, so a later `@` always retargets the span.
  const at = value.lastIndexOf("@", caret - 1);
  if (at >= start && at < caret && mentionBoundaryAt(value, at)) {
    return { start: at, end, token: value.slice(at, end) };
  }
  return null;
}

// `@` is a mention marker, not path syntax: the request carries the token
// without it and the completion is re-prefixed, so the inserted text keeps the
// `@` the user typed.
function splitTrigger(typed) {
  return typed.startsWith("@") ? { prefix: "@", query: typed.slice(1) } : { prefix: "", query: typed };
}

function nameFragment(query) {
  return query.slice(query.lastIndexOf("/") + 1);
}

// This runtime owns the anchored completion session and network state only.
// The composer remains the sole writer of the shared picker DOM.
function createPathCompletionRuntime({ api, onChange, setTimeout: timerSet = (...args) => setTimeout(...args), clearTimeout: timerClear = (id) => clearTimeout(id), AbortController: AbortCtor = globalThis.AbortController, debounceMs = 120 }) {
  let anchor = null;
  let sessionId = null;
  let end = 0;
  let typed = "";
  let prefix = "";
  let token = "";
  let entries = [];
  let dir = "";
  let loading = false;
  let truncated = false;
  let timer = null;
  let abort = null;
  let seq = 0;

  function snapshot() {
    return { open: anchor !== null, start: anchor, end, token: typed, fragment: nameFragment(token), sessionId, entries: entries.slice(), dir, loading, truncated };
  }

  function cancelRequest() {
    seq += 1;
    if (timer !== null) timerClear(timer);
    timer = null;
    if (abort) abort.abort();
    abort = null;
  }

  function close() {
    cancelRequest();
    anchor = null;
    sessionId = null;
    end = 0;
    typed = "";
    prefix = "";
    token = "";
    entries = [];
    dir = "";
    loading = false;
    truncated = false;
  }

  function schedule() {
    cancelRequest();
    // Keep the previous listing visible while the replacement fetch is in
    // flight: completions stay valid (they replace the whole token), and
    // blanking the menu on every keystroke would flash "No matches".
    loading = true;
    const requestSeq = seq;
    const requestSession = sessionId;
    const requestToken = token;
    const requestPrefix = prefix;
    timer = timerSet(async () => {
      timer = null;
      const controller = typeof AbortCtor === "function" ? new AbortCtor() : null;
      abort = controller;
      let result = null;
      try {
        result = await api(`/api/sessions/${encodeURIComponent(requestSession)}/file/complete?path=${encodeURIComponent(requestToken)}&limit=50`, { signal: controller ? controller.signal : undefined });
      } catch (_) {
        // Unavailable directories and network failures are an empty listing.
      }
      if (requestSeq !== seq || sessionId !== requestSession || anchor === null) return;
      abort = null;
      const raw = result && result.ok === true && Array.isArray(result.entries) ? result.entries : [];
      entries = requestPrefix
        ? raw.map((entry) => (entry && typeof entry === "object"
          ? { ...entry, completion: `${requestPrefix}${String(entry.completion || "")}` }
          : entry))
        : raw;
      dir = result && result.ok === true ? String(result.dir || "") : "";
      truncated = Boolean(result && result.ok === true && result.truncated);
      loading = false;
      onChange(snapshot());
    }, debounceMs);
  }

  function sync(value, caret, selectedSessionId) {
    if (anchor !== null) {
      const active = value.slice(anchor, caret);
      // A whitespace at the caret boundary ends the token: interior spaces
      // stay legal (anchored span across accepted names containing spaces),
      // but a trailing space can only come from the user finishing the path.
      if (!selectedSessionId || selectedSessionId !== sessionId || caret < anchor
          || !hasPathPrefix(active, anchor) || /[\n\r]/.test(active)
          || /\s$/.test(active)) {
        close();
        return null;
      }
      end = tokenEnd(value, caret);
      if (active !== typed) {
        typed = active;
        const split = splitTrigger(typed);
        prefix = split.prefix;
        token = split.query;
        schedule();
      }
    } else {
      const span = selectedSessionId ? pathTokenAt(value, caret) : null;
      if (!span) return null;
      const active = value.slice(span.start, caret);
      if (!hasPathPrefix(active, span.start)) return null;
      anchor = span.start;
      sessionId = selectedSessionId;
      end = span.end;
      typed = active;
      const split = splitTrigger(typed);
      prefix = split.prefix;
      token = split.query;
      schedule();
    }
    return snapshot();
  }

  return Object.freeze({ sync, snapshot, close });
}

export { pathTokenAt, createPathCompletionRuntime };
