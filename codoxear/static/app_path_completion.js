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

// ---- Path *field* completion (New Session cwd, file viewer path) ----
//
// A path field holds one path, not prose: the whole value is the token. Tab
// completes the fragment after the last `/` against the listing of that
// directory. This is deliberately *not* the composer's token model above
// (which rewrites a span inside a message and has an `@` mention prefix).

function commonPrefix(names) {
  if (!names.length) return "";
  let prefix = names[0];
  for (const name of names.slice(1)) {
    let index = 0;
    while (index < prefix.length && index < name.length && prefix[index] === name[index]) index += 1;
    prefix = prefix.slice(0, index);
    if (!prefix) return "";
  }
  return prefix;
}

// The completion of one path fragment, given the names in its directory.
// `truncated` means the listing was capped: a longest common prefix over a
// *subset* can be longer than the real one, so a partial fragment is never
// inserted then — only a complete name the caller actually received.
function completePathFragment(fragment, names, { truncated = false } = {}) {
  const typed = String(fragment || "");
  const available = (Array.isArray(names) ? names : []).filter((name) => typeof name === "string" && name !== "");
  if (!typed || !available.length) return null;
  // Exact case wins; a case-insensitive fallback repairs the case the user typed
  // ("App" -> "app_file.js") without ever inserting a name whose case is wrong.
  let matches = available.filter((name) => name.startsWith(typed));
  if (!matches.length) {
    const folded = typed.toLowerCase();
    matches = available.filter((name) => name.toLowerCase().startsWith(folded));
  }
  if (!matches.length) return null;
  if (matches.length === 1) return { insert: matches[0], unique: true };
  const shared = commonPrefix(matches);
  if (shared.length <= typed.length) return null;
  if (truncated && !matches.includes(shared)) return null;
  return { insert: shared, unique: false };
}

// Map a field value + its directory listing to the replacement value Tab should
// write, or null when nothing can be completed.
//
// `defaultDir` is the directory a slashless value belongs to: the file picker
// resolves a bare fragment against the session cwd (`""`), the New Session cwd
// field keeps its long-standing reading of a bare fragment as a name under `/`.
function completionPlan({ value, defaultDir = "", names = [], truncated = false } = {}) {
  const text = String(value == null ? "" : value);
  if (!text) return null;
  const slash = text.lastIndexOf("/");
  const dirPart = slash < 0 ? String(defaultDir || "") : text.slice(0, slash + 1);
  const fragment = slash < 0 ? text : text.slice(slash + 1);
  const entries = (Array.isArray(names) ? names : []).filter((entry) => entry && typeof entry.name === "string" && entry.name !== "");
  const completed = completePathFragment(fragment, entries.map((entry) => entry.name), { truncated });
  if (!completed) return null;
  const hit = entries.find((entry) => entry.name === completed.insert);
  const next = dirPart + completed.insert + (hit && hit.isDir ? "/" : "");
  if (next === text) return null;
  return { value: next, caret: next.length, unique: completed.unique };
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

// ---- Tab in a path field: the decision, and its fall-through ----
//
// Typing and pressing Tab immediately is the normal gesture, so the answer is
// usually not on the wire yet when the key arrives. A key event's default action
// cannot be cancelled after a network round trip (the browser runs it before the
// first macrotask), which is why the field consumes Tab unconditionally and
// decides afterwards: from the listing it already has, or from a request it
// starts now. Only the latency of the effect varies, never the outcome — the
// earlier design made the *decision* depend on whether a fetch had landed, which
// reads to a user as "Tab sometimes does nothing at all".

function completionFromListing(value, defaultDir, listing) {
  if (!listing || !Array.isArray(listing.names)) return null;
  return completionPlan({ value, defaultDir, names: listing.names, truncated: Boolean(listing.truncated) });
}

// A listing belongs to exactly one value. Anything else is another directory's
// answer (a superseded request, an aborted prefetch) and must never be combined
// with this value's text — that is how a name from the wrong directory gets
// inserted.
function listingFor(listing, text) {
  if (!listing || !Array.isArray(listing.names) || listing.query !== text) return null;
  // A failed request is not "this directory is empty": caching it as an empty
  // listing would make the next Tab skip the request and find nothing forever.
  if (listing.ok === false) return null;
  return listing;
}

// An empty fragment (the value ends with "/") leaves nothing to complete by
// prefix, but a directory holding exactly one entry is still unambiguous.
function singleEntryPlan(text, defaultDir, listing) {
  // An empty field has nothing to complete: Tab there belongs to focus movement.
  if (!text) return null;
  const slash = text.lastIndexOf("/");
  const fragment = slash < 0 ? text : text.slice(slash + 1);
  if (fragment || listing.names.length !== 1) return null;
  const only = listing.names[0];
  const name = only && typeof only.name === "string" ? only.name : "";
  if (!name) return null;
  const dirPart = slash < 0 ? String(defaultDir || "") : text.slice(0, slash + 1);
  const next = `${dirPart}${name}${only.isDir ? "/" : ""}`;
  return next === text ? null : { value: next };
}

function planFromRow(text, firstRow) {
  const row = typeof firstRow === "function" ? firstRow(text) : firstRow;
  const path = row && typeof row.path === "string" ? row.path : "";
  // `fromRow` tells the caller the value names something the list offered (a
  // directory, for both fields that have rows), not a fragment extension.
  return path && path !== text ? { apply: path, fromRow: true } : { fallthrough: true };
}

function planFromListing(text, defaultDir, listing, firstRow) {
  const applied = completionFromListing(text, defaultDir, listing) || singleEntryPlan(text, defaultDir, listing);
  return applied ? { apply: applied.value } : planFromRow(text, firstRow);
}

// Resolve what Tab should do. **A visible list makes Tab a take key, not a focus
// key**: while rows are on screen there is an answer, so Tab consumes the answer
// and never moves focus; focus moves only when nothing is listed to take. That is
// the difference between "Tab does nothing while I type" and a completion a user
// can actually use.
//
// Every input is scoped to THIS text by the caller and re-validated here, so the
// answer cannot depend on when anything arrived:
//   picked    - the row the user explicitly arrow-keyed onto
//   listing   - the directory listing for this text (cached or fetched)
//   firstRow  - the top row currently listed for this text (synchronous, keyed to
//               it); null when no list is showing, so an empty list falls through
//   { apply: text, fromRow? } | { fallthrough: true }
async function resolveTabCompletion({ value, defaultDir = "", picked = null, cachedListing = null, firstRow = null, fetchListing = null } = {}) {
  const text = String(value == null ? "" : value);
  const pickedPath = picked && typeof picked.path === "string" ? picked.path : "";
  if (pickedPath && pickedPath !== text) return { apply: pickedPath, fromRow: true };
  // An empty field has nothing to complete, whether or not a listing for "" is
  // cached — that check has to come first or the answer depends on the cache.
  if (!text) return { fallthrough: true };
  const settled = listingFor(cachedListing, text);
  if (settled) return planFromListing(text, defaultDir, settled, firstRow);
  if (typeof fetchListing === "function") {
    let fetched = null;
    try {
      fetched = await fetchListing(text);
    } catch (_) {
      fetched = null;
    }
    const listing = listingFor(fetched, text);
    if (listing) return planFromListing(text, defaultDir, listing, firstRow);
  }
  return planFromRow(text, firstRow);
}

const FOCUSABLE_SELECTOR = "a[href], area[href], button, input, select, textarea, iframe, [tabindex]";
const POPUP_SELECTOR = ".filePickerMenu, .dialogPickerMenu, #newSessionCwdMenu, [inert], [aria-hidden='true']";

function isFocusTarget(node) {
  if (!node || typeof node.focus !== "function") return false;
  if (node.disabled) return false;
  if (typeof node.matches === "function" && node.matches(":disabled")) return false;
  if (typeof node.getAttribute === "function") {
    const tabindex = node.getAttribute("tabindex");
    if (tabindex !== null && Number(tabindex) < 0) return false;
  }
  if (typeof node.closest === "function" && node.closest(POPUP_SELECTOR)) return false;
  if (typeof node.checkVisibility === "function" && !node.checkVisibility()) return false;
  const rect = typeof node.getBoundingClientRect === "function" ? node.getBoundingClientRect() : null;
  return !rect || (rect.width > 0 && rect.height > 0);
}

// The fall-through for a consumed Tab: move focus on, the way the browser would
// have if the key had not been spent on completion. `container` bounds the
// search (a modal dialog, the viewer panel) and the walk wraps inside it, so a
// control that sits *before* the field is still reachable; the move is verified,
// so an ineffective focus (visibility:hidden, disabled ancestor) falls through
// to the next candidate instead of reporting success and leaving focus behind.
function focusNextControl(from, { container = null } = {}) {
  if (!from || typeof from.focus !== "function") return false;
  const doc = from.ownerDocument;
  if (!doc || typeof doc.querySelectorAll !== "function") return false;
  const scope = container && typeof container.querySelectorAll === "function" ? container : doc;
  const nodes = Array.from(scope.querySelectorAll(FOCUSABLE_SELECTOR)).filter(isFocusTarget);
  if (!nodes.length) return false;
  const start = nodes.indexOf(from);
  const ordered = start < 0 ? nodes : nodes.slice(start + 1).concat(nodes.slice(0, start));
  for (const node of ordered) {
    if (node === from) continue;
    try {
      node.focus();
    } catch (_) {
      continue;
    }
    if (doc.activeElement === node) return true;
  }
  return false;
}

export {
  pathTokenAt,
  completePathFragment,
  completionPlan,
  resolveTabCompletion,
  focusNextControl,
  createPathCompletionRuntime,
};
