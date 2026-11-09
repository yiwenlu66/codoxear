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

// Resolve what Tab should do. Never throws, never depends on timing:
//   { apply: text }       -> write `text` into the field, caret at the end
//   { fallthrough: true } -> there is nothing to complete: close the list and
//                            let the caller move focus on
async function resolveTabCompletion({ value, defaultDir = "", rows = [], pickedIndex = -1, cached = null, fetchListing = null } = {}) {
  const text = String(value == null ? "" : value);
  const list = Array.isArray(rows) ? rows : [];
  const pickedRow = pickedIndex >= 0 && list[pickedIndex] ? list[pickedIndex] : null;
  const pickedPath = pickedRow && typeof pickedRow.path === "string" ? pickedRow.path : "";
  // A row the user picked with the arrow keys outranks everything else, but a
  // row that already equals the text is not a decision: fall through to the
  // listing so the field can drill into it instead.
  if (pickedPath && pickedPath !== text) return { apply: pickedPath };
  const applied = completionFromListing(text, defaultDir, cached);
  if (applied) return { apply: applied.value };
  if (!text) return { fallthrough: true };
  // The field's own listing is unknown for this text: ask. A listing that is
  // known and yielded nothing is final — never re-ask for it, because a second,
  // identical listing could only produce the same answer.
  if (!cached && typeof fetchListing === "function") {
    let listing = null;
    try {
      listing = await fetchListing(text);
    } catch (_) {
      listing = null;
    }
    const fetched = completionFromListing(text, defaultDir, listing);
    if (fetched) return { apply: fetched.value };
  }
  // One row left is an unambiguous answer even when it is not a prefix of what
  // was typed (a recent path, say), and a list with a single item that Tab
  // cannot take reads as broken. A "create this path" row is not an answer: it
  // is the text retyped, so taking it would only rewrite what the user wrote.
  const onlyRow = list.length === 1 ? list[0] : null;
  const onlyPath = onlyRow && !onlyRow.createNew && typeof onlyRow.path === "string" ? onlyRow.path : "";
  if (onlyPath && onlyPath !== text) return { apply: onlyPath };
  return { fallthrough: true };
}

const FOCUSABLE_SELECTOR = "a[href], area[href], button, input, select, textarea, iframe, [tabindex]";
const POPUP_SELECTOR = ".filePickerMenu, .dialogPickerMenu, #newSessionCwdMenu, [aria-hidden='true'], [inert]";

// The fall-through for a consumed Tab: move focus to the next control, the way
// the browser would have if the key had not been used for completion. Walking
// the document is deliberate — the two surfaces have different neighbours, and a
// list row must never be a landing spot.
function focusNextControl(from) {
  if (!from || typeof from.focus !== "function") return false;
  const doc = from.ownerDocument;
  if (!doc || typeof doc.querySelectorAll !== "function") return false;
  const nodes = Array.from(doc.querySelectorAll(FOCUSABLE_SELECTOR));
  const start = nodes.indexOf(from);
  if (start < 0) return false;
  for (let index = start + 1; index < nodes.length; index += 1) {
    const node = nodes[index];
    if (!node || typeof node.focus !== "function" || node.disabled) continue;
    if (typeof node.getAttribute === "function" && node.getAttribute("tabindex") === "-1") continue;
    if (typeof node.closest === "function" && node.closest(POPUP_SELECTOR)) continue;
    const rect = typeof node.getBoundingClientRect === "function" ? node.getBoundingClientRect() : null;
    if (rect && (rect.width <= 0 || rect.height <= 0)) continue;
    node.focus();
    return true;
  }
  return false;
}

export { pathTokenAt, completePathFragment, completionPlan, resolveTabCompletion, focusNextControl, createPathCompletionRuntime };
