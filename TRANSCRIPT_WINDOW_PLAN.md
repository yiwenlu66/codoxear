# Plan: one window model for the transcript (paging, tail attachment, and the working bubble)

Bug report: *"if i go back to a particular old message and then scroll to bottom, it
fails to scroll down to the actual 'last' message. The 'working' bubble follows
after an old message in the loaded page, not the latest message of this session."*

Status: **proposal only, nothing implemented.** Sections 1–2 are the diagnosis and
the evidence for it; 3–5 are the model and the change list; 6 lists the behaviour
changes a reviewer must accept; 7 records rejected alternatives; 8–11 are edge
cases, tests, milestones and the decisions I do not want to take alone.

> **Delta since this plan was written (commit 63bad03d).** The *viewport* authority
> below no longer lives entirely in `createTranscriptScrollRuntime`: the
> per-session reading position is now owned by `app_transcript_position.js` as
> either `{kind: "tail"}` or an anchor (`messageId` + `historyCursor` +
> intra-row offset), restored once per render generation and re-asserted on
> content-size change (see AGENTS.md, "A session's viewport position has one
> owner"). `createTranscriptScrollRuntime` keeps `autoScroll`/`isNearBottom`, the
> bottom pin, and the write/foreign-scroll counters. Read "scroll memory" in the
> table below as a pointer to that module; do not re-introduce a pixel memory
> inside the scroll runtime. The window/tail-attachment half of this plan is
> untouched by that change.

---

## 1. Diagnosis

The transcript renders a sliding **window** of the session log into the DOM. Two
different code paths install a window whose last row is *not* the log's last
event, and every surface that belongs to the log's end keeps asserting otherwise.

### 1.1 Path A — paging upward silently discards the tail

`prependOlderEvents` (`app_transcript.js:826-863`) inserts an older page above the
first message row and then bounds the DOM by removing rows from the **bottom**
(`domRuntime.trimRenderedRows({ fromTop: false, maxRows: historySlackRows })`,
`:852`). `CHAT_DOM_WINDOW` is 260 and the history slack is `+OLDER_PAGE_LIMIT`
(`app_application_composition.js:188-191`), so the ceiling is 320 rows; the
initial tail window is `INIT_PAGE_LIMIT = 24` rows and each older page is 60. The
fifth prepend crosses the ceiling and starts deleting the newest rows; the sixth
deletes what remains of the tail.

The only signal emitted by that deletion is `scrollRuntime.setRenderedAtLiveTail(Boolean(fromTop))`
(`app_transcript.js:932-938`) — i.e. "we trimmed from the bottom, so we are no
longer at the tail". This is reached by ordinary use, not an exotic case:

- scroll-top auto-load (`maybeAutoLoadOlder` → `loadOlderMessages({auto:true})`);
- the previous-user-message button: `materializeNeighbor` pages up to 20 pages of
  60 rows in a loop (`app_chat_navigation.js:246-268`), so a single click can walk
  straight past the 320-row ceiling.

After that, the DOM's last row is some old message; the log's end is not loaded at
all, and nothing in the client can fetch it by scrolling.

### 1.2 Path B — cursor windows are born detached

`loadTranscriptWindowAtCursor` (`app_message_history.js:180-200`) installs
`/messages/window?before=30&after=30` around a search hit or a cross-log
neighbour, replacing the DOM, and **discards the server's `has_newer`**
(`message_routes.py:1045-1068` returns it). It is used by chat search
(`app_chat_search.js:437`) and by `materializeNeighbor` for next/cross-log
targets. From that window there is no forward paging at all: the reader can only
page further *back*.

### 1.3 What the UI then claims

| Surface | Code | What it does while the tail is not in the DOM |
| --- | --- | --- |
| working / subagent bubble | `typingRowRuntime.setVisible` (`app_transcript.js:520-560`, `632-651`) driven by `createTypingRowStoreProjection` (`app_transcript_render.js:14-70`) | inserted before `bottomSentinel`, i.e. **directly after the last old row** |
| live deltas | `appendEvent` gate (`app_transcript.js:762-767`) | dropped — but the transport cursor still advances (`app_message_flow.js:479-482`), so they are not re-delivered |
| "scroll to bottom" | `scrollRuntime.scrollToBottom` (`app_transcript.js:1124-1140`) | scrolls to the bottom of the *rendered window*, i.e. a stale byte offset in the middle of history |
| `G` / jump button | `jumpToLatest` (`app_message_history.js:446-470`) | first calls `view.scrollToBottom({force:true})`, which calls `markLiveTail()` (`app_transcript_view.js:213-222`) and so **asserts** tail attachment before any content exists, then reloads via `openSession(forceRender)` |
| optimistic send echo | `sendText` (`app_message_flow.js:690-698`) → `appendEvent` with `pending:true` | `pending` bypasses the attachment gate (`:756`), so the echo is spliced onto the old page; its trim runs `fromTop: stick` and therefore **sets attachment back to true** (`:768-771`) |

`jumpToLatest` mostly works, which is why the bug reads as "scroll to bottom does
not work" rather than "jump does not work". But its claim-then-reload order is a
lie in the window between the two steps, and it becomes permanent whenever the
reload does not render: `openSession` aborts the previous tail request
(`prepareSessionOpen`), returns early on `ignoredStaleBound`
(`app_session_lifecycle.js:243-248`), or fails the fetch. In those cases the view
is left claiming "attached" with the jump button hidden
(`shouldStickToBottom()` → `app_transcript.js:1037-1039`), and every later live
event is spliced onto the old page — the state the reporter describes as
permanent.

### 1.4 Why the flag cannot be trusted: six writers, two meanings

`renderedAtLiveTail` is written by:

1. `renderTranscript` → `markLiveTail()` — *content*: a tail window was installed.
2. `renderDetachedTranscriptWindow` → `setRenderedAtLiveTail(false)` — *content*: a cursor window.
3. `trimRenderedRows` → `setRenderedAtLiveTail(Boolean(fromTop))` — *content*: which edge was trimmed.
4. `scrollRuntime.reset`, `replaceWith`, `replaceWithPlaceholder`, `scrollToBottom`
   → `markLiveTail()` — *intent*: "the reader is now at the bottom".
5. `appendEvent`'s `pending` path trimming from the top — *intent* leaking into a content fact.
6. `prependOlderEvents` reacting to (3) — reading the fact, then disabling auto-scroll.

A content invariant ("does the rendered window reach the log's end") is being
stored as mutable placement state and written by scroll code. That is the
architectural defect; every symptom above is downstream of it.

---

## 2. Evidence (behavioural, real modules)

I drove the real `app_transcript.js`, `app_transcript_view.js` and
`app_transcript_render.js` in a Node VM with a stub DOM (the same technique as
`tests/test_chat_transcript_runtime.py`) and reproduced the state transition
table. Rows are named `tail-*` (initial tail window), `old-*` (prepended pages),
`live-*` (live deltas).

| step | action | observed |
| --- | --- | --- |
| 1 | install tail window (24 rows) | 24 rows, `tail-23` last, `renderedAtLiveTail = true` |
| 2 | six older pages of 60 (`prependEvents`) | 320 rows, last row `old-119`, `tail-*` gone, `renderedAtLiveTail = false`, jump button shown |
| 3 | live delta `live-1` | `appendEvents` returns false, no row — dropped (and marked seen by `:764`) |
| 4 | `sessionState.set("running", true)` | `<WORKING BUBBLE>` inserted **immediately after `old-119`** — the reported symptom |
| 5 | scroll to bottom of the window | viewport rests on `old-119`; jump button still visible; nothing fetches the log's end |
| 6 | `view.scrollToBottom({force:true})` (what `jumpToLatest` calls first) | returns true, `renderedAtLiveTail = true`, bottom row still `old-119`, **jump button now hidden** |
| 7 | live delta `live-2` after step 6 | appended after `old-119` — the whole live turn now renders on the old page |

Steps 4–7 reproduce both halves of the report exactly. Step 6 is the
stranding mechanism: the assertion survives if the reload that follows it does not
render.

Limits of this evidence: stub DOM (no layout engine), so I verified row identity
and ordering, not pixels; and I did not run the Docker end-to-end script for the
real browser path. Both belong to the verification plan (§9).

---

## 3. The model

Three authorities, each with one job:

| authority | owns | module |
| --- | --- | --- |
| **window** | which contiguous range of the log is rendered, and whether its last row is the log's last event | `app_transcript_view.js` (policy), fed by facts returned from `app_transcript.js` |
| **viewport** | where the reader is: `autoScroll`, `isNearBottom`; the remembered position (tail intent or row anchor) | `createTranscriptScrollRuntime` + `app_transcript_position.js` |
| **agent** | running / turn open / tokens / subagents | `sessionState` (`app_session_state.js`) |

### 3.1 Window descriptor

```
window = {
  olderCursor: string|null,   // anchor for the page before the first rendered row
  hasOlder: boolean,
  tailAttached: boolean,      // the last rendered row IS the log's last event
}
```

`tailAttached` is a *content* fact. It is written only where window content
changes, and only from fetch payloads or trim results:

| event | `tailAttached` |
| --- | --- |
| `installTail(events)` (tail fetch) | `true` |
| `installWindowAt(cursor)` (`/messages/window`) | `!data.has_newer` |
| `extendOlder(events)`, no rows removed from the bottom | unchanged |
| `extendOlder(events)`, rows removed from the bottom | `false` |
| `extendNewer(events)`, `data.has_newer === false` | `true` |
| `extendNewer(events)`, `data.has_newer === true` | unchanged (`false`) |
| live `appendEvent`, trim from top (sticking) | unchanged |
| live `appendEvent`, trim from bottom (not sticking) | `false` |
| `clear` / placeholder | `true` (empty window; nothing to be behind) |

`createTranscriptScrollRuntime` loses `renderedAtLiveTail`, `markLiveTail`,
`markDetachedWindow` and `setRenderedAtLiveTail` entirely, and receives
`isWindowTailAttached()` instead — the same lazy-closure pattern already used for
`shouldCancelOlderLoad` (`app_transcript_render.js:536`). `shouldStickToBottom()`
becomes `tailAttached && (autoScroll || isNearBottom())`.

### 3.2 The invariant

> **Rows that belong to the log's end render only in a window whose last row is
> the log's end. "Go to the latest message" is an operation that installs such a
> window; it is never a scroll, and it never asserts attachment it does not have.**

Three consequences, and they are the whole fix:

1. The working bubble, the subagent bubble, the activity peek and the optimistic
   prompt echo are *tail presentations*: `running && tailAttached`. While the
   reader is behind, they are not in the transcript, so nothing can appear after
   an old message.
2. Live deltas are appended only when `tailAttached`, and their drops no longer
   masquerade as "seen" (`:764` currently marks a dropped event as rendered).
3. There is one operation for "be at the end" — `attachTail()` — used by the jump
   button, `G`, and the send path. It fetches, installs, rebases the live cursor,
   then scrolls. It cannot fail silently into a false state, because the state is
   derived from what was installed.

### 3.3 While detached, the reading position is honest

- The jump control stays visible (`!shouldStickToBottom()`) and now also carries
  the agent's state — `data-running` on `#jumpBtn`, title "Jump to latest message
  (agent working)". This follows the existing principle that a count rides the
  control that acts on it (queue badge, not a topbar status chip); the transcript
  is not the place to say "the agent is working over there", and the sidebar card
  badge already reports busy independently of what the transcript is showing.
- Reaching the bottom edge of a detached window **pages forward** rather than
  jumping (§4), so downward scrolling is continuous and terminates by itself at
  the log's end. The jump button remains the instant path.

---

## 4. Paging protocol

Four primitives, each mapping a fetch to window facts:

| primitive | request | facts set |
| --- | --- | --- |
| `installTail({scroll})` | `GET /messages/tail?limit=INIT_PAGE_LIMIT` | replace window; `tailAttached = true`; `olderCursor`/`hasOlder` from `history_cursor`/`has_older`; live cursor from `live_cursor` |
| `installWindowAt(cursor)` | `GET /messages/window?cursor&before=30&after=30` | replace window; `tailAttached = !has_newer`; older edge from `history_cursor`/`has_older`; if `!has_newer`, rebase live cursor |
| `extendOlder()` | `GET /messages/history?cursor=<oldest rendered row>&limit=OLDER_PAGE_LIMIT` | prepend (existing `prependOlderEvents`); `tailAttached &&= !trimmedFromBottom`; older edge advances |
| `extendNewer()` | `GET /messages/window?cursor=<last rendered row>&before=0&after=OLDER_PAGE_LIMIT` (or a new `GET /messages/newer`, see §5.1) | append rows not already rendered (dedupe by `data-message-id`, exactly as `prependOlderEvents:826-846`); trim from `trimRowsBeforeViewport`; `tailAttached = !has_newer`; if attached, rebase live cursor |

Trim direction follows the reader: paging up trims the newest rows (they are far
below the viewport), paging down trims rows entirely above the viewport
(`domRuntime.trimRowsBeforeViewport` already exists, `app_transcript.js:940-946`).
Therefore detachment on upward paging is *correct* — the view genuinely no longer
shows the log's end — and the model's job is only to say so and to keep the way
back working.

`app_chat_navigation.window` (`:246-268`) becomes simpler too: a previous-target
prepend may now run without the 20-page cap turning into a detached window, and a
next-target may be reached by forward paging from the current window.

### 4.1 The one server gap: the right-edge transport cursor

`handle_messages_window` (`message_routes.py:1015-1068`) already reports
`has_newer`, but not the byte position of the window's right edge. Every fetch
that *establishes* attachment must also rebase the live transport, otherwise
events written during the fetch's round trip are never delivered (the transport
cursor may already be past them, and the window was read before them). Two
acceptable shapes:

- **(a) recommended** — add `live_cursor` to the window response when
  `has_newer === false`: `encode_message_cursor(kind="live", pos=size)`.
  `read_chat_window_around` already stats the file (`transcript_search.py:453`),
  so returning `size` is free.
- **(b)** add a symmetric `GET /messages/newer?cursor&limit` that returns
  `{events, has_newer, live_cursor?}` and leave `window` a pure jump primitive.

Rebasing *backward* is safe and is the desired behaviour: rows already rendered
are in the seen ring (`renderTranscript` marks its rows seen, `app_transcript.js:791-800`)
and are dropped on re-delivery; rows that were dropped while detached were never
rendered, so they arrive in the right order after the installed window. This
requires the accompanying rule from §3.2(2): **mark seen exactly the rows that
were rendered** — extend that to page inserts (`prependOlderEvents` currently
passes `markSeen:false`) and remove it from the drop path.

---

## 5. Change list

### 5.1 Server (`message_routes.py`, `transcript_search.py`)

1. `read_chat_window_around` returns the log size (or `handle_messages_window`
   re-stats) so the route can mint a right-edge `live_cursor`.
2. `handle_messages_window` adds `"live_cursor": ...` when `!has_newer`.
   (Variant (b) instead adds `handle_messages_newer`.)

No change to `tail`, `history`, `live`, or `neighbor`.

### 5.2 `app_transcript.js`

- `createTranscriptRenderRuntime`: mutators return the facts the policy needs —
  `{ changed, trimmedFromBottom }` from `appendEvent`, `prependOlderEvents`,
  `extendNewer`; `renderTranscript`/`renderDetachedTranscriptWindow` are replaced
  by one `installWindow(events, { markSeen })`. `renderTranscript`'s
  `firstUnreadEventId` option has no caller and should not survive the merge.
- `createTranscriptDomRuntime.trimRenderedRows` returns `{ removedLastRow }`
  instead of writing `setRenderedAtLiveTail`.
- `createTranscriptScrollRuntime`: drop `renderedAtLiveTail` (and its 3 setters);
  add `isWindowTailAttached`; `shouldStickToBottom` combines it with the reader's
  position; `syncJumpButton` also renders `agentRunning` (§3.3).
- `createTypingRowRuntime`: no logic change; it is gated by its caller. Its
  `anchor()` remains the append point.
- Keep `createOlderLoadRuntime` but generalize it to `createPageLoadRuntime` so
  the newer lane gets the same currentness / cooldown / abort discipline
  (one instance per direction; only the older lane has the button/UI hooks).

### 5.3 `app_transcript_view.js` (the policy boundary)

- `window` descriptor + `tailAttached` as the only writer (§3.1).
- One `installWindow` path replacing `replaceWith` / `replaceWithPlaceholder` /
  detached rendering; `attachTail` is a caller-provided async operation
  (`installTail` content comes from `app_message_history`, the controller that
  already owns fetching).
- `extendNewer` mirroring `prependEvents`, including the `queuedLiveEvents`
  discipline (a delta arriving mid-paging is queued, not spliced).
- `scrollToBottom` keeps only the geometric meaning. Delete the `force/events`
  branch and the `markLiveTail` claim.
- `state()` returns `{ state, scrollTop, tailAttached, hasOlder, hasNewer }`
  (`renderedAtLiveTail` is renamed; ~12 test fixtures reference it).
- Retire `VIEW_STATES.LIVE/BROWSING` as attachment signals: the machine becomes
  `window` facts plus `INSTALLING | LOADING_OLDER | LOADING_NEWER`. Nothing else
  reads `state().state` (`app_message_history.js:269-273`'s `maybeAutoLoadOlder`
  is exported but unused), so this removes machinery rather than adding it.

### 5.4 `app_message_history.js`

- `loadTranscriptWindowAtCursor` → `installWindowAt` (pass `has_newer` through).
- `loadOlderMessages` unchanged except that it reports `trimmedFromBottom`.
- New `loadNewerMessages({auto})`, mirroring the older lane, plus a bottom-edge
  trigger in the scroll runtime (`isNearBottom() && delta > 0`, cooldown
  `OLDER_AUTO_COOLDOWN_MS`).
- New `attachTail()`: identity check (thread/log unchanged, else
  `openSession(useCache:false)`), tail fetch, `applySessionRuntimeFromTail`,
  install, rebase cursor, enable auto-scroll, scroll to bottom. It replaces the
  body of `jumpToLatest` (which keeps only the error toast and `kickPoll(0)`).

### 5.5 `app_message_flow.js`

- Delete the dead `renderedAtLiveTail` dependency (`:109`; nothing calls it).
- `applyLiveMessageData`: unchanged except it no longer decides anything about
  attachment; the view controller does.
- `sendText`: when the window is not attached, `await attachTail()` before the
  optimistic echo (`:690-698`). If the attach fails, still `addPendingUser` (so
  the row reappears with the next tail render) but do not splice the row into a
  detached window; toast "sent — latest messages could not be loaded".

### 5.6 `app_transcript_render.js` / `app_wiring.js`

- `createTypingRowStoreProjection({ sessionState, typingRowRuntime, isWindowTailAttached })`
  → `setVisible(running && isWindowTailAttached())`, `setSubagentVisible(!running && ...)`.
- The projection needs a resync trigger when attachment changes: rename
  `afterReplace` to `afterWindowChange` and call it on install *and* on any trim
  that detaches (`app_transcript_render.js:813` wiring is one line).
- Wiring: add `isWindowTailAttached`/`agentRunning` to `createTypingRowOptions`
  and `createTranscriptScrollOptions`; drop `renderedAtLiveTail` from
  `createMessageFlowOptions`; rename it to `tailAttached` in
  `createChatNavigationOptions`. `scripts/check_wiring.py` and
  `scripts/wiring_guard_allowlist.json` need the matching updates (the allowlist
  already carries one `app_message_flow.js` entry).
- `jumpBtn.onclick` (`app_application_composition.js:1219`) → `attachTail()`;
  the `G` shortcut (`app_transcript_render.js:303-318`) → the same call.

---

## 6. Behaviour changes that a reviewer must accept

1. **The working bubble disappears while the reader is paging through history.**
   The work signal moves to the jump control (`data-running`). Alternative if
   rejected: keep a compact chip pinned outside the scroller — a new surface, and
   a second place that has to know about attachment.
2. **Scrolling down in a detached window pages forward continuously** (60 rows per
   fetch, 450 ms cooldown) instead of jumping. The jump button remains instant.
3. **Sending from history attaches the tail first** (one extra fetch, ~ms). This
   is what the existing comment at `app_message_flow.js:691-692` already claims to
   do ("Just scroll to bottom so the user sees the new message").
4. **`G` / jump button no longer claim attachment before fetching**, so a failed
   jump leaves the jump button visible instead of hiding it.

---

## 7. Rejected alternatives

- **Move the bubble to just after the last *user* message.** The tail is not in
  the DOM, so there is no row that means "here is where the agent is working".
  It would also break the two-lane projection rule that the bubble is re-projected
  from `sessionState` after any DOM rebuild.
- **Gate the bubble on `isNearBottom()`.** Wrong predicate: the reader can be
  pinned at the bottom of a stale window — steps 4–5 above are exactly that case.
- **Keep the tail rows during upward paging** (trim only above the viewport).
  Then the rows the reader is about to read get deleted instead, and the DOM can
  no longer be bounded, because a contiguous window that always contains the tail
  grows without limit. Paging cannot keep both ends for free.
- **Auto-jump to the tail when the reader reaches the bottom edge.** Fixes the
  symptom with a discontinuity: it skips every message between the window's edge
  and the log's end and destroys the reader's position. Forward paging costs the
  same one fetch and keeps the transcript a continuum.
- **Raise `CHAT_DOM_WINDOW` / `CHAT_DOM_WINDOW_WITH_HISTORY_SLACK`.** Delays
  detachment by a few pages; the defect is the missing edge, not the ceiling.
- **Make `scrollToBottom` scroll harder / re-issue `scrollTop = scrollHeight`.**
  There is nothing below the rendered window to scroll to; the scroll container is
  already at its maximum.
- **Keep `renderedAtLiveTail` and add an `attachedByContent` flag next to it.**
  Two flags for one fact is the current defect with a second name.

---

## 8. Edge cases the implementation must cover

- **Transcript identity change mid-page** (Pi `/new`, resume, rotation): a paging
  fetch returns `409 transcript_pending` or a cursor that no longer decodes; the
  policy must route to `openSession(useCache:false)` (the older lane already does,
  `app_message_history.js:246-249`) and reset both lanes' state.
- **Session switch while paging**: both lanes' in-flight requests must abort
  (the existing `requestId`/`AbortController` currentness check) and the window
  descriptor must reset with the transcript (`resetChatRenderState`).
- **Page-boundary projection parity**: a forward page must not drop or duplicate a
  row that the tail projection would show (adjacent assistant dedupe is
  stream-local per page, `transcript_search.py:299-360`). Needs an explicit test
  (§9.2) rather than an assumption.
- **Day separators** are rebuilt by `rebuildDecorations` after trims; verify no
  separator survives for a removed day.
- **Chat search while detached**: `applyChatSearchMarks` / `refreshLoaded` operate
  on rendered rows only; selecting a match installs a window at the target — the
  attachment facts of that window must come from `has_newer`, not from "detached".
- **Copy conversation / loaded user navigation**: `loadedUserJumpTarget` and
  `loadedCopyJumpTarget` are window-local; the next-button modal mode must key on
  `tailAttached` (the truthful fact) and keep its click-time fallback.
- **iOS keyboard scroll work**: `app_ios_viewport.js` calls
  `transcriptScrollRuntime.scrollToBottom()` — keep that geometric and never let
  it mean "attach".
- **Notifications / activity peek / subagent liveness**: unaffected in wiring, but
  the peek lives inside the bubble and therefore inherits the gate.
- **Pending-echo recovery**: `restorePendingUserRowsForSession` appends through
  `appendEvent`; while detached it must not splice (the row is restored with the
  next tail install).

---

## 9. Tests

### 9.1 Unit / VM (behaviour, no source strings)

1. **Detachment fact**: install tail (24), prepend 6×60, assert `tailAttached`
   false, `hasOlder` true and the last rendered row is not the log end. (This is
   the harness from §2 rewritten as a test.)
2. **Tail-owned presentation**: with `running` true and `tailAttached` false, the
   typing row is absent and the jump control carries `data-running`; flipping the
   key through `extendNewer` to `has_newer:false` makes the row appear at the
   bottom (after the newest row, not after an old one).
3. **No splice**: a live delta while detached renders no row and does not mark the
   event seen; a later forward page renders it.
4. **Attach is an operation**: `attachTail()` installs the tail, rebases the live
   cursor from the payload, and `state().tailAttached` is true only after install;
   a failed fetch leaves `tailAttached` false and the jump control visible.
5. **Send from history**: echo appears only after the attach, and the subsequent
   live delta lands after the echo.
6. **Forward paging continuity**: page forward from a mid-log window until
   `has_newer === false`; the concatenation of pages equals the projection of the
   same byte range read in one window (page-boundary parity, §8).
7. **View-controller contract**: update `tests/test_transcript_view_controller.py`
   for the new `state()` shape and the removal of the LIVE/BROWSING attachment
   signal.

### 9.2 Integration (server)

`handle_messages_window` returns `live_cursor` iff `!has_newer`, and it decodes as
a `live` cursor at the right edge (extend `tests/test_frontend_message_flow.py`,
`tests/test_chat_scrollback_source.py` style server tests).

### 9.3 Docker end-to-end (the reported scenario, real browser)

New `scripts/docker_transcript_paging_verify.sh`: a scripted session with ≥400
messages; open it, page up (scroll and the previous-user button) until the tail is
detached; assert (a) the working bubble is not rendered after an old row while the
agent runs, (b) scrolling to the bottom pages forward and the last rendered row is
the log's last message, (c) `G` and the jump button both land on the last message,
(d) sending while detached puts the prompt at the end. This is the acceptance
boundary; the VM tests only prove the mechanics.

---

## 10. Milestones

| # | scope | why first | risk |
| --- | --- | --- | --- |
| M0 | the failing behavioural test from §9.1(1)+(2) | locks the invariant before code moves; fails today | none |
| M1 | `tailAttached` as a derived window fact; scroll runtime loses the flag; `state()` rename; projection gate; jump control carries `data-running` | fixes the reported bubble placement and the false-claim stranding with no new fetch | medium: touches ~12 test fixtures and 4 wiring contracts |
| M2 | `attachTail()` + `G`/jump button/send path; delete `scrollToBottom`'s force branch | makes "go to latest" one honest operation; send stops splicing | low |
| M3 | forward paging (`extendNewer`, bottom-edge trigger, generalized paging runtime) + server `live_cursor` on `window` | makes downward scrolling terminate at the log's end without a jump | medium: new fetch path, page-boundary parity |
| M4 | Docker end-to-end script; AGENTS.md contract paragraph | acceptance evidence | low |

M1 alone removes both halves of the reported symptom for readers who use the jump
button; M1+M3 removes it for readers who only scroll.

---

## 11. Decisions I do not want to take alone

1. **Work signal while browsing history**: jump-control marker (my recommendation)
   or a new pinned chip, or accept that the transcript simply shows nothing while
   the agent works.
2. **Forward paging vs. jump-at-the-bottom-edge** (§7): continuous paging is more
   code but keeps the reader's place; a jump is one fetch and no new trigger.
3. **Do we want an "N new messages" count while detached?** It needs a counter
   fed by the dropped deltas; the jump control is the natural carrier, but it is
   extra state I would rather not add unless asked.
4. **Rename `renderedAtLiveTail` → `tailAttached`**: honest, but touches every
   fixture that stubs the scroll runtime. If that churn is unwelcome in one
   review, M1 can keep the old name on `state()` while moving the writer.
