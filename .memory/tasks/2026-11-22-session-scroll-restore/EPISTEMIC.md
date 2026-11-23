# Epistemic model — session-switch scroll restore

## Phenomenon to explain

After selecting a session, the transcript viewport can land mid-history when the reader
expects the bottom, and can land at an arbitrary offset when the reader expects the
message they were reading. Images in recent messages make it much more likely.

## Measured outcome (OPS 11:05Z + 11:2xZ)

Whole app, two seeded Pi sessions, cold image-dimension cache:
- at-bottom return: gap 0 at 50 ms (image box 26.8 px, no source) -> gap **1204 px** at 2 s and 4 s
  (image box 1230.8 px), `#jumpBtn` hidden the whole time. Requirement 1 fails.
- mid-history return (anchor `byte-5764-user` at scrollTop 1837): snapped to the tail (5807).
  Requirement 2 fails.
- same mid-history return with the dimension cache warm: the pixel was restored exactly (1837, top row
  `byte-5764-user` at -37) -- but four frames into the same render the viewport was observed at the
  bottom and back at 1837 in alternate samples (6.9/25/40.8/57.4 ms), i.e. two pending writers fighting.
Isolated layout probe (real Chromium, real `app_markdown.js`, 300 ms dimension delay): restore-frame
height 10145 px -> 11799 px at 1.5 s (+1654); with `overflow-anchor: none` (what WebKit effectively does)
a mid-history view moves six messages (row 16 -> row 10); with UA anchoring the content is compensated.

So: the cache state alone decides whether the pixel branch is taken, and whether the view ends at the
bottom or above it is decided by which scheduled write lands after hydration. Both are timing, not
position.

## Mechanism (primary, supported by code and by the measurements above)

The remembered position is a pixel pair `{scrollTop, scrollHeight}` per session id
(`codoxear/static/app_transcript.js:989`). The restore
(`app_transcript.js:999-1016`) runs two animation frames after the new DOM is rendered
and restores the pixel **only if the whole scroller height is byte-identical**, else it
calls `scrollToBottom()`. Therefore:

- The two required behaviors are conflated into one heuristic. When heights match, a
  pixel restore runs (correct only for mid-history + unchanged content). When they do
  not, the view goes to the bottom (correct only for the at-bottom case). It cannot be
  right for both, and it is wrong for mid-history whenever anything changed: the return
  render is a 24-event tail page (`INIT_PAGE_LIMIT`, app_application_composition.js:188),
  older pages loaded while browsing are dropped, and the rendered window trims at 260
  rows.
- Content height is not final at the decision frame. For a local image whose dimensions
  are not in the localStorage cache, `rewriteMarkedImages`
  (`codoxear/static/app_markdown.js:406-424`) inserts the `img` with `src` removed and
  **no** box, and dimensions are applied later from an async fetch
  (`hydrateMarkedImage`, `app_markdown.js:359-372`). So the box is ~0 height at frames 1-2
  and full height afterwards. Consequently: (a) the equality test flips on fetch timing
  (hence "random"); (b) after a bottom restore, late hydration grows the content below
  the viewport, leaving the viewport above the real bottom while `autoScroll` is still
  true, so `syncJumpButton` hides `#jumpBtn` and `#chatTimeChip` — the UI asserts "at
  live tail" and offers no way back. Chrome/Firefox UA scroll anchoring masks part of
  this; iOS Safari has none.

## Contributing defects (same subsystem)

- No session/generation guard on the restore rAF chain (`app_transcript.js:1004`), so a
  pending restore can apply to a different session's DOM. The repo's own guarded pattern
  is `loadTranscriptWindowAtCursor` (`app_message_history.js:188`).
- Restore is called from four places in `openSession`
  (`app_session_lifecycle.js:208, 231, 277`), including immediately after a cached-tail
  render that the live tail render replaces milliseconds later.
- `scrollPositions` entries are never invalidated on transcript replacement (/new,
  resume renewal); `sessionScrollPosition` has no production caller.
- Multiple writers of `chat.scrollTop` (replaceWith's double schedule, rebuildDecorations,
  typing-row setters, appendEvent, composer autoGrow, iOS viewport controller, restore)
  with ordering determined by rAF registration order — no declared owner.
- Stickiness is an event, not an invariant: `scrollToBottom()` writes once; nothing
  re-asserts it when the content box grows later.
- Orphaned anchoring primitive: `firstUnreadEventId`/`scrollToFirstUnread`
  (`app_transcript.js:780-806`) has no caller, `preserveScroll` is never true,
  `trimRenderedRowsBeforeViewport` has no caller.

## Ruled out

- Reader-attribution logic (`noteUserScroll`/`readerOwnsScroll`, `handleScroll` delta<0
  && readerOwnsScroll && !pinnedToTail) — this is the phone-keyboard strobe fix and is
  sound; it is not the cause. It should be preserved by any redesign.
- Typing-row ordering: the row is always re-inserted before the bottom sentinel
  (`createTypingRowRuntime.setVisible`), so the "working bubble appears last" requirement
  fails only because the viewport is not at the bottom, never because of DOM order.

## Delivered (2026-11-22)

Implemented as the model described under "Current best claim" below: position = intent
(`tail`) or anchor (row id + history cursor + intra-row offset); one owner restored once per
render generation and guarded by sequence + session; a settle loop re-asserting on content-size
change and standing down for the reader, another writer, or a vanished row; the tail
re-asserted by a content-size pin in the scroll runtime; image boxes reserved at first paint;
the two tail affordances rendered from geometry. Verified in the isolated app: A returns to a
true bottom (was 1204px short), B returns to the remembered row at its exact offset (was
snapped to the tail). Details and numbers in OPS.md (14:0xZ).

Residual: the anchor settle depends on `ResizeObserver` (no settle in browsers without it);
an anchor outside the tail window now materializes a detached window (same semantics as
search navigation: no live appends, jump control visible); the ad-hoc bottom writers
(`updateActivityPeek`, typing-row setters, composer auto-grow) remain and are now redundant
with the pin — leaving them in place is deliberate (they are harmless and removing them is a
separate, separately verified cleanup).

## Current best claim (pre-implementation)

Neither requirement can be met by a pixel memory whose restore gate is a document-height
equality test. The unit of position must be the message row (id + history cursor +
intra-row offset) plus an explicit "at tail" state, and stickiness must be an invariant
maintained across content-size changes, with one owner invoked once per render
generation.

## Question that would most change the model

Whether the reported failure also occurs with zero images (i.e. whether mid-history
restore fails for a session that grew while unselected). The mechanism predicts yes;
a clean no would mean something else (e.g. the reading position is being lost before the
save) dominates.
