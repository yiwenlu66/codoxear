# EPISTEMIC

## Phenomenon
`#chatTimeChip` (grey `Today (YYYY-MM-DD) · HH:MM` band directly under the topbar,
screenshot /tmp/tip-probe/top-strip.png) appears/disappears while the user types on a
phone and settles hidden when typing stops.

## Identity — settled
Only time readout above the transcript; single writer
`syncVisibleTimeIndicator` inside `createTranscriptScrollRuntime`
(`codoxear/static/app_transcript.js`), reached only via `syncJumpButton()`,
transcript reset, chat-search open/close, or `handleComposerFocus`. No timer.
Visibility = `selected && !searchOpen && !shouldStickToBottom() && firstVisibleRowHasTs`
where `shouldStickToBottom() = renderedAtLiveTail && (autoScroll || isNearBottom())`
and `isNearBottom()` = `scrollHeight - (scrollTop + clientHeight) <= 80`
(`bottomThresholdPx`). Consequence: `#jumpBtn` shares the predicate and must flip with it.

## Live mechanisms
1. **Intent latch built from raw scroll deltas** (leading).
   `handleScroll`: `delta = scrollTop - lastScrollTop; if (delta < 0) autoScroll = false;`
   `scrollToBottom()`/`setScrollTop()` write `chat.scrollTop` and then record
   `lastScrollTop` from the *read-back*. On iOS a `scrollTop` write during/adjacent to a
   momentum, rubber-band or keyboard-driven adjustment can be ignored or overridden, so
   `lastScrollTop` can hold a position the scroller never settled at; the next scroll
   event then carries a negative delta that no gesture produced. `autoScroll` is a
   latch with no provenance and no hysteresis, and `autoScroll === false` is the only
   state in which the chip can be visible while `renderedAtLiveTail` is true.
   Typing reaches this once per keystroke: `autoGrow` -> `onAutoGrow` ->
   `scheduleScrollToBottom()`, and on iOS `onViewportShift` -> `scheduleScrollToBottom()`.
2. **`--appH` / `--vvTop` compensation race.** `.app` is `position: fixed; top: var(--vvTop);
   height: var(--appH)`; `updateAppHeightVar()` writes both from `visualViewport`. If the
   write lands a frame late relative to the platform's caret pan, the top strip (topbar
   padding absorbs the shift, the 44px chip band does not) slides off the top edge.
   Predicts the transcript text visibly sliding tens of px per keystroke, and the topbar
   buttons jittering.

## Ruled out (measured, real Chromium + iPhone metrics, Docker, 24-row Pi transcript)
- Typing at the live tail never shows the chip, including composer growth, viewport
  shrink 852->500, and an iOS-faithful fake `visualViewport` (layout viewport fixed,
  visual height/offsetTop changed, app compensation running).
- Pure geometry change (`--appH` 852->500 with no event) repaints nothing: a chip
  transition requires a `.chat` scroll event or a render.
- Chip is a pure function of detachment: appears at >80px off the tail, hides at the tail
  (8px-step sweep). All rendered rows carry `dataset.ts`, so the "no ts" branch is inert.
- `handleScroll` delta bookkeeping is consistent under Chromium: no phantom deltas, so
  mechanism 1 needs a platform that overrides scroll writes — hence mobile-only.

## Open question (one device observation)
Does `#jumpBtn` blink in lockstep (mechanism 1) or does the transcript slide (mechanism 2)?

## Implemented (2026-11-13)
Root cause as implemented: `autoScroll` — the one authority behind
`shouldStickToBottom()` and therefore both `#chatTimeChip` and `#jumpBtn` — was
cleared by any scroll event with a negative `scrollTop` delta. A phone soft
keyboard moves this container with nobody scrolling it (browser clamp when the
composer's auto-grow or the keyboard changes chat height; the platform's
caret-reveal pan settling), so typing re-derived "the reader left the tail" at
keystroke cadence.

Fix, in `createTranscriptScrollRuntime` (`codoxear/static/app_transcript.js`):
- `readerOwnsScroll` provenance: set by `noteUserScroll()` (wheel, touch,
  `pointerdown` on `#chat`, `j/k/d/u` shortcuts), cleared whenever the module
  places the viewport (`scrollToBottom`, `setScrollTop`, `enableAutoScroll`) or
  re-renders. A negative delta detaches only while a reader owns the scroll.
- `pinnedToTail()`: a scroll event that ends at the tail is evidence of *being*
  at the tail, never of leaving it (covers the keyboard-close clamp).
- `scrollToBottom()` targets the exact bottom offset and writes nothing when the
  viewport already holds it — removes the per-keystroke write storm that
  provoked the platform corrections.

Verified (Docker, real Chromium, iPhone metrics, fixture transcript, pre-fix HEAD
vs working tree, same probe): platform-initiated 300px upward move with no
gesture shows chip+jump pre-fix and shows nothing after; reader wheel, pointer
grab, and wheel-after-app-pin still show both; typing with the composer growing,
keyboard up, keyboard typing, and keyboard dismissed all stay hidden in both.

Confidence: high that non-reader scroll traffic can no longer surface the chip.
Unverified residual: whether the reporter's trigger was this latch or the
`--appH`/`--vvTop` compensation (no device access). If the chip still blinks,
that is the remaining candidate and needs a device log of
`visualViewport.offsetTop`/`scrollTop` at keystroke time.
