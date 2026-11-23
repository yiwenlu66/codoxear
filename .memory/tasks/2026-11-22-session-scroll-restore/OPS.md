# Evidence trail — session-switch scroll restore

All timestamps UTC. Repo state: HEAD dc69e561 (+ untracked .memory from other sessions).

## 2026-11-22T10:44Z — code read (no execution)

Sites established:
- save: `codoxear/static/app_session_lifecycle.js:172` -> `app_transcript.js:989`
  (`{scrollTop, scrollHeight}` per session id)
- restore: `app_transcript.js:999-1016`, two rAFs, gate = `chat.scrollHeight === saved.scrollHeight`,
  else `scrollToBottom()`; call sites `app_session_lifecycle.js:208, 231, 277`
- render path: `app_transcript_view.js:186-196` (`replaceWith` -> enableAutoScroll, markLiveTail,
  scheduleScrollToBottom({double:true})), `app_transcript.js:791-804` (renderTranscript),
  `app_transcript.js:927` (rebuildDecorations conditional bottom write)
- image cold path: `app_markdown.js:406-424` (`image.removeAttribute("src")` when dimensions are
  uncached), `359-372` (`hydrateMarkedImage` -> async fetch), `379-386` (microtask hydration),
  `316-322` (`applyImageDimensions`: aspectRatio + width/height)
- tail window: `INIT_PAGE_LIMIT = 24` (`app_application_composition.js:188`), DOM window 260 rows

Prediction before any measurement: at the two restore frames the scroller height is not final when a
local image's dimensions are uncached; a bottom restore therefore ends above the real bottom, and a
mid-history pixel restore drifts by the late growth. Both worse on engines without scroll anchoring
(iOS Safari).

## 2026-11-22T11:05Z — independent browser probe of the layout mechanism (main agent)

Isolated page, real Chromium via agent-browser (`scrollprobe-main`), real
`codoxear/static/app_markdown.js` imported (module served from a symlink; no repo writes).
Server: `python3 /tmp/scroll-probe/serve.py` on 127.0.0.1:18999, dims endpoint delayed 300 ms.
Scroller 600 px tall, 40 text rows, one 900x1600 PNG row, `scrollTop = max` written synchronously.

Bottom intent, cold dimension cache (`/tmp/scroll-probe/probe2.html?mode=bottom&img=1`):
| t | scrollHeight | scrollTop | gap to bottom | image box |
|---|---|---|---|---|
| sync | 10145 | 9545 | 0 | 24 px |
| rAF1 (1 ms) | 10145 | 9545 | 0 | 24 px |
| rAF2 (12 ms) | 10145 | 9545 | 0 | 24 px |
| 300 ms | 10145 | 9545 | 0 | 24 px |
| 1500 ms | 11799 | 9545 | **1654** | 1678 px |

Mid-history, image ABOVE the anchor (`probe3.html?mode=mid&imgpos=5`), cold cache:
- `overflow-anchor: auto` (Chromium): scrollTop is rewritten by the UA 4261 -> 5915 and the viewport-top
  row stays row 16 at the same offset. UA anchoring hides the defect.
- `overflow-anchor: none` (what an engine without scroll anchoring, e.g. WebKit, effectively does):
  scrollTop stays 4261, the content is pushed down by 1654 px and the viewport-top row becomes
  **row 10** — six messages above the remembered one.

Control: with the dimension cache warm, image box is 1678 px at `sync` and every later sample is
stable in all three modes — i.e. the failure needs a cold/evicted dimension cache.

## 2026-11-22T10:48Z — isolated Docker app repro (delegated, in progress)

Image `codoxear-design-final:3ddc8a1ff868` + host repo mounted read-only at /workspace + bundle rebuilt
with esbuild; two seeded Pi sessions (A: 12 messages, last one a tall local PNG; B: text only), port
19688. Full-app samples for scenario A (scroll to bottom in A, switch to B, return to A), cold cache:

| t | scrollHeight | scrollTop | gap to bottom | #jumpBtn | image |
|---|---|---|---|---|---|
| 50 ms | 5345 | 4603 | 0 | none | h=26.8 px, naturalWidth 0, source not yet assigned |
| 2000 ms | 6549 | 4603 | **1204** | none | h=1230.8 px, naturalWidth 900 |
| 4000 ms | 6549 | 4603 | 1204 | none | unchanged |

## 2026-11-22T11:2xZ — whole-app scenario B (mid-history), same harness

Anchor recorded in session A before switching away: `scrollTop 1837`, `scrollHeight 6549`,
`firstVisibleRowId byte-5764-user` at offset -36.8, `row.offsetTop - scrollTop = 7`.

After switching to B and back to A (cold dimension cache):

| t | scrollTop | scrollHeight | gap | #jumpBtn | viewport-top row |
|---|---|---|---|---|---|
| 50 ms | 4603 | 5345 | 0 | none | byte-14206-user (-206) |
| 2000 ms | 5807 | 6549 | 0 | none | bb4aaf71... (-977) |
| 4000 ms | 5807 | 6549 | 0 | none | bb4aaf71... (-977) |

Verdict: the remembered message (`byte-5764-user` at 1837) was discarded; the view snapped to the tail.
Note the difference from scenario A in the same app: there the late bottom write never happened
(gap 1204 persisted for 4 s), here the view ended exactly at the bottom — i.e. whether the late
scheduled write lands after the image hydration is timing-dependent, which is the observed
"random".

## 2026-11-22T11:40Z — delegated full run (executor) consolidated, plus frame-trace analysis (main agent)

Container `codoxear-scroll-repro-19688`, two real Pi brokers, Chromium 151, 1280x900, chat clientHeight
742, no artificial network delay. Artifacts: `/tmp/codoxear-scroll-repro.wXqRk2/`.

Cases (pixels; `lastΔ` = viewport bottom − last row bottom; `auto` inferred from the jump control):
| case | top | height | max | gap | first@offset | jump | lastΔ | imageH |
|---|---|---|---|---|---|---|---|---|
| A 0.05 s | 4603 | 5345 | 4603 | 0 | U11@-206 | none | +15.2 | 26.8 |
| A 2 s / 4 s | 4603 | 6549 | 5807 | **1204** | U11@-206 | **none** | **-1188.8** | 1230.8 |
| B 0.05 s | 4603 | 5345 | 4603 | 0 | U11@-206 | none | +15.2 | 26.8 |
| B 2 s / 4 s | **5807** | 6549 | 5807 | 0 | LA@-977 | none | +15.2 | 1230.8 |
| B2 0.05/2/4 s | **1837** | 6549 | 5807 | 3970 | **U5@-36.8** | inline-flex | -3954 | 1230.8 |
| C 0.05/2/4 s | **1837** | 5271 | 4529 | 2692 | **U5@-36.8** | inline-flex | -2677 | — |

- A FAIL: viewport bottom is 1188.8 px above the last row's bottom, yet the last row is *partially*
  visible (its own height is 1230.8 px), and `#jumpBtn` is hidden (gap 1204 > 80 with `auto` inferred T):
  the UI reports "at live tail" while 1.2 screens of tail are below the fold.
- B FAIL: saved anchor U5 at −36.8125 px (scrollTop 1837, row.offsetTop−scrollTop = 7); returned at the
  tail (5807) — anchor displacement −3970 px. The transient bottom gap healed at 523 ms; the anchor
  never returned.
- B2 PASS (dimension cache warm) and C PASS (text-only session): U5 restored exactly, 0 px drift, jump
  shown. Failure is therefore conditional on late layout, not on switching.

### Frame traces isolate the corrective write

`measurements.json` → `runs[*].frames` (253 frames each), transitions only:

```
initial-A:                ms=55  height 5345->6140 img 27->822 ; ms=72 height->6549 img->1231 ;
                          ms=889 top 4603->5807  height/img/jump UNCHANGED
A-bottom-screenshot-repeat: ms=74 height 5345->6549 img 27->1231 ;
                          ms=1990 top 4603->5807  height/img/jump UNCHANGED
A-bottom-uncached:        ms=75 height 5345->6549 img 27->1231 ; no top change through 4191 ms
```

So the correction is a *content-independent* scroll write at a poll boundary (889 ms ≈ the
`kickPoll(900)` after `openSession`; 1990 ms on the repeat), not a layout event: the image finished
hydrating 800+ ms earlier. Among writers that can fire on a `/messages/live` poll with no new rows, the
only bottom writer is `typingRowRuntime.updateActivityPeek` (`app_transcript.js:606`:
`!hadPeek && activityPeek && shouldAutoScroll()`), i.e. the first time the activity peek appears.
Ranked hypothesis; the confirming observation is a peek element present at the heal frame (or a patched
`scrollToBottom` recording its caller).

Consequence: whether the at-bottom requirement is met depends on whether an activity peek happens to
arrive while `autoScroll` is still true. That is the residual "random". It also means the defect is
partly masked in trials that heal, which is why it reads as intermittent rather than deterministic.

## 2026-11-22T14:0xZ — implementation (this session)

Files:
- new `codoxear/static/app_transcript_position.js` — the position model
  (`{kind:"tail"}` | `{kind:"anchor",messageId,cursor,offset}`), save/restore/clear/cancel,
  ResizeObserver settle that stands down on reader gesture (`unownedScrollSeq`), another
  writer (`writeSeq`), session change, or a vanished row; anchored materialization through
  `resolveAnchorWindow(cursor)`.
- `app_transcript.js` — removed the pixel memory (`save/restore/clear/sessionScrollPosition`,
  `scrollPositions`); added the content-size pin ("following the tail" is an invariant),
  `writeSeq`/`unownedScrollSeq`, `cancelScheduledBottomWrites()`, geometry-based
  `isAtLiveTail()` for `syncJumpButton`/`syncVisibleTimeIndicator`.
- `app_transcript_render.js` — instantiates the controller (`content: chatInner`,
  `resizeObserver`), exports save/restore/clear, cancels on `resetChatRenderState`.
- `app_session_lifecycle.js` — renamed seam; one restore per open (after the render the
  session shows); no restore after the cached tail; a transcript replacement clears the
  position.
- `app_wiring.js` — `createTranscriptPositionOptions`; scroll options gained
  `content`/`resizeObserver`.
- `app_markdown.js` — a pending local image reserves the default 16:9 box instead of
  collapsing (src still withheld until the real ratio arrives).
- `AGENTS.md` — the contract; `.memory/project/ARCHITECTURE.md` — durable summary.

Tests: `tests/test_transcript_position.py` (6 new behavioral cases), `test_app_markdown_extended.py`
(box reserved / cached ratio), updated `test_transcript_scroll_intent.py` (affordances =
geometry) and `test_chat_transcript_runtime.py` (superseded pixel test removed, projection
expectation now geometry), renamed stubs in `test_chat_scrollback_source.py`.
Full suite: 3019 passed, 3 pre-existing/flaky failures (2 fail at HEAD: hint-mode coverage,
pi prompt receipt; 1 SSE timing test passes on rerun).

### Behavioral verification (isolated container `codoxear-scroll-repro-19688`, real Chromium)

Bundle rebuilt from the working tree and served into the running container.

| scenario | 50 ms | 1 s | 2 s | 4 s |
|---|---|---|---|---|
| A: left at the bottom, cold image | top=max=5987, gap 0, last row 15px above viewport bottom, jump hidden | same | same | same |
| B: left on `byte-5764-user` @ -36.81, cold | top 1837, first row `byte-5764-user` @ **-36.81**, jump shown | same | same | same |
| B2: same, warm dimension cache | same as B (exact) | same | same | same |
| C: text-only control (B→A→B) | `byte-5764-user` @ -36.81 | same | same | same |

Before the change the same container measured A = gap 1204 with the last row half off-screen
and jump hidden, and B = snapped to the tail (+3970).

Mechanism checks in the same session:
- at the bottom, injecting 600px of content below the viewport -> `top` follows to the new max
  (5987 → 6587), gap 0, jump hidden (the pin);
- after a wheel gesture at scrollTop 500, the same 600px growth leaves `top` at 500 (the pin
  stands down);
- after an anchor restore, injecting 500px **above** the anchor -> scrollTop 1837 → 2337 with
  the anchor still at -36.81 (the settle; this is the case WebKit has no UA anchoring for).
Browser console: no exceptions (only the CSP meta warning and the pre-login 401 probe).
