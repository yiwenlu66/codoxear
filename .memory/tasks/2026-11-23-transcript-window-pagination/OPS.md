# OPS — transcript window / paging (diagnosis only, no code changed)

- 2026-11-23T03:50Z: Code survey (read-only). Owners of paging/attachment:
  `app_transcript.js` (render runtime, dom runtime, scroll runtime, typing row
  runtime), `app_transcript_view.js` (view-state policy), `app_message_history.js`
  (older pages, cursor windows, `jumpToLatest`), `app_message_flow.js` (live
  deltas, send echo), `app_transcript_render.js` (composition + typing-row store
  projection), `message_routes.py` (`tail`/`history`/`window`/`neighbor`),
  `transcript_search.py::read_chat_window_around` (already returns `has_newer`).
- 2026-11-23T03:55Z: Built a Node VM harness driving the real
  `app_transcript.js`, `app_transcript_view.js`, `app_transcript_render.js` and
  `app_message_rows.js` (projected through `tests/frontend_module_loader.py`) with
  a stub DOM, mirroring the stub pattern of
  `tests/test_chat_transcript_runtime.py:597`. Harness:
  `/tmp/paginate-repro/repro.js` (disposable; the plan proposes the equivalent as
  a committed test).
- 2026-11-23T03:58Z: Predictions before running (recorded so the result is
  meaningful): after six older pages the tail rows should be gone and
  `renderedAtLiveTail` false; a live delta should be dropped; the typing row
  should be inserted after the last old row; `view.scrollToBottom({force:true})`
  should claim attachment without installing the tail.

  Observed (exactly as predicted, `repro.js` JSON):

  | step | observation |
  | --- | --- |
  | tail install (24 rows) | 24 rows, last `tail-23`, attached=true |
  | 6 × `prependEvents(60)` | 320 rows, last `old-119`, no `tail-*`, attached=false, jump shown |
  | live delta | `appendEvents` → false, no row (event marked seen at `app_transcript.js:764`) |
  | `sessionState.set("running", true)` | `<WORKING BUBBLE>` inserted directly after `old-119` |
  | scroll to bottom | viewport rests on `old-119`; jump shown; no fetch of the log end |
  | `view.scrollToBottom({force:true})` | returns true, attached=true, bottom row still `old-119`, jump **hidden** |
  | live delta after that | appended after `old-119` (live turn spliced onto the old page) |

  Trim arithmetic corroborated by the harness: 24 + 5×60 = 324 > 320
  (`CHAT_DOM_WINDOW` 260 + `OLDER_PAGE_LIMIT` 60), so page 5 starts deleting the
  newest rows and page 6 deletes the tail entirely.
- 2026-11-23T04:02Z: Cross-checked the second entry path: `loadTranscriptWindowAtCursor`
  (`app_message_history.js:180-200`) installs a `before=30&after=30` window and
  **discards** the server's `has_newer` (`message_routes.py:1064`), so cursor
  windows (chat search, cross-log neighbour) can never page forward.
- 2026-11-23T04:05Z: Confirmed the transport/`seen` interaction that a forward
  pager depends on: `applyLiveMessageData` advances the live cursor
  (`app_message_flow.js:479-482`) whether or not the rows are rendered, and the
  drop path marks dropped rows seen (`app_transcript.js:762-767`) while page
  inserts do not (`prependOlderEvents` passes `markSeen:false`). Recorded as a
  required change in the plan (§4.1).
- 2026-11-23T04:07Z: Recorded dead code found on the way: `app_message_flow.js`'s
  `renderedAtLiveTail` dependency has no call site; `renderTranscript`'s
  `firstUnreadEventId` has no caller; `maybeAutoLoadOlder` in
  `app_message_history.js:269-273` is exported but unused. None is fixed here
  (no implementation in this task).
- 2026-11-23T04:09Z: Wrote `TRANSCRIPT_WINDOW_PLAN.md`. No source file touched;
  `git status` other than this task dir + the plan is pre-existing WIP memory
  dirs (untouched).
