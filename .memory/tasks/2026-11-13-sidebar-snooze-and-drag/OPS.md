# OPS — sidebar snooze + drag (planning)

Append-only. All times local (2026-11-13).

## 2026-11-13 — investigation (no product code changed)

Read for the plan (paths relative to repo root):

- `codoxear/static/app_snooze.js` — presets only. No "indefinite"; `snoozePresetEpoch("none")`
  returns null, so "none" is expressed by the absence of a timestamp.
- `codoxear/static/app_session_edit.js` — snooze mode state machine
  (`none|evening|tomorrow|weekend|nextWeek|custom`), custom value written from
  `<input type=date>` + `<input type=time>`, save posts `/api/sessions/<id>/edit` with
  `name + priority_offset + snooze_until + dependency_session_id` (all fields always sent).
- `codoxear/static/app_shell.js:527-553` — chip construction; `:579-582` field layout;
  `editSnoozeCustomRow` holds the two native inputs.
- `codoxear/static/app.css:3212-3216` (date/time at `--ctl` = 44px), `:3325-3331` (chips at
  `--dialog-control-h` = 32px), `:4095-4106` (coarse-pointer `font-size: var(--font-xl)
  !important` on `input[type=date|time]`), `:3755` (`.customSnoozeRow` grid `1fr 140px`),
  `:3546` (`#editViewer .choiceChips` wraps), `:2626-2674` (`.filePickerMenu` /
  `.fileMenuItem`), `:258-264` (`#sessions` is the scroll container), `:363-373`
  (`.session { touch-action: pan-y }`).
- `codoxear/session_listing.py:153` (`clip01(time_priority + offset)`), `:425` (`snoozed`
  predicate), `:646-694` (`SIDEBAR_GROUP_RANKS`, `sidebar_group_key`, `sort_session_rows`),
  `:670` single order authority.
- `codoxear/server_config.py:349-350` — `SIDEBAR_PRIORITY_HALF_LIFE_SECONDS = 8h`,
  `SIDEBAR_PRIORITY_BUCKET_SECONDS = 30s`.
- `codoxear/session_cleaners.py:41-43` — offset range `[-1, 1]`.
- `codoxear/session_store.py:346-367` — `sidebar_state_for_session` prunes dead dependency and
  expired snooze (the pattern a rank prune would follow).
- `codoxear/session_ui_state.py:52-125` — `sidebar_meta_get` / `_validated_sidebar_entry` /
  `sidebar_meta_set` / `edit_session` (whole-row write, no partial semantics).
- `codoxear/control_routes.py:104-115` post route table; `:166-188` `_handle_edit` (requires
  `name`, always writes all four fields).
- `codoxear/session_routes.py:88-113` — non-session POST paths (`/api/quick-sessions`) are
  matched here, the precedent for a new `/api/sidebar-layout`.
- `codoxear/server_routing.py:23-34` — `match_session_route` requires exact arity, so a
  non-session `/api/...` path cannot be mistaken for a session route.
- `codoxear/diagnostics_routes.py:71-122` — recomputes `snoozed` and shows
  `priority_offset` / `final_priority` independently of the listing.
- `codoxear/session_snapshot_runtime.py:552-556` — `remap_session_identity` moves the whole
  `sidebar_meta` entry, so new meta keys follow a resume with no code change.
- `codoxear/static/app_sessions.js:82-141` (`bindSwipe`; axis gate at `:112-116`), `:165-172`
  (`refreshDeferred`), `:287-334` (touch/desktop card branches), `:171` + `:330` (render signature,
  empty-hint handling).
- `codoxear/static/app_session_helpers.js:43-64` — `sessionSidebarGroupKey` /
  `sidebarSessionEntries` partition only, never re-sort.
- `codoxear/static/app_dialog_menu.js:39-71` — `positionDialogMenu` (flip + max-height + width
  clamp) already suited to a 28-row / 48-row menu.
- `codoxear/static/app_transcript_render.js:303-305` — the direct-shortcut handler bails on
  `altKey`, so `Alt+↑/↓` is free.
- `tests/test_snooze_presets.py`, `tests/test_sidebar_render_order.py`,
  `tests/test_session_listing.py:417-442` — the behavioural harness style and the sort tests that
  will need new cases.
- `scripts/docker_theme_verify.sh` — the established docker + loopback-port + agent-browser
  harness to copy for a drag verifier.

## 2026-11-13 — negative results (kept so they are not re-proposed)

- Styling the native `<input type=date|time>` to match the chips was rejected: the coarse-pointer
  `font-size: var(--font-xl) !important` rule (`app.css:4105`) exists to stop iOS focus zoom
  and cannot be lowered for these two inputs without reintroducing that bug.
- Encoding "indefinite" as `math.inf` or a far-future stamp was rejected: `json.dumps(float("inf"))`
  emits `Infinity`, which `JSON.parse` rejects, so the listing row could hand the browser invalid
  JSON.
- Writing `priority_offset` from the drag was rejected on arithmetic, not taste: with
  `clip01(tp + offset)` an old row cannot out-rank a saturated row (score exactly 1.0 loses the
  `updated_ts` tie), and a constant offset on a decaying recency term inverts a young-row-below-
  older-row order in ~16 hours (numbers in `PLAN.md` §4.2).
- A month-grid calendar was rejected: 7 columns cannot give 44px targets inside a 366px phone
  dialog, and the app already ships a list-menu primitive that honours the touch rule.

## 2026-11-13 — artifact note

`PROMPT.md` in this directory is local-only: the repo `.gitignore` ignores `PROMPT.md` (25 older
task specs were force-added before the rule). The committed deliverable is `PLAN.md`, which is
self-contained; `OPS.md` carries the evidence trail.

## 2026-11-13 — ladder math verified against the real sort (v2 plan)

Prediction before running: a whole-block offset ladder with `TOP=0.99`, `BOTTOM=0.02`,
`offset_i = target_i - tp_i` reproduces any requested order exactly, needs no clamping, and survives
one 30s bucket of decay.

Harness (`/tmp/ladder_check.py`, `/tmp/ladder_stress.py`, throwaway): builds rows with real
`listing_priority` / `sidebar_time_priority_from_elapsed_seconds` (8h half-life, 30s bucket), writes
the ladder, and re-runs `sort_session_rows` on the result.

Observed:
- 5 fixed ages (1 min … 30 d) × 4 hand-picked permutations, and a block where three rows were already
  saturated at `final_priority = 1.0` by pre-existing offsets: every requested order reproduced
  exactly, including dragging the oldest row to the top of the saturated block.
- +30 s later (one bucket of decay): all orders still exact.
- 400 random trials, N ∈ {1,2,3,5,8,13,21,40,60}, ages from 0 to 30 days, random permutations,
  checked again at +30 s: 0 mismatches; `max |offset| = 0.99`, i.e. the `[-1, 1]` cleaner bound is
  never reached and no clamp is needed.
- Activity behaviour confirmed as documented: a row demoted while fresh stays demoted when it next
  receives a message (`['c','b','a']`), a row demoted while stale clips back to the top
  (`['z','y','x']`). This is why the v2 plan states the stickiness instead of claiming "fresh always
  rises".

Conclusion: the user's rule (compute the offset for the desired position at that instant) is exactly
satisfiable at block granularity; v1's manual-rank field was unnecessary.

## 2026-11-13 — v3: single-row offset, preview built

User rulings taken as binding (supersede v2): a drop computes the offset for the **dragged row only**;
a UI preview is required; no unrequested surface may be added (the Later-card snooze badge idea is
dropped, and nothing new is displayed on a card).

Changes made to the plan:
- §4.2 replaces the whole-block ladder with a single-row target: `(score_above + score_below)/2`,
  or `score_below + 0.005` at the top, or `score_above - 0.005` at the bottom; one
  `priority_offset` write for the dragged row, computed server-side against the same clock/bucket as
  the sort.
- Two boundaries are now stated instead of engineered around: a neighbour pinned at exactly 1.0
  (clip) falls to the `updated_ts` tiebreak, and a target outside `[tp-1, tp+1]` is clamped. The route
  reports `applied: false` in those cases.
- `/edit` no longer becomes partial; the two gestures get two narrow routes
  (`/snooze`, `/reorder`), so no existing endpoint changes contract.
- Removed from the plan because they were not asked for: `Alt+↑/↓` reordering, a hover grip
  affordance, success toasts, and the Later-card state question.

Preview (new): `mockup.html` imports the real `app.css` + `themes/clay.css` and uses the real dialog
classes; screenshots `preview-desktop.png` (1100×1000) and `preview-phone.png` (430×1200) were taken
in Chrome via `agent-browser --session sidemock`.

Measured while building the preview (implementation notes, not opinions):
- `.dialogPickerBtn` has no width rule (only `.sidePickerBtn` does): `getBoundingClientRect()` gave
  118.6px inside a 170px grid column, i.e. shrink-to-fit. The two new pickers need `width: 100%`.
- The native `input[type=date]` renders 44px tall with 16px text next to 32px chips at the same
  dialog width — visible in the preview's "before" shot, matching the user's screenshot.
- `#editViewer`-scoped rules (chips wrap, `formBody` overflow visible) are what make the mock behave
  like the real dialog; the implementation inherits them.

## 2026-11-13 — preview correction: chip plane, not field plane

User: "still ugly! don't you notice size difference?" — correct. Measured with
`getBoundingClientRect()` + `getComputedStyle()` in the preview page, before the fix:

| element | height | font-size |
| --- | --- | --- |
| `.choiceChip` | 32px | 13px |
| my custom date picker | 44px | 16px (`--font-lg`) |

So v3 §3 had put the custom row on the same plane as the name input, which left the *field-internal*
size difference the user was complaining about. Corrected: custom pickers now use
`--dialog-control-h`, `--font-md`, `0 var(--space-5)` padding and the chips' `::after { inset: -2px }`
hit area, two equal columns at every width. Re-measured after the change: chip 32px/13px, date picker
32px/13px, time picker 32px/13px; the "Depends on" field correctly stays 44px/14px because it is a
field, not a chip.

Screenshots regenerated (`preview-desktop.png` at 1100×1000, `preview-phone.png` at 430×1150).

## 2026-11-13 — dropdowns rejected; calendar + wheel built and measured

User: "you use dropdown for datetime?" — correct, v3 proposed a 28-day list menu plus a 30-minute list
menu, which is a dropdown pair, not a date/time control.

Two further errors surfaced while fixing it:
- The reason I had written for rejecting a calendar ("7 columns cannot reach the 44px touch target
  inside a phone dialog") is arithmetically false. Measured content width: 393px viewport → 322.6px
  content → 46.1px per column; 430px viewport → 356.6px → 50.9px. Desktop dialog → 64px. A calendar
  fits.
- The earlier preview carried wrong weekday labels (`Wed 12 Nov`, `Fri 14 Nov`). 2026-11-12 is a
  Thursday and 2026-11-14 is a Saturday. The calendar in the mock is now generated from the real
  month, not typed by hand.

Replaced in the plan and the mock: `.snoozeCalendar` (7 columns, 44px-tall day cells, past days
disabled, today ringed, selected ink-filled) + `.snoozeTimeWheel` (132px = 3 × 44px rows,
`scroll-snap-type: y mandatory`, centred row in ink with 1px rules). No `.filePickerMenu` remains in the
date/time path.

Measured after the change: day cell 64×44 (desktop) / ≈46×44 at 393px, wheel row 44px, chips 32px,
`Depends on` field 44px.

Screenshots regenerated: `preview-custom.png` (the Custom… block alone), `preview-desktop.png`,
`preview-phone.png`.

## 2026-11-13 — design delegated; evaluation rubric fixed BEFORE seeing results

User: "still ugly. make a decent subagent do design." Two independent designers are running in
parallel (design-a: claude-opus-5-5 high; design-b: gpt-6.1-sol high), same brief, separate output
dirs, forbidden from reading each other.

Rubric fixed in advance so the choice is not after-the-fact taste. Each item is measurable or
falsifiable from the artifacts:

1. **Height parity.** Visual height of the custom control either matches the chip plane
   (`--dialog-control-h` = 32px) or is an overlay that costs zero inline height. Anything that adds a
   >150px inline block to the field is the failure that has now been rejected three times.
2. **Type parity.** Nothing inside the custom control renders above `--font-md` (13px desktop /
   14px coarse). The original sin was `--font-xl` (16px) next to 13px chips.
3. **Touch targets ≥ 44×44** in both axes, measured, whether visual or via `::after`.
4. **Token purity.** No color literal, no new font family, all geometry via `--radius-*`/`--space-*`.
   Must still read correctly under paper (square, no shadow) and slate (pills, ghost chrome).
5. **Legibility of state.** The chosen day and time must be readable without opening anything, and
   there must be no hidden commit step.
6. **Not a re-skin.** If the answer is rejection #3 or #4 with different padding, it fails regardless
   of polish.

Prior (recorded before the results): the untried combination is a chip-plane (32px) trigger whose
picker is an *overlay* — calendar for the day, wheel/stepper for the time — rather than inline. That
satisfies #1 and #2 without a dropdown list, and it is the shape the platforms use (Android date
dialog, iOS wheels). If neither designer lands there, that is a signal to steer, not to accept.

## 2026-11-13 — design-b (gpt-6.1-sol) result, judged

Artifacts: `design-b/` (mockup.html, shot-phone.png, shot-desktop.png, check-{clay,paper,slate}[-dark].png,
measure-*.json, RATIONALE.md). Verified the reported numbers myself from `measure-phone.json`: chip
32px/14px, value 32px/14px, nudge 32px visual with 44px pseudo-element, stepper block 120px tall,
4 chip rows at 393px, body does not scroll at 852px height. Six theme/mode renders exist and the
control holds its 120px/32px/44px geometry in all of them. Its critique of rejection #4 (calendar
consumes 264px of rows before headers; the 44px selected day out-shouts the 32px chip that revealed
it) is accurate and reusable.

Verdict against the rubric: 1 height pass (120px, under the 150px limit), 2 type pass, 3 touch pass,
4 tokens pass, 5 state legible, 6 not a re-skin. It fails the *purpose*: ten floating chevrons with no
grouping read as an unfinished debug widget, and a date three weeks out becomes a repeated-tap chore —
which the designer itself named as the weakest point and shipped anyway.

Model-feedback submitted: rating 2, request-id 73ddb205-9d37-492d-8517-1973b242e02e.

## 2026-11-13 — design-a (opus-5-5) round 1: two chip strips, steered away

Artifacts: `design-a/` (mockup with `?theme=`/`?only=` switches, shot-phone.png, shot-desktop.png,
shot-families.png, RATIONALE.md). Measured: mode chip and strip chips both 32px/14px (13px desktop),
strip hit 44px tall, revealed block 88px, dialog 520/614/520 at 393 wide, 4 mode chip rows.

Its critique of rejection #4 is the best work product of the whole exercise and is accurate:
44px *visible* boxes against the constitution's "hit area is not visual size"; three competing
selection languages (filled mode chip, filled day cell, ringed today, bold wheel row); bare day digits
while every other target is a bordered chip; `November 2026` at `--font-md` 600 duplicating
`.fieldLabel`; the wheel's implicit scroll-settle commit; a vertical scroller nested in `.formBody`.

Verdict: consistent with the system (it *is* `.choiceChip`) but functionally wrong — its own weakest
point says it: a linear strip makes "the 9th of next month" a ~1800px fling with no month landmark,
and the half-clipped `er 17` / `0pm` edge fragments read as rendering glitches. Three scroll gestures
in one field (two horizontal strips inside a vertical body).

Steered (same session, run 6510f682): keep the design language, the two-weight selection rule, the
`· Wed 18 Nov, 2pm` readout and the no-commit-step property; replace the mechanism with a chip-plane
trigger plus an overlay month-calendar panel reusing `filePickerMenu` chrome and
`positionDialogMenu`, with the time slots in the same panel. Output to `design-a2/`, must report panel
rect vs viewport and prove it does not clip or scroll the body.

## 2026-11-13 — design-a2: chip trigger + calendar popover, judged

Artifacts: `design-a2/` (mockup, shot-phone.png, shot-desktop.png, shot-families.png,
shot-helper-stock.png, RATIONALE.md). Verified by eye in `shot-families.png`: clay/paper/slate ×
light/dark all hold — paper square with no shadow, slate rounded, dark inverts the ink edge; day cells
are bordered chips, past days borderless, `18` carries the 2px ink edge, `2pm` likewise.

Rubric: 1 inline growth **0px** (best possible), 2 day/time chips at `--font-md` (14/13), 3 day cell
32×32 visual in a 44×44 track (measured, tiling without overlap), 4 tokens only, 5 state legible via
`· Wed 18 Nov, 2pm` in the mode chip's existing slot, 6 not a re-skin. All pass.

Real findings worth carrying into the plan:
- `positionDialogMenu` clamps the panel to half the viewport and only flips under 220px, so a 414px
  calendar gets cut (their `shot-helper-stock.png` shows it at 393×660 and 1100×900). Their proposal:
  an opt-in `data-fit-whole` flag (full `scrollHeight`, open on the larger side). Three lines, no
  change for menus without the flag.
- Document click-away must use `event.composedPath()`: a day tap re-renders the grid, so
  `target.closest()` sees a detached node and closes the panel mid-selection.
- At 360px wide the day tracks shrink to 42.6px, so neighbouring 44px hit areas overlap by 1.4px.

Rejected remainder: the time well is a nested scroller — 8 of 48 slots visible, no cue that it
scrolls, one 37px row at 393×660. That is the same defect the designer criticised in rejection #4.
Steered (run 06587768, `design-a3/`): no inner scrolling anywhere (`scrollHeight === clientHeight`),
all 48 half-hours within two taps via a period selector plus a fixed 3-row grid, panel ≤ 460px, shots
with a non-default time.

## 2026-11-13 — design-a3: period tabs + fixed slot grid, judged

Artifacts: `design-a3/`. Verified in the shot: panel = nav row (`November 2026`, ‹ ›), weekday row, six
week rows, period tabs `Night · Morning · Afternoon · Evening`, then a fixed 6×2 grid of the period's
twelve half-hours. No inner scroll (their measurement: no element in the panel computes overflow
auto/scroll; panel scrollHeight 443 = clientHeight 443 at five viewport sizes). Period tab carries two
marks: `--wash` fill = on view, 2px ink edge = holds the answer, and browsing periods never hides the
answer's location. Every half-hour ≤ 2 taps; "9th of next month at 9am" = 4 taps from a Morning answer.
Panel 322×445, and the placement table shows it opens *over* the anchor at 393×852 and 393×660 because
445 > 417px of free space below.

Rubric: all pass except one new consequence, which the designer found itself: with the panel over the
chip, the only written readout of the answer is covered; state survives only as three ink edges.

Decision: keep the design, fix the consequence. Steered (run a97a2364, `design-a4/`): put the live
answer in the panel header row alongside a still-identified month, no extra row, panel ≤ 460px,
shots at 393×852 / 393×660 / 1100×900 plus six family variants, header measured at 360 and 393 for
truncation.

## 2026-11-13 — design-a4 accepted as the final control

Artifacts: `design-a4/` (mockup with `?state=&theme=&mode=&pick=`, shot-phone.png, shot-desktop.png,
shot-families.png, shot-header-widest.png, RATIONALE.md). Copied into the task root as
`preview-custom.png` and `preview-custom-families.png`; `PLAN.md` §3 rewritten to this design.

Accepted because, measured rather than asserted: 0px inline growth (dialog 558px in every state);
panel 322×445; day chip 32×32 inside a 44×44 tiling track; period tab 72.5→78.5×44; slot 46.3→52.3×44;
header 308×32 and the widest real answer (`Wed 12 May, 12:30am`, 139.6px) untruncated at 360 with 18.4px
slack; zero scroll containers inside the panel at five viewport sizes; `.formBody` scrollTop 0; four taps
for "the 9th of next month at 9am"; six family/mode variants render from tokens only.

Two product-level findings carried into the plan: `positionDialogMenu` clamps a menu to half the viewport
and flaps only under 220px free, so an opt-in `data-fit-whole` branch is required (below → above → over),
and click-away must use `event.composedPath()` because a day tap re-renders the grid.

Known and accepted: on phones the panel opens over its own chip (445px fits neither side of 417/367), so
the live answer lives in the panel header; the header then carries two dates, distinguished by
`--font-md`/`--text` vs `--font-sm`/`--muted`, which is weakest in slate's ghost-button header.

Model feedback submitted for the designer session: rating 5, request-id dd237ea5-49b8-4326-ba70-137e4650d653
(the session produced a1→a4 under three steers; rated once, at the end).

## 2026-11-13 — user ruling: native pickers stay, only the closed box was too large

User: "i actually believe browser-native selector ui is ok. my actual comment is the data and time boxes
are too large when not opening picker ui."

That reclassifies everything above §3-of-the-plan: the calendar panel (`design-a4/`) is not the answer to
the request. The artifact and its measurements stay in the repo as the record, and the plan's §3 is
replaced by a four-rule change to the closed field.

Measured after the change in the root `mockup.html` (Chrome):
- desktop viewport: chip 32px/13px, date field 32px/13px, time field 32px/13px (was 44px/16px-ish);
- narrow viewport (`max-width: 880px` branch): chip 32px/14px, fields 32px/16px — height parity holds;
  the 2px type step is `app.css:4095-4106`'s anti-zoom `!important`, left alone.
- `::-webkit-datetime-edit` padding and the picker indicator are what made the field read larger than
  its own height; both are addressed in the proposed rules.

New constraint discovered while writing this: an `<input>` is a replaced element and cannot carry the
`::after` hit-area outset, so a 32px box is a 32px tap target. Plan now wraps each field in a
`min-height: var(--ctl)` label for the 44px region.

## 2026-11-13 — implementation (all three features)

Server (implementer 1): sparse `snoozed_indefinitely` + one predicate consumed by listing and
diagnostics; `/snooze` and `/reorder` routes; `reorder_target_offset` pure function; `/edit` accepts the
flag. 162 + 31 tests green.

Edit dialog (implementer 2): `Indefinitely` chip, open/save mapping, the four field rules.
17 tests green. Reviewed: its CSS placed the fields at 32px but I had to remove its
`font-size: var(--font-md)` declaration — `tests/test_form_typography_roles.py` forbids any dialog-scoped
rule from owning typography (the dialog's type planes retune as a unit on coarse pointers). The fields
therefore ride the entry plane (`--font-lg`, 16px on coarse), which is the same plane as the name input.

Edit dialog, my fixes after review: the snooze field container was `<label class="field">` wrapping chips
and now nested `<label class="snoozeInputCell">`s — nested labels forward taps to the outer label's first
labelable descendant ("No snooze"), so a tap on the date field would have fired both. The field is now
`div role="group" aria-labelledby`. Also scoped `.snoozeInputCell`, deleted the two declarations the new
rule supersedes, and dropped the dead `.customSnoozeRow` entry from the phone single-column rule.

Drag: the first implementer wrote only `app_sidebar_drag.js` (pure helpers, good) and timed out after an
hour without integration. I wrote the gesture in `app_sessions.js` (deferral, click suppression, drop
resolution, API commits) and the card-region CSS; a second implementer added the loader registration,
66 VM assertions, and completed five test stubs with `addEventListener`/`getBoundingClientRect`.

Full suite: 2954 passed, 1 failed — `test_hint_mode_coverage` (`draftUndoBtn`), which I reproduced in a
clean worktree at HEAD, so it is pre-existing and unrelated.

## 2026-11-13 — Docker verification of 71ffff93

Verifier: `scripts/docker_sidebar_snooze_drag_verify.sh` (commit-pinned, own container/port/HOME, real
server + broker + tmux, server-only restart). Evidence root
`.memory/tasks/2026-11-13-sidebar-snooze-and-drag/verify/20261113T050833Z-2198078/` (`report.json`,
per-step JSON, 12 picker screenshots + resting-field screenshots).

Steps 1–13 pass with evidence in hand:
- indefinite set/clear round trip, `snoozed:true` and grouped after both Now rows; the flag is on disk
  and survives a server-only restart (broker PIDs unchanged);
- an indefinite snooze is not pruned while a timed snooze one hour in the past on another session is
  (`snoozed:false`, timestamp null) — the two representations really are distinct;
- reorder: `[75,57,39] → [75,39,57]` with `priority_offset` written **only** for the dragged row
  (−0.65 → −0.35; the other two unchanged), and 400/400/404/401 for the error cases;
- `tmux list-windows` equals the sidebar order (`gamma, alpha, beta`);
- dialog, all six variants: seven chips with `Indefinitely`, native date/time fields at **32px** in
  **44px** wrappers beside 32px chips, `scrollWidth === clientWidth` (148/148 phone, 210/210 desktop),
  indicator clicks open the native popups (12 screenshots), save → Later → reopen selects the chip →
  clear → Now; no console errors or unhandled rejections;
- drag at 393×852 (400ms hold, `desktop:false`) and 1100×900 (8px mouse pull): reorder
  `[75,39,57] → [57,75,39]` and `[57,75,39] → [39,57,75]`, API and DOM agreeing; Now→Later sets
  `snoozed_indefinitely:true`; Later→Now wakes it first in the block.

Step 14 reported a "failure" that is an error in my brief, not in the product: the touch card's action
groups are Delete on the LEFT (`leftMax: 72`) and Edit/Duplicate on the RIGHT (`rightMax: 104`), so a
left swipe correctly reveals Edit/Duplicate (`swipeX: -104`) and a **right** swipe reveals Delete. The
verifier's observation is right; my expected direction was inverted, and the action-group wiring is
untouched by this change. What step 14 did establish: the swipe still engages and reveals its actions
after drags on the same cards, and a post-drop click leaves selection unchanged while the next tap
selects.

Verifier also corrected a fact I had wrong twice: the canonical file is `session_sidebar.json`
(`server_config.SIDEBAR_META_PATH`); `sidebar_meta` is only the Python attribute name. Both `AGENTS.md`
and the plan are corrected.

Not proven, and not claimed: physical iOS Safari / Android Chrome (intrinsic field height, real touch
and scroll arbitration), and a rapid second tap inside the 550ms post-drop suppression window.
