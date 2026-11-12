# Sidebar snooze + drag — feature plan (v4)

Plan only, nothing implemented. Previews: `preview-custom.png` / `preview-custom-families.png` is the
final snooze control (`design-a4/`); `preview-desktop.png` / `preview-phone.png` are the drag states
(`mockup.html`). Everything rendered from the real `app.css` + theme tokens in Chrome.

---

## 1. What decides the shapes

**The date row.** Three mechanisms stack, so it cannot be restyled into place:

- `.formViewer input[type=date|time]` is forced to `height: var(--ctl)` = 44px (`app.css:3212-3216`),
  while every snooze chip is `--dialog-control-h` = 32px (`app.css:3325-3331`).
- On coarse pointers `input[type=date|time]` takes `font-size: var(--font-xl) !important` — the iOS
  anti-zoom floor (`app.css:4095-4106`) — while chip labels stay at 13/14px. That is the bigger number
  in your screenshot, and it cannot be lowered.
- UA chrome: inner segments, calendar indicator, platform padding, on `minmax(0,1fr) 140px`
  (`app.css:3755`).

**The score.** `final_priority = clip01(time_priority + priority_offset)`, `offset ∈ [-1,1]`
(`session_listing.py:153`, `session_cleaners.py:41-43`), `time_priority` decaying with an 8h half-life
on a 30s bucket (`server_config.py:349-350`). A row's placement is `score`, so the drag is a
one-row offset computation (§4.2). One boundary follows from the clip: a row whose score is exactly
1.0 cannot be outranked by score at all, only by the `updated_ts` tiebreak — so if the row directly
above the drop slot is at exactly 1.0, the dragged row lands at 1.0 too and the tiebreak decides. That
state only exists while a row is within its first bucket after activity, or until a boosted row decays
below 1.0 again.

**Snooze.** `snooze_until` means "a finite timestamp or nothing" in the row contract, the expiry prune
(`session_store.py:358`) and the `snoozed` predicate. It cannot also mean "forever".

---

## 2. Indefinite snooze

One sparse key beside the existing ones, never a sentinel timestamp and never `math.inf`
(`json.dumps(float("inf"))` emits `Infinity`, which `JSON.parse` rejects):

```
sidebar_meta[sid] = { "priority_offset": float, "snooze_until": float,
                      "snoozed_indefinitely": true, "dependency_session_id": str }
```

Mutual exclusion lives in the one validator that already rejects a self-dependency
(`session_ui_state._validated_sidebar_entry`); downstream:

```
snoozed = snoozed_indefinitely or (snooze_until is not None and snooze_until > now_ts)
```

Touch points: `session_cleaners`, `session_store.SidebarSessionState` + `sidebar_state_for_session`
(the prune never touches an indefinite snooze), `session_ui_state`, `session_listing` facts + row,
`diagnostics_routes`, `launch_ledger` default row. `remap_session_identity` moves the whole meta dict,
so a resume carries the key for free.

UI: an `Indefinitely` chip between **Next week** and **Custom…** (see preview); opening a session that
is snoozed indefinitely selects it; saving sends the flag and no timestamp. Chip row already wraps.

Write path: `POST /api/sessions/<id>/snooze` with `{"indefinite": true}` or `{"indefinite": false}`
(clear). Narrow on purpose — a gesture must not be able to overwrite a name, an offset or a
dependency, so `/edit` keeps its current whole-row contract and is not touched by this feature.

---

## 3. Custom date/time control — final design

Designed and measured by a subagent; artifacts and reasoning in `design-a4/` (mockup with
`?state=&theme=&mode=&pick=`), rendered in `preview-custom.png` and `preview-custom-families.png`.

**Shape.** `Custom…` is the trigger. The picker is a panel anchored to that chip, reusing the
`.filePickerMenu.dialogPickerMenu` chrome (1px `--border`, `--radius-card`, `--shadow-pop`,
`--paper`) and `positionDialogMenu`. **Inline growth is 0px** — the dialog is 558px at 393 in every
state — and the panel is 322×445px.

**Panel contents, top to bottom**

1. *Header, one 32px row.* Left: the live answer `Wed 18 Nov, 9:30am` — `--font-md`, `--text`,
   `nowrap`, `flex: 0 0 auto`, `aria-live="polite"`, written by the same formatter call that fills the
   chip's `.chipTime` so the two cannot drift. Right: `‹ Nov 2026 ›` — month at `--font-sm`/`--muted`
   in a 60px `min-width` so the arrows never move, arrows are the standard 32px `.icon-btn` with 44px
   hit areas, `‹` disabled on the current month.
2. *Weekday row* `Mo … Su`, `--font-sm`, muted.
3. *Day grid* — Monday-first, **six fixed week rows** so the panel never changes height. Each day is a
   32×32 `.choiceChip` centred in a 44×44 track whose `::after` fills the track, so hit areas tile
   without overlap. Past days disabled and borderless; today at weight 600; the chosen day carries a
   2px ink edge (1px border + 1px inset ink ring — the `.agentBackendTab.active` language the app
   already uses).
4. *Period tabs* `Night · Morning · Afternoon · Evening`, 32px, with **two independent marks**:
   `--wash` fill = the period on view, 2px ink edge = this period holds the chosen time. A fully past
   period is disabled; the Today clamp moves the period on view.
5. *Slot grid* — a fixed 6 × 2 grid of that period's twelve half-hours, labels `h:mm` with the
   meridiem carried by the tab, each slot 32px visual in a 44px pitch, chosen slot with the same 2px
   ink edge.

**Interaction.** Opening `Custom…` fills it like a preset and opens the panel. A day tap commits and
keeps the panel open; a time tap commits and closes it; a period tap commits nothing; an outside tap or
picking a preset closes it. Nothing is ever pending. When closed, the chip's existing `.chipTime` slot
reads `· Wed 18 Nov, 2pm`, exactly the way presets read `· 9am`. "The 9th of next month at 9am" is four
taps: `Custom… › 9 9:00`.

**Measured** (Chrome, `getBoundingClientRect`/`getComputedStyle`, hit box = border box + `::after`):
day chip 32×32 → 44×44 at a 44px pitch; period tab 72.5×32 → 78.5×44; slot 46.3×32 → 52.3×44; header
308×32 with the widest real answer (`Wed 12 May, 12:30am`, 139.6px) fitting at 360 with 18.4px of slack
and no truncation; **no scroll container anywhere in the panel** (`scrollHeight === clientHeight` at
393×852, 393×660, 360×640, 430×932, 1100×900); `.formBody` never scrolls.

**Two narrow product changes this design needs**, both found by measurement:

- `positionDialogMenu` caps a menu at half the viewport and flips only under 220px of free space, so a
  445px panel gets cut. Add an **opt-in `data-fit-whole` branch**: use the full `scrollHeight`, place
  below if it fits, else above if it fits, else over the anchor from the larger side. Menus without the
  flag behave byte-identically.
- Click-away must use `event.composedPath()`. A day tap re-renders the grid, so `target.closest()` sees
  a detached node and closes the panel mid-selection.

**Consequence to keep in mind:** at 393×852 the panel fits neither side (417px below, 367px above), so
it opens *over* the `Custom…` chip — which is exactly why the live answer lives in the panel header. At
430×932 it opens below; at 1100×900 above.

**Rejected in review, with the reason each time** — native `<input type=date|time>` (44px box and 16px
type beside 32px chips, UA chrome); 32px pickers on the *field* plane at 44px (the size difference, once
more); dropdown lists of days and times ("you use dropdown for datetime?"); an inline month calendar plus
time wheel (470px block; 44px *visible* boxes where the app's rule is that hit area is not visual size;
three competing selection languages; nested vertical scroller); per-segment arrow steppers (ten targets,
repeated tapping for distant dates); horizontal chip strips (a month becomes a 1,800px fling; clipped
half-chips read as glitches).

## 4. Drag

### 4.1 Gesture map

```
source  │ within Now            │ Now header      │ Later block/header     │ Waiting
────────┼───────────────────────┼─────────────────┼────────────────────────┼────────
Now     │ reorder               │ move to top     │ snooze indefinitely    │ refuse
Later   │ —                     │ wake, at slot   │ cancel                 │ refuse
```

- Insertion inside Now is per-card halves (upper half → above, lower half → below).
- The block header is the guaranteed drop target, and while dragging, an empty opposite block gets its
  header rendered so Now→Later is always possible. The drag module owns that ephemeral node and
  removes it on drag end; `app_sessions.render()` runs after the flush (`app_sessions.js:165-172`).
- Later keeps recency order; Later→Later is a no-op. Launch-attempt rows are not draggable and keep
  sorting at the top by score (`session_listing.py:619-628`).

### 4.2 Offset for the dragged row (your rule)

On drop, the dragged row `D` sits between `A` (above) and `B` (below). The server computes, from the
same projection the listing uses:

```
sa = score(A), sb = score(B)          # final_priority at this instant
target = (sa + sb) / 2                # between two neighbours
       = sb + 0.005                   # dropped at the top of Now
       = sa - 0.005                   # dropped at the bottom of Now
offset_D = round(target - tp_D, 4)    # tp_D = time_priority of the dragged row right now
```

- Only `D` is written. No other row's offset changes, so nothing else moves.
- Clamped into `[-1, 1]` by the existing cleaner; if the clamp bites (target outside
  `[tp_D - 1, tp_D + 1]`) the row lands at the nearest legal score. That plus the exact-1.0 tie in §1
  are the only two ways a drop can miss the requested slot, and both are reported by the response
  (`applied: false`), so the UI can say so instead of appearing broken.
- Server-side rather than client-side so the arithmetic uses the same clock and bucket as the sort;
  it also means the client sends ids, not numbers.

### 4.3 Pointer machinery

`codoxear/static/app_sidebar_drag.js`, one gesture session:

- Touch/pen: 380ms still hold (movement < 8px) — the sidebar scrolls and `.session` declares
  `touch-action: pan-y` (`app.css:372`), so a still hold is the only activation that cannot be a
  scroll. Mouse: 6px with `|dy| > |dx| * 1.2`. The horizontal swipe keeps its own gate
  (`app_sessions.js:112-116`); the two are exclusive by geometry.
- Non-passive `touchmove` guard added at activation (`preventDefault` on `pointermove` does not stop
  scrolling).
- Refresh during a drag reuses the swipe deferral (`refreshDeferred`), flushing after commit, so a poll
  cannot destroy the dragged node.
- Geometry from the live DOM at activation: `#sessions` (the scroll container, `app.css:258-264`) and
  card rects, each card's group from the nearest preceding `.sessionGroupHeader`. No second copy of the
  session list.
- Auto-scroll within 48px of the container edges (6–14px/frame, rAF, stops with the drag).
- The card holds its dropped position until the authoritative snapshot re-renders; no local order
  cache, no second sort. Failure → revert + toast with the error. A completed drag swallows the
  following click so the drop never selects the session.

### 4.4 Visuals (tokens only, shown in the preview)

- Dragged card: `border-color: var(--ink)`, `box-shadow: var(--shadow-pop)` (paper's is `none`),
  `cursor: grabbing`, measured width, lifted.
- Insertion point: 2px `var(--ink)` line, `--radius-pill`, inside `#sessions`.
- Drop target header: `data-drop="snooze"` → `var(--wash)` fill + ink border, label
  `Later · drop to snooze indefinitely`; `data-drop="wake"` → `Now · drop to wake`;
  `data-drop="refuse"` (Waiting) → `opacity .5`, `cursor: not-allowed`.
- No backdrop, no dimming, nothing reflows under the pointer, nothing animated under
  `prefers-reduced-motion`.

---

## 5. API

- `POST /api/sessions/<id>/snooze` — `{"indefinite": true|false}`. One field, one coordinator method,
  one `save_sidebar_meta()`. Used by the Edit dialog chip and by both cross-block drags.
- `POST /api/sessions/<id>/reorder` — `{"above": sid|null, "below": sid|null}`. Writes one
  `priority_offset` (§4.2) and answers `{ok, priority_offset, applied}`.

Both are authenticated like every other control route. Later→Now sends `reorder` first (while the row
is still in Later, so it appears already in the right slot) and then `snooze {indefinite:false}`.
`/edit` keeps its whole-row contract unchanged.

---

## 6. Tests

| File | Pins |
| --- | --- |
| `tests/test_snooze_indefinite.py` (new) | cleaner mutual exclusion; sparse keys; prune ignores an indefinite snooze; listing `snoozed`; diagnostics parity; remap carries the key |
| `tests/test_snooze_routes.py` (new) | `/snooze` set + clear, 401, unknown session 404, atomic sidecar write |
| `tests/test_reorder_offset.py` (new) | midpoint/top/bottom targets; only the dragged row's offset changes; result order via `listing_priority` + `sort_session_rows`; clamp path reports `applied: false`; the exact-1.0 neighbour tie |
| `tests/test_snooze_presets.py` (extend) | presets unchanged; `nextSnoozeDays`/`snoozeTimeSlots` local and DST-safe |
| `tests/test_sidebar_drag_module.py` (new, VM) | `classifyDrop` truth table from §4.1; insertion-index math (above first, below last, header, refuse zone, single card, source=Later); payload builder; same-index no-op |
| `tests/test_frontend_session_edit_module.py` (extend) | indefinite chip mapping on open/save; panel open/close per §3; day tap keeps the panel, time tap closes it; period tap commits nothing; Today clamp |
| `tests/test_dialog_menu_placement.py` (new) | `data-fit-whole`: full `scrollHeight`, below → above → over, and unflagged menus unchanged |
| `tests/test_snooze_panel_layout.py` (new, VM) | no scroll container inside the panel; header not truncated at 360 with the widest real answer; day/period/slot hit boxes ≥44 in both axes |
| `scripts/docker_sidebar_drag_verify.sh` (new) | real server + broker: synthesized pointer sequence reorders Now and the next `GET /api/sessions` agrees; Now→Later sets `snoozed_indefinitely`; Later→Now wakes at the dropped slot; tmux order follows |

`tests/test_sidebar_render_order.py` is untouched — `sort_session_rows` does not change.

Real-browser acceptance before calling it done: one touch drag at `set device "iPhone 15 Pro"` and one
desktop drag; the synthesized-pointer harness proves our handler, not the platform's arbitration.

---

## 7. Phasing

1. **Indefinite snooze** — store/listing/diagnostics + `/snooze` + the chip.
2. **Custom picker** — `design-a4/` markup and CSS, the `data-fit-whole` branch in
   `app_dialog_menu.js`, `composedPath()` click-away, `app_snooze.js` month/period helpers, and the
   dialog's panel wiring.
3. **`/reorder`** — the offset computation, tests, `curl` a permutation and check
   `GET /api/sessions` plus the tmux window order.
4. **Drag gesture** — `app_sidebar_drag.js`, `app_sessions.js` integration, the §4.4 rules.
5. **Docs** — one paragraph under "Sidebar order is server-owned" in `AGENTS.md`: a drop writes the
   dragged row's `priority_offset` for that instant, the two boundaries in §1/§4.2, and the drag
   refresh-deferral contract.

---

## 8. Risks

- **iOS gesture arbitration**: a still hold means nothing has scrolled yet; the non-passive `touchmove`
  guard is the standard fix. If a real device still fights it, the fallback is a small grip handle —
  decided by the device test.
- **The two boundaries** (§1, §4.2) are visible: a drop above a row pinned at exactly 1.0 falls to the
  tiebreak, and a target outside the offset range is clamped. Both are reported, not hidden.
