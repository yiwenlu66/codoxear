# Sidebar snooze + drag ordering — feature plan

Status: plan only. Nothing below is implemented. §9 lists the four decisions I need from you.

---

## 1. Three facts from the current code that decide the design

**(a) The "ugly" date row is not a styling accident — it is three mechanisms stacked.**

- `app.css:3212-3216` forces `.formViewer input[type=date|time]` to `height: var(--ctl)` = **44px**,
  while every snooze chip is `.formViewer .choiceChip` at `--dialog-control-h` = **32px**
  (`app.css:3325-3331`). Two different vertical planes inside one field.
- `app.css:4095-4106` applies `font-size: var(--font-xl) !important` (**16px**) to
  `input[type=date|time]` on coarse pointers — the iOS anti-zoom floor — while chip labels stay at
  `--font-md` (13px). This is why the numbers in your screenshot are visibly larger than
  "This weekend".
- The rest is UA chrome: inner field segments, the calendar indicator, and the platform's own
  padding/box, on a `minmax(0,1fr) 140px` grid (`app.css:3755`).

Consequence: restyling the native widgets cannot converge them onto the dialog's type plane,
because the 16px floor on a text-entry input is `!important` for a real reason. A date control that
belongs to the dialog has to stop being an `<input type=date>`.

**(b) `priority_offset` cannot express an order.** `base_priority = clip01(time_priority + offset)`
(`session_listing.py:153`) with `offset ∈ [-1, 1]` (`session_cleaners.py:41-43`). Two consequences:

- *Saturation*: an old session dragged above a saturated row (score exactly 1.0) is unreachable —
  the dragged row clamps to 1.0 and loses the tie to the newer `updated_ts`. Saturation is not rare:
  any row with `offset ≥ 1 - time_priority` sits at 1.0 for hours.
- *Drift*: the recency term decays (8h half-life, 30s bucket, `server_config.py:349-350`). A
  constant offset added to a decaying score changes its relative position over time. Worked example
  in §4.2: a young row dragged *below* an older row inverts within ~16 hours.

So the drag must not write `priority_offset`. It needs its own ordering channel. The slider keeps
its current meaning.

**(c) `snooze_until` has no way to say "forever", and one `snooze_until` value cannot carry two
meanings without lying about its type.** The row contract, the prune (an expired stamp is deleted at
`session_store.py:358`) and the `snoozed` predicate all assume "a finite timestamp or nothing".

---

## 2. Feature A — indefinite snooze

### Representation

Store a second, sparse key. Never a sentinel timestamp, never `math.inf` (it serializes to invalid
JSON), never `-1` smuggled into a float field.

```
sidebar_meta[sid] = {
  "priority_offset": float,
  "snooze_until": float,            # present only for a timed snooze
  "snoozed_indefinitely": true,     # present only for an indefinite snooze
  "dependency_session_id": str,
  "manual_order": float,            # §4
  "manual_order_ts": float,         # §4
}
```

Mutual exclusion is validated in exactly one place — the same `_validated_sidebar_entry` that
already rejects a self-dependency:

```
snoozed = snoozed_indefinitely or (snooze_until is not None and snooze_until > now_ts)
```

Call sites that must learn the new flag: `session_cleaners.clean_snoozed_indefinitely`,
`session_store.SidebarSessionState` + `sidebar_state_for_session` (the prune never touches an
indefinite snooze), `session_ui_state._validated_sidebar_entry` / `edit_session`,
`session_listing` facts + row, `diagnostics_routes` (it recomputes `snoozed` itself),
`launch_ledger` default row. `session_snapshot_runtime.remap_session_identity` moves the whole meta
dict, so the new keys inherit resume correctness with no code change (pin that with a test).

### UI

A seventh chip in the existing row, between **Next week** and **Custom…**:

```
[ No snooze ] [ This evening ·6pm ] [ Tomorrow ·9am ]
[ This weekend ·Sat 10am ] [ Next week ·Mon 9am ] [ Indefinitely ] [ Custom… ]
```

- `openEditSession` maps a stored indefinite snooze to that chip.
- Save sends `snoozed_indefinitely: true` and no `snooze_until`.
- The chip row already wraps (`app.css:3546`), so no layout work.
- No card badge. The Later block plus the Edit dialog carry the state; adding a badge is a separate
  visual decision (question 3 in §9).

### Wire format

`POST /api/sessions/<id>/edit` becomes **partial**: an absent field means "leave unchanged"
(`name` absent today is an error; it becomes "keep the alias", `snooze_until: null` still means
"clear"). This is needed beyond this feature — every gesture write stops being able to clobber a
name/offset the user edited concurrently. The dialog always sends every field, so its behaviour is
unchanged. `snoozed_indefinitely: true` together with a non-null `snooze_until` is a 400.

---

## 3. Feature B — custom date/time control that belongs to the dialog

**Replace the two native inputs with two dialog picker buttons and two menus**, reusing the
primitive the "Depends on" field already uses (`dialogPickerBtn` + `dialogPickerMenu` +
`positionDialogMenu`, `app_dialog_menu.js:39`).

```
Custom…
[ Wed 12 Nov ▾ ]  [ 9:00am ▾ ]
```

- Buttons sit on the dialog *field* plane (44px, `.formViewer .dialogPickerBtn`) — the same plane as
  the name input and the dependency picker, so the field stops looking like a foreign widget and
  starts looking like the dialog's own control. Type comes from the field plane (`--font-lg`, which
  the dialog retunes to 16px on coarse pointers, same as the anti-zoom floor), with no UA chrome and
  no iOS zoom, because a `button` is not an entry field and the 16px `!important` rule does not
  apply to it.
  Two planes in one dialog (32px chips for choices, 44px fields for values) is already the dialog's
  existing hierarchy and reads as hierarchy; what reads as a defect today is a *third*, foreign
  widget with its own type size, chrome and corner treatment.
- Row: `grid-template-columns: minmax(0,1fr) minmax(0,1fr); gap: var(--space-2)` (equal halves, no
  more `1fr 140px`).
- **Date menu**: the next 28 days as `.fileMenuItem` rows — `Today · Thu 13 Nov`,
  `Tomorrow · Fri 14 Nov`, then `Sat 15 Nov`, … — with the current value marked `.active`. 28 rows
  scrolls inside the menu's existing `max-height: min(44vh, 360px)` and `positionDialogMenu` already
  flips it above the field when the space below is short.
- **Time menu**: 30-minute slots `00:00 … 23:30`, current value marked and scrolled into view on open.
- Pure helpers land in `app_snooze.js` (`nextSnoozeDays(count, nowMs)`, `snoozeTimeSlots()`), so the
  list construction is unit-testable without a DOM, and the DST behaviour is inherited from the
  local-`Date` arithmetic the presets already use.
- Save path unchanged in shape: `new Date(y, m, d, hh, mm)` → epoch seconds, i.e. exactly what
  `Date.parse("YYYY-MM-DDTHH:MM")` produced before.

Rejected: (i) keep native inputs, clamp CSS — the 16px `!important` floor keeps the type mismatch on
phones; (ii) `<input type="datetime-local">` — one field, same chrome, worse; (iii) a month-grid
calendar — 7 columns cannot reach the 44px touch target inside a 366px phone dialog, and a 28-row
list already covers every snooze horizon this UI needs.

---

## 4. Feature C — drag: reorder, snooze, wake

### 4.1 Gesture map

```
        ┌─────────────────────────── drop zone → action ────────────────────────────┐
source  │ within Now (any slot)      │ on Now header/block │ on Later block/header │ Waiting │
────────┼────────────────────────────┼─────────────────────┼───────────────────────┼─────────┤
Now     │ reorder (rank write)       │ reorder to index 0  │ snooze indefinitely   │ refuse  │
Later   │ — (no reorder in Later)    │ wake + land at slot │ cancel (no-op)        │ refuse  │
────────┴────────────────────────────┴─────────────────────┴───────────────────────┴─────────┘
```

- Insertion inside Now is per-card halves: pointer in a card's upper half → insert above it, lower
  half → insert below it.
- Cross-block drops have **no slot semantics in the destination block** except one: Later→Now lands
  the row at the slot you dropped it into (it is being ranked in the same request, §5).
- Dropping anywhere else (chat area, sidebar chrome, below the list) cancels.
- **The target block's header is the guaranteed drop zone.** While a drag is active, the opposite
  block's header is rendered if it does not exist (an empty Later block has no header, which would
  otherwise make Now→Later impossible). The ephemeral header is owned by the drag module and removed
  on drag end; `app_sessions.render()` runs strictly after that (the existing `refreshDeferred` flush
  already fires on gesture close, `app_sessions.js:165-172`), so the container never has two writers
  at once.
- Later is recency-ordered and stays that way; a Later→Later drag is a deliberate no-op.
- Non-session rows (launch attempts, orphan recovery) are not draggable and are excluded from the
  `order` payload. They never carry a rank, so they keep sorting at the top of Now by score — exactly
  where they sort today (`final_priority: 1.0`, `session_listing.py:619-628`).

### 4.2 The ordering contract (the load-bearing decision)

Within **Now only**, the sort key grows two leading tiers:

```
manual_rank := the value below, or None for any row that is not in the Now block

(group_rank,                                     # unchanged: now/waiting/later
 0 if manual_rank is None else 1,                # within Now: unranked floats above the manual list
 manual_rank if manual_rank is not None else 0.0, # ascending
 -final_priority, -updated_ts, -start_ts, sid)   # today's chain, unchanged
```

with

```
manual_rank(row) = manual_order
    if manual_order is a number
       and manual_order_ts is a number
       and updated_ts <= manual_order_ts
    else None
```

and a session that leaves Now (snoozed, or blocked by a dependency) has its rank pruned where the
store already prunes (`sidebar_state_for_session`), so ranks exist only for rows in the last-written
Now order.

Why this shape:

- **Exact at drop time.** The drag posts the whole Now order; the server writes dense ranks. No
  arithmetic against a decaying score, no clipping, no unreachable slot.
- **Stable.** A rank is a position, not a boost. It does not decay, so the order you set holds.
- **Recency still wins on new information.** A row whose `updated_ts` advances past `manual_order_ts`
  loses its rank and floats to the top of Now by score. That is the same promise the sidebar makes
  today ("the thing that just answered is at the top"), and it is what makes the manual order safe
  to use: it can never bury new content. A brand-new session is unranked by construction and also
  appears at the top.
- **Zero effect until used.** A sidebar that has never been dragged has no ranks at all, so all rows
  sit in tier 0 and the key reduces to today's key exactly.

Worked example of why `priority_offset` was rejected (§1b): Now = A (age 1 min, tp 1.0), B (age 3h,
tp 0.77), C (age 2d, tp 0.0016). Dragging C above A needs `score_C > 1.0` — impossible while the
score clips, and a normalized ladder (C 0.99 / A 0.90 / B 0.81) quietly re-sorts when a young row is
placed *below* an older one: `score_A - score_B` starts at +0.09 and reaches −0.08 by hour 16, because
A's recency term loses 0.5 while B's loses 0.23. A constant offset plus a decaying term cannot hold
an order.

### 4.3 Pointer machinery

One gesture session, owned by a new module `codoxear/static/app_sidebar_drag.js`.

- **Activation.** Touch/pen: 380 ms still hold (movement < 8px). Mouse: 6px movement with
  `|dy| > |dx| * 1.2`. Horizontal intent keeps belonging to the existing swipe:
  `bindSwipe`'s axis gate (`app_sessions.js:112-116`) already refuses a vertical drag, and a still
  hold never engages a swipe, so the two gestures are exclusive by geometry — no coordination code,
  just a test that pins it.
- **Scroll conflict.** `pointermove` + `preventDefault` does not stop scrolling; a non-passive
  `touchmove` listener added at activation does, because nothing has scrolled yet at that point.
  `.session` already declares `touch-action: pan-y` (`app.css:372`), which is what makes the still
  hold the right activation trigger in the first place.
- **Refresh during drag** reuses the swipe deferral: `app_sessions` reports
  `draggingSessionId != null` into the same `refreshDeferred` path, then flushes after commit. A poll
  must never destroy the dragged node.
- **Geometry.** Everything is measured from the live DOM at activation: `#sessions` (the scroll
  container, `app.css:258-264`) and the card rects, with each card's group taken from the nearest
  preceding `.sessionGroupHeader`. The module holds no second copy of the session list.
- **Auto-scroll** while the pointer is within 48px of the container's top/bottom edge, 6–14px per
  frame, on a rAF loop that stops with the drag.
- **Commit holds, it does not optimistically reorder.** The dropped card stays where you put it while
  the request is in flight; the authoritative snapshot then re-renders the block. On a LAN that is
  tens of milliseconds, and it keeps the single-writer rule (no local order cache, no second sort)
  intact. Failure → snap back + toast naming the error.
- **Click suppression.** A completed drag swallows the click that follows the release, so dropping a
  card never selects it.
- Horizontal swipe, vertical scroll, tap-to-select and hint mode all keep working unchanged.

### 4.4 Visuals (tokens only, no new vocabulary)

- Dragged card: `position: fixed`, measured width, `border-color: var(--ink)`,
  `box-shadow: var(--shadow-pop)`, `cursor: grabbing`; paper's `--shadow-pop: none` already means
  "no decorative shadow" there.
- Insertion point: a 2px `var(--ink)` line with `--radius-pill`, absolutely positioned inside
  `#sessions`.
- Drop zone states on the block header/region: `data-drop="snooze"` → `var(--wash)` fill + ink
  border, label becomes `Later · drop to snooze indefinitely`; `data-drop="wake"` → same treatment with
  `Now · drop to wake`; `data-drop="refuse"` (Waiting) → `opacity .5`, `cursor: not-allowed`.
- No backdrop, no dimming (design rule), nothing that shifts layout under the pointer, and no
  transition at all under `prefers-reduced-motion`.
- Desktop: hovering a Now card reveals a small grip in the existing hover-reveal group
  (`:hover` / `:has(... :focus-visible)`, never `:focus-within`), purely as an affordance — the press
  target stays the whole card.

### 4.5 Keyboard

`Alt+↑` / `Alt+↓` on a focused card moves it one slot inside Now and posts the same order payload.
`Alt` is free: the direct-shortcut handler bails out on `altKey` (`app_transcript_render.js:303-305`).
Cross-block moves from the keyboard stay in the Edit dialog (chips are reachable); no drag-only
capability is introduced.

### 4.6 Failure and race handling

- Unknown ids in `order` (the other tab deleted a session mid-drag): dropped from the payload,
  reported in the response as `ignored`, request succeeds. The follow-up refresh reconciles.
- Unknown id in `snooze`/`wake` (the gesture's own subject vanished): whole request fails with 404,
  nothing is written, the drag reverts and a toast says the session is gone.
- A duplicate id, or a session in both `order` and `snooze`: 400.
- Two clients dragging at once: the last whole-block write wins — stated, not silently merged.

---

## 5. Server API

**`POST /api/sidebar-layout`** (new; authenticated like every other control route; matched in
`handle_session_post_route`'s non-session branch, next to `/api/quick-sessions`):

```json
{ "order": ["sid1", "sid2", ...],   // authoritative Now order, may be []
  "snooze": ["sidX"],               // set indefinite snooze (and drop its rank)
  "wake": ["sidY"] }                // clear any snooze (rank comes from "order")
```

One coordinator method, one lock, one `save_sidebar_meta()`:

1. `order` → `manual_order = index`, `manual_order_ts = now`; **every other live session loses
   `manual_order`/`manual_order_ts`** (the payload is authoritative for the whole block, so no
   orphans and no colliding ranks).
2. `snooze` → `snoozed_indefinitely: true`, `snooze_until` deleted, rank cleared.
3. `wake` → both snooze keys deleted.
4. Respond `{ok, ignored, sessions: {sid: {priority_offset, snooze_until, snoozed_indefinitely,
   dependency_session_id}}}`.

Everything else routes through the existing `/edit` (partial, §2). The gesture uses
`/api/sidebar-layout` rather than two `/edit` calls because "snooze this row and re-rank the block" is
one user action: two calls would render a visible intermediate state and could half-apply.

The client computes `order` from what it rendered (Now rows in DOM order, dragged id inserted at the
drop index). It needs no knowledge of scores.

`manual_order` / `manual_order_ts` stay on the public listing row like `priority_offset` and
`snooze_until` do: they are two floats, they are what diagnostics should show, and hiding them would
force a second derived field.

---

## 6. Tests and verification

Behavioural only (no source-string assertions):

| File | What it pins |
| --- | --- |
| `tests/test_snooze_indefinite.py` (new) | cleaner mutual exclusion; sparse keys; store prune never touches an indefinite snooze; listing `snoozed` derivation; diagnostics parity |
| `tests/test_sidebar_manual_order.py` (new) | sort tiers (unranked above ranked, group rank still first, waiting/later never consult a rank); rank validity vs `updated_ts`; prune-on-leave; remap carries the keys |
| `tests/test_control_routes.py` (extend) | `/edit` partial body leaves untouched fields alone; indefinite round trip; layout: 401, 400 duplicate/overlap, 404 unknown subject, `ignored` for unknown order ids, atomic sidecar write, non-listed ranks cleared |
| `tests/test_snooze_presets.py` (extend) | presets unchanged; `nextSnoozeDays` / `snoozeTimeSlots` produce local, DST-safe sequences |
| `tests/test_sidebar_drag_module.py` (new, VM) | `classifyDrop` truth table from §4.1; insertion-index math (above first card, below last, on header, refuse zone, single card, source=Later); payload builder; same-index no-op |
| `tests/test_frontend_session_edit_module.py` (extend) | indefinite chip mapping on open/save; picker value round-trip |
| `scripts/docker_sidebar_drag_verify.sh` (new) | real server + broker: synthesized pointer sequence reorders Now and the next `GET /api/sessions` agrees; Now→Later sets `snoozed_indefinitely`; Later→Now clears it at the dropped slot; tmux windows follow the new order; capture screenshots at 390px and desktop, mouse and touch emulation |

Real-browser acceptance (not substitution) before calling it done: `agent-browser` at
`set device "iPhone 15 Pro"` with a real touch drag, plus one desktop drag — the synthesized-pointer
harness proves our handler, not the platform's gesture arbitration.

---

## 7. Phasing (each phase ends at a state you can look at)

1. **Indefinite snooze** — cleaners/store/ui_state/listing/diagnostics/launch_ledger, partial `/edit`,
   dialog chip. Look at: chip saves, row moves to Later, survives a server restart, never expires.
2. **Custom pickers** — `app_snooze.js` helpers, shell markup, controller wiring, CSS. Look at:
   dialog in all three families × light/dark at 390px and desktop.
3. **Ordering plumbing, server only** — facts/row fields, `manual_rank`, sort key, prune, layout
   endpoint. Look at: `curl` the endpoint, then `GET /api/sessions`, and the tmux window order.
4. **Drag gesture** — `app_sidebar_drag.js`, `app_sessions.js` integration, CSS, Alt+↑/↓.
   Look at: the docker verifier plus a real phone drag.
5. **Documentation** — one paragraph in `AGENTS.md` under "Sidebar order is server-owned" recording:
   the Now two-tier key, the rank-validity rule, the rank-cleared-on-leaving-Now rule, the
   layout endpoint as the gesture writer, and the drag/refresh deferral contract.

---

## 8. Risks

- **iOS gesture arbitration** (long-press + touchmove suppression inside a scrolling container).
  Mitigation: activation requires a still hold, so nothing has scrolled yet. If a real device still
  fights it, the fallback is a dedicated grip handle (smaller hit region, no long press) — decide
  from the device test, not from speculation.
- **`updated_ts` clock assumption.** Rank validity compares a server-written stamp against a
  backend-generated chat timestamp; both come from this machine. If a backend ever writes a future
  timestamp, that row loses its rank silently (it lands at the top). Acceptable, and visible.
- **tmux churn.** Order writes produce one listing generation → one coalesced `tmux_order_sync` pass,
  which is exactly what the mirror is for. Dragging a row in the browser will move a tmux window; that
  is the existing mirror contract, not a new one.
- **Frozen order for dormant rows.** After a reorder, rows that receive no activity keep their manual
  position instead of sinking with age. That is the point of a manual order, but it does change the
  default feel of a block that has been dragged once.

---

## 9. Decisions I need from you

1. **New reply on a manually placed row** → it floats to the top of Now until you reorder again
   (my recommendation; it keeps "just answered is on top" true). Alternative: your rank wins and the
   reply does not move the row.
2. **Drop into Later is always indefinite** — no timed variant from a drag. Confirm.
3. **Snooze state on the card**: show `until Fri 9am` / `indefinitely` in the Later block, or leave
   Later silent as it is today?
4. **`Indefinitely` placement**: its own chip in the row (my recommendation) or only reachable
   through Custom…?
