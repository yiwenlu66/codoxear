# Sidebar snooze + drag — feature plan (v2)

Status: plan only, nothing implemented. v2 replaces v1's manual-rank idea with your rule: a drag
computes `priority_offset` so the row lands at the desired position in that instant.

---

## 1. What decides the shapes

**The date row.** Three mechanisms stack, so it can't be restyled into place:

- `.formViewer input[type=date|time]` is forced to `height: var(--ctl)` = 44px (`app.css:3212-3216`)
  while every snooze chip is `--dialog-control-h` = 32px (`app.css:3325-3331`).
- On coarse pointers `input[type=date|time]` takes `font-size: var(--font-xl) !important` — the iOS
  anti-zoom floor (`app.css:4095-4106`) — while chip labels stay at 13/14px. That is the visibly
  larger number in your screenshot, and it cannot be lowered without reintroducing focus zoom.
- The rest is UA chrome: inner segments, calendar indicator, platform padding, on a
  `minmax(0,1fr) 140px` grid (`app.css:3755`).

So the custom control is rebuilt from the dialog's own picker primitive (§3), not restyled.

**The score.** `base_priority = clip01(time_priority + priority_offset)`, `offset ∈ [-1, 1]`
(`session_listing.py:153`, `session_cleaners.py:41-43`), `time_priority` decaying with an 8h half-life
and a 30s bucket (`server_config.py:349-350`). Two facts that the drag arithmetic must respect:

- Clipping: any row whose `tp + offset ≥ 1` sits at exactly 1.0, so its position is decided by the
  `updated_ts` tiebreak, not by its offset.
- The decay is multiplicative per row, so gaps between rows are not conserved.

Both are handled by writing a **whole-block ladder** (§4.2) rather than one row's offset.

**The snooze field.** `snooze_until` means "a finite timestamp or nothing" in the row contract, the
expiry prune (`session_store.py:358`) and the `snoozed` predicate. It cannot also mean "forever".

---

## 2. Feature A — indefinite snooze

A second, sparse key. Not a sentinel timestamp, not `math.inf` (`json.dumps(float("inf"))` emits
`Infinity`, which `JSON.parse` rejects), not `-1` in a float field.

```
sidebar_meta[sid] = {
  "priority_offset": float,
  "snooze_until": float,          # only for a timed snooze
  "snoozed_indefinitely": true,   # only for an indefinite snooze
  "dependency_session_id": str,
}
```

Mutual exclusion is enforced in the one place that already rejects a self-dependency
(`session_ui_state._validated_sidebar_entry`); everything downstream uses

```
snoozed = snoozed_indefinitely or (snooze_until is not None and snooze_until > now_ts)
```

Call sites: `session_cleaners`, `session_store.SidebarSessionState` + `sidebar_state_for_session`
(the prune never touches an indefinite snooze), `session_ui_state`, `session_listing` facts + row,
`diagnostics_routes` (recomputes `snoozed` itself), `launch_ledger` default row.
`session_snapshot_runtime.remap_session_identity` moves the whole meta dict, so a resume carries the
new key for free — pin that with a test.

UI: a seventh chip between **Next week** and **Custom…** (`[ Indefinitely ]`); `openEditSession` maps a
stored indefinite snooze to it; save sends `snoozed_indefinitely: true` and no `snooze_until`. The chip
row already wraps (`app.css:3546`), so no layout work.

`POST /api/sessions/<id>/edit` becomes **partial** (absent field = leave unchanged; `name` absent keeps
the alias; `snooze_until: null` still clears). The gesture writes must never be able to clobber a name
or offset edited in another tab. `snoozed_indefinitely: true` plus a non-null `snooze_until` → 400.

---

## 3. Feature B — custom date/time control

Two `dialogPickerBtn` buttons + two `dialogPickerMenu` menus — the primitive the "Depends on" field
already uses (`app_dialog_menu.js:39`):

```
Custom…
[ Wed 12 Nov ▾ ]   [ 9:00am ▾ ]
```

- Field plane (44px, `.formViewer .dialogPickerBtn`), type from `--font-lg` (the dialog retunes it to
  16px on coarse pointers), no UA chrome, and no zoom because a `button` is not an entry field. This
  is the same plane as the name input and the dependency picker; the thing that reads as broken today
  is a *third* widget with its own type size, corner treatment and indicator.
- Row: `minmax(0,1fr) minmax(0,1fr)` with `gap: var(--space-2)`.
- Date menu: next 28 days as `.fileMenuItem` rows (`Today · Thu 13 Nov`, `Tomorrow · Fri 14 Nov`,
  `Sat 15 Nov`, …), current value `.active`. Fits the menu's existing `max-height: min(44vh, 360px)`
  and `positionDialogMenu` already flips it above the field.
- Time menu: 30-minute slots, current value marked and scrolled into view on open.
- Pure helpers in `app_snooze.js` (`nextSnoozeDays(count, nowMs)`, `snoozeTimeSlots()`) so the list
  construction is testable without a DOM and inherits the local-`Date` arithmetic the presets use.
- Save: `new Date(y, m, d, hh, mm)` → epoch seconds (today's `Date.parse("YYYY-MM-DDTHH:MM")`
  produced the same).

---

## 4. Feature C — drag

### 4.1 Gesture map

```
source  │ within Now            │ Now header/block │ Later block/header     │ Waiting
────────┼───────────────────────┼──────────────────┼────────────────────────┼────────
Now     │ reorder               │ move to top      │ snooze indefinitely    │ refuse
Later   │ —                     │ wake, at slot    │ cancel                 │ refuse
```

- Insertion inside Now is per-card halves (upper half → above, lower half → below).
- Later→Now lands the row at the slot you dropped it into; Now→Later has no slot semantics.
- The block header is the guaranteed drop zone: while dragging, the opposite block's header is
  rendered if the block is empty (otherwise Now→Later is impossible with nothing snoozed yet). The
  drag module owns that ephemeral node and removes it on drag end; `app_sessions.render()` runs after
  the flush (`app_sessions.js:165-172`), so the container never has two writers.
- Later stays recency-ordered; Later→Later is a no-op.
- Launch-attempt / orphan-recovery rows are not draggable and are not part of the payload; they are
  unranked rows with `final_priority: 1.0`, so they keep sitting at the top of Now as they do today
  (`session_listing.py:619-628`).

### 4.2 The offset ladder (your rule, made exact)

A drop posts the desired Now order. The server recomputes `priority_offset` for **every** Now row so
that the resulting scores are one descending ladder in that instant:

```
TOP    = 0.99                     # a freshly active row (tp = 1.0) still outranks the whole ladder
BOTTOM = 0.02                     # keep the last rung off 0 so nothing ties with a dead score
step   = (TOP - BOTTOM) / max(1, N - 1)
target_i = TOP - i * step         # i = position in the dropped order
offset_i = round(target_i - tp_i, 4)    # tp_i from the same projection the listing uses
```

- `offset_i ∈ [-0.98, 0.99]` for every possible `tp ∈ [0,1]`, so it is always a legal offset and never
  needs clamping. `final_priority = clip01(tp_i + offset_i) = target_i`, because targets are < 1.
- Targets are distinct by `step`, and 30s-bucket jitter moves a score by at most 0.0007, so the order
  is exact for any realistic block size.
- Why the whole block and not just the dragged row: a midpoint between two neighbours has no room when
  both neighbours are clamped at 1.0 (that is exactly the case you hit — the top of Now is usually a
  saturation cluster), and repeated drops between the same pair exhaust the gap. Re-laddering the
  block has room for any number of rows, and it heals drift left by earlier drags at the same time.
- Why `TOP = 0.99` and not 1.0: a fully fresh row scores 1.0, so the ladder's ceiling sits just under
  freshness and nothing else has to be special-cased. What activity does to a rung follows directly
  from `score = tp + offset`: a message restores up to `1 - tp_at_placement` of score. A row you pushed
  down while it was stale (its `tp` was near 0) therefore clips to 1.0 on its next message and comes
  back to the top; a row you demoted straight after it had been active keeps the rung you gave it.
  Your ordering is not silently reverted by activity — only your own drags change it.
- Rounding offsets to 4dp is safe: rungs differ by `0.97 / (N-1)`, which stays far above 1e-4 for any
  realistic block.
- Behaviour over hours, stated once and not defended: rungs are absolute while decay is per-row, so a
  young row placed below an older one closes the gap and can eventually cross it. Any later drag
  re-ladders the whole block and heals it. You asked for the desired position in that instant; that
  is what this delivers.
- Nothing else changes: `sort_session_rows` keeps today's key, the client keeps rendering server
  order, no new store field, no rank validity rule, no `tmux_order_sync` change.

Side effect worth knowing: the Edit dialog's slider will now show ladder values (`+0.97`, `+0.71`,
`+0.45`). Those are the real offsets; dragging it by hand still works as before.

### 4.3 Pointer machinery

One gesture session in a new module `codoxear/static/app_sidebar_drag.js`.

- Activation: touch/pen = 380ms still hold (movement < 8px); mouse = 6px with `|dy| > |dx| * 1.2`.
  The sidebar scrolls and `.session` declares `touch-action: pan-y` (`app.css:372`), so a still hold is
  the only activation that cannot be a scroll. The horizontal swipe keeps its own gate
  (`app_sessions.js:112-116`) and is exclusive by geometry.
- Scroll suppression: a non-passive `touchmove` listener added at activation (`preventDefault` on
  `pointermove` does not stop scrolling).
- Refresh during drag reuses the swipe deferral (`refreshDeferred`), so a poll can't destroy the
  dragged node; it flushes after commit.
- Geometry comes from the live DOM at activation: `#sessions` (the scroll container,
  `app.css:258-264`) and the card rects, each card's group taken from the nearest preceding
  `.sessionGroupHeader`. The module keeps no second copy of the session list.
- Auto-scroll within 48px of the container's top/bottom edge, 6–14px per frame, rAF loop that stops
  with the drag.
- Commit holds instead of optimistically reordering: the card stays where you dropped it until the
  authoritative snapshot renders the block. No local order cache, no second sort. Failure → snap back
  + toast with the error.
- A completed drag swallows the click that follows the release, so a drop never selects the session.

### 4.4 Visuals (tokens only)

- Dragged card: `position: fixed`, measured width, `border-color: var(--ink)`,
  `box-shadow: var(--shadow-pop)` (paper's is `none`), `cursor: grabbing`.
- Insertion point: 2px `var(--ink)` line, `--radius-pill`, absolutely positioned in `#sessions`.
- Drop zone on the header/region: `data-drop="snooze"` → `var(--wash)` fill + ink border, label
  `Later · drop to snooze indefinitely`; `data-drop="wake"` → `Now · drop to wake`;
  `data-drop="refuse"` (Waiting) → `opacity .5`, `cursor: not-allowed`.
- No backdrop, no dimming, nothing that reflows under the pointer, nothing animated under
  `prefers-reduced-motion`.
- Desktop affordance: a small grip inside the existing hover-reveal group (`:hover` /
  `:has(… :focus-visible)`, never `:focus-within`); the press target stays the whole card.

### 4.5 Keyboard

`Alt+↑` / `Alt+↓` on a focused card moves it one slot in Now and posts the same payload. `Alt` is free
— the direct-shortcut handler bails on `altKey` (`app_transcript_render.js:303-305`). Cross-block moves
stay in the Edit dialog, so nothing is drag-only.

### 4.6 Failure and races

- Unknown ids in `order` (deleted mid-drag): dropped from the ladder, reported as `ignored`, gesture
  succeeds and the refresh reconciles.
- Unknown id in `snooze`/`wake`: whole request fails 404, nothing written, drag reverts, toast names it.
- Duplicate id, or a session in both `order` and `snooze`: 400.
- Two clients dragging at once: the last whole-block write wins.

---

## 5. Server API

**`POST /api/sidebar-layout`** (new; authenticated; matched in `handle_session_post_route`'s
non-session branch next to `/api/quick-sessions`):

```json
{ "order": ["sid1", "sid2", ...],   // desired Now order → one ladder write for the whole block
  "snooze": ["sidX"],               // indefinite snooze
  "wake": ["sidY"] }                // clear snooze
```

One coordinator method, one lock, one `save_sidebar_meta()`:

1. `order` → `reorder_offsets(rows_in_payload_order + any other current Now row appended, now_ts)`
   → `priority_offset` per row (§4.2).
2. `snooze` → `snoozed_indefinitely: true`, `snooze_until` deleted. No ladder rewrite needed: removing
   a row doesn't disturb the remaining rungs.
3. `wake` → both snooze keys deleted; the row's position comes from the same request's `order`.
4. Respond `{ok, ignored, sessions: {sid: {priority_offset, snooze_until, snoozed_indefinitely,
   dependency_session_id}}}`.

`reorder_offsets` lives in `session_listing.py` (the module that owns priority math) as a pure
function of `(rows, now_ts, half_life, bucket)`, so the ladder is unit-testable without a server.

The client sends only ids in render order with the dragged id inserted at the drop index; it computes
no offsets. That keeps the arithmetic next to the sort it has to satisfy, and immune to the
client/server clock or bucket skew.

---

## 6. Tests

| File | Pins |
| --- | --- |
| `tests/test_sidebar_reorder_offsets.py` (new) | ladder is exact (feed the result back through `listing_priority` + `sort_session_rows` and assert the requested order); offsets always in `[-1,1]`; N=0/N=1; a saturated top row still reorders; a row with `tp = 1.0` dragged to the top and to the bottom; order survives a 30s advance; a stale-demoted row clips back to the top on activity while a fresh-demoted row keeps its rung |
| `tests/test_snooze_indefinite.py` (new) | cleaner mutual exclusion; sparse keys; prune ignores an indefinite snooze; listing `snoozed`; diagnostics parity |
| `tests/test_control_routes.py` (extend) | `/edit` partial body leaves other fields alone; indefinite round trip; layout: 401, 400 duplicate/overlap, 404 unknown subject, `ignored` for unknown order ids, offsets visible on the next `GET /api/sessions` |
| `tests/test_snooze_presets.py` (extend) | presets unchanged; `nextSnoozeDays` / `snoozeTimeSlots` are local and DST-safe |
| `tests/test_sidebar_drag_module.py` (new, VM) | `classifyDrop` truth table from §4.1; insertion-index math (above first, below last, on header, refuse zone, single card, source=Later); payload builder; same-index no-op |
| `tests/test_frontend_session_edit_module.py` (extend) | indefinite chip mapping on open/save; picker value round trip |
| `scripts/docker_sidebar_drag_verify.sh` (new) | real server + broker: synthesized pointer sequence reorders Now and the next `GET /api/sessions` agrees (including a saturated top row); Now→Later sets `snoozed_indefinitely`; Later→Now wakes at the dropped slot; tmux windows follow; screenshots at 390px and desktop |

`tests/test_sidebar_render_order.py` is untouched — the sort is unchanged, which is the point of this
design.

Real-browser acceptance before calling it done: one `agent-browser` touch drag at
`set device "iPhone 15 Pro"` and one desktop drag. The synthesized-pointer harness proves our handler,
not the platform's gesture arbitration.

---

## 7. Phasing

1. **Indefinite snooze** — cleaners/store/ui_state/listing/diagnostics/launch_ledger, partial `/edit`,
   chip. Look at: saves, lands in Later, survives a restart, never expires.
2. **Custom pickers** — `app_snooze.js` helpers, shell markup, controller wiring, CSS. Look at: the
   dialog in three families × light/dark at 390px and desktop.
3. **Ladder + endpoint, server only** — `reorder_offsets`, `/api/sidebar-layout`, tests. Look at:
   `curl` a permutation, then `GET /api/sessions` and the tmux window order.
4. **Drag gesture** — `app_sidebar_drag.js`, `app_sessions.js` integration, CSS, Alt+↑/↓. Look at: the
   docker verifier plus a real phone drag.
5. **Docs** — one paragraph under "Sidebar order is server-owned" in `AGENTS.md`: the drag writes a
   whole-block offset ladder with a 0.99 top rung, the sort key is unchanged, and the gesture route is
   `/api/sidebar-layout`.

---

## 8. Risks

- **iOS gesture arbitration**: a still hold means nothing has scrolled yet, and the non-passive
  `touchmove` guard is the standard fix. If a real device still fights it, the fallback is a grip
  handle (smaller region, no long press) — decided by the device test.
- **Decay drift** over hours (§4.2): accepted as instant semantics, healed by the next drag.
- **tmux follow**: one drag → one listing generation → one coalesced `tmux_order_sync` pass. Window
  order follows the sidebar, as it already does for every other order change.

---

## 9. Open question

Only one: should a snoozed card in **Later** show its state (`until Fri 9am` / `indefinitely`), or stay
silent as today? Everything else is settled; I'll flag at build time if the device test contradicts the
iOS assumptions in §4.3.
