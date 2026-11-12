# Snooze · Custom… — design A: two chip strips

Mockup: `mockup.html` (real `app.css` + theme file, real dialog classes; `?theme=paper|slate&mode=dark`
switches family, `?only=a|b|c` renders one state). Shots: `shot-phone.png` (393 wide),
`shot-desktop.png` (1100 wide), `shot-families.png` (state (b) in clay/paper/slate × light/dark at 393).

## What is wrong with rejection #4 (`../preview-custom.png`)

- **It is a 470px block inside a 32px-chip field.** Calendar header 32 + weekday row ≈ 18 + six week rows ×
  44 = 264 + wheel 132 + gaps. The Snooze field grew from three chip rows to roughly 3× the height of the
  rest of the dialog, so the dialog on a phone became a calendar with a form attached.
- **It spends visible height on touch targets.** Day cells and wheel rows are 44px *visible* boxes; every
  other compact control in the dialog is a 32px box whose 44px target lives in `::after`. That broke the
  constitution's "hit area is not visual size" and is exactly why it reads taller than its neighbours.
- **Three selection languages in one field.** `Custom…` is ink-filled, the selected day is *also*
  ink-filled (two equal-weight black blocks competing for "the answer"), today is a 1px ring, and the wheel's
  chosen row is bold text on `--wash` between two rules. None of the latter three exists anywhere else in
  the dialog.
- **Day numbers are not buttons.** Bare digits with no border or fill, while every other tap target in the
  dialog is a bordered paper chip; 11 past days sit at 45% opacity as dead grid.
- **A second header plane.** `November 2026` is `--font-md` 600 — byte-for-byte the `.fieldLabel` style —
  so the field gets two labels; the weekday row is `--font-xs` uppercase letter-spaced, a type style used
  nowhere else in the app's dialogs.
- **The wheel's commit is implicit.** The value is "whatever row is centred when scrolling settles"; a
  half-scrolled wheel shows a row that is visually centred but not committed, and it is a vertical
  scroller nested inside the vertically scrolling `.formBody` (scroll trapping on phones).

## Design decisions

1. **Nothing new is drawn.** The revealed control is two more rows of the very `.choiceChip` the user just
   pressed: a **day strip** (`Today`, `Sat 14`, `Sun 15` … 60 days; the 1st of a month reads `Tue 1 Dec`)
   and a **time strip** (`12am` … `11:30pm`, 48 half-hours, formatted with `app_snooze.js`'s
   `compactLocalTime`, so `2pm` here looks exactly like `· 6pm` on the presets). Same 32px box, same 13px
   (14px coarse) type, same border, radius, paper fill. Paper squares them, slate pills nothing extra,
   clay rounds them — for free, because they *are* `.choiceChip`.
2. **One row each, scroll sideways.** Each strip is the base `.choiceChips` scroller (`nowrap`,
   `overflow-x: auto`) — the component other dialogs already use for chip rows that do not wrap. The
   revealed block is 88px tall (two 44px lanes), not 470px. The scrollbar is hidden; the clipped chip at
   either edge is the scroll affordance, and on open the selected chip is placed second with its
   predecessor half-cut so the row visibly continues both ways.
3. **Two-level selection, two weights.** The *mode* keeps the ink fill (`Custom…`). The chosen day and time
   are *outlined* — 1px border + 1px inset ink ring = a 2px ink edge, the `.agentBackendTab.active`
   language that already exists in the new-session dialog — so the field shows one filled answer and two
   sub-answers beneath it, never three equal black blocks. No weight change, so the box does not reflow.
4. **The answer is written where the presets write theirs.** `Custom…`'s existing `.chipTime` slot — empty
   today — shows the resolved instant, `· Wed 18 Nov, 2pm`, exactly like `Tomorrow · 9am`. Tap a day: it
   updates. Tap a time: it updates. That *is* the commit; there is no wheel settle, no OK button, nothing
   pending. (This text is the only content added to an existing element; drop it if it reads as a hint —
   the outlined chips alone still carry the state.)
5. **No past instant is reachable.** On `Today`, times at or before now are `disabled` (`--ink-disabled`
   text, `--border-faint` edge, transparent fill); choosing `Today` while the chosen time has passed
   clamps to the next half-hour. Later days enable all 48 slots.
6. **Geometry consequence for the chip row: none.** The mode chips keep their wrap, 6px gap, labels and
   hit areas. The strips align their first chip exactly with the first mode chip (both at x = 29 in the
   mock) and continue the mode rows' rhythm (6px row gap above, 12px lane gap between strips).
7. **Desktop.** A vertical mouse wheel over a strip scrolls it horizontally (one `wheel` listener,
   `deltaY → scrollLeft`, only when `|deltaY| > |deltaX|`, so trackpads keep native behaviour).

## Proposed CSS (tokens only; selectors as they would land in `app.css`)

```css
/* Custom… reveals two chip strips, stacked. No gap of its own: each strip's
   6px vertical padding is the chips' hit-area lane and the visual gap. */
#editViewer .customSnoozeRow {
  grid-template-columns: minmax(0, 1fr);
  gap: 0;
}

/* A strip is the base .choiceChips scroller, retuned: 6px vertical padding so
   the 44px hit area is not clipped by overflow, no scrollbar (the clipped
   edge chip is the affordance), horizontal overscroll does not chain. */
#editViewer .choiceChips.snoozeStrip {
  flex-wrap: nowrap;
  overflow-x: auto;
  overflow-y: hidden;
  gap: var(--space-2);
  padding: var(--space-2) 2px;
  scrollbar-width: none;
  overscroll-behavior-x: contain;
}
#editViewer .snoozeStrip::-webkit-scrollbar {
  display: none;
}

/* Strip chips are the same 32px .choiceChip; the 44px target comes from
   ::after (6px up/down, 3px into each 6px gap) — hit area is not visual size. */
#editViewer .snoozeStrip .choiceChip {
  min-width: var(--ctl);
  font-variant-numeric: tabular-nums;
}
#editViewer .snoozeStrip .choiceChip::after {
  inset: calc(var(--space-2) * -1) -3px;
}

/* Sub-selection is an ink outline, not a fill: the fill belongs to the mode. */
#editViewer .snoozeStrip .choiceChip[aria-checked="true"] {
  border-color: var(--ink);
  box-shadow: inset 0 0 0 1px var(--ink);
}
#editViewer .snoozeStrip .choiceChip:disabled {
  color: var(--ink-disabled);
  border-color: var(--border-faint);
  background: transparent;
}
```

The existing phone rule `.customSnoozeRow { grid-template-columns: 1fr; }` (app.css ~4274) and the base
`.customSnoozeRow { grid-template-columns: minmax(0,1fr) 140px }` become dead and should be deleted with the
native inputs. Markup: `.customSnoozeRow > .choiceChips.snoozeStrip[role=radiogroup][aria-label="Snooze day"]`
and `…[aria-label="Snooze time"]`, children `button.choiceChip[role=radio][aria-checked]` with the usual
`.chipLabel`. Save computes `new Date(y, m, d + dayIndex, slot >> 1, (slot & 1) * 30)` → epoch seconds,
replacing `Date.parse("YYYY-MM-DDTHH:MM")`; `fillCustomSnoozeInputs(ts)` becomes "select day index + slot
(rounded up to the next half-hour) and scroll both strips".

The literal `-3px` is half the `--space-2` gap; written as `calc(var(--space-2) / -2)` if the reviewer
prefers no literal. The mock's `.mockDialog` duplicates of each selector exist only because the page holds
three dialogs and cannot reuse `#editViewer`.

## Measurements

Measured in Chrome (agent-browser) with `getBoundingClientRect()` and `getComputedStyle()` on the mock;
hit area = border box grown by the computed `::after` insets.

| element | 393 px viewport (coarse planes) | 1100 px viewport |
| --- | --- | --- |
| mode chip (`.choiceChip`) | 32 px tall, 14 px type | 32 px tall, 13 px type |
| mode chip hit area | 36 px tall (existing `inset:-2px`; pre-existing, unchanged) | 36 px tall |
| day / time strip chip | **32 px tall, 14 px type** | **32 px tall, 13 px type** |
| strip chip hit area | **44 px tall × (width + 6) px**; narrowest chip 53.2 px → 59 px wide | 44 px tall; narrowest 51.3 px → 57 px wide |
| `Custom…` readout (`.chipTime`) | 13 px | 12 px |
| Conversation name input / Depends on picker | 44 px / 44 px (field plane, unchanged) | 44 / 44 |
| revealed `.customSnoozeRow` | **88 px** (2 × 44 px lanes) | **88 px** |
| whole dialog: No snooze / Custom… / Indefinitely | 520 / **614** / 520 px | 478 / **572** / 478 px |
| vertical rhythm | mode rows 6 px apart · Custom→day 14 px · day→time 12 px · time→next label 18 px | same |
| first chip x: mode row vs strip content | 29 / 29 | 537 / 537 |

Interaction checks in the same page: clicking `Custom…` sets `.customSnoozeRow` to `display: grid`, then a
preset sets it to `none`; on `Today` at the mock clock (Fri 13 Nov 14:10) 29 of 48 time chips are disabled
and the selection clamps to 2:30pm, readout `· Fri 13 Nov, 2:30pm`. Day strip labels checked against the
real calendar (`Tue 1 Dec`, `Fri 1 Jan` are the true weekdays).

Rejection #4 for comparison: ~470 px revealed block, 44 px visible cells.
