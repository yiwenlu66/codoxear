# Snooze · Custom… — design A3: period tabs and a fixed time grid, no scroll

A3 changes one thing from A2: it replaces the scrolling time well. Everything else is carried over
unchanged:
- the chip trigger and the `.filePickerMenu.dialogPickerMenu` panel chrome;
- placement through `positionDialogMenu` with `data-fit-whole`;
- 32px day chips in 44px tracks, and six fixed week rows;
- today shown by weight, and past days disabled;
- the 2px ink edge on the chosen day and time;
- the `· Wed 18 Nov, 9:30am` readout in the chip;
- "a day tap commits and stays open, a time tap commits and closes";
- zero inline growth.

Files:
- `mockup.html` takes these URL parameters:
  - `?state=a|b-open|b-open-other|b|c` selects the state; `b-open-other` has 9:30am chosen with
    Afternoon on view.
  - `&theme=…&mode=…` selects the family; `&helper=stock` uses the unmodified helper.
  - With no `state`, the page shows a gallery.
- `shot-phone.png` at 393×852 shows (a), (b) open with Morning on view, (b) open with Afternoon on
  view, (b) open at 393×660, (b) closed, and (c).
- `shot-desktop.png` shows the same states at 1100×900.
- `shot-families.png` shows the panel in clay, paper and slate, light and dark, with 9:30am chosen
  and Afternoon on view.

## What changed

**The time well is now a period row above a fixed 6 × 2 slot grid.**

- **Period row.** Four 32px tabs, each covering six hours:
  - `Night` 12–6am
  - `Morning` 6am–12pm
  - `Afternoon` 12–6pm
  - `Evening` 6pm–12am
- **Slot grid.** The twelve half-hours of the period on view, in two rows of six (`6:00 … 11:30`).
  The am/pm is given by the period, so the chips carry only `h:mm`; each chip's `aria-label`
  carries the full `9:30am`.
- **No scroll anywhere.** The section is 127px tall and nothing inside it moves.
- **Taps.**
  - Every one of the 48 half-hours is at most two taps away: a period, then a slot.
  - A slot in the period already on view is one tap.
  - Both rows are always fully visible.
- **The panel opens on the chosen time's period.** The 9:30am answer opens on Morning.

**Each period tab has two independent marks, so there is no hidden state:**

- **On view** (the tab's 12 slots are showing): `--text` label on a `--wash` fill. The other tabs
  are `--muted` on paper. This mark is navigation only and never takes the ink fill.
- **Holds the answer** (the chosen time lives in this period): the same 2px ink edge as the chosen
  day and time.
  - When you browse to another period, the ink edge stays on the period holding 9:30am. You can
    always see where your answer is, even when its slot is not on screen; this is the
    `b-open-other` shot.
  - The chosen day, the chosen time and the period holding that time all wear the same ink edge.

**Tapping a period commits nothing.** The readout does not change and the panel stays open. Only a
day tap or a time tap writes the value.

**Past times:**

- Today's past slots are disabled.
- A period that is entirely in the past today is disabled. At the mock clock, Fri 14:10, Night and
  Morning are off.
- Choosing Today clamps the answer to the next free half-hour (2:30pm) and puts its period on view.

### Tap count for "the 9th of next month at 9am"

Starting from a 9:30am answer, it takes **4 taps**: `Custom…`, `›`, `9`, `9:00`. Morning is
already on view, because the panel opens on the answer's period.

From an answer in a different period, it takes **5 taps**: `Custom…`, `›`, `9`, `Morning`, `9:00`.

Measured by scripted clicks: the readout went `· Wed 9 Dec, 9:30am`, then `· Wed 9 Dec, 9am`, and
the panel closed.

### Height: the panel cannot fit in the 417px below the chip at 393×852 — measured, and why

The requested budget (≤ 460px and fits below without flipping) is internally inconsistent: 460 is
more than the free space. The helper's free space below `Custom…` at 393×852 is
852 − 12 (margin) − (chip bottom 417 + 6 gap) = **417px**, not 423. 423 is where the panel's top edge
would sit.

The panel needs ten tappable rows at the 44px pitch:

| part | rows | height |
| --- | --- | --- |
| month nav | 1 | 32px |
| weekday labels | — | 14px |
| weeks | 6 | 6 × 32 + 5 × 12 = 252px |
| period | 1 | 32px |
| slots | 2 | 32 + 12 + 32 = 76px |
| lanes, rule and padding | — | rest of the 445px |

I already removed every lane that does not separate two sets of hit areas:

- The first and last week rows' hit areas overhang the weekday text and the time section's top
  lane, which hold no targets.
- Measured, the nav button's hit area ends at y 430 and the first week row's begins at 432. The
  last week row's ends at 696 and the period row's begins at 697. Nothing overlaps.

The result is **445px**, the minimum under the brief's own rules. Each way of fitting into 417px
breaks a requirement:

- Dropping a week row breaks "six fixed week rows".
- Dropping the period row means 48 slots in one view, which is a scroll or a 26px pitch.
- One slot row of twelve at 322px width is a 26px pitch.
- Shrinking the week pitch to 40px breaks the 44px hit target.

**So at 393×852 it does not fit either side.** The space above is 367px. The shot shows what
happens instead.

**When neither side fits, the panel opens over its own chip.** This is the third branch of the
`data-fit-whole` placement:

- The panel opens toward the larger side.
- Its edge sits flush on the anchor's near edge.
- It slides back over the anchor by the shortfall, so it is always whole, unclipped and unscrolled.

| viewport | room below / above | side | panel rect (left, top → right, bottom) |
| --- | --- | --- | --- |
| 393×852 | 417 / 367 | **over** | 36, 385 → 358, 830: top edge on the chip's top edge, covers the chip |
| 393×660 | 225 / 367 | **over** | 36, 12 → 358, 457: pinned to the top margin, covers the chip |
| 360×640 | 205 / 367 | **over** | 36, 12 → 348, 457 |
| 430×932 | 535 / 329 | below | 36, 385 → 358, 830, no overlap |
| 1100×900 | 341 / 491 | **above** | 327, 58 → 649, 503, no overlap |

In every case the panel is inside the viewport and `.formBody` did not scroll: scrollTop is 0 and
scrollHeight equals clientHeight (438 = 438 on the phone, 358 = 358 on desktop).

## Inner-scroll measurement (requirement 1)

On `?state=b-open` and `b-open-other`, at 393×852, 393×660, 1100×900, 360×640 and 430×932:

- **The panel itself**: scrollHeight 443 = clientHeight 443.
- **Scroll containers inside the panel**: none. I listed every element of the panel and its
  descendants whose computed `overflow-x`/`overflow-y` is `auto` or `scroll`, and the list is empty
  (`innerScrollers: []`). `.snoozeTimes` is a plain grid with no overflow declaration.
- **Content boxes**: some report scrollHeight greater than clientHeight. Each chip reports 36 vs 30,
  and `.snoozeTime` reports 131 vs 126.
  - That is the 6px `::after` hit-area overhang, counted as ink overflow.
  - These boxes are `overflow: visible` and cannot scroll.
  - The overhang is the same mechanism the base `.choiceChips` comment in `app.css` describes.

## Proposed CSS (tokens only; replaces all of A2's snooze-panel rules)

```css
/* Panel: the existing .filePickerMenu.dialogPickerMenu surface. Its only
   own geometry is an intrinsic minimum width — seven 44px day tracks — so
   the helper's 280px list width cannot squeeze the grid below the touch
   pitch; on very narrow phones it yields to the viewport. */
#editViewer .snoozePanel {
  min-width: min(calc(7 * var(--ctl) + 2 * var(--space-2) + 2px), calc(100vw - 4 * var(--space-5)));
  max-height: none;   /* never clamped: data-fit-whole places it whole instead */
  padding: var(--space-2);
  overflow: hidden;
}
/* Open, it is a column of fixed-height parts; nothing inside yields or scrolls. */
#editViewer .snoozePanel.open {
  display: flex;
  flex-direction: column;
}

/* Month nav: plain month text + the ‹ › pair of standard 32px dialog icon
   buttons (44px hit area from the existing .formViewer .icon-btn::after). */
#editViewer .snoozeMonthNav {
  display: flex;
  align-items: center;
  gap: var(--space-1);
  padding-left: var(--space-3);
}
#editViewer .snoozeMonth {
  flex: 1 1 auto;
  font-size: var(--font-md);
  color: var(--text);
}
#editViewer .snoozeMonthNav .icon-btn:disabled {
  color: var(--ink-disabled);
  border-color: var(--border-faint);
}

/* Weekday row and day grid share seven equal 44px tracks. */
#editViewer .snoozeWeekdays,
#editViewer .snoozeDays {
  display: grid;
  grid-template-columns: repeat(7, minmax(0, 1fr));
}
#editViewer .snoozeWeekdays {
  font-size: var(--font-sm);
  color: var(--muted);
  text-align: center;
  line-height: var(--space-6);
}
/* Always six week rows (height never changes between months). Rows are
   32px chips 12px apart (44px pitch); the outer 6px of the first and last
   rows' hit areas lie over the weekday text and the time section's top
   lane, which hold no targets. */
#editViewer .snoozeDays {
  grid-template-rows: repeat(6, var(--ctl-chrome));
  row-gap: calc(var(--ctl) - var(--ctl-chrome));
  place-items: center;
}
#editViewer .snoozePanel .choiceChip {
  padding: 0;
  font-variant-numeric: tabular-nums;
}
#editViewer .snoozeDays .choiceChip {
  width: var(--ctl-chrome);
  min-width: var(--ctl-chrome);
}
#editViewer .snoozePanel .choiceChip::after {
  inset: calc(var(--space-2) * -1);
}

/* Time: a period row (Night · Morning · Afternoon · Evening, 6 h each)
   above a fixed 6 × 2 grid of that period's twelve half-hours. Nothing
   scrolls: every slot is at most two taps away and both rows are always
   visible. Rows sit at the 44px pitch (32px chip + 12px lane). */
#editViewer .snoozeTime {
  display: flex;
  flex-direction: column;
  gap: calc(var(--ctl) - var(--ctl-chrome));
  margin-top: var(--space-2);
  padding-top: var(--space-2);
  border-top: 1px solid var(--border-faint);
}
#editViewer .snoozePeriods,
#editViewer .snoozeTimes {
  display: grid;
  gap: calc(var(--ctl) - var(--ctl-chrome)) var(--space-2);
  grid-auto-rows: var(--ctl-chrome);
}
#editViewer .snoozePeriods {
  grid-template-columns: repeat(4, minmax(0, 1fr));
}
#editViewer .snoozeTimes {
  grid-template-columns: repeat(6, minmax(0, 1fr));
}
/* In the gapped rows the hit area fills half of each 6px gap sideways and
   the 12px lane vertically, so neighbouring targets tile without overlap. */
#editViewer .snoozeTime .choiceChip::after {
  inset: calc(var(--space-2) * -1) calc(var(--space-2) / -2);
}
/* Periods are tabs. The one on view reads by text + fill (--text on
   --wash; the others --muted on paper) — the text step carries it in dark
   modes, where --wash sits close to --paper. Navigation only: never the
   ink fill. */
#editViewer .snoozePeriods .choiceChip {
  color: var(--muted);
}
#editViewer .snoozePeriods .choiceChip[aria-selected="true"] {
  color: var(--text);
  background: var(--wash);
}
#editViewer .snoozePeriods .choiceChip[aria-checked="true"] {
  color: var(--text);
}

/* Two-weight selection: the mode chip keeps the ink fill; the answer path
   (chosen day, the period holding the chosen time, the chosen time) gets
   the 2px ink edge of .agentBackendTab.active; today is weight only. */
#editViewer .snoozePanel .choiceChip[aria-checked="true"] {
  border-color: var(--ink);
  box-shadow: inset 0 0 0 1px var(--ink);
}
#editViewer .snoozeDays .choiceChip[aria-current="date"] {
  font-weight: 600;
}
#editViewer .snoozePanel .choiceChip:disabled {
  color: var(--ink-disabled);
  border-color: transparent;
  background: transparent;
}
```

The rule `#editViewer .snoozePanel .choiceChip::after { inset: -6px }` applies to the day chips. The
time section's more specific `#editViewer .snoozeTime .choiceChip::after` overrides it for periods
and slots, whose 6px column gaps need a 3px sideways overhang.

### Proposed helper change (`app_dialog_menu.js`, opt-in through `data-fit-whole`)

This replaces A2's three-line patch. A fixed-height panel does not need a `menuHeight` estimate or a
clamp. It needs a whole-panel decision with a defined fallback for when neither side fits. Menus
without the flag keep the stock code path byte for byte. It goes in after the existing left/width
assignments and `menu.style.maxHeight = ""`:

```js
if ("fitWhole" in menu.dataset) {
  const h = menu.offsetHeight;
  const below = rect.bottom + 6;
  const above = rect.top - 6 - h;
  let top;
  if (below + h <= viewportBottom - margin) top = below;                 // whole, below
  else if (above >= viewportTop + margin) top = above;                   // whole, above
  else top = viewportBottom - below >= rect.top - viewportTop            // neither: over the anchor,
    ? Math.max(viewportTop + margin, Math.min(rect.top, viewportBottom - margin - h))    // from the larger side
    : Math.min(viewportBottom - margin - h, Math.max(viewportTop + margin, rect.bottom - h));
  menu.style.top = `${top - hostRect.top}px`;
  return;
}
```

### Markup delta from A2

```
.snoozePanel > … > .snoozeTime
  > .snoozePeriods[role=tablist]  > button.choiceChip[role=tab][aria-selected][aria-checked]   (4)
  > .snoozeTimes[role=radiogroup] > button.choiceChip[role=radio][aria-checked][aria-label="9:30am"]  (12)
```

State adds one transient field, `viewPeriod`. It is set to the answer's period on every open and
after the Today clamp, and is never saved. The saved value is still `(y, m, d, slot)`, written as
`new Date(y, m, d, slot >> 1, (slot & 1) * 30)`.

## Measurements

Taken in Chrome (agent-browser) with `getBoundingClientRect` and `getComputedStyle`. The hit area
is the border box plus the computed `::after` insets.

| element | 393×852 (coarse type planes) | 1100×900 |
| --- | --- | --- |
| day chip → hit area | 32×32 → **44×44**, row pitch 44 | same |
| period tab → hit area | 72.5×32 → **78.5×44**, column pitch 78.5 | same |
| time slot → hit area | 46.3×32 → **52.3×44**, pitch 52.3 × 44 | same |
| type: day / period / slot | 14 / 14 / 14px | 13 / 13 / 13px |
| time section (rule + period row + 2 slot rows) | 127px | 127px |
| panel | 322 × **445** | 322 × 445 |
| dialog height in (a), (b) closed, (c) | 558 / 558 / 558, **+0px inline** | 478 / 478 / 478 |

At 360×640 the period tabs measure 70×32 → 76×44 and the slots 44.7×32 → 50.7×44, still at least
44px in both dimensions.

**Interaction**, run as scripted clicks:
- Custom → › → 9 → `9:00` gives `· Wed 9 Dec, 9am` in **4 taps**, and the panel closes.
- A period tap leaves the readout unchanged and the panel open.
- The Evening slots list `6pm … 11:30pm` (by `aria-label`).
- Morning keeps the ink edge while Evening is on view.
- Today clamps to 2:30pm with Afternoon on view; Night and Morning are disabled, as are 12 past days.
- An outside tap closes the panel; choosing a preset clears the readout.
- `.formBody` scrollTop stays at 0 throughout.

**Token check:** the mock's stylesheet has no hex or `rgb()` literal and no font declaration.

## Single weakest thing about this version

**On a phone the panel covers the `Custom…` chip and its readout while it is open.**

- At 393×852 the 445px panel fits neither the 417px below nor the 367px above. It opens over the
  chip, with its top edge on the chip's top edge.
- This is a regression from A2, whose 414px panel fitted below on that phone. A3 is 31px taller.
- While the panel is open, the only place the answer is written as text (`· Wed 18 Nov, 9:30am`) is
  hidden. You read the answer from the three ink edges inside the panel (day, period, slot), and see
  the text again when the panel closes.
- This holds on every phone up to roughly 900px viewport height. Only tall phones (430×932) and
  desktops see the readout next to the open panel.
- **Fix, if the readout must stay visible:** replace the static month title with the live answer
  (`Wed 18 Nov · 9:30am` in place of `November 2026`), since the grid already shows the month. I did
  not do this because it changes the month nav, which the brief said to keep.

Secondary:
- The slot labels drop am/pm, so `12:00` under Night means midnight. The period tab is the only
  meridiem cue on screen; the `aria-label` carries the full time.
- At 360px wide the day tracks are still 42.6px, unchanged from A2.
