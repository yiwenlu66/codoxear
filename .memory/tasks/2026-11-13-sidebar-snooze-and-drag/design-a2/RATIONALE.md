# Snooze · Custom… — design A2: chip trigger + calendar popover

Files in this folder:
- `mockup.html` — the real `app.css` + `themes/clay.css`, the real Edit dialog built in a real
  `<dialog>.showModal()`, and the panel placed by a verbatim copy of `positionDialogMenu`.
  - `?state=a|b-open|b|c` renders one state at the true viewport.
  - No `state` gives a gallery of iframes; add `size=desktop` for desktop frames.
  - `&theme=paper|slate&mode=dark` switches family and mode.
  - `&helper=stock` runs the unmodified helper.
- `shot-phone.png` — 393×852: (a), (b) open, (b) closed, (c), plus (b) open at 393×660 (Safari with
  its toolbars showing) to show the panel opening above.
- `shot-desktop.png` — 1100×900: (a), (b) open, (b) closed, (c).
- `shot-families.png` — the open panel in clay, paper and slate, light and dark, at 393×852.
- `shot-helper-stock.png` — what the **unmodified** helper does to this panel. See "What changed",
  point 2.

## What changed from A, and why

- **Mechanism: inline strips → a panel anchored to `Custom…`.** Tapping `Custom…` selects the mode
  and opens the panel. The panel uses the existing `.filePickerMenu.dialogPickerMenu` surface:
  `1px var(--border)`, `var(--radius-card)`, `var(--shadow-pop)` and `var(--paper)`. It is appended
  to `#editViewer` exactly as `editDependencyMenu` is, and placed by `positionDialogMenu(panel,
  customBtn)`.
  - Inline growth is **0px** in every state: the dialog is 558px tall at 393 and 478px at 1100,
    whether `No snooze`, `Custom…` (open or closed) or `Indefinitely` is selected.
  - The three scroll gestures of A are gone. There are no horizontal scrollers at all.
- **Inside the panel, one surface, top to bottom:**
  - **Month nav row.** The month is plain `--font-md` text at regular weight. It is not a
    `.fieldLabel` clone, which was the "second header plane" fault of #4. The control is the ‹ ›
    pair: standard 32px `.icon-btn`s with their existing 44px `::after`. ‹ is disabled on the
    current month, because no earlier month holds a future day.
  - **Weekday row.** `Mo … Su`, `--font-sm` `--muted`, no uppercase or letter-spacing (the other
    #4 fault).
  - **Day grid.** Seven 44px tracks, Monday-first.
    - Every day is a 32×32 `.choiceChip`: a bordered paper button, the same object as the preset
      chips. Its `::after` fills its 44×44 track, so hit areas tile edge to edge with no overlap
      and no dead strip.
    - The grid always has six rows, so the panel's height is the same in every month: a panel
      opened upward does not move ‹ › out from under the finger.
  - **Time well.** Under a `--border-faint` hairline: 48 half-hour `.choiceChip`s in four columns,
    labelled with `app_snooze.js`'s `compactLocalTime` (`2pm`, `2:30pm`).
    - It shows two rows. On open it is scrolled so the chosen slot's row is on top.
    - It scrolls with no snapping; tapping a slot commits.
- **Selection language, kept from A.**
  - `Custom…` carries the ink fill.
  - The chosen day and the chosen time carry the `.agentBackendTab.active` 2px ink edge (1px border
    plus a 1px inset ring).
  - **Today is marked by weight (600) only.** It gets no ring and no fill, so it never reads as a
    second selection; #4 had three competing marks.
  - Past days and past times are `disabled`: `--ink-disabled` text, no border, no fill. They read
    as calendar context, not as buttons.
  - `· Wed 18 Nov, 2pm` stays in `Custom…`'s existing `.chipTime` slot.
- **No commit step.**
  - Tapping a day commits it and the readout updates at once; the panel stays open, because time
    is the next choice.
  - Tapping a time commits it and closes the panel, because that is the last choice.
  - Tapping outside, tapping `Custom…` again, or tapping any preset closes the panel. Nothing is
    pending in any of these cases, because every tap has already written the value.
  - Choosing `Today` when the chosen time has already passed clamps the time to the next free
    half-hour. "The 9th of next month at 9am" is four taps: `Custom…`, ›, 9, 9am.
- **Reopening** opens on the chosen day's month, not on the current month.

### 1. Panel width

The helper sets the panel's width to the anchor's width, floored at 280px. That gives 40px tracks,
below the touch pitch. The panel therefore carries one intrinsic `min-width`: seven 44px tracks,
`calc(7 * var(--ctl) + 2 * var(--space-2) + 2px)` = 322px. That is capped at
`100vw − 4 * var(--space-5)`, so a 360px phone still fits.

### 2. The stock helper clips this panel — a three-line, opt-in helper change is required

Measured with `?helper=stock`, `shot-helper-stock.png`:

| viewport | space below `Custom…` | stock decision | result |
| --- | --- | --- | --- |
| 393×852 | 423 | below | fits (needs 414) |
| 393×660 | 231 | **below** | panel clamped to 223px; the calendar is cut at the 16–22 row and the panel itself scrolls |
| 1100×900 | 347 | **below** | panel clamped to 339px; the time well is cut off |

**Cause.** `openAbove = spaceBelow < Math.min(220, menuHeight) && spaceAbove > spaceBelow`. The
helper only flips when there is less than 220px below, and otherwise it clamps `max-height`. For a
list, clamping just adds scrolling. For a calendar, it cuts the grid in half.

The helper also caps `menuHeight` at half the viewport. For a 414px panel on a 660px screen, that
under-reports the panel's height as 330.

**Proposed change** (`app_dialog_menu.js`, opt-in per menu through a `data-fit-whole` attribute):

```js
const fitWhole = "fitWhole" in menu.dataset;
const menuHeight = fitWhole ? (menu.scrollHeight || 260)
  : Math.min(menu.scrollHeight || 260, Math.floor((viewportBottom - viewportTop) * 0.5));
...
const minVisible = fitWhole ? menuHeight : Math.min(220, menuHeight);
const openAbove = spaceBelow < minVisible && spaceAbove > spaceBelow;
```

- Menus without the attribute behave exactly as before; the dependency, model and cwd menus are
  untouched.
- With the attribute, the panel goes below only when it fits whole. Otherwise it goes to the larger
  side, and the clamp falls on the time well. The panel is a flex column and the well is
  `flex: 0 1 …`, so the well is the only part that shrinks.
- The mock runs this as `helper=patched`, its default.

## Proposed CSS (tokens only; scoped to `#editViewer`)

```css
/* Panel = existing .filePickerMenu.dialogPickerMenu surface (border, radius-card, shadow-pop,
   paper, absolute). Own geometry: an intrinsic width of seven 44px day tracks. */
#editViewer .snoozePanel {
  min-width: min(calc(7 * var(--ctl) + 2 * var(--space-2) + 2px), calc(100vw - 4 * var(--space-5)));
  max-height: none;   /* the helper measures scrollHeight with max-height cleared; the base
                         320px list cap would shrink the time well and under-report */
  padding: var(--space-2);
  overflow: hidden;
}
/* Open, it is a column: the calendar keeps its height; only the time well yields to a clamp. */
#editViewer .snoozePanel.open {
  display: flex;
  flex-direction: column;
}

/* Month nav: plain month text + the ‹ › pair of standard 32px dialog icon buttons
   (44px hit area from the existing .formViewer .icon-btn::after). */
#editViewer .snoozeMonthNav {
  display: flex;
  align-items: center;
  gap: var(--space-1);
  padding-left: var(--space-3);
  margin-bottom: var(--space-1);
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

/* Seven equal tracks; each day track is a 44px square holding a centred 32px chip whose
   ::after fills the track, so hit areas tile without overlap. Always six week rows. */
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
#editViewer .snoozeDays {
  grid-template-rows: repeat(6, var(--ctl));
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

/* Time well: 48 half-hour chips, four columns, two 44px rows visible, plain scroll,
   no snap, tap commits. */
#editViewer .snoozeTimes {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  grid-auto-rows: var(--ctl-chrome);
  gap: calc(var(--ctl) - var(--ctl-chrome));
  flex: 0 1 calc(2 * var(--ctl));
  min-height: calc(var(--ctl-chrome) + var(--space-2));
  padding: var(--space-2);
  overflow-y: auto;
  overscroll-behavior: contain;
  scrollbar-width: thin;
  scrollbar-color: var(--border) transparent;
  border-top: 1px solid var(--border-faint);
}

/* Two-weight selection: mode = ink fill (existing .choiceChip.active); day/time = 2px ink edge;
   today = weight only; past = disabled, no border. */
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

**Markup**, built in `app_shell.js` next to `editDependencyMenu`:

```
div#editSnoozePanel.filePickerMenu.dialogPickerMenu.snoozePanel[role=dialog][data-fit-whole]
  > .snoozeMonthNav (span.snoozeMonth, button.icon-btn ‹, button.icon-btn ›)
  > .snoozeWeekdays
  > .snoozeDays[role=radiogroup] > button.choiceChip[role=radio][aria-checked][aria-current?]
  > .snoozeTimes[role=radiogroup] > button.choiceChip[role=radio][aria-checked]
```

- `Custom…` gains `aria-haspopup="dialog"`, `aria-controls` and `aria-expanded`.
- `applyMenus()` gains one line: `positionDialogMenu(editSnoozePanel, customBtn)`. It runs on open
  only, so stepping months never moves the panel.
- The existing document click-away handler closes the panel. It must use `event.composedPath()`
  rather than `target.closest()`. A day tap re-renders the grid, so by the time the click reaches
  `document` its target is detached, and `closest()` reports "outside". I hit this in the mock;
  the dependency menu does not re-render on click, so it never sees the problem.
- `.customSnoozeRow` and the two native inputs are deleted, along with both of their CSS rules.
- Save computes `new Date(y, m, d, slot >> 1, (slot & 1) * 30)`.

## Measurements

Taken in Chrome (agent-browser) with `getBoundingClientRect` and `getComputedStyle`, on
`?state=b-open` at the true viewport. The hit box is the border box grown by the computed `::after`
insets.

**Placement** (patched helper, the proposal):

| viewport | `Custom…` chip y | opened | panel rect (x, y → right, bottom) | inside viewport | content clipped | `.formBody` scrollTop / scrollable |
| --- | --- | --- | --- | --- | --- | --- |
| 393×852 | 385–417 | **below** | 36, 423 → 358, 838 (322×415) | yes | no (413/413) | 0 / no (438 = 438) |
| 393×660 | 385–417 | **above** | 36, 14 → 358, 379 (322×365) | yes | no (364/363); time well shrinks to 37px, one row | 0 / no |
| 1100×900 | 509–541 | **above** (347 below < 414 needed) | 327, 89 → 649, 505 (322×416) | yes | no | 0 / no (358 = 358) |

Other widths, open: at 360×640 and 375×667 the panel opened above and stayed inside the viewport;
430×740 opened below; 1440×900 opened above. The panel never overlaps its own chip, and
`.formBody` never scrolls.

**Control sizes:**

| element | 393 (coarse planes) | 1100 |
| --- | --- | --- |
| mode chips incl. `Custom…` | 32px, 14px type | 32px, 13px type |
| day cell box → hit box | **32×32 → 44×44**; track pitch 44 × 44 (tiles exactly) | same |
| time chip box → hit box | 65×32 → 77×44; row pitch 44 | same |
| ‹ › nav buttons | 32×32 → 44×44 | same |
| month text / weekday text | 14px / 13px | 13px / 12px |
| panel | 322 wide; 414px natural height | 322 × 416 |
| time well | 86px (two rows); scroll height 528 | 87px |
| inline growth when `Custom…` is chosen | **0px** (dialog 558 in a / b / c) | **0px** (dialog 478) |

**Interaction**, run as scripted clicks in the page:
- Custom opens the panel.
- 27 Nov → readout `· Fri 27 Nov, 2pm`; panel stays open.
- › → December; panel `top` unchanged.
- 9 → `· Wed 9 Dec, 2pm`.
- 9am → `· Wed 9 Dec, 9am`; panel closes.
- Reopen shows December. ‹ stops at November and is disabled there.
- Today → 29 of 48 times disabled; clamps to `· Fri 13 Nov, 2:30pm`. 12 past days are disabled.
- An outside tap closes the panel and keeps the value. A preset clears the readout.

**Token check:** no hex or `rgb()` literal in the mock's stylesheet, and no new font. Every value is
a `--ctl`, `--ctl-chrome`, `--space-*`, `--font-*`, `--radius-*` or colour token, or the 2px border
pair.

## Weakest thing about this version

**The time well is a two-row peephole, and it is the nested scroller I criticised in #4.**
- It shows 8 of 48 slots. From the default `2pm`, reaching `9am` or `7pm` means scrolling a
  86px-tall box inside a floating panel.
- On a short phone (393×660, toolbars showing) the clamp leaves exactly **one** row, 37px, so it
  reads as a strip of four times with no visible sign that more exist above or below.
- It is better than #4's wheel: no snap, no settle commit, `overscroll-behavior: contain`, and it
  lives in the overlay rather than in `.formBody`. But for the half of the day not shown, it is
  still a scroll-to-find.
- **Fix:** a coarser first level, eight 3-hour row headers, would remove the scroll. It would add a
  step, so I left it out.

**Second, the design depends on a helper change.** The stock `positionDialogMenu` does *not*
already flip this panel correctly. It stays below whenever 220px are free and cuts the calendar:
measured at 393×660 and 1100×900, see `shot-helper-stock.png`. The three-line opt-in `fitWhole`
change above is part of this proposal, not an optional extra.

**Third, at 360px wide the tracks are 42.6px**, not 44. The `min-width` yields to the viewport cap,
so adjacent day hit areas overlap by about 1.4px.
