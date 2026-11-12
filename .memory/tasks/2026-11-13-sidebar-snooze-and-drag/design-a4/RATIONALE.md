# Snooze · Custom… — design A4: live answer in the panel header

A4 changes one thing from A3: the panel's header row. The old header was `November 2026` with ‹ ›
beside it. The new header puts **the live answer on the left and a month pager `‹ Nov 2026 ›` on
the right**, in the same 32px row. Everything else is A3 byte for byte:
- no scroll containers;
- 32px day chips in 44px tracks, and six fixed week rows;
- period tabs carrying "on view" and "holds the answer" as separate marks;
- a day tap commits and stays open, a time tap commits and closes;
- zero inline growth, and `· Wed 18 Nov, 9:30am` in the mode chip when the panel is closed.

**Files:**
- `mockup.html` — `?state=a|b-open|b-open-other|b|c`, plus `&theme=…&mode=…`.
  `&pick=y,m,d,slot` sets the answer; I used it for the widest-answer test.
- `shot-phone.png` — 393×852: (a); (b) open; (b) open with Afternoon on view; (b) open at 393×660;
  (b) closed; (c).
- `shot-desktop.png` — the same states at 1100×900.
- `shot-families.png` — six family/mode variants at 393, with 9:30am chosen and Afternoon on view.
- `shot-header-widest.png` — the widest possible answer at 360 (clay light and paper dark) and at
  393, zoomed 2×.

## Header design

- **Answer** (`.snoozeAnswer`): `Wed 18 Nov, 9:30am`, `--font-md`, `--text`, regular weight,
  `white-space: nowrap`, `flex: 0 0 auto` (it never shrinks).
  - It comes from the same formatter call that writes the chip's `.chipTime`, so the two strings
    cannot drift. On phones, where the panel covers the chip, the header is the readout.
  - It carries `aria-live="polite"`.
  - A 600 weight first cut made the header the loudest thing in the panel, so it is regular weight.
- **Pager** (`.snoozePager`): ‹ `Nov 2026` ›.
  - The month is `--font-sm` `--muted`, centred in a fixed `min-width` of
    `calc(var(--ctl) + var(--space-7))`, which is 60px. The widest `MMM yyyy`, "May 2027", measures
    57px, so the arrows never move when the month changes.
  - The arrows are the unchanged 32px `.icon-btn`s, with 44×44 hit areas.
- **Two kinds of date.** The answer is `--text` at `--font-md`; the month is `--muted` at
  `--font-sm` and sits between arrows. The answer reads as the value, the month as the page you are
  on.
- **Pager versus answer.** Paging the month never changes the answer, measured: `Nov 2026` →
  `Dec 2026` while the answer stays `Wed 18 Nov, 9:30am`. Only a day tap or a time tap writes it.
- **Height.** The row is still 32px, so the panel is still **445px** (≤ 460).

### Proposed CSS (replaces A3's `.snoozeMonthNav` / `.snoozeMonth` rules; everything else unchanged)

```css
/* Header row (one 32px row): the live answer on the left, the month pager
   ‹ Nov 2026 › on the right. The answer is the panel's readout of the chip
   text it covers on phones; the month is a caption of the pager, muted and
   one step smaller, so the two dates never read as the same kind of thing.
   Neither may shrink: both fit whole at 360px with the widest real answer. */
#editViewer .snoozeHead {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-3);
  padding-left: var(--space-3);
}
#editViewer .snoozeAnswer {
  flex: 0 0 auto;
  font-size: var(--font-md);
  color: var(--text);
  white-space: nowrap;
  font-variant-numeric: tabular-nums;
}
#editViewer .snoozePager {
  display: flex;
  align-items: center;
  gap: var(--space-1);
  flex: 0 0 auto;
}
#editViewer .snoozeMonth {
  min-width: calc(var(--ctl) + var(--space-7));   /* 60px ≥ widest "MMM yyyy" (57px): arrows never move */
  text-align: center;
  font-size: var(--font-sm);
  color: var(--muted);
  white-space: nowrap;
  font-variant-numeric: tabular-nums;
}
#editViewer .snoozePager .icon-btn:disabled {
  color: var(--ink-disabled);
  border-color: var(--border-faint);
}
```

The markup changes from `.snoozeMonthNav > [month, ‹, ›]` to:

```
.snoozeHead
  > span.snoozeAnswer[aria-live=polite]
  > .snoozePager > [button.icon-btn ‹, span.snoozeMonth, button.icon-btn ›]
```

The month string changes from `November 2026` to `Nov 2026`, the same `MON[]` table the answer uses.

## Header measurements

Measured in Chrome with `getBoundingClientRect` and `scrollWidth`/`clientWidth`. **Worst case:** I
ran every real date from 13 Nov 2026 to 12 Nov 2028 combined with the widest time label (`12:30am`).
The widest is `Wed 12 May, 12:30am`, at 139.6px.

| viewport | header box | answer box | answer scrollW / clientW | month box | ‹ › box → hit area | gap answer → ‹ | truncated |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 393×852, default `Wed 18 Nov, 9:30am` | 308×32 | 130.2×16 | 130 / 130 | 60×15 | 32×32 → 44×44 | 37.8px | no |
| 393×852, widest `Wed 12 May, 12:30am` | 308×32 | 139.6×16 | 140 / 140 | 60×15 | 32×32 → 44×44 | 28.4px | no |
| 360×640, default | 298×32 | 130.2×16 | 130 / 130 | 60×15 | 32×32 → 44×44 | 27.8px | no |
| 360×640, widest | 298×32 | 139.6×16 | 140 / 140 | 60×15 | 32×32 → 44×44 | **18.4px** | no |

- Type is 14px for the answer and 13px for the month on coarse-pointer planes; on desktop it is 13
  and 12.
- No `text-overflow`, ellipsis or `overflow: hidden` is declared on any header element, so nothing
  can be eaten silently.
- `scrollWidth === clientWidth` holds for the answer and the month in every row.
- Below 360px the header still fits:
  - At 340 wide, the gap shrinks to 11.9px.
  - At 320 wide, it reaches the 8px `--space-3` floor. At that width the panel itself yields to its
    viewport cap, and the day tracks were already under 44px in A3.

## Placement table (unchanged from A3: the panel is still 445px)

| viewport | room below / above | side | panel rect (left, top → right, bottom) | in viewport | scroll containers in the panel · panel scrollH = clientH | `.formBody` scrollTop · scrollH = clientH | dialog height |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 393×852 | 417 / 367 | **over** | 36, 385 → 358, 830 | yes | 0 · 443 = 443 | 0 · 438 = 438 | 558 |
| 393×660 | 225 / 367 | **over** | 36, 12 → 358, 457 | yes | 0 · 443 = 443 | 0 · 438 = 438 | 558 |
| 360×640 | 205 / 367 | over | 36, 12 → 348, 457 | yes | 0 · 443 = 443 | 0 · 438 = 438 | 558 |
| 430×932 | 535 / 329 | below | 36, 385 → 358, 830 | yes | 0 · 443 = 443 | 0 · 400 = 400 | 520 |
| 1100×900 | 341 / 491 | **above** | 327, 58 → 649, 503 | yes | 0 · 443 = 443 | 0 · 358 = 358 | 478 |

Inline growth is zero: the dialog is 558px at 393 in (a), in (b) closed and in (c), and 478px at
1100.

On phones the panel still opens over the `Custom…` chip. The answer the chip carried is now in the
header, top left of the panel, which was the point of this round. `shot-phone.png` (b) shows
`Wed 18 Nov, 9:30am` in the header while the chip beneath is covered.

## Interaction (scripted taps)

- `Custom…` opens: header `Wed 18 Nov, 9:30am`, month `Nov 2026`.
- `›` gives `Dec 2026`; the answer is unchanged.
- `9` gives header `Wed 9 Dec, 9:30am` and chip `· Wed 9 Dec, 9:30am`; the panel stays open.
- `9:00` gives header `Wed 9 Dec, 9am` and chip `· Wed 9 Dec, 9am`; the panel closes. That is
  **4 taps** for "the 9th of next month at 9am", unchanged.
- A period tap leaves the answer unchanged.
- `‹` returns to `Nov 2026` and is disabled there.

## Families (`shot-families.png`, inspected by eye)

- **Clay:** rounded nav buttons, warm muted month, `--text` answer.
- **Paper:** square buttons with ink borders; in paper dark the button borders drop to hairline (an
  existing paper-dark rule) and the answer stays light ink.
- **Slate:** ghost nav buttons (borderless until hover, an existing slate rule), so the header reads
  as text + glyphs. The disabled ‹ fades to `--ink-disabled`.
- **Dark modes:** every header colour comes from `--text`/`--muted`/`--ink-disabled`, so they all
  invert. No literal is used anywhere: the mock's stylesheet has no hex or `rgb()` literal and no
  font declaration.

## Single weakest thing about this version

**The header now carries two dates, `Wed 18 Nov, 9:30am` and `Nov 2026`, and when they disagree
nothing says which one is the answer except styling.**

- After a `›`, the header reads `Wed 18 Nov, 9:30am  ‹ Dec 2026 ›`. Typographically the answer wins:
  `--text` at `--font-md` beats `--muted` at `--font-sm`, and the month sits between arrows.
- In slate, whose nav buttons are ghost, the month and the answer are both plain text, one step
  apart in size and tone. A user may read `Dec 2026` as "the snooze is in December" before tapping a
  day.
- The ink edge on the chosen day is off screen in that state, because 18 Nov is not in the December
  grid. So the only cue is the size/tone step.
- A label (`Until Wed 18 Nov, 9:30am`) would remove the ambiguity, but `Until ` costs about 32px, and
  the widest answer at 360 has only 18.4px of slack. I left it out rather than shrink or wrap the
  answer.

Secondary: at 360 wide the widest answer leaves 18px of slack. A longer locale format, for example
`Wednesday`, or a 24h/12h change, would need the header to wrap; this design assumes the app's
fixed English `compactLocalTime` format.
