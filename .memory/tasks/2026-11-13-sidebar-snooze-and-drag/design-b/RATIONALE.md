# Custom snooze: a compact segmented stepper

## What fails in rejection #4

- Six calendar rows at 44px already consume 264px before the month header and weekday labels. Add the 132px time wheel and there are 396px of selection surfaces alone; the plan records roughly 470px expanded. A subordinate part of Snooze becomes the dialog's largest region.
- The selected day is a full 44px-high ink block, versus a 32px active preset chip immediately above it. The rare custom path becomes visually stronger than the choice that revealed it.
- November's leading blanks and disabled past dates consume the same grid area as useful choices. The screenshot devotes its upper calendar rows to dates the user cannot select.
- The wheel displays three times, but only the middle one is the value. The two full-width rules around that row introduce an additional selection treatment and require the user to understand a centre-position commit, unlike the chips' explicit tap selection.
- Day navigation, a month header, weekday capitals, a today outline, a selected-date fill, a nested scrolling wheel, and wheel separators all arrive together. They introduce several affordances for choosing just two values, while pushing Depends on and Save toward the phone's bottom edge.

## Design decisions

**Show the instant, not its surrounding calendar.** Two compact segmented groups read `18 Nov 2026` and `09 : 00`. Day/month/year and hour/half-hour each have an up/down pair of ghost controls. This is a segmented stepper, not a month grid, dropdown, snapping wheel, or pair of field-plane picker buttons.

**One change, one result.** Each arrow updates its segment and the visible local instant immediately. The middle row is always the value Save uses; no hidden Apply, scroll-settle state, open picker, or second commit exists. Up/down keyboard keys also change a focused segment. Hour changes are one hour, minute changes are 30 minutes. Day and time arithmetic carries across boundaries; month/year changes preserve the day when possible and otherwise clamp to that month's last day.

**Stay on the chip plane.** Both the value band and arrow buttons are 32px high. Values use the same `--font-md` as chip labels (14px on phone, 13px on desktop); dates and times use the existing mono token for data. The arrow glyphs are 20px SVG view boxes, not new text symbols or fonts. There is no outer panel, calendar heading, hint, icon badge, or supplementary option.

**Pay honestly for touch targets.** Each 32px arrow sits in a 44px lane and has the existing borderless icon button's 44×44px hit area. The complete control is therefore `44 + 32 + 44 = 120px`, not a 32px control with undisclosed overlapping hit zones. It adds 126px to the dialog including the normal 6px field gap, rather than the rejected roughly 470px block.

The original chips' 2px pseudo-element outset does not meet a 44px target. Here the chip pseudo-element is explicitly 44px high and vertically centred, and chip rows have a 12px gap. Horizontal targets do not need outsets: the narrowest chip is already about 79px wide. Reducing horizontal chip padding from 12px to the existing 8px token preserves paired presets on the phone. All seven labels remain unchanged. The consequence is four chip rows on this 393px phone and three on a 480px desktop dialog, rather than pretending the larger touch area fits in a 6px row gap.

**Keep family ownership.** Fill, ink, rules, corners and focus come from existing tokens. In Paper dark, the new chrome perimeter mirrors that family's existing hairline button adaptation instead of retaining a bright ink rectangle. Clay's corners remain 6px, Slate's 8px, and Paper's 0. No new custom property is declared.

## Exact proposed CSS

The following is the mockup's `style#proposal` verbatim. `.mockDialog` is only an alternate selector for the two static comparison copies; the shipped dialog would use `#editViewer`. Page layout, comparison captions and stacking are in a separate `style#scaffolding` and are not proposed product rules. Existing border widths and structural quantities remain literal; appearance, spacing, sizing and typography use existing tokens.

```css
/* Proposed field geometry, not page scaffolding. */
#editViewer .snoozeField .choiceChips,
.mockDialog .snoozeField .choiceChips {
  flex-wrap: wrap;
  overflow: visible;
  row-gap: var(--space-5);
  column-gap: var(--space-2);
  padding: var(--space-2) var(--space-1);
}
.formViewer .snoozeField .choiceChip { padding-inline: var(--space-3); }
.formViewer .snoozeField .choiceChip::after {
  inset: auto 0;
  top: 50%;
  height: var(--ctl);
  transform: translateY(-50%);
}
.snoozeField { gap: var(--space-2); }
.snoozeStepper {
  display: grid;
  grid-template-columns: minmax(0, 3fr) minmax(0, 2fr);
  column-gap: var(--space-5);
  padding-inline: var(--space-2);
}
.snoozeSegments {
  display: grid;
  grid-template-columns: repeat(3, minmax(var(--ctl), 1fr));
}
.snoozeSegments.timeSegments { grid-template-columns: repeat(2, minmax(var(--ctl), 1fr)); }
.snoozeSegment {
  display: grid;
  grid-template-rows: var(--ctl) var(--dialog-control-h) var(--ctl);
  justify-items: center;
  align-items: center;
  min-width: 0;
}
.formViewer .snoozeNudge {
  padding: 0;
  background: none;
  border: 0;
  color: var(--muted);
}
.formViewer .snoozeNudge:hover { background: var(--paper); color: var(--ink); }
.snoozeNudge .icon { width: var(--font-3xl); height: var(--font-3xl); }
.snoozeValue {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: var(--dialog-control-h);
  border-block: 1px solid var(--border);
  background: var(--paper);
  color: var(--ink);
  font: var(--font-md)/1 var(--font-mono);
  font-variant-numeric: tabular-nums;
}
.snoozeSegment:first-child .snoozeValue {
  border-left: 1px solid var(--border);
  border-top-left-radius: var(--radius-control);
  border-bottom-left-radius: var(--radius-control);
}
.snoozeSegment:last-child .snoozeValue {
  border-right: 1px solid var(--border);
  border-top-right-radius: var(--radius-control);
  border-bottom-right-radius: var(--radius-control);
}
.snoozeSegment [data-part="hour"] { position: relative; }
.snoozeSegment [data-part="hour"]::after {
  content: ":";
  position: absolute;
  right: 0;
  transform: translateX(50%);
}
.snoozeValue:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 1px; position: relative; z-index: 1; }
:where(:root[data-theme="paper"][data-mode="dark"]) .snoozeValue,
:where(:root[data-theme="paper"][data-mode="dark"]) .snoozeSegment:first-child .snoozeValue,
:where(:root[data-theme="paper"][data-mode="dark"]) .snoozeSegment:last-child .snoozeValue {
  border-color: var(--hairline);
}
```

## Measured geometry and method

Chrome, via `agent-browser --session design-b`. Phone viewport **393×852**; desktop viewport **1100×900**. The desktop image is full-page (1100×1096) to show all three state copies. The phone image shows the complete custom dialog; the standalone page has No snooze, Custom…, and Indefinitely copies in order.

| Measurement | Phone | Desktop |
|---|---:|---:|
| Custom dialog | 361×676px | 480×628px |
| Revealed custom control | 311×120px | 430×120px |
| Visible value-band height | 32px | 32px |
| Arrow button visible bounds | 32×32px | 32×32px |
| Arrow effective hit area | 44×44px | 44×44px |
| Preset chip visible height | 32px | 32px |
| Preset chip effective hit height | 44px | 44px |
| Chip label / date-time font | 14px / 14px | 13px / 13px |
| Date/time segment width | ~57.4px | ~81.2px |
| Name / dependency controls | 44px | 44px |
| Preset row pitch | 44px | 44px |
| Custom dialog body client / scroll height | 552 / 552px | 504 / 504px |

- Dimensions: `getBoundingClientRect()`. Type, border radius and pseudo-element dimensions: `getComputedStyle(element)` and `getComputedStyle(element, "::after")`. Raw measurements are in `measure-phone.json`, `measure-desktop.json`, and `measure-themes-{phone,desktop}.json` beside the mockup.
- Hit areas were not inferred from CSS declarations alone: `elementFromPoint()` at each arrow/chip's vertical centre ±21px resolved to the intended button for all 17 controls. The chip target is about 79–174px wide ×44px high, depending on label. Arrow targets are 44px wide; neighbouring segment centres are ≥57px apart on phone.
- An intermediate `inset: -6px` on bordered chips measured **42px**, not 44px, because their absolute-positioning origin is the inner padding box. The explicit centred 44px height corrects that discrepancy. Borderless arrow buttons do measure 44px with the existing inset.
- All six family×mode combinations were measured at both viewport widths: 32px chip/value heights, 120px custom block and 44px arrow targets remained unchanged. None required body scrolling at 852px phone height or 900px desktop height. The actual `.formBody` retains scrolling and the mock dialog retains a viewport height cap for shorter screens.
- Read back the phone and desktop PNGs, plus Paper/Slate light and all three dark renders. Enlarged the initially undersized chevrons, corrected overlapping/undersized chip targets, reduced chip padding so Tomorrow and This weekend share a row, and corrected Paper-dark border weight before final capture.

## Interaction observed

Clicked Increase day, then Increase half hour: `18 Nov 2026, 09:00` became `19 Nov 2026, 09:30` without any additional confirmation. Clicking Save recorded the same local instant as `snooze_until: 1795051800` in the mock's local `savedDraft` object. Choosing Indefinitely hid the control; choosing Custom… revealed the retained `19 Nov, 09:30` value. The initial screenshot value was then restored by reloading.

This is a visual prototype, not a server integration: its clock is fixed at 13 November 2026 for the pictured scenario, and Save records a local preview payload. No sidebar or real session is changed. A production implementation would reuse the app's actual current-time validation and snooze write path.

## Weakest point

**Ten arrow targets are an interaction tax.** They are quiet and compact, but selecting a substantially different day can require repeated taps, and the absence of a weekday overview makes “the Thursday after next” less direct. This trades calendar overview and fast spatial day selection for a small, always-explicit control; it is the main usability question to judge, not a claim that the stepper is universally faster.
