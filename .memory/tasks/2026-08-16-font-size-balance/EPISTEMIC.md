# Epistemic model

## Phenomenon
In Edit conversation → Snooze → Custom, the native date and time values visually overpowered the adjacent Snooze label and choice chips.

## Established mechanism
The desktop imbalance came from CSS ownership, not native-control unpredictability. The generic `.formViewer input` rule set every form input to `--font-xl` (16px), while `.fieldLabel` and `.choiceChip` use `--font-md` (13px). Date/time inputs had no component-specific typography override, so they inherited the larger dialog-input size. On small/coarse contexts, the anti-zoom rule omitted date/time entirely.

## Intervention
The Edit dialog now owns date/time typography explicitly: date/time inputs inherit the dialog font family and use `--font-lg` (14px) on desktop. They remain covered by the mobile anti-zoom `--font-xl` (16px) rule where browser zoom behavior makes that tradeoff necessary.

## Evidence
- Screenshot observation: date/time values dominated labels/chips in the reported crop.
- Docker/browser observation: after the change, real UI computed styles are date/time 14px and label/chips 13px at 1280×720; date/time remain 16px at 390×844 with one-column layout and no horizontal overflow.
- Independent visual comparison: fixed row reads balanced; 1px editable-value emphasis is acceptable hierarchy.
- Full tests: 1668 passed, 103 subtests.
- Scoped code commit: `4fbadf0e` (`Balance custom snooze date and time typography`).
- Provenance: OPS.md entries for 2026-08-16.

## Ruled out
- A missing global base style as the primary desktop cause: `.formViewer input` was already forcing 16px before any native-default explanation.
- A viewport-specific design branch: the fix is component typography plus the existing sanctioned anti-zoom exception, not a media-query restyle.

## Current justified claim
The reported desktop imbalance is fixed with a scoped 16px → 14px date/time control change. Mobile preserves 16px deliberately for input-focus anti-zoom.

## Remaining boundary
Verification used Chromium in Docker with a fake Pi broker. Native date/time rendering can vary by OS/browser, but explicit author CSS now controls the relevant font family/size instead of leaving the controls to the generic dialog-input rule.
