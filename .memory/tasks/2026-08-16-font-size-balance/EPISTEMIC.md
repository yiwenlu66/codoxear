# Epistemic model

## Phenomenon
Form dialogs accumulated adjacent 13px, 14px, and 16px text variants. The reported Edit conversation → Snooze → Custom row made the failure visible when date/time values overpowered their labels and choices.

## Established mechanism
The design system defined a type scale but did not assign scale ownership by role. Global entry styling enumerated only some input types; form containers added broad 16px overrides; buttons fell back to UA defaults; date/time and number controls were missing from anti-zoom coverage; per-row mobile rules added further exceptions. The 16px desktop variant originated as mobile anti-zoom logic that had escaped its media scope.

## Design rule
Form-dialog typography is role-based:

- 13px (`--font-md`): labels and actions.
- 14px (`--font-lg`): editable values, picker values, and dialog titles.
- 12px (`--font-sm`): hints, status, and secondary/meta values.
- 16px (`--font-xl`): the small-viewport/coarse-pointer text-entry anti-zoom floor only, not a desktop dialog style.

Font size belongs to role rules, not `.formViewer` or feature-specific containers.

## Intervention
The shared entry selector now covers text, password, search, date, time, datetime-local, number, select, and textarea at 14px desktop. The existing media rule covers the same entry types at 16px for anti-zoom. Container-level 16px rules, the date/time special case, login's 16px password exception, and per-row mobile label restyles were removed. Base buttons and action labels are tokenized at 13px; secondary picker text and slider values are 12px meta. AGENTS.md now records the role-to-token mapping.

## Evidence
- Git history: f2640641 introduced the unscoped 16px form rule during mobile work; later media anti-zoom rules duplicated the intent.
- Parsed-CSS tests pin entry, action, value/title, meta, mobile anti-zoom, and no form-container ownership of entry typography.
- Full suite: 1672 passed, 103 subtests.
- Docker/browser Edit conversation: desktop title/name/date/time/dependency 14px, labels/chips/Reset/Save/Cancel 13px, priority value 12px monospace; mobile name/date/time 16px, labels/chips/buttons 13px, one-column row, no horizontal overflow.
- Docker/browser New Session dialog follows the same roles.
- Independent visual review found the typography coherent and no remaining type defect.
- Scoped code commit: `16082b8a` (`Define form dialog typography by role`).
- Provenance: OPS.md entries for 2026-08-16.

## Ruled out
- A date/time-only defect: the same ownership failure affected all form dialogs.
- A need for a viewport-specific typography branch: mobile differs only through the existing sanctioned anti-zoom token floor.
- 16px as a desktop form style: it was an escaped mobile rule, not a chosen hierarchy level.

## Current justified claim
The correct design is now implemented as a role system rather than a local patch. Desktop form dialogs have two content planes—13px labels/actions and 14px values/titles—with 12px meta; 16px appears only where text entry needs mobile anti-zoom protection.

## Remaining boundary
Browser verification used Chromium in Docker with a fake Pi broker. Native date/time rendering varies by OS/browser, but both base and mobile sizes now come from explicit shared role rules. A visual reviewer noticed a minor mobile modal placement asymmetry unrelated to typography; it was not expanded into this scoped change.
