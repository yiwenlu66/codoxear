# Epistemic model

## Phenomenon
Form dialogs appeared to accumulate adjacent 13px, 14px, and 16px text variants. The reported Edit conversation → Snooze → Custom row made the failure visible when date/time values overpowered their labels and choices.

## Established mechanism
The original failure was escaped ownership: a mobile-oriented 16px entry rule applied on desktop, global input selectors omitted some entry types, buttons used UA defaults, and feature containers patched typography locally.

The later 13px/14px split is different in kind. It is the app-wide boundary between chrome and content: buttons, labels, chips, menu rows, and sidebar chrome use 13px; text the user reads or edits, picker values, and titles use 14px. A 1px difference is weak as a local hierarchy signal, but collapsing it inside dialogs would create a larger system inconsistency by making the same button or entry role change size by container.

## Design rule
Form dialogs own zero private text sizes. They inherit three shared planes:

- 13px (`--font-md`): chrome—labels, buttons, chips, toggles, and menu rows.
- 14px (`--font-lg`): content—editable text, picker values, and dialog titles.
- 12px (`--font-sm`): metadata—hints, status, and secondary values.
- 16px (`--font-xl`): only the small-viewport/coarse-pointer text-entry anti-zoom floor.

Title hierarchy should be carried by weight; if 14/600 versus 13/600 ever proves too weak, adjust title weight rather than adding a dialog-local size.

## Intervention
The shared entry selector now covers text, password, search, date, time, datetime-local, number, select, and textarea at 14px desktop. The existing media rule covers the same entry types at 16px for anti-zoom. Container-level 16px rules, the date/time special case, login's 16px password exception, and per-row mobile label restyles were removed. Base buttons and action labels are tokenized at 13px; secondary picker text and slider values are 12px meta.

After the second design challenge, AGENTS.md was tightened to state the zero-private-size ownership invariant, and the parsed-CSS test now forbids font-size declarations on `.formViewer`, `.formDialog`, `#editViewer`, `#newSessionViewer`, and `.formActions`.

## Evidence
- Git history: f2640641 introduced the unscoped 16px form rule during mobile work; later media anti-zoom rules duplicated the intent.
- Independent critique established that 13/14 is an app-wide chrome/content boundary, while a dialog-local collapse would require new container-owned overrides.
- Parsed-CSS tests pin entry, action, value/title, meta, mobile anti-zoom, and zero form-container typography ownership.
- Full suite after the role-based correction: 1672 passed, 103 subtests. Focused ownership test after hardening: 6 passed.
- Docker/browser Edit conversation: desktop title/name/date/time/dependency 14px, labels/chips/Reset/Save/Cancel 13px, priority value 12px monospace; mobile name/date/time 16px, labels/chips/buttons 13px, one-column row, no horizontal overflow.
- Docker/browser New Session dialog follows the same shared planes.
- Independent visual review found the typography coherent and no remaining type defect.
- Scoped code commits: `16082b8a` (`Define form dialog typography by role`) and `623d659a` (`Clarify form typography ownership invariant`).
- A later full-suite rerun reached 1671 passed with one unrelated failure from concurrent uncommitted transcript-lifecycle work; the typography-focused tests remained green.
- Provenance: OPS.md entries for 2026-08-16.

## Ruled out
- A date/time-only defect: the ownership failure affected all form dialogs.
- 16px as a desktop form style: it was an escaped mobile rule, not a chosen hierarchy level.
- A dialog-local 14px-everywhere collapse: it would reintroduce container-owned typography and make action buttons differ by container.
- A dialog-local 13px-everywhere collapse: it would shrink editable data values and fragment entry typography across dialogs, chat search, and composer.

## Current justified claim
The correct design is not “fewer pixel values inside one dialog.” It is zero dialog-owned variants. Form dialogs use the app's shared 13px chrome, 14px content, and 12px meta planes, with 16px reserved strictly for mobile text-entry anti-zoom.

## Remaining boundary
Browser verification used Chromium in Docker with a fake Pi broker. Native date/time rendering varies by OS/browser, but both base and mobile sizes now come from explicit shared role rules. A visual reviewer noticed a minor mobile modal placement asymmetry unrelated to typography; it was not expanded into this scoped change.
