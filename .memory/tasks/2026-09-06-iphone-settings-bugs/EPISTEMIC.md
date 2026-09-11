# EPISTEMIC — iPhone settings bugs (task complete)

## Bug 1 "borders get cropped" — FIXED (dcf3f3da)

Phenomenon: the selected theme swatch's outer selection ring was missing on its outermost
side (left of Paper, right of Slate) while the other three sides showed the 1px gap.

Mechanism (established, OPS #1–#3, #6): `.themeSwatch.active` paints a 2px outline at 1px
offset (3px outside the box); `.themeSwatches` was full-width inside `.formBody`
(`overflow-x: hidden`, no horizontal padding), so the outermost 3px lay in the clipped
region. Fix: 3px geometry padding on the grid (same device `.choiceChips` already used).
Docker probe: ring left edge 31 == body clip left 31; scrollWidth == clientWidth.

Ruled out: iOS auto-zoom / page magnification. The 3× JPEG measures identically to
Docker Chrome (92 CSS px card height, 3 device px per 1px border, dialog 354.7 vs Chrome
355 CSS px). Anti-zoom broadening (untyped/email/url/tel) is hardening only; every present
text entry was already covered.

## Bug 2 drill-down row — FIXED (71bea229)

One flat dialog: Appearance section then Voice & notifications section. Ownership stays
split: app_voice.js builds/renders/saves the voice form (`createVoiceDom` returns a
`<section>`); app_settings.js mounts it and calls activate/deactivate on show/hide; the voice
controller opens/closes the dialog via injected `openSettings`/`closeSettings`. Save closes
the dialog after persisting (Docker: reopen seeds the saved URL). Wiring guard passes without
allowlist changes.

## Bug 3 theme-color lag — FIXED within app control (cfbd9a75)

Boot script writes `meta[name=theme-color]` from stored family×resolved mode before first
paint; table duplicated inline (cannot import) and pinned equal to `THEME_COLORS` by a test
that executes the real inline script. iOS applying theme-color on navigation (so live
switches catch up on next load) is platform behavior, documented in AGENTS.md — not
verifiable here (no iOS device); user confirmation needed.

## Open

- Real-iOS confirmation of all three fixes (deploy is the user's call).
