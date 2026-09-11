# EPISTEMIC — clay default theme

Mutable model.

## Phenomenon / goal
Reposition the theme family lineup: clay default, slate secondary, paper
third — in both the Settings swatch row order and the no-stored-preference
boot default (previously paper first/default).

## Mechanism
- Order is data, not code structure: `FAMILIES` in app_theme.js drives the
  swatch DOM order (settings iterates themeController.families) and
  `DEFAULT_FAMILY` drives normalize/persist/boot fallback.
- index.html boot script carries a duplicated default-family fallback and a
  static theme-color for the pre-boot state; both moved to clay.
- Everything asserting paper-default colors/order (unit tests, Docker
  verifier, AGENTS.md) was downstream of those two constants and has been
  updated to clay values.

## Verified
- Full pytest green at 04c58542.
- Docker behavioral verify PASS (29/29): fresh boot renders clay light
  (#faf7f0 meta, 12px session radius), reset boots clay/system, system-mode
  dark resolves to clay dark body rgb(27,24,21), Monaco follows clay light
  rgb(242,237,226) and dark rgb(31,27,22) in the real UI.

## Ruled out / notes
- Mode default unchanged ("system").
- Paper remains a selectable family and the base stylesheet (app.css is paper
  light) — only its default/first position changed.
- Existing users with a stored non-clay family keep it (persist semantics
  unchanged); storage key only removed when family == DEFAULT_FAMILY (now
  clay).
