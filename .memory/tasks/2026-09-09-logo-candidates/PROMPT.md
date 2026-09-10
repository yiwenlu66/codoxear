# Paper logo regression: restore monochrome in-app mark

## Objective

Correct the deployed Paper-theme sidebar mark. Its approved dog-ear geometry
stays unchanged; only the in-app Paper paint returns to the monochrome
ink-on-paper language.

## Contract

- The in-app Paper logo page uses Paper's `--paper` role, its contour and
  terminal `>_` use `--ink`, and the fold uses Paper's `--bg` role with the
  same ink contour.
- The result is charcoal ink in Paper light and light ink in Paper dark. No
  independent taupe, terracotta, or blue logo paint remains in Paper.
- CSS tokens remain the sole paint owner. Clay and Slate token presentations,
  canonical SVG geometry/viewport, favicon, PWA, and installation assets stay
  unchanged.
- CSS tests resolve the actual family cascade in both Paper modes. Docker
  browser validation switches Settings through Paper light and dark, captures
  screenshots, and spot-checks Clay and Slate are unchanged.
- No deployment occurs before independent review.

## Success criteria

1. The browser-visible Paper sidebar mark reads as a monochrome ink-on-paper
   document in both modes.
2. Resolved CSS paints are page `--paper`, fold `--bg`, and contour/terminal
   `--ink` in both Paper modes.
3. The correction is committed atomically after staged-diff inspection.
