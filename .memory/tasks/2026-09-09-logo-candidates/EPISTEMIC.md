# Epistemic model

## Phenomenon

The deployed Paper sidebar dog-ear carries a stale tinted-paper palette: taupe
contours/fold and a terracotta terminal in light mode. Those colors contradict
Paper's declared ink-on-paper language and make the in-app mark read like a
Clay variant. The approved canonical dog-ear geometry, tight viewport, browser
favicon, and PWA artwork are separate concerns and are not implicated.

## Mechanism

The sidebar SVG has semantic page, fold, and terminal paths. `app.css` assigns
their paints exclusively through `--brand-logo-*` tokens. The base tokens still
encoded the prior tinted-paper literals, and Paper dark gave its fold a separate
wash literal. Consequently, Paper's semantic mark did not resolve from its own
surface and ink roles.

The correction makes Paper page resolve from `--paper`, fold from `--bg`, and
both contour and terminal from `--ink` in the base light palette and the Paper
dark override. CSS remains the only paint writer. Clay and Slate retain their
independent family token assignments.

## Evidence

- Before the correction, the resolved Paper-light signature was page
  `#fdfbf6`, contour `#a79a84`, fold `#e8e1d2`, terminal `#c96442`.
- `tests/test_brand_logo_theme.py` had behaviorally resolved and pinned that
  stale signature. Its revised contract resolves actual CSS tokens and requires
  the Paper signature to equal `(paper, ink, background, ink)` in light and
  dark modes.
- Docker browser validation and final rendered screenshots are recorded in the
  task OPS entry added for this correction.

## Current commitment

The Paper in-app mark alone is monochrome: Paper light uses charcoal ink on
Paper surfaces and Paper dark uses light ink on its dark Paper surfaces. The
fold uses the theme background role to remain visibly folded without introducing
an independent tinted paint. No deployment occurs until parent independent
review.
