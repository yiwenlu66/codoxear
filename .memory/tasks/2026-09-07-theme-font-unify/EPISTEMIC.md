# EPISTEMIC — theme font unification

## Goal
One typeface language across clay/slate/paper, defined once (paper's tokens in
app.css), never retuned per family.

## Mechanism
- Fonts were the last per-family type language: clay overrode --font-ui to a
  system stack and --font-prose to serif (+ a serif-heading weight rule);
  slate overrode --font-ui to a system stack. Paper (base app.css) stayed
  generic sans-serif for UI and prose.
- Unify = delete the family font overrides so every theme falls through to the
  base tokens (--font-ui: sans-serif; --font-prose: var(--font-ui)); remove
  clay's serif-heading structural rule; keep fonts declared only in app.css.
- Guard: test_theme_token_coverage.py::test_family_stylesheets_never_retune_typeface
  (mirrors the geometry token guard).

## Verified
- Resolution table: paper/clay/slate all -> ui='sans-serif', prose='sans-serif',
  body='sans-serif', .md h1='sans-serif'; family files declare no --font-*.
- Full pytest green (1859 passed).
