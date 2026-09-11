# OPS — theme font unification

- 2026-09-07: task opened. Request: unify fonts; paper's font authoritative;
  sync clay and slate with paper.
- 2026-09-07: removed --font-ui/--font-prose overrides from clay.css and
  --font-ui from slate.css; removed clay .md h1-h3 weight/letter-spacing rule;
  updated both file header comments; AGENTS.md now declares typeface
  family-invariant (paper authoritative); added family-stylesheet font guard
  test (test_family_stylesheets_never_retune_typeface). Verified via
  css_tokens resolution: ui/prose/body/h1 all resolve to 'sans-serif' in all
  three families; no family font tokens remain.
- 2026-09-07: full pytest green (1859 passed, 112 subtests).
- 2026-09-07: scripts/docker_theme_verify.sh 40ebddee PASS — all 29 checks.
  Artifacts: /tmp/codoxear-theme-verify.XN9dSj. Real-app bodyFont is
  "sans-serif" under slate-dark (05/06/07), clay light default (01/02/13),
  clay system-dark (14) and clay light (15) — unified typeface confirmed.
