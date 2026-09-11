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

The scoped Paper correction was deployed through the prescribed script. OPS 2026-09-10T12:44:53+08:00 records the exact tmux command, resolved SHA, successful output, health-boundary result, and restart scope; OPS 2026-09-10T12:45:42+08:00 records exact snapshot and service confirmation. The active immutable snapshot is `79d71532e03b946d78bf912a269fc1e04e8e3c04`; the actual prior release retained for script-based rollback is `2bec55ddbf137c436f7a5a9f887e4a2ad1287d19`. The optional authenticated browser smoke check was disabled as directed; the script's required active-service and HTTP boundary checks succeeded. Broker, agent CLI, and runtime session processes were preserved because only the server service was restarted.
