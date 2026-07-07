# Coarse-pointer code-copy touch-target epistemic model

## Phenomenon
Rendered markdown code-block copy buttons must remain usable on companion touch devices. The accepted code-block copy invariant treats mobile code-copy controls as part of the companion-device contract: at least 44x44 CSS px and no page-level horizontal overflow.

## Accepted mechanism
`.code-copy-btn` remains compact at the base 30x30 size for non-coarse desktop surfaces. When the browser matches `@media (hover: none) and (pointer: coarse)`, the later media rule explicitly sets `.code-copy-btn` width/height/min-width/min-height to 44px and raises `.md pre` right padding to 58px. This matches the existing phone accommodation while covering wider touch devices that do not hit the `max-width: 520px` phone breakpoint. The copied payload and nearest-code event behavior are unchanged because only CSS changed.

## Evidence
- Pre-fix discriminator failed as predicted: the coarse-pointer media block lacked `.code-copy-btn` and `.md pre` 58px padding rules (OPS 2026-07-07T19:10:00Z).
- Post-fix local validation passed: focused code-copy/mobile-touch tests, full pytest, and diff check (OPS 2026-07-07T19:10:00Z).
- Docker/browser proof used Chromium CDP media emulation, not viewport-only checks: tablet and phone matched `pointer:coarse` + `hover:none` and measured 44x44 buttons with 58px pre padding; desktop measured 30x30 with `pointer:coarse=false`; all scenarios had no horizontal overflow and exact block-local clipboard text (OPS 2026-07-07T19:24:00Z; `browser-artifacts/coarse-code-copy-19532/VERIFICATION-REPORT.md`).
- Clean-room review accepted the slice: CSS scoping cannot enlarge desktop controls, source tests discriminate the missing coarse-pointer rule, browser proof is media-query-driven, artifacts are sanitized, and `.msg-copy-btn` exclusion is the correct scope boundary (OPS 2026-07-07T19:32:00Z; `reviews/coarse-pointer-code-copy-cleanroom-review.md`).

## Ruled out
- Viewport width alone is insufficient evidence; the accepted proof requires and records `pointer:coarse` and `hover:none` media matches.
- Enlarging `.msg-copy-btn` is out of scope: the documented 44px invariant here is for code-copy controls.
- Desktop compact sizing is not threatened because the new rule requires `pointer:coarse`; the desktop proof measured `pointer:coarse=false` and 30x30 controls.

## Residual boundaries
Coarse-pointer tablet opacity remains the base 0.72 until hover/focus; the button is visible and the accepted invariant is target size/overflow. Headless Chromium did not positively report `pointer:fine` for desktop, but `pointer:coarse=false` plus 30x30 measurement proves the coarse rule did not fire. Browser proof copied the first of two blocks; source tests continue to cover nearest-code isolation.

## Current claim
The coarse-pointer code-copy touch-target slice is accepted: `.code-copy-btn` now meets the 44x44 touch-target contract on coarse-pointer companion devices without changing fine-pointer desktop sizing or block-local copy behavior.
