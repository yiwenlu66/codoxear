# Epistemic model

## Phenomenon
Codoxear is moving from test-green refactor recovery to user-perspective usability plus cleaner architecture. Browser evidence remains the decisive signal: pytest green previously missed live-route and load-order failures.

## Current justified claims
- The real-backend browser path is usable for the core Pi flow: create session, send/response, queue/drain, interrupt, transcript search, file viewer, and mobile live view were proven in Docker sandbox 19083 before the current architecture tranche.
- First-run/empty-state UX is improved: sidebar hint, centered main CTA, muted disabled composer, clearer backend tab label, and desktop toast pill were browser-verified and committed.
- New Session dialog state has a dedicated controller authority (`CodoxearNewSession`): provider/model + reasoning moved in `ecb934b`; cwd/recent-cwd, resume candidate state, and worktree/tmux UI sync moved in `31a08b0`.
- Script-load order is now an explicit invariant: `app_launch.js < app_display.js < app_new_session.js < app_dom.js`. Pass 2 initially violated this because `app_new_session.js` gained a hard `CodoxearDisplay` dependency while loading before display; browser showed blank body with `hasDisplay=true`, `hasNewSession=false`. The order fix preserves fail-loud semantics.
- Pass-2 browser verification succeeded after the order fix: cwd input `/workspace` opens recent-cwd menu; resume menu loads prior session candidates and selection updates the label; a throwaway git repo in the sandbox shows the worktree field, toggling enables the branch input and changes start text; Pi start path creates a live session (`broker-255`). Evidence screenshots: d29, d30; raw observations in OPS.
- Internal monkeypatch seams are shrinking. Converted clusters: message routes, transcript export, pending-log idle, file inspect, session resume, launch defaults, unattended sweep. The mechanism is consistent: tests now call route/coordinator/impl seams with injected deps instead of patching server module globals or constructing incomplete managers.

## Open work
- Cleanly land in-flight test conversions: session sidebar priority (`c53efff5` child 2), queue sweep idle guard and server queue persistence (`56e5353b`).
- After the working tree is clean, run full local pytest, Docker test, Docker smoke, then browser re-check the New Session dialog under the committed order.
- Continue architecture tranche: remaining monkeypatch clusters include stale sidecars, launch provenance, queue-related tests if not completed, and remaining app.js concentrations outside New Session (chat search/navigation, queue/recovery panels).
- Final acceptance still requires clean-room adversarial review after the current tranche.

## Ruled out
- "Tests alone prove usability": falsified by live `/api/sessions` 500s, failed-launch UX bugs, and the pass-2 script-order blank page. Browser and real-server evidence are mandatory.
- Soft-loading `app_new_session.js` without `CodoxearDisplay`: rejected. The dependency is real; load order must satisfy it rather than hiding the violation.


## Acceptance judgment after clean-room review
Clean-room critic c6109f6e accepted the current tranche with no blockers. The accepted mechanism is: New Session state now has a single controller authority and correct fail-loud script order; browser evidence proves the load-order contract from the user's perspective; converted tests exercise real coordinators with injected external boundaries rather than server-global monkeypatches; validation covers local pytest, Docker test/smoke, clean browser, and independent review.

Non-blocking backlog from review: dispose the resume debounce timer on dialog close to avoid one hidden trailing request; replace/drop the vacuous `__codoxearLoadError` probe; continue reducing residual source-text assertions; continue remaining app-shell extractions beyond New Session.
