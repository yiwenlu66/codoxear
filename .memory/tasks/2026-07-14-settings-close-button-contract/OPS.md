# OPS — evidence trail

## 2026-07-14 investigation

- Screenshot `IMG_3094.jpeg`: Settings dialog, clay dark, iPhone. Close button
  = narrow vertical pill + accent ring. Pixel sample at ring (1063,295 device
  px) = rgb(196,128,107) ≈ clay dark `--accent #d97757` anti-aliased over
  paper; rules out `--border #4a4236`.
- `app_settings.js:54` and `app_new_session.js:824` build identical
  `icon-btn` + `iconSvg("x")` closes; difference is CSS context only.
- Settings surface: native `dialog.formViewer.formDialog`; New session:
  `div.formViewer.newSessionViewer`; queue/diag/help: `queueViewer` /
  `diagViewer` / `helpViewer`. All six dialogs share the `.queueHeader` +
  icon-btn close pattern.
- Breaker rule: `.formViewer .icon-btn { width:auto; min-width:0 }`.
  Introduced with the dialog-control sizing in 7a058d9b ("Redesign new
  session dialog layout"), which simultaneously added the ID rescue
  `#newSessionViewer .agentBackendTab, #newSessionCloseBtn { width:
  var(--dialog-control-h) }` — symptom patch for New session only. Settings
  (`#settingsCloseBtn`) and Edit conversation (`#editCloseBtn`) share the
  `.formViewer` chain and had no rescue.
- Second rescue found: `.diagViewer .queueHeader .icon-btn` (square +
  border:0, via `--ctl-chrome`, aliased to the same 32px). Duplicate
  border:0 statement: `#newSessionCloseBtn { border:0; background:transparent;
  overflow:visible }` (fully redundant before the fix).
- Focus ring layer: `focusModalCloseButton` auto-focuses the close button on
  open (settings/diag always; new session on mobile). iOS Safari
  :focus-visible matches after programmatic focus → accent outline shows on
  open. Present in both dialogs; on a square button it reads as deliberate,
  on the pill as broken. No focus-behavior change made.

## 2026-07-14 fix (commit be8b4920)

- `app.css`: one square-chrome rule for `.icon-btn` in all four modal
  surfaces + `.formViewer .agentBackendTab`; `.choiceChip` keeps
  content-sizing; `.queueHeader .icon-btn { border:0 }` un-scoped; deleted
  `#newSessionCloseBtn` blocks (×2) and `.diagViewer .queueHeader .icon-btn`
  rescue.
- `tests/test_dialog_control_contract.py`: parses app.css, asserts the
  contract declarations, resolved var values (32px), absence of close-button
  ID selectors, and no later surface-scoped rule re-content-sizing icon
  buttons.

## 2026-07-14 verification

- Unit: `.venv/bin/python -m pytest tests/ -q` → 1864 passed, 112 subtests.
- `scripts/docker_verify.sh be8b4920` → PASS
  (/tmp/codoxear-docker-verify-results.3Y0pK5).
- Dialog scenario (same isolation pattern, throwaway container, agent-browser
  through real UI): login → Settings → Dark chip → measured
  `#settingsCloseBtn` 32×32 radius 6px border 0 → closed via that button →
  queue close 32×32 border 0 → New session: `#newSessionCloseBtn` 32×32
  border 0, three backend tabs 32×32. clay/dark confirmed via
  `documentElement.dataset`. Screenshot:
  /tmp/codoxear-dialog-verify-results.UHxNRb/settings-clay-dark.png.

## 2026-07-14 follow-up: phantom focus ring (commit 74e449df)

- User report after be8b4920 deploy: "still border around close button in
  settings". Confirmed mechanism: focusModalCloseButton script-focuses the
  close button on open; iOS Safari never focuses tapped buttons → no
  previously focused element → :focus-visible matches → accent outline.
  Chromium didn't show it (pointer-initiated focus heuristic), which is why
  the first Docker verification missed it.
- Fix: focusModalCloseButton → focusModalSurface(viewer); open focuses the
  dialog surface (tabindex=-1 declared on settings/new-session/queue/help/
  diag/fileViewer surfaces). Focus-ring CSS targets button/input/textarea/
  select only → a surface cannot ring in any browser. New session mobile
  branch: surface instead of close button (desktop cwd caret unchanged).
  Tab flow intact (lands on close button). restoreModalFocus unchanged.
- Renamed across wiring (app_wiring select lists, composition, 6 modules)
  and VM-test stubs; overlay-accessibility test re-pinned to
  focus:settingsViewer.
- Trap found: dist/app.bundle.js is a tracked derived artifact; git-archive
  Docker builds serve the committed bundle, so JS changes require an esbuild
  rebuild + commit (amended into 74e449df).
- Verification: 1864 tests pass; wiring guard pass; check_js_refs pass;
  real-browser (Docker Chromium): after open activeElement=settingsViewer,
  close button focused=False outline=none 3px, Tab→settingsCloseBtn,
  geometry 32×32 everywhere; docker_verify PASS; deployed 74e449df, health
  boundary green.

## 2026-07-14 follow-up 2: UA focus ring on the surface (commit 6c7bf345)

- User screenshot IMG_3095: blue border around the whole Settings dialog on
  iOS. Mechanism: focusModalSurface focuses the dialog; author CSS had no
  outline declaration for it, so Safari's UA default focus ring (system
  blue) painted. Chromium suppresses its ring on programmatic focus, so
  Docker verification could not observe it — cross-browser UA differences
  are invisible to a single-engine harness.
- Fix: [tabindex="-1"]:focus { outline: none } next to the focus-ring block.
  tabindex="-1" is by definition keyboard-unreachable, so no focus indicator
  belongs on those elements (modal surfaces + copy button, whose feedback is
  the copied state). Pinned in test_dialog_control_contract.py.
- Deployed 6c7bf345; health boundary green; served app.css contains the rule.
