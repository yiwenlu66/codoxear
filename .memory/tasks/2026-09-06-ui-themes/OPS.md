# OPS — evidence trail (append-only)

## 2026-09-06 — analysis phase (pre-task, session work)

- Measured codoxear/static/app.css: 3661 lines, 776 var(--*) usages, three
  :root blocks. Python scan: 0 hex/rgb(a) literals outside :root blocks —
  palette is fully tokenized. 81 `border-radius: 0` literals. No box-shadow,
  no backdrop-filter, no ::selection or scrollbar styling.
- index.html: app.css loaded via ?v=__CODOXEAR_ASSET_VERSION__; CSP allows
  unsafe-inline style + script; theme-color meta = stale #1d4ed8.
- Real DOM structure read from app_sessions.js / app_topbar.js /
  app_shell.js / app_display.js: session card = stateDot + ▸N marker +
  titleLine + badges, meta line = backend logo img + ownerIconBadge +
  " | "-separated segments (rel age, model ·effort, cwd base, branch).
  Badges: failed/starting/unattended/queue N/unread N (no "busy" badge).
  Topbar ctx chip = "Ctx NN%" only when token data exists.
- v1 mock (hand-invented meta) was called out by user for infidelity; rebuilt
  with exact renderer structure and real icons/logos.

## 2026-09-06 — preview build & fixes (evidence for theme contract)

- Built /tmp/codoxear-themes/{preview.html,themes.css,shoot.py}; Playwright
  chromium screenshots at 1440x900@2x + 390x844 mobile.
- Six variants rendered and visually QA'd: paper-light (base, unchanged),
  paper-dark, clay-light, clay-dark, slate-light, slate-dark.
- System-mode proof: slate+system under emulated dark OS is pixel-identical
  (0.0% diff, PIL ImageChops) to explicit slate-dark.
- Four specificity-leak defects found+fixed in preview (the failure class
  the token-coverage/computed-style tests must catch):
  1. theme .md code background leaks into pre code (per-line stripes)
  2. theme `button` rule fills .icon-btn (specificity (0,2,1) > (0,1,0))
  3. slate .badge transparent reset blanks .badge.unattended inversion
  4. backend-logo brightness(0) filter = black-on-black in dark modes
- git state at task start: clean tree on main (cd203ca6), untracked:
  .memory/tasks/*, .pi-subagents/, docker/iso-broker-16447.json, node_modules/

## 2026-09-06 — task start

- Created PROMPT.md (objective, architecture contract, iteration protocol).
  Scope: Phase 0 token extraction + theme engine + settings dialog + v1 themes
  + tests + docs + docker verification. Explicit non-goal: pixel polish.
- While architecture runs: read docker_verify.sh fully. Key facts: container
  builds from git archive of a commit; Pi runs OFFLINE inside (bootstrap turn
  with expected provider failure; transcripts won't have rich markdown unless
  a session log is pre-seeded synthetically); agent-browser (host Chromium
  via CDP) drives login/snapshot/eval/screenshot over loopback port 19643;
  stock harness takes a single verification.png. Iteration harness will reuse
  this pattern with theme-switching + multi-surface captures.
- Wrote DESIGN_REVIEW.md: universal invariants (measurable), surface capture
  list (real app), per-theme watchlists, round protocol (fast mock inner loop
  vs Docker outer loop), defect taxonomy.
- Wrote design_fixture.py: synthetic Pi session JSONL (v3 format, cwd
  /workspace, model dexgem-messages/claude-fable-5-1) with a rich transcript
  (headings, lists, inline+fenced code, blockquote, table, thinking block,
  usage with totalTokens, two day separators) for Docker design reviews.

## 2026-09-06 — architecture delivery (subagent completed)

- executor run c607d4c4 completed. 8 commits bf96aa91..b8ef4527 on main:
  token extraction (5th token --radius-dot added; correct call), theme engine
  (app_theme.js sole writer; boot script; versioned theme link), settings
  dialog (app_settings.js, swatches/mode chips/custom CSS/reset/voice handoff),
  AGENTS.md language split, docker_theme_verify.sh sibling harness (21/21
  PASS with DOM probes), bundle rebuilt.
- Independent verification by me: theme tests 43/43 pass; wiring guard passes;
  viewed 02-settings-dialog-paper-light.png + 06-slate-dark-app.png — dialog
  and live dark switch render correctly in the real app (incl. error-bubble
  variant visible from offline Pi provider failure).
- Deviations accepted: :where() theme prefixes (kills the preview's
  specificity-leak class), --on-accent token, slate sessionContent keeps
  --panel (swipe bleed).
- Subagent-flagged rough edges seeding the iteration backlog: hint badges
  invisible over showModal dialogs (pre-existing, now affects Settings);
  paper-dark ink perimeters loud; clay-dark active card heavy; slate-light
  inline code faint + ghost Ctx chip; --sw-* swatch duplication; surfaces not
  yet reviewed (login, new-session, pickers, queue/file viewers, diag, toasts,
  hover/focus).

## 2026-09-06 — architecture implementation (subagent)

- Baseline: `.venv/bin/python -m pytest tests` → 1819 passed before changes
  (host python3 lacks pytest/tinycss2; use `.venv/bin/python`).
- Phase 0 (commit bf96aa91): added --radius-control/card/bubble/pill/dot,
  --font-ui/--font-prose, --focus-ring, --icon-muted-filter, --shadow-pop.
  81 radius literals mapped by role via a CSS walker; 12 focus outlines →
  --focus-ring (navPulse keyframes deliberately kept --ink: not a focus
  indicator); .md h1–h3 → --font-prose; 8 floating surfaces gained
  `box-shadow: var(--shadow-pop)`; .md blockquote radius became
  `0 R R 0` (R=0 in paper) so families get the right-rounded shape via token.
- Parity proof (host Playwright on the /tmp/codoxear-themes mock DOM with
  old vs new app.css, paper light): 300 elements × all computed properties
  incl. ::before/::after, custom props excluded → 0 diffs / 427,500 props.
  Also tinycss2 declaration diff old→new: only additive (box-shadow: none ×8,
  .md h1-h3 font-family: sans-serif).
- Theme files adapted from /tmp/codoxear-themes/themes.css. Component rules
  rewritten as `:where(:root[data-theme=X]) <base>` so specificity equals the
  base rule (preview used bare `:root[...]` prefix, +(0,2,0) — the leak class
  found in the preview). Mock-vs-preview pixel diff after adaptation:
  paper-dark 0.0%; clay/slate 1.4–3.6% confined to (a) sidebar card padding
  (preview restated `.session` radius via its own rule ordering) and (b) the
  slate topbar Ctx chip, which the preview themed with a filled pill via
  `button{}` (the specificity leak); new files intentionally leave it ghost.
- Theme engine (f717336e), Settings (236d47ff), AGENTS split (e1e22214),
  bundle (21086445), chip wrap fix (c6c26904), verifier (6a428302, b8ef4527).
- Stock `scripts/docker_verify.sh HEAD` first run FAILED on
  chrome_rows_not_scrollable: #settingsModeChips (.choiceChips) was a
  horizontal scroll container; fixed by wrapping (c6c26904). Re-run on
  6a428302 PASS → /tmp/codoxear-docker-verify-results.PGwpoW/.
- `scripts/docker_theme_verify.sh HEAD` (new sibling verifier, port 19663)
  → PASS, 21/21 checks, artifacts copied to /tmp/codoxear-theme-verify-final/
  (13 PNGs + report.json). Checks include: boot renders paper light with
  versioned link adjacent to app.css; Escape keeps Settings open; hint mode
  collects ≥8 dialog buttons; slate+dark applies live (body #212121, sidebar
  #171717, session radius 12px, meta #171717); persists across reload with
  a single theme link; custom CSS applies live and clears; reset restores
  defaults and empties storage; emulated dark scheme re-resolves system mode
  to paper dark (#181613) and back.
- Final: full suite 1820 passed; wiring guard passes with no new allowlist
  entries; check_js_refs ok. No deploy performed.

## 2026-09-06 — design round 1 (mock loop)

- Reviewed all six landed variants on the mock repointed at repo CSS.
  Defects fixed: paper-dark chrome glare (chrome buttons/seams/badges/toast →
  hairline; content containers keep ink), clay active-card de-heavied
  (hairline perimeter + single accent left edge), clay-dark blockquote and
  user-bubble surfaces separated from assistant surface, slate-light inline
  code stepped from surface-code (#f7f7f7, faint) to wash (#f1f1f1).
- Contract tests updated: the zero-specificity :where() scoping is the
  invariant, not the literal prefix string — mode conditions
  ([data-mode="light|dark"] / :not([data-mode="dark"])) now allowed inside
  :where(); paper.css may hold dark-scoped component rules. AGENTS.md updated.
- Measured (computed-style probes on mock, not eyeball): paper-dark chrome
  #3a362d / content ink #e6e1d7; clay-dark active card accent left edge.
  First attempt missed the .topActions/.pill/.sidebar-header compound rule
  (base (0,2,0) beats theme .icon-btn (0,1,0)) — fixed by mirroring the
  compound. This is the second instance of the specificity-leak class;
  the correct workflow is probe-first, not eyeball-first.
- Commits: 74c7ebea (contract tests), 2685a001 (paper-dark chrome),
  0acdc029 (clay), 2d987161 (slate light code), 66eb2a3e (AGENTS.md).
- Fixed pre-existing bug surfaced by the architecture run: hint-mode badges
  invisible over showModal() top-layer dialogs (settings/voice/edit/file
  viewer). Badge layer now enters the top layer as a manual popover
  (app_hint_mode.js) with graceful fallback; VM test with show/hide spies;
  bundle rebuilt. Commit d3068593.
- Dispatched design-matrix harness build: executor (terra), run
  68c17227-c87b-4fce-81f3-448b42bfa7aa — task-local design_review.sh reusing
  the docker_theme_verify container pattern + design_fixture.py, capturing
  8 surfaces × 6 variants desktop + 2 mobile with report.json probes.

## 2026-09-06 — design round 2 (real-app matrix)

- Harness delivered + rerun at HEAD b7ca1d5b: /tmp/codoxear-design-review.nwLyes
  (60 captures, probe-matched, PASS). Fixture renders a rich transcript in the
  real app (ctx chip 93% live from fixture usage tokens; serif h2 confirmed by
  crop zoom).
- Reviewed per rubric: paper-dark (all 8 surfaces: PASS after round-1 chrome
  fix), clay-light (PASS), clay-dark (PASS), slate-light (PASS), slate-dark
  (PASS), paper-light (regression-clean), mobile spot checks (PASS).
- Measured contrast (WCAG rel-lum) on the mock, all 6 variants x 5 pairs:
  only failure = clay text-bearing primaries (light 3.84, dark 3.07 vs 4.5
  target). Fixed: clay light gains --accent-strong #b35739 for primaries
  (hover #a04c31), decorative accent stays #c96442; clay dark keeps lighter
  terracotta fill and flips to dark text --on-accent #2b2118. Re-measured:
  4.76 / 5.04. Commit (clay contrast).
- NOTE measurement artifact: slate flat assistant has transparent bg; naive
  backgroundColor parsing reads rgba(0,0,0,0) as black. Contrast probes must
  walk to the effective backdrop. (paper/clay measure true bubble surfaces.)
- Extended harness (resumed run 6c3a2c9d): +fileviewer/filepicker/edit
  surfaces per variant.

## 2026-09-06 — design round 3 (extended surfaces + Monaco defect)

- Harness extension delivered: /tmp/codoxear-design-review.55chym — 78
  captures (added filepicker/fileviewer/edit per variant), report pass=true,
  no skips.
- Verdicts: edit dialog in clay-light/slate-dark cohesive (Save shows the new
  accent-strong in clay-light — confirms the contrast fix in the real app);
  file picker field + viewer chrome themed correctly in dark variants.
- ONE real defect found by the extended pass (harness subagent flagged it):
  the Monaco code surface is always the hardcoded codoxear-github-light theme
  (app_file_editor.js), white on dark shells in all three dark variants.
  Root cause class: third-party widget with its own theme system outside CSS
  tokens — the class of thing token coverage cannot see.
- Dispatched Monaco theming to executor (fable-5-1, run 8d2727de): six
  codoxear-<family>-<mode> monaco themes (values specified per variant,
  paper-light byte-preserved), live setTheme via themeController.subscribe
  wired through the loader, VM tests, bundle rebuild, docker_theme_verify +
  design_review re-run as evidence.

## 2026-09-07 Monaco follows the UI theme (commits 60c2f8b0, 185273ad)

- Cause of the white-editor-on-dark-shell finding: app_file_editor.js defined a single
  hardcoded `codoxear-github-light` Monaco theme and both editor create paths used it
  as a constant. Fixed by a six-theme table (`codoxear-<family>-<mode>`), a loader
  subscription to the theme store, and creation using the loader's current name.
- Full suite: 1822 passed, 112 subtests (.venv pytest). Wiring guard passed.
- scripts/docker_theme_verify.sh HEAD: PASS (22 checks incl. new
  monaco_opens_paper_light / monaco_follows_live_dark_switch);
  artifacts /tmp/codoxear-theme-verify.vwO704 (17-fileviewer-paper-dark.png shows the
  live switch; probe bg rgb(32,29,23), class vs-dark).
- design_review.sh HEAD: PASS 78 captures, 0 mismatches; artifacts
  /tmp/codoxear-design-review.kCM3mn. Editor-region means: paper-dark (38,35,29),
  clay-dark (36,33,28), slate-dark (30,30,30). paper-light editor region is
  pixel-identical to the 55chym baseline; clay-light moved from white to the
  design's #f2ede2 cream (design item 1), slate-light to #f7f7f7.

## 2026-09-06 — Monaco theming + task close-out

- Monaco fix landed (60c2f8b0 + 185273ad): six codoxear-<family>-<mode>
  themes; live setTheme via themeController.subscribe wired through the loader
  chain; creation requires the current name (throws rather than silently
  defaulting); verifier gained a live-switch check (existing editor repaints
  vs-dark under emulated OS scheme flip).
- Deviation accepted: clay-light/slate-light editor surfaces moved from white
  to family code surfaces (cream / #f7f7f7) per the spec table — my
  "unchanged" wording for light variants was wrong; paper-light is the only
  byte-preserved one (pixel-identical baseline diff).
- My verification: viewed slate-dark-fileviewer (dark editor ✓) and
  clay-light-fileviewer (cream ✓) from /tmp/codoxear-design-review.kCM3mn;
  full suite 1822 passed + 112 subtests; wiring guard passed; tree clean.
- FINAL: 18 commits beb463ef..185273ad. docker_theme_verify PASS (23 checks
  incl. Monaco live-switch), design_review 78 captures 0 mismatches at HEAD.
- Deferred (documented, user-aware): queue viewer + login page uncaptured
  (queue needs live queue, login themes from stored prefs on revisit);
  --sw-* swatch palette duplication in app.css (intentional); server-side
  theme sync (per-device localStorage is the designed scope).

## 2026-09-06 — design round 4 (luminance-boundary analysis, e-ink)

- Method: computed luminance deltas between adjacent-surface token pairs for
  all 6 variants from the actual theme files, plus pure-grayscale conversions
  of the real-app captures (e-ink approximation: hue disappears, luminance +
  borders carry everything).
- Result: paper-light — zero weak boundaries; grayscale capture confirms all
  structure survives. E-ink premise holds by construction.
- Defects found + fixed (border-strengthening round): paper-dark
  border-subtle #36322a→#454037, border-faint #2e2b24→#3a362d; clay-dark
  border-subtle #38332a→#453f33, border-faint #322d25→#3e382d; slate-dark
  surface-code #181818→#161616 + border-subtle #333→#3a3a3a, monaco slate-dark
  editor bg followed (#161616) + pinned test updated + bundle rebuilt.
  Commits 5a2097d5, 3242c8de, 1e641854.
- Judged NOT defects: slate-light card/bg 0.0000 and transparent assistant
  (ChatGPT borderless pattern — separation via hover/active fills);
  paper-dark weak fill deltas are all backed by strong ink borders.
- Harness extension running (run 9f38db85): hover-card/hover-icon/focus-ring/
  queueviewer/slashmenu/unattended/modelpicker captures per variant.

## 2026-09-06 — design round 5 (interaction states) + final gates

- Harness extended: 120 captures at /tmp/codoxear-design-review.BY7gEJ (HEAD
  1e641854, includes all fixes). New surfaces per variant: hover-card,
  hover-icon, focus-ring, queueviewer (seeded session_queues.json), slashmenu,
  unattended, modelpicker. report: pass=true, 0 mismatches, 0 skips.
- Reviewed: hover-reveal actions on slate-light (opaque fills, no bleed),
  queue viewer on paper-dark (Recovery labels, danger delete, hairline item
  boxes), focus ring on clay-dark (visible on queue button), slash menu on
  clay-light (wash highlight), model picker on slate-dark (mono list,
  shadow-pop elevation), unattended menu on paper-dark (inputs legible).
- Subagent's "dark file viewers still bright" note was STALE — the actual
  capture at 1e641854 shows the dark editor; verified by eyeballing the PNG.
  (Lesson reconfirmed: observations about artifacts go stale; check the
  artifact, not the recollection.)
- Loop-convergence rule met: two consecutive outer rounds with zero new
  defects; measurable invariants (contrast, boundary deltas) pass.
- Final gates at HEAD 1e641854: full suite 1822+112 pass;
  docker_theme_verify.sh PASS (/tmp/codoxear-theme-verify.fNNmw7).
- TASK COMPLETE. 22 commits beb463ef..1e641854. Deploy is the user's call:
  scripts/deploy.sh 1e641854.

## 2026-09-06 — deploy

- scripts/deploy.sh 1e641854: snapshot worktree updated, bundle rebuilt from
  snapshot source (608.4kb), wiring guard passed, pipx reinstall, smoke test
  passed ("initial load and reload rendered a session card"), service
  restarted. Only codoxear-server.service touched; live brokers/sessions
  untouched.
- Post-deploy (read-only): / → 200, /api/sessions → 401 unauth, boot script
  key codoxear.ui.theme.family present in served index.html, all three
  /static/themes/{paper,clay,slate}.css → 200.

## 2026-09-06 — post-deploy user bug report (iPhone screenshots)

User-reported, from deployed 1e641854:
1. "borders get cropped" — screenshots show the whole page uniformly ~1.2x
   magnified behind/over the settings dialog (chat clipped both sides,
   topbar off-screen). Signature = iOS Safari auto-zoom on a <16px text
   field, persisting after keyboard close. Existing anti-zoom rule covers
   text/password/date/time/datetime-local/search/number/select/textarea but
   MISSES url/email/tel and inputs with no type attribute. Alternative
   mechanism (dialog width overflow at 393pt) to be ruled out by measurement
   in Docker at 393x852.
2. Settings dialog structure weird — voice & notifications behind a
   drill-down row. Design decision (mine): flatten to ONE settings dialog,
   voice section inline below Appearance, retire the voice viewer.
3. iOS top status-bar area lags theme switch — meta IS written live by
   app_theme.js render(), but index.html hardcodes #ffffff until app boot
   (login page/first paint ignore stored theme) and iOS Safari applies
   theme-color on navigation, not continuously. Fix: boot script writes meta
   from stored prefs; document the platform residual.

Dispatched all three to executor (fable-5-1, run f80d72ca-284e-45da-adf5-9845462c5211)
with measurement-first mandate for bug 1.

## 2026-09-06 — user design feedback: active-card edge bar vs rounded corners

User screenshot (IMG_3091, clay-light iPhone): the active card's accent edge
renders as a detached pill sticking past the card's rounded corners. Root
cause chain: base 3px border-left artifact → my inset-shadow fix is fragile
on iOS composited layers (sessionContent is translate3d/will-change) — and
deeper: the leading-edge bar is a SQUARE-design idiom that doesn't transfer
to rounded cards.
Fix (design): clay active card = warm fill + uniform 1px accent perimeter
(borders follow border-radius; no bar, no shadow). Paper keeps the 3px edge
(square, works there); slate unchanged (fill-only, ChatGPT parity).
Commit 897bc50d; mock-verified both clay modes (perimeter follows radius);
deployed immediately (deploy.sh 897bc50d, smoke passed) since the user is
hitting it on their phone.
Lesson recorded: iOS compositing makes inset box-shadow on transformed
rounded elements unreliable — prefer plain borders for state indication.

## 2026-09-06 — active-card iteration 2: nested-radius/iOS clip fix

User: perimeter fix "even uglier than leading bar... card and border have
different corner radius". Mechanism: .sessionContent (the fill layer) never
had its own radius — it relied on .sessionSwipe's overflow:hidden clip, which
iOS Safari renders badly for composited children (the fill is
translate3d/will-change for swipe). Chromium clips correctly, which is why my
desktop checks were clean. Fix in base app.css: .sessionContent self-rounds
concentrically — border-radius: max(0px, calc(var(--radius-card) - 1px)) —
no reliance on the ancestor clip. Paper unaffected (0 -> 0); extraction test
now evaluates the concentric expression numerically and .sessionContent is
registered under --radius-card. Commit 6c898185, deployed.
Durable lesson for AGENTS.md: layered components must self-round; never rely
on ancestor border-radius+overflow clipping of composited children.

## 2026-09-06 — bug batch landed (f80d72ca) + deploy

- FALSIFIED HYPOTHESIS: my iOS auto-zoom theory for "borders get cropped" was
  wrong. Subagent measured the 3x-DPR screenshot: swatch card = 92 CSS px on
  both phone and Chrome; 1px borders exactly 3 device px — no magnification.
  The real defect: .themeSwatch.active outline (2px + 1px offset) paints 3px
  outside the button; the full-width grid inside .formBody's overflow clip
  cut the ring on the outermost side. Fix: padding: 3px on .themeSwatches
  (the same outset-absorption device .choiceChips uses). Lesson: measure the
  pixels before theorizing the platform.
- Anti-zoom rule broadened anyway (input:not([type]), email/url/tel) —
  audit found all existing entries were already typed+covered.
- Bug 2 landed: one flat Settings dialog; voice & notifications inline
  section; voice dialog viewer retired; voice controller drives via injected
  openSettings/closeSettings. Wiring guard clean, no allowlist additions.
- Bug 3 landed: boot script writes theme-color from stored prefs pre-paint
  (login page + first paint themed); iOS applies it on navigation (documented
  platform residual).
- Verified myself: 1847+112 tests pass at HEAD; flattened dialog reviewed in
  clay-light desktop + paper-dark mobile captures (ring fully visible, all
  sections inline, narrow layout fits).
- Deployed bddd680c (includes all bug fixes + card radius fixes + docs).

## 2026-09-06 — settings dialog polish round (user report IMG_3092)

User: content overlaps scrollbar; various reset buttons; custom CSS text too
large; "review design language before editing".
Design-language mapping (the constitution already had the answer):
- Labels already uniform (.fieldLabel = --font-md); section titles --font-lg;
  hints --font-sm. The real defects:
  1. .formBody had no scrollbar lane -> content ran under the iOS overlay
     scrollbar. Fix: padding-right var(--space-2) + scrollbar-gutter: stable.
  2. "Reset to default" rendered full-width because .field is
     flex-column/stretch (not a design choice). Both resets now content-width
     via .settingsSection .field > .text-btn; renamed "Reset to built-in
     prompt"; prompt hint dropped the reset-flow sentence.
  3. .customCssInput is --font-sm (12px) by design; the anti-zoom floor
     (!important 16px on coarse/small) made it the biggest text on the phone.
     Floor kept (removing it reintroduces the zoom class); field calmed
     (rows 6->4, hairline border). Unattended prompt rows 14->6 (the dialog's
     runaway-height source).
- Commit 08fa7a36; suite 1847 green; design_review at 08fa7a36 PASS
  (/tmp/codoxear-design-review.VYica0; mobile clay-light settings verified:
  swatch ring complete, quiet resets); deployed.
- PROCESS NOTE: ran design_review once against HEAD before committing the
  fix (harness builds from a COMMIT) — stale artifact; reran after commit.

## 2026-09-06 — form-system review approved; implementation delegated

User approved the plan ("delegate subagent to implement"). Dispatched executor
(fable-5-1, run 107837c3-68cf-4292-a0b6-76de1a8f57f1) with the exact plan:
unified dialog scroll-gutter (padding-inline 3px + scrollbar-gutter stable +
right lane; removes swatch/formBody piecemeal patches), global textarea
de-squash (composer owns its single-line sizing; dialog textareas share a
multi-line treatment; prompt becomes real textarea), mobile form-scale retune
(.formViewer/.formDialog: md 14 / lg 16 / sm 13 under coarse-or-narrow media;
AGENTS.md clause), Reset appearance removal (verifier resets via localStorage
clear), verification = suite + guard + docker_theme_verify + design_review
with capture review. Deploy NOT delegated — my call after capture review.

## 2026-09-06 — form-system fix pass (user: "system-level fix, not patches")

Six commits 10c18768..4bee4254 on main:
- 10c18768 one grouped gutter rule (.formBody, .helpBody, .queueList):
  padding 3px var(--space-3) 3px 3px + scrollbar-gutter: stable. Removed
  .themeSwatches padding: 3px and the 08fa7a36 .formBody lane patch.
  Grouped rule chosen over a shared class: three JS owners would each need
  a class edit and the wiring guard/tests gain nothing; the grouped
  selector keeps the contract in one CSS location the CSSOM test pins.
  .detailsGrid stays out (bordered table; its rows carry their own inset;
  a body gutter would detach row separators from the border).
- 45db50d4 bare textarea rule no longer squashes (height/min/max/resize/
  overflow moved to .composer textarea); .formViewer textarea shared
  multi-line treatment; .customCssInput keeps mono/meta/pre only.
  Side effect: .filePlainEditTextarea is no longer capped at 180px by the
  element rule (it has min-height 260px; the cap was a latent bug).
- ff1251a9 coarse/small .formViewer plane retune 13/14/16 + AGENTS clauses.
- 8abe473f Reset appearance removed (button, row rule, handler,
  themeController.reset()); verifier clears localStorage keys + reload.
- 1e07c155/4bee4254 verifier probes prompt/ring/planes + prompt capture.
Evidence: suite 1855 + 112 subtests; wiring guard clean, no allowlist
change; docker_theme_verify PASS (/tmp/codoxear-theme-verify.x48avQ);
design_review PASS 120 captures (/tmp/codoxear-design-review.nhdBu3).
Mobile probe 393x852: prompt height 161px rows 6 resize vertical; custom
CSS focused outset 3, leftClear 0, rightClear 20; planes title 16 = entry
16 > label 14 > hint 13. The 3px body shift vs. the header reads fine in
captures (header text at 375, body labels at 375 — the flex column's
padding sits on the outer body edge where nothing aligns to it).

## 2026-09-06 — form-system pass landed + deployed (4bee4254)

Executor (107837c3) implemented the approved plan in 6 commits
(10c18768..4bee4254):
1. Gutter: grouped rule (.formBody,.helpBody,.queueList) padding 3px left /
   space-3 right + scrollbar-gutter stable; .detailsGrid deliberately excluded
   (bordered table, rows own their insets; classification pinned by test).
   Piecemeal paddings removed.
2. Textarea de-squash: bare textarea rule no longer forces 44px single-line;
   composer owns its sizing; .formViewer textarea is the shared multi-line
   treatment; prompt renders 161px with resize handle. Latent bug found: the
   plain file editor was silently capped at 180px by the old element rule.
3. Mobile form scale: .formViewer retunes --font-sm/md/lg to 13/14/16 under
   the coarse/small media; measured title=16=entry > label 14 > hint 13.
4. Reset appearance removed (themeController.reset deleted as callerless);
   verifier resets via localStorage clear.
My review of captures (paper-light desktop, clay-light mobile, slate-dark
desktop, phone prompt): all clean. Suite 1855+112 green, wiring guard green.
Deployed 4bee4254.

## 2026-09-06 — checkbox component spec + unification

User: "checkboxes feel weird; how does the design language specify this?"
Answer was: it didn't. Two duplicated ad-hoc checkbox rules (.checkField and
.voiceToggleRow) with an 18px box (oversized vs --font-md labels), a
font-dependent text "✓" glyph, and a hard --ink border (harsh in clay).
Fix: design language gained a checkbox clause; one component (.checkField)
with 16px box / var(--border) ring / --radius-control corners, checked = fill
--accent + SVG-mask check in --on-accent (base gains --on-accent: var(--paper);
clay already had it; slate/paper resolve via fallback). voiceToggleRow
migrated to checkField and its rules deleted. Suite 1855 green; micro-render
verified all variants; design_review PASS at 40373f32.
