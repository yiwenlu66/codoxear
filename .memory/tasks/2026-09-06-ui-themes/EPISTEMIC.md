# EPISTEMIC — current model of the theming problem

## Phenomenon

Codoxear's single "paper" design language is now fully tokenized (color +
geometry + type + focus + icon filter + elevation) and carries three families ×
two modes plus custom CSS, selectable from a real Settings dialog. Architecture is on main (bf96aa91…b8ef4527); design iteration is mid-flight:
round 1 (mock loop) and round 2 (real-app matrix) are done for the core
surfaces; extended surfaces (file viewer, picker, edit dialog) are being
captured next.

## What's established (with evidence)

- Phase 0 is appearance-neutral: computed-style diff old→new on the mock DOM
  is 0/427,500 properties (OPS 2026-09-06 architecture). Tests pin the token
  values and role wiring (tests/test_theme_token_extraction.py).
- A theme = palette blocks + a few structural rules. Structural rules must be
  `:where(:root[data-theme])`-prefixed to keep base specificity; the preview's
  bare `:root[...]` prefix was the leak mechanism (Ctx chip filled by
  `button{}`, .badge inversion blanked). Enforced by
  test_theme_token_coverage.py::test_family_component_rules_keep_base_specificity.
- Token coverage per family/mode is enforced (every color-literal :root token
  overridden; derived `var()` tokens follow their source automatically).
- Boot: inline script after app.css writes attrs + parser-inserted theme link
  + custom style via document.write (render-blocking → no flash; verified
  `reload_boots_into_slate_dark` with a single theme link). app_theme.js
  adopts those nodes (test: boot-node adoption) and swaps links on load.
- Docker: all 21 behavioral checks pass through the real Settings dialog,
  including system-mode live re-resolution under emulated scheme change.
- Custom CSS is client-only, debounced 250ms, flushed on dialog close.

## Resolved in design rounds 1–2 (OPS 2026-09-06)

- Hint badges over top-layer dialogs: FIXED — badge layer joins the top layer
  as a manual popover (d3068593), spy-tested.
- Paper-dark chrome glare: FIXED — chrome buttons/seams/badges → hairline,
  content keeps ink. Required mirroring the base `.topActions/.pill/...`
  compound selector (base (0,2,0) vs theme (0,1,0)) — measured, not eyeballed.
- Clay active card heaviness: FIXED twice — hairline perimeter + accent edge
  as radius-clipped inset box-shadow (straight 3px border-left sticks out past
  rounded corners; inset shadow clips correctly).
- Clay dark blockquote/user-bubble surface separation: FIXED (token bumps).
- Slate-light faint inline code: FIXED (wash fill, light mode only).
- Clay primary contrast (the one measured a11y failure: 3.84 light / 3.07
  dark vs 4.5): FIXED — light primaries use --accent-strong #b35739 (4.76),
  dark keeps #d97757 with dark --on-accent #2b2118 (5.04). Decorative accent
  uses keep --accent.
- Real-app matrix at b7ca1d5b: all 6 variants × 8 surfaces + mobile reviewed,
  no blocking defects. Serif h2 confirmed in app (crop zoom). Fixture
  transcript + ctx chip flow verified.

## Live beliefs / rough edges for the design phase

- Slate: topbar Ctx chip is a ghost button (transparent) — preview showed a
  filled pill only via the leak; decided deliberately: acceptable (ChatGPT
  parity), watch user feedback. Slate `.sessionContent` uses `--panel` (not
  transparent) so swipe actions don't bleed; hover fill is `--wash`.
- Swatch miniatures duplicate family palettes in app.css (`--sw-*`); they are
  the one intentional place a family's colors appear outside its file.
- `--radius-dot` was added as a fifth radius role (state/typing dots need 50%
  in clay/slate while pills need 999px) and clay has `--on-accent` for text on
  accent fills (preview hardcoded #fffdf9).
- Contrast probes must walk to the effective backdrop: slate's flat assistant
  is transparent; naive backgroundColor parsing reads rgba(0,0,0,0) as black.
- Still uncaptured: queue viewer (offline sessions have no queue), login page
  (themed via stored localStorage on revisit; first visit is paper-light by
  design), hover/focus states per variant.

## Ruled out

- Server-side persistence (v1). Separate dark boolean theme. Bare
  `:root[data-theme]` component prefixes.

## Next decisive evidence

None — task complete. All six variants passed the real-app matrix (78
captures, 0 mismatches), the Monaco third-party-widget gap is closed and
covered by a live-switch verifier check, and the full suite (1822) is green
at 185273ad. Deployment is the user's call: scripts/deploy.sh 185273ad.
