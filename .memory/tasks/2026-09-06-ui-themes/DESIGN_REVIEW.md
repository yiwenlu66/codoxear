# DESIGN_REVIEW — rubric for per-theme iteration rounds

Purpose: make every screenshot review answer the same sharp questions, so a
theme is "done" by evidence, not by vibes. Apply per family×mode. A theme
passes a round only when every applicable line reads clean at desktop AND
mobile widths.

## 0. Universal invariants (all themes, both modes)

- State-dot language intact: busy = filled + pulse, idle = hollow,
  suppressed = filled no-pulse, pending = amber + pulse. Motion, not hue.
- 44px touch targets unchanged (hit-area ::after, not visual size).
- Focus visible: one clearly visible focus ring on every control class.
- Text contrast: body text ≥ 4.5:1 against its surface; muted ≥ 3:1;
  timestamps/placeholders may go lower but must remain legible on e-ink
  (paper) at arm's length.
- Active session card is unambiguous at a glance in a 10-card sidebar.
- Busy session is unambiguous in sidebar AND topbar (interrupt visible).
- No element relies on a hue pair alone (error/warning/info differ in
  surface + border + text weight, not just tint).

## 1. Surfaces to capture per round (real app, Docker)

Login page · sidebar with ≥4 cards in mixed states (busy+subagents, idle,
queue badge, terminal-owned, unattended badge) · topbar with Ctx chip ·
transcript with: user bubble, assistant markdown (heading, inline code, code
block + copy btn, table, blockquote, list), typing row w/ thinking tokens,
subagent activity row, error bubble, warning bubble · composer (empty,
multiline, staged attachment chips, queue badge) · slash-command menu ·
model picker popover · search bar with current/other marks · Settings dialog
· new-session dialog (backend tabs visible) · voice dialog · queue viewer ·
diagnostics viewer · help overlay · file viewer (text + pdf) · toast ·
network banner · empty state · confirmation dialog (destructive) ·
hint-mode overlay (f).

Mock loop covers: sidebar, chat/markdown, composer, topbar. Everything else
requires the Docker pass.

## 2. Per-theme watchlist

### paper-light
Regression only. Diff computed styles vs pre-theme base; zero visual change
is the requirement, not an aspiration.

### paper-dark (highest risk)
- 1px light-ink rules on near-black: check chrome seams (sidebar/topbar/
  composer) for harshness; if harsh, borders drop to --hairline and only
  bubbles/cards keep full ink. Decide by screenshot, not principle.
- bubble-user (#262219) vs bubble-assistant (--paper #211e19) separation is
  subtle by design; confirm the user right-alignment carries it.
- Code block surface vs assistant bubble surface: need visible edge.
- mark.searchHit / searchHitCurrent legibility.
- badge.unattended (ink bg, paper text) — inversion glare on dark?
- Amber pending/queue on warm black: warm-on-warm mush risk.

### clay-light
- Accent #c96442 on cream: primary button contrast (white text on
  terracotta ≥ 4.5:1? measure; if short, white→#fffdf9 is already there,
  else darken accent for the button only via --ink-hover).
- Hairline chrome seams may vanish on some displays — check sidebar/topbar
  separation from bg.
- Serif headings: weight 600 at 18px — check it reads editorial, not noisy.
- session.active: accent left edge + wash fill — confirm not muddy.
- Hover fills on cards vs session-active fill: distinct?

### clay-dark
- bubble-user (#2d271f) vs bubble-assistant (--paper #26211b): warm-on-warm;
  likely needs the user bubble one step lighter or a hairline edge.
- Terracotta #d97757 on umber: vibrant enough as accent, not orange-neon.
- Code surface #1f1b16 inside paper bubble: edge visible.
- Serif on dark at small sizes: check h2/h3 don't look fuzzy.
- Queue/amber tokens: warm-dark harmony (amber must not look like dirt).

### slate-light
- Ghost icon buttons (borderless): affordance risk. Hover fill exists, but
  at rest they must still read as buttons — check against ChatGPT reference:
  they rely on icon familiarity + 32px grid. If weak, add hairline border.
- User bubble #f1f1f1 on white bg: visible boundary?
- Flat assistant messages: separation between consecutive assistant rows and
  vs the user bubble rhythm — timestamp position still sensible?
- session.active #ececec vs hover wash #f1f1f1: distinct fills.
- --shadow-pop on composer/dialogs: visible but not heavy.

### slate-dark
- Muted #a0a0a0 on #212121: ≥ 4.5:1 (measure; likely ok).
- Code block #181818 inside flat assistant on #212121: needs border-subtle
  edge — verify visible.
- White primary button glare in dark room: acceptable (ChatGPT parity) but
  check disabled state doesn't look active.
- Error/warning flat assistant variants keep surfaces (rule exists).
- mark.searchHit (#5c5320) on dark: current mark is ink-filled — ok.

## 3. Round protocol

1. Inner loop (fast): mock pointed at the repo's current app.css +
   themes/*.css; shoot all variants; fix tokens/rules; repeat.
2. Outer loop (truth): commit → docker screenshot matrix (all 6 variants ×
   surface list) → review against §0 + §2 → defect list → fix → next round.
3. A theme is DONE when: two consecutive outer-loop rounds produce zero new
   defects, and §0 universal invariants all pass by measurement where
   measurable (contrast ratios computed from computed styles).
4. Record each round in OPS.md: commit, screenshots dir, defects found/fixed,
   residual judgments.

## 4. Defect taxonomy (use in round notes)

- contrast: measurable ratio failure
- separation: two adjacent surfaces indistinguishable
- affordance: control doesn't read as interactive
- hierarchy: wrong thing draws the eye first
- coherence: element belongs to a different language than its theme
- inheritance-leak: base rule survives where theme meant to override
  (the preview's four bugs were all this class)
