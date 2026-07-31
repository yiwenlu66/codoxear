## Objective
Complete the mobile touch-target floor for Codoxear's always-used shell command surfaces. On phone-sized viewports, selected-session utility controls, chat navigation controls, sidebar header controls, topbar controls, and New Session backend tabs must expose at least 44x44 CSS-pixel targets without changing desktop compact sizing or backend/session semantics.

## Workbench
- Add mobile-scoped CSS rules that override existing 34px shell/backend-tab selectors at phone widths.
- Add source/regression tests pinning the mobile shell selectors and preserving desktop/base compact behavior.
- Validate locally and with Docker/browser evidence at mobile viewport.
- Run clean-room review before closing.

## Context
- Active checkout: `/home/yiwen/codex-web-product-recovery`, branch `recovery/product-gaps`.
- Protected checkout: `/home/yiwen/codex-web`; do not edit or promote.
- Current-head scout: `/tmp/codoxear-next-slice-current-scout.md`.
- Current-head critic: `/tmp/codoxear-next-slice-current-critic.md`.
- Accepted prior slices: staged-upload expansion and Claude Code context-chip parity.

## Task specifications
- Current defect mechanism: selected-session command rails and New Session backend tabs keep 34px controls on phone-sized viewports because base selectors are more specific than the generic coarse-pointer `.icon-btn` 40px rule.
- Target surfaces:
  - `.pill > .icon-btn` and `.topActions .icon-btn` for topbar controls.
  - `.sidebar header .icon-btn` for mobile drawer header actions.
  - `.sessionContextBar .icon-btn` for File, Copy conversation, Details, Unattended.
  - `.chatNavRail .icon-btn` for transcript search and previous/next navigation.
  - `.agentBackendTab` for Codex/Pi/Claude backend selection in New Session.
- At `max-width: 520px`, every target above must have explicit 44px minimum/actual touch dimensions where needed to beat existing 34px rules.
- Preserve desktop/base compact sizing; do not globally enlarge every icon button.
- Preserve horizontal scrolling/wrapping behavior and prove the mobile shell has no body horizontal overflow.
- Do not change send, queue, attachment, busy/idle, backend launch, transcript parsing, Monaco, or upload behavior.

## Constraints
- Docker-only for browser/server/session verification; avoid port 8743.
- Do not touch live host runtime dirs or protected checkout.
- Keep functional, proof, review, and memory commits separate.
- Do not stage broad paths; stage explicit files only.
- Browser evidence must exercise the real served UI at phone viewport.
