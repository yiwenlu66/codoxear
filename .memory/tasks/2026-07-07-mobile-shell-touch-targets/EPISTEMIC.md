# Epistemic model

## Phenomenon
Codoxear is a mobile companion UI, but several always-used shell controls remain below the 44x44 touch-target floor on phone-sized viewports.

## Current mechanism
- Accepted mobile fixes already give composer and file-viewer controls 44px targets.
- Current CSS still pins selected-session utility rails, chat navigation rails, topbar actions, and backend tabs to 34px or at most generic 40px. More-specific shell selectors beat the generic coarse-pointer override.
- The affected controls are frequent session-operation controls: File, Copy conversation, Details, Unattended, Search, previous/next navigation, sidebar header actions, topbar actions, and Codex/Pi/Claude backend selection.

## Working hypothesis
A mobile-scoped CSS patch can complete the shell touch-target floor without changing product semantics: override only the affected shell/backend-tab selectors in the phone media block with 44px dimensions, add source tests, and prove real browser geometry at 390x844.

## Evidence to collect
- Source tests showing the intended selectors live in the phone media block and retain base compact rules outside it.
- Local full-suite validation.
- Docker/browser proof at mobile viewport measuring target bounding boxes and body overflow, including New Session backend tabs.

## Current justified claim
No implementation is accepted yet. The target is current-code, bounded, and product-facing; acceptance requires code, tests, Docker/browser proof, clean-room review, and memory update.
