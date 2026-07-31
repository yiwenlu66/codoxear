## Objective
Make the context usage chip a truthful accessible control in the Codoxear shell while preserving backend token semantics. The slice is done when the chip is keyboard/screen-reader actionable, its visible/hidden behavior remains driven only by token updates, Docker/browser proof exercises click and keyboard activation, and clean-room review accepts the result.

## Workbench
1. Confirm the current chip construction, styling, and interaction path.
2. Implement accessible control semantics for `#ctxChip` without changing context token math or backend/session APIs.
3. Add focused regression coverage for DOM semantics, CSS preservation, and activation wiring.
4. Verify locally, in Docker, and through browser proof with a deterministic session.
5. Record review, evidence, and accepted invariant in task/project memory.

## Context
Repository: `/home/yiwen/codex-web-product-recovery` on branch `recovery/product-gaps`.
Protected checkout: `/home/yiwen/codex-web` on `main`; do not edit or promote it.
Project memory: `.memory/project/ARCHITECTURE.md` and `.memory/project/VALIDATION.md`.
Next-slice scout: `/tmp/codoxear-next-slice-current-scout.md` recommended this after mobile shell touch-target closure.
Docker validation skill: `.codex/skills/codoxear-docker-test/SKILL.md`.

## Task specifications
The existing `#ctxChip` displays context pressure and opens Details on click, but it is constructed as a `span.status-chip`. A clickable span is not a native keyboard/screen-reader control. Replace it with a native button styled as the existing status chip, or provide equivalent role/tabindex/key semantics if a button is mechanically impossible.

Acceptance criteria:
- `#ctxChip` is a native control or equivalent accessible control with a stable accessible name describing the context/detail action.
- Enter/Space activation opens the same Details path as pointer activation.
- Hidden/no-token state remains hidden and non-focusable.
- Token extraction/math, context-window mapping, `/api/sessions`, `/messages/tail`, backend launch/session semantics, and upload behavior do not change.
- Existing status-chip visual density is preserved; the control must not introduce desktop/mobile layout regressions.
- Source tests cover the semantic contract.
- Browser proof in Docker shows visible chip text/title, click activation, keyboard activation, and no backend send/key calls.

## Constraints
Do not edit `/home/yiwen/codex-web` or `main`.
Do not touch live runtime dirs: `~/.local/share/codoxear`, `~/.claude`, `~/.codex`, host Pi logs/sockets, systemd/tailscale.
Use Docker-only for broker/server/session/browser verification; avoid port `8743`.
Cleanup must be exact-PID/container scoped; no `pkill -f`, `killall`, or broad kills.
Keep functional, proof/evidence, review, and memory commits separate.
Run clean-room adversarial review before final yielding.
Monaco remains required; do not introduce plain textarea/diff fallback paths.
Do not change context token math or context-window mappings in this slice.
