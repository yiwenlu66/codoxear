## Objective
Fix and prove sidebar model-name display. The slice is done when sidebar session metadata includes the session model for sessions whose API row has a meaningful `model`, so users can distinguish same-project sessions by model without opening Details, while empty/default model values remain omitted to avoid noise.

## Workbench
1. Confirm the current sidebar projection omits `s.model` despite API availability.
2. Implement the smallest frontend projection change in the sidebar row builder.
3. Add source/runtime tests proving model text is included, default/empty values are omitted, and existing age/cwd/branch/effort behavior is preserved.
4. Run focused/local validation plus Docker/browser proof with multiple fake sessions using different models and a long model name on mobile.
5. Commit functional, proof, review, and memory changes separately.

## Context
Repository: `/home/yiwen/codex-web-product-recovery` on branch `recovery/product-gaps`.
Protected checkout: `/home/yiwen/codex-web` on `main`; do not edit or promote it.
Project memory: `.memory/project/ARCHITECTURE.md` and `.memory/project/VALIDATION.md`.
Next-slice scout: `/tmp/codoxear-next-slice-after-effort-markers.md`.
Current sidebar metadata line is built in `codoxear/static/app.js` around the `.sessionMetaLine` / `.metaText` row and currently renders `stateTxt | cwdBase | branchTxt` plus separate backend/owner/effort icons.
Session rows already expose `model` from `/api/sessions`; Details/diagnostics expose it, but the sidebar scan surface does not.

## Task specifications
Sidebar metadata must include a compact model segment when `s.model` is a meaningful string:
- Render order: `stateTxt | model | cwdBase | branchTxt`.
- Trim whitespace before display.
- Omit model when absent, empty, or case-insensitive `default`.
- Preserve existing cwd/branch behavior and existing backend logo, owner icon, effort marker, fast marker, title, and swipe behavior.
- Do not change server/session/API schema, backend launch defaults, provider choice semantics, Details, diagnostics, or model parsing.
- Long model names must truncate through the existing `.metaText` ellipsis; do not add horizontal overflow or new wrapping.

Acceptance criteria:
- A sidebar row with model `gpt-5.4` renders model text in the metadata line.
- A sidebar row with model `claude-sonnet-4-5` renders model text in the metadata line.
- A sidebar row with a long provider/model string renders without horizontal overflow on a 390px mobile viewport.
- Rows with `model: null`, empty, or `default` omit the model segment.
- Source tests cover the helper/projection and preserve existing metadata order.
- Docker/browser proof exercises actual sidebar rows and records desktop/mobile metadata text.

## Constraints
Do not edit `/home/yiwen/codex-web` or `main`.
Do not touch live runtime dirs: `~/.local/share/codoxear`, `~/.claude`, `~/.codex`, host Pi logs/sockets, systemd/tailscale.
Use Docker-only for broker/server/session/browser verification; avoid port `8743`.
Do not copy secrets into committed artifacts.
Cleanup must be exact-PID/container scoped; no `pkill -f`, `killall`, or broad kills.
Keep functional, proof/evidence, review, and memory commits separate.
Run clean-room adversarial review before yielding.
