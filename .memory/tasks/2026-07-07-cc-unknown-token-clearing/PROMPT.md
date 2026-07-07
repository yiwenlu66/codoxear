## Objective
Fix and prove the Claude Code context-chip stale-token boundary for unknown/unmapped model usage rows. The slice is done when a newer CC assistant usage row whose model has no conservative context-window mapping clears/hides any older known-model token projection instead of keeping a stale context chip, while known-model token math and no-usage rows keep their existing behavior.

## Workbench
1. Confirm the current stale-token mechanism with a synthetic CC log discriminator.
2. Design the smallest token-signal change that distinguishes no new token observation from a newer unsupported token observation.
3. Implement the fix without guessing context windows or changing the accepted CC token formula.
4. Add focused tests for batch extraction, latest-log scan, live/session token clearing, and preserved no-usage behavior.
5. Run focused/local validation, Docker/browser proof if the UI-facing claim requires it, clean-room review, and separate functional/proof/review/memory commits.

## Context
Repository: `/home/yiwen/codex-web-product-recovery` on branch `recovery/product-gaps`.
Protected checkout: `/home/yiwen/codex-web` on `main`; do not edit or promote it.
Project memory: `.memory/project/ARCHITECTURE.md` and `.memory/project/VALIDATION.md`.
The accepted CC context-token formula is `input_tokens + cache_read_input_tokens + cache_creation_input_tokens`; `output_tokens` is excluded.
The accepted selectable-backend proof used mapped model `claude-sonnet-4-5`; this task targets future/unmapped CC model rows after a mapped row.

## Task specifications
Unknown CC models must not project guessed context pressure. If the latest CC assistant row with a `message.usage` object has an unmapped model or otherwise cannot yield a conservative token update, Codoxear must clear the token projection for that session instead of retaining an older token from a previous mapped row.

A CC assistant row without `message.usage` is not a context-pressure observation and must not clear an existing token by itself.
Known-model usage rows must continue to project the exact accepted token dict, including cache input tokens and excluding output tokens.
The fix must reach all product token surfaces: session rows (`/api/sessions`), message polls/tail (`/messages/tail`/live), and browser `#ctxChip` through the shared token state.
The implementation should keep the clear signal internal. API responses should expose either a valid token dict or `null`, never an internal sentinel.

Acceptance criteria:
- Synthetic known→unknown CC usage log returns no latest token and stops scanning past the unknown row.
- Existing session token is cleared when refresh/live processing sees a newer unknown-model usage row.
- Known→no-usage CC assistant row preserves the previous token.
- Known-only and unknown-only behavior remain correct.
- Focused tests cover rollout token extraction/scanning and session/runtime token clearing.
- Browser/API proof is required if the final claim includes `#ctxChip` behavior.

## Constraints
Do not edit `/home/yiwen/codex-web` or `main`.
Do not touch live runtime dirs: `~/.local/share/codoxear`, `~/.claude`, `~/.codex`, host Pi logs/sockets, systemd/tailscale.
Use Docker-only for broker/server/session/browser verification; avoid port `8743`.
Do not copy secrets into committed artifacts.
Cleanup must be exact-PID/container scoped; no `pkill -f`, `killall`, or broad kills.
Keep functional, proof/evidence, review, and memory commits separate.
Run clean-room adversarial review before yielding.
Do not change CC context-window mappings unless direct evidence justifies the specific mapping.
Do not guess context pressure for unknown models.
