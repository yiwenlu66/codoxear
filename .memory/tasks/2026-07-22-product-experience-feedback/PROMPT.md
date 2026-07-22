## Objective
Iterate the refactored Codoxear product from the user's hands-on experience with the running Docker preview, until the user ends the feedback campaign. Each user comment is either fixed immediately (delegated subagent) or queued here as a tracked issue; this file's Task specifications section is the temporary issue tracker.

Done when the user declares the experience satisfactory or ends the campaign. No individual issue closes the task.

## Workbench
- No open issues yet. Awaiting first comment.
- Incoming comment handling rule: see Task specifications §Protocol.

## Context
- Product source: `/home/yiwen/codex-web-product-recovery`, branch `recovery/product-gaps`.
- Running preview: container `codoxear-preview-19580` → tailnet HTTPS `https://yiwen-workstation.tail0de6f7.ts.net:19581/`; password = same as main service; cookie name `codoxear_preview_auth`.
- Preview project mount: `/home/yiwen/codex_ws` (read-write) at `/home/tester/codex_ws`.
- Operations: `scripts/codoxear-docker-preview {status,logs,shell,restart,stop}`; restart only when active turns are idle.
- Prior task `2026-07-16-docker-product-preview-feedback` deployed the preview and fixed the cross-service cookie collision (`CODEX_WEB_COOKIE_NAME`); resolved items do not re-enter this tracker.
- Protected: `/home/yiwen/codex-web` (main), `codoxear-server.service`, port `8743`, host `~/.local/share/codoxear`.
- Validation baseline: full local pytest + Docker test/smoke + browser evidence for UX claims.

## Task specifications

### Protocol
For each incoming user comment:
1. Reproduce / locate the mechanism in the preview (not host live runtime).
2. Decide:
   - **Immediate fix** — delegate to a subagent with a bounded contract when the issue is well-scoped and the fix is clear. Redeploy the preview, verify from the browser, record evidence in OPS.md, mark the issue closed here.
   - **Queue** — when nontrivial, ordering-sensitive, or blocked on a decision, append a numbered issue to §Issues below with status `open`. Work open issues in priority order.
3. Every closed issue must record: observation → mechanism → intervention → verification (browser/user-facing, not tests alone).

### Issues

1. **[resolved] Codex reasoning effort list incomplete** — Added `minimal` to Codex SUPPORTED_REASONING_EFFORTS; sidebar markers already had `minimal:m` and `off:–`. Codex has no `off` or `max` (those are Pi/Anthropic concepts). Commit `9fa7805`.

2. **[open] CWD combobox: combine recent + filesystem listing** — Current logic: recent-cwd fuzzy filter only. User wants historical cwds merged with live filesystem directory listing, non-blocking. Design needed for async filesystem enumeration without UI lag.

3. **[resolved] Codex provider/model dropdown only showed openai-api/default** — Root cause: `read_codex_launch_defaults` returned no model list for custom providers, and preview lacked models_cache.json. Fixed by adding `provider_models_from_config()` to extract per-provider `models` arrays from config.toml, and updated preview to declare dexgem models + copy models_cache.json. Commit `9fa7805`.

## Constraints
- Do not touch host `codoxear-server.service`, port `8743`, or host `~/.local/share/codoxear`.
- Fix against `/home/yiwen/codex-web-product-recovery`; deploy only to the preview container.
- Never expose credential values in git, memory, logs, screenshots, or responses.
- No `pkill -f` / `killall` / broad kills; cleanup is exact named-container or exact preview-session scoped.
- Restart the preview only when no backend turn is active.
- Functional commits, docs, and memory commits stay atomic.
- Do not claim fixed from tests alone; UX claims require browser/preview verification.
