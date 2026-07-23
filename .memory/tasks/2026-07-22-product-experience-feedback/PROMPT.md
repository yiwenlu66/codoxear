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

1. **[resolved] Codex reasoning effort list incomplete** — Commit `9fa7805`.

2. **[open] CWD combobox: combine recent + filesystem listing** — Current logic: recent-cwd fuzzy filter only. User wants historical cwds merged with live filesystem directory listing, non-blocking.

3. **[resolved] Codex provider/model dropdown only showed openai-api/default** — Commit `9fa7805`.

4. **[open] Session utility buttons cluttering layout** — file/copy/details/unattended buttons are in a separate `sessionContextBar` below the topbar, creating visual clutter. User wants them back in the topbar (pre-refactor position). Interacts with #5/#6/#8.

5. **[open] "Voice settings" should be "Settings"** — The sidebar button and dialog say "Voice settings" but should be the general "Settings" entry point to allow future settings items.

6. **[open] Confusing "down" button near jump-to-last and search** — The `nextUserBtn` (next user message) sits beside search and jump-to-last with only a down-arrow icon. Its affordance is unclear. Needs better icon/tooltip or grouping with prev-user.

7. **[resolved] Sidebar highlight sluggish on session switch** — Added `data-session-id` to sidebar cards and optimistic DOM class toggle in `openSession()` immediately after `selected = sessionId`. Commit `b007911`.

8. **[in progress] Redundant camera icon** — Separate `captureBtn` removed; camera merged into attach button via `accept="image/*,video/*,*/*"`. Tests being updated.

## Constraints
- Do not touch host `codoxear-server.service`, port `8743`, or host `~/.local/share/codoxear`.
- Fix against `/home/yiwen/codex-web-product-recovery`; deploy only to the preview container.
- Never expose credential values in git, memory, logs, screenshots, or responses.
- No `pkill -f` / `killall` / broad kills; cleanup is exact named-container or exact preview-session scoped.
- Restart the preview only when no backend turn is active.
- Functional commits, docs, and memory commits stay atomic.
- Do not claim fixed from tests alone; UX claims require browser/preview verification.
