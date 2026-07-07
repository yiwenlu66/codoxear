## Objective
Prove and, if necessary, fix the browser-visible launch/send/outcome contract for selectable backend tabs at current HEAD. The slice starts with Claude Code (`cc`) using max reasoning effort because the Claude tab and CC token projection were recently touched. It is done when a Docker/browser run from the New Session UI shows the selected backend either launches, binds, accepts a send, and renders a truthful answer/error/no-response/recovery outcome, or fails at launch with a truthful visible failed-launch row and disabled session actions; silent idle or disappearing rows are failures.

## Workbench
1. Confirm current backend-tab launch paths, CC max-effort plumbing, and Docker sandbox constraints.
2. Drive the browser New Session flow for Claude Code in an isolated Docker sandbox.
3. If the backend cannot run in the sandbox, prove the visible failed-launch semantics and send/queue/attach blocking; if it can run, send a sentinel prompt and prove visible outcome semantics.
4. Fix any silent or misleading browser state exposed by the proof without adding sandbox-only UI or changing backend parser/token math unnecessarily.
5. Record Docker/browser artifacts, clean-room review, and accepted project memory.

## Context
Repository: `/home/yiwen/codex-web-product-recovery` on branch `recovery/product-gaps`.
Protected checkout: `/home/yiwen/codex-web` on `main`; do not edit or promote it.
Project memory: `.memory/project/ARCHITECTURE.md` and `.memory/project/VALIDATION.md`.
Current gap review: `/tmp/codoxear-next-slice-current-critic.md` recommended selectable-backend live parity and confirmed CC max reasoning effort is already implemented.
Docker validation skill: `.codex/skills/codoxear-docker-test/SKILL.md`.

## Task specifications
Backend tabs are product promises. Parser-only and fake-sidecar proofs are insufficient for the browser New Session path: the user selects a backend, starts it, sends a turn, and must see a truthful state. For this slice, start with Claude Code (`cc`) using reasoning effort `max`.

Acceptance criteria:
- Browser New Session exposes the Claude backend and `max` effort option without adding new controls.
- Starting a Claude session from the browser uses the requested backend/model/effort in launch metadata/argv when launch reaches broker creation.
- If the backend executable/config/provider is unavailable in Docker, the browser shows a truthful failed-launch row with useful Details/Copy/New-like-this behavior and no real-session actions; send/queue/attach/file-view must remain blocked for that failed row.
- If the backend launches and binds a log, a sentinel send must produce either an assistant answer or an explicit terminal backend error/no-response/recovery row in the transcript; silent idle is a defect.
- If assistant usage is present for a mapped model, `/api/sessions`, `/messages/tail`, and `#ctxChip` must agree; if no usage is present, no stale/guessed token is required.
- Docker/browser artifacts must prove the user-visible branch exercised, not only API internals.

## Constraints
Do not edit `/home/yiwen/codex-web` or `main`.
Do not touch live runtime dirs: `~/.local/share/codoxear`, `~/.claude`, `~/.codex`, host Pi logs/sockets, systemd/tailscale.
Use Docker-only for broker/server/session/browser verification; avoid port `8743`.
Do not copy secrets into committed artifacts. If authentication/config is needed, use only scoped Docker copies that are excluded from artifacts and never print values.
Cleanup must be exact-PID/container scoped; no `pkill -f`, `killall`, or broad kills.
Keep functional, proof/evidence, review, and memory commits separate.
Run clean-room adversarial review before yielding.
Do not implement CC max effort; it already exists. Treat `max` as a verification condition.
Do not change context token math or context-window mappings unless a directly observed backend-path defect requires it.
