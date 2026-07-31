## Objective
Make copy-conversation export-size failures understandable in the browser: when `/messages/export` returns the known transcript-size 413, the user should see a specific "conversation too large to copy" style message instead of a generic copy failure.

## Workbench
1. Implement a bounded frontend/API projection for transcript-export-too-large failures.
2. Preserve the existing 50MiB export guard and do not weaken server-side protection.
3. Add tests proving the specific user-facing copy failure message and the existing route shape.
4. Prove in Docker/browser with an oversized container-only transcript log that the toast is specific and no large artifact is committed.

## Context
Active checkout: `/home/yiwen/codex-web-product-recovery` on branch `recovery/product-gaps`.
Protected checkout: `/home/yiwen/codex-web` on `main`; do not edit.
Roadmap evidence: `/tmp/codoxear-next-slice-after-effort-markers.md` listed export 413 messaging as an open edge-case product gap after sidebar/model/reasoning/confirm slices.

## Task specifications
- Current mechanism: `copyConversation()` calls `/api/sessions/${sid}/messages/export`; server rejects logs larger than `transcript_export_max_bytes` with 413 and `{error, max_bytes}`; frontend catch renders generic `copy failed: ...`.
- Target mechanism: known export-too-large responses produce a concise user-facing toast such as `Conversation too large to copy (max 50 MiB). Use search or copy a smaller range.` Exact wording may differ, but it must be specific to conversation/export size and must not imply clipboard failure.
- Preserve generic error handling for non-413 copy/export/clipboard failures.
- Do not increase the export cap, stream huge logs to the browser, or commit oversized proof logs.
- Browser proof must create the oversized log only inside Docker runtime and commit only reduced evidence.

## Constraints
Do not touch `/home/yiwen/codex-web` or `main`.
Do not touch live runtime dirs (`~/.local/share/codoxear`, `~/.claude`, `~/.codex`, host Pi logs/sockets, systemd/tailscale).
Use Docker-only for broker/server/session/browser verification; avoid port 8743.
Cleanup must be exact container/PID scoped via sandbox commands; no `pkill -f`, `killall`, or broad kills.
Keep functional, proof/evidence, review, and memory commits separate.
Browser + Docker evidence is required for product usability claims.
Do not commit secrets, cookies, auth headers, credential values, private file contents, bulky logs, or ignored/oversized artifacts.
