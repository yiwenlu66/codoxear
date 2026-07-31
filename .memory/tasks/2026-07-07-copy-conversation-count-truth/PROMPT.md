## Objective
Make the Copy Conversation success toast count the messages actually copied, not the raw exported event count.

Done when the implementation is committed, locally validated, Docker/browser proof shows truthful copied-message counts, clean-room review accepts the slice, and task/project memory records the accepted invariant.

## Workbench
1. Prove the current overcount with a failing discriminator: export events can include non-copyable roles or blank text that `formatConversationForCopy()` filters out while `copyConversation()` toasts `events.length`.
2. Implement a single-source counting mechanism aligned with the formatter's copyable parts.
3. Preserve the existing copied text format, transcript-too-large message, generic copy/export failure behavior, and message/code-block copy behavior.
4. Prove in Docker/browser with a transcript/export containing copyable user/assistant messages plus filtered events/blank rows.
5. Run clean-room adversarial review and record durable memory.

## Context
Active checkout: `/home/yiwen/codex-web-product-recovery` on branch `recovery/product-gaps`.
Protected checkout: `/home/yiwen/codex-web` on `main`; do not touch.
Next-slice scout: `/tmp/pi-subagents-uid-1000/artifacts/ec79d9f9-168f-4b39-a004-cf34b2f91b85_theorist_output.md` and `/tmp/codoxear-next-slice-after-code-copy.md`.
Relevant prior slices: `.memory/tasks/2026-07-07-copy-export-too-large-message/`, `.memory/tasks/2026-07-07-code-block-copy-buttons/`.
Project memory: `.memory/project/ARCHITECTURE.md`, `.memory/project/VALIDATION.md`.
Relevant source/tests: `codoxear/static/app.js`, `codoxear/static/app_conversation_copy.js`, `tests/test_frontend_conversation_copy_source.py`, `tests/test_transcript_export.py`.
Docker skill: `.codex/skills/codoxear-docker-test/SKILL.md`.

## Task specifications
Current mechanism: `copyConversation()` fetches `/api/sessions/<id>/messages/export`, receives `events`, formats copy text through `formatConversationForCopy(events)`, writes that text to the clipboard, and then shows `Copied ${events.length} messages`. The formatter filters to user/assistant events with non-empty trimmed text. Therefore raw events can outnumber copied sections, and the success toast can overstate what was copied.

Target mechanism: the success toast count must come from the same copyable-message selection used to build the clipboard payload. The formatter may keep returning a string for existing callers, but any count helper/result must share the same filtering path rather than reimplementing divergent logic. Toast grammar should be truthful for 0/1/many messages. The copied text format must remain unchanged.

Discriminator examples should include system/tool/non-user roles, blank assistant/user text, and valid user/assistant text. Expected copied count is the number of formatted `## User` / `## Assistant` sections, not raw event array length.

Browser proof should exercise the real Copy Conversation button and clipboard path in Docker. Use a deterministic synthetic session/export that includes filtered events so the old UI would overcount. Verify the clipboard text contains exactly the copyable sections and the toast count matches those sections.

## Constraints
Do not edit/promote/merge protected `/home/yiwen/codex-web` or `main`.
Do not touch live runtime dirs: `~/.local/share/codoxear`, `~/.claude`, `~/.codex`, host Pi logs/sockets, systemd/tailscale.
Docker-only for broker/server/session/tmux/browser verification; avoid port `8743`.
Cleanup must be exact-PID/container scoped; no `pkill -f`, `killall`, broad kills.
Keep functional, proof/evidence, review, and memory commits separate.
Browser + Docker evidence required for browser/product usability claims.
Delegate concrete implementation/validation work to executor subagents where possible.
Run clean-room adversarial review before yielding.
Do not copy secrets into committed artifacts; exclude cookies, auth headers, credential values, private file contents, bulky logs.
Monaco remains required; no plain textarea/diff fallback certification.
