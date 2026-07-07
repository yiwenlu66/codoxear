## Objective
Add per-code-block copy buttons to assistant transcript markdown so a user can copy one command/snippet from a multi-block agent response without selecting text or copying the whole message.

Done when the implementation is committed, locally validated, Docker/browser proof shows desktop and mobile code-block copy behavior, clean-room review accepts the slice, and task/project memory records the accepted invariant.

## Workbench
1. Render a copy control for every fenced/indented markdown code block in assistant transcript messages.
2. Wire click handling through the existing clipboard/toast infrastructure so the copied payload is exactly the target code block text, not surrounding prose or other blocks.
3. Preserve the existing per-message raw-markdown copy button and markdown rendering semantics.
4. Prove multi-block independence and mobile touch-target/no-overflow behavior in Docker/browser.
5. Run clean-room adversarial review and record durable memory.

## Context
Active checkout: `/home/yiwen/codex-web-product-recovery` on branch `recovery/product-gaps`.
Protected checkout: `/home/yiwen/codex-web` on `main`; do not touch.
Project memory: `.memory/project/ARCHITECTURE.md`, `.memory/project/VALIDATION.md`.
Next-slice scout: `/tmp/pi-subagents-uid-1000/artifacts/8ee46a64-a6e2-4202-aeb5-eb486c46a1a7_theorist_output.md`.
Current code: `codoxear/static/app_markdown.js`, `codoxear/static/app.js`, `codoxear/static/app.css`, `codoxear/static/app_message_rows.js`.
Relevant tests: `tests/test_markdown_renderer_source.py`, `tests/test_frontend_message_rows_source.py`, new source/VM tests as needed.
Docker skill: `.codex/skills/codoxear-docker-test/SKILL.md`.

## Task specifications
Current mechanism: `app_markdown.js` renders code blocks as `<pre><code...>...</code></pre>`. The only copy affordance in transcript rows is the message-level `.msg-copy-btn`, which copies the whole raw markdown message.

Target mechanism: each code block gets its own visible copy control associated with that `<pre>`. Clicking the control copies `code.textContent` for that block only, uses existing `copyToClipboard()`, prevents row/file-reference click side effects, gives local/user feedback, and leaves per-message copy unchanged. HTML escaping must remain correct; `textContent` should decode escaped entities back to the original code.

Mobile target: code-block copy controls are at least 44x44 CSS pixels on ~390px wide mobile viewport and do not create horizontal page overflow. Desktop may be visually compact but must remain accessible by keyboard/pointer.

Optional polish: language label may be included if it does not widen the scope or compromise the functional copy behavior. Do not let label polish delay the copy feature.

## Constraints
Do not edit/promote/merge protected `/home/yiwen/codex-web` or `main`.
Do not touch live runtime dirs: `~/.local/share/codoxear`, `~/.claude`, `~/.codex`, host Pi logs/sockets, systemd/tailscale.
Docker-only for broker/server/session/tmux/browser verification; avoid port `8743`.
Cleanup must be exact-PID/container scoped; no `pkill -f`, `killall`, or broad kills.
Keep functional, proof/evidence, review, and memory commits separate.
Browser + Docker evidence required for browser/product usability claims.
Delegate concrete implementation/validation work to executor subagents where possible.
Run clean-room adversarial review before yielding.
Do not copy secrets into committed artifacts; exclude cookies, auth headers, credential values, private file contents, bulky logs.
Monaco remains required; no plain textarea/diff fallback certification.
