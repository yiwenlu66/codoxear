## Objective
Make Claude Code sessions surface the same topbar context-usage chip (`Ctx N%`) that Codex and Pi sessions already surface, using Claude Code assistant usage records and a conservative model context-window mapping.

## Workbench
- Add backend token extraction for Claude Code logs.
- Preserve existing Codex/Pi token behavior and frontend context-chip rendering.
- Validate locally and with Docker/browser evidence against a synthetic Claude Code session.

## Context
- Active checkout: `/home/yiwen/codex-web-product-recovery`, branch `recovery/product-gaps`.
- Protected checkout: `/home/yiwen/codex-web`; do not edit or promote.
- Scout: `/tmp/codoxear-next-product-slice-after-upload.md`.
- Implementation contract: `/tmp/cc-context-chip-next-slice-contract.md`.
- Current mechanism gap: `codoxear/rollout_tokens.py::_extract_token_update()` has Pi and Codex token branches but no Claude Code branch, so CC session rows keep `token:null` and `codoxear/static/app.js::setContext(null)` hides `#ctxChip`.

## Task specifications
- Add a Claude Code token parser for assistant rows, preferably in `codoxear/cc_log.py`, using existing `_message(obj, role="assistant")` helper semantics.
- Read `message.usage` and `message.model` from assistant records.
- Define CC `tokens_in_context` as `input_tokens + cache_read_input_tokens + cache_creation_input_tokens`. Anthropic context-window docs state all three count toward the request window under prompt caching. Do not include `output_tokens` in this input-context chip because the frontend labels the metric `Context input`.
- Resolve context window through a conservative model mapping. Current Anthropic docs: Sonnet 4.6+/Sonnet 5, Opus 4.6+/4.7+/4.8, Fable 5, Mythos 5/Preview have 1M-token windows; other Claude models including Sonnet 4.5 and Haiku 4.5 have 200k. Unknown/unmapped models must return no token update rather than a guessed number.
- Emit the existing token-update shape (`context_window`, `tokens_in_context`, `tokens_remaining`, `percent_remaining`, `reserved_tokens`, `max_input_tokens`, `as_of`) so the frontend needs no change.
- Wire the CC parser into `codoxear/rollout_tokens.py::_extract_token_update()` without changing Codex/Pi behavior.
- A CC session with a known model and assistant usage must project non-null `token` in `/api/sessions` and render `#ctxChip` in the browser.
- Unknown CC model must keep `token:null` and hide the chip.
- Do not change send, queue, attachment, busy/idle authority, frontend context rendering, or backend launch behavior.

## Constraints
- Docker-only for browser/server/session verification; avoid port 8743.
- Do not touch live host runtime dirs or protected checkout.
- Keep functional, proof, review, and memory commits separate.
- Do not read or commit secrets, cookies, raw bulky logs, npm artifacts, or ignored runtime files.
- Use synthetic Claude Code logs for proof; no real Claude inference or live credentials required.
