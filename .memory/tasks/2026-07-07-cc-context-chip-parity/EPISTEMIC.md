# Epistemic model

## Phenomenon
Claude Code sessions lack the context-pressure signal that Codex and Pi sessions already expose via the topbar `Ctx N%` chip.

## Current mechanism
- Frontend `setContext(tok)` is backend-agnostic: when a session row or message poll carries a token dict, it renders `#ctxChip`; when token is null, it hides the chip.
- Codex token updates come from `event_msg`/`token_count` rows carrying `model_context_window` and `last_token_usage.total_tokens`.
- Pi token updates come from `pi_token_update()`, which reads assistant `usage.totalTokens` and resolves model context windows through Pi model config/registry.
- Claude Code logs have assistant `message.usage` and `message.model`, but `rollout_tokens._extract_token_update()` has no CC branch, so CC token remains null.

## Working hypothesis
A bounded additive parser can restore CC parity: parse CC assistant usage, compute input-context tokens from prompt-side usage fields, resolve a known model context window conservatively, and return the existing token-update shape. No frontend or runtime authority changes are required.

## Key decisions
- `tokens_in_context = input_tokens + cache_read_input_tokens + cache_creation_input_tokens`.
- `output_tokens` are excluded from the chip because the frontend metric is input context and should match the pre-response prompt load.
- Unknown CC model means no token update, not a guessed context window.

## Evidence to collect
- Unit tests for known/unknown model parsing and rollout token extraction.
- Local full-suite validation.
- Docker/browser proof using a synthetic CC sidecar/log showing `/api/sessions[*].token` non-null and visible `#ctxChip` for a known model.

## Current justified claim
No implementation is accepted yet. The target is mechanism-identified and bounded; acceptance requires code, tests, Docker/browser proof, clean-room review, and memory update.
