# Epistemic model

## Phenomenon
Claude Code sessions previously lacked the context-pressure signal that Codex and Pi sessions expose through the topbar `Ctx N%` chip, even when Claude Code logs contained assistant usage records.

## Accepted mechanism
- Frontend `setContext(tok)` is backend-agnostic: a non-null token dict on session rows or message polls renders `#ctxChip`; null hides it.
- Claude Code assistant records carry the needed input-side usage under `message.usage` and the context-window key under `message.model`.
- `cc_token_update()` now parses only assistant rows, requires a known conservative model window, computes `tokens_in_context = input_tokens + cache_read_input_tokens + cache_creation_input_tokens`, excludes `output_tokens`, and emits the existing token dict shape.
- `rollout_tokens._extract_token_update()` now checks CC token updates after Pi and before existing Codex token-count rows, preserving the prior Pi/Codex mechanisms.

## Boundaries
- Unknown/unmapped CC models return no token update rather than a guessed window. Unknown-only CC logs keep `token:null` and hide the chip by the existing frontend rule.
- Stable session model identity is assumed, matching existing token-state semantics. A hypothetical known-model token followed by a later unknown-model row would retain the last known token until a future clear-token semantic exists; this was reviewed as a residual nonblocker, not part of the current slice.
- No send, queue, attachment, busy/idle, launch, or frontend rendering authority changed.

## Evidence
- Functional implementation and validation are recorded in OPS `2026-07-07T05:00:00Z`: focused tests passed, full local suite passed, and `git diff --check` was clean.
- Docker/browser proof is recorded in OPS `2026-07-07T05:08:00Z`: a synthetic CC session projected token state in `/api/sessions` and `/messages/tail`, and the browser rendered `#ctxChip` as `Ctx 18%`.
- Clean-room review is recorded in OPS `2026-07-07T05:13:00Z`: no blockers; reviewer independently confirmed scoped changes, token math, proof credibility, and unknown-model parser behavior.

## Current justified claim
Claude Code context-chip parity is accepted for known conservatively mapped CC models. The staged-upload workbench remains closed, and there are no known blockers in this CC context-chip slice.
