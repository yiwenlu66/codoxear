# EPISTEMIC

## Phenomenon
Claude Code context-pressure projection must never show guessed or stale pressure for a newer unsupported model. A CC assistant row with `message.usage` is an explicit context observation; if the row's model cannot be conservatively mapped to a context window, the correct public projection is absence of token data (`null`/hidden chip), not fallback to an older known-model row.

## Accepted mechanism
Codoxear now represents token extraction as a three-state internal observation:

- `TOKEN_NONE`: no token observation, e.g. a CC assistant row without `message.usage`; preserves prior session token state.
- `update`: a valid token dict; updates session/runtime/public token state.
- `TOKEN_CLEAR`: an observed token event that has no safe public token, e.g. an unmapped/invalid CC usage row; clears stored/runtime token state and stops latest-log scanning.

The public boundary consumes the internal observation through `.public_token`, so API and browser surfaces still expose only a token dict or `null`. `TokenObservation` itself is not serialized.

## Accepted claim
The stale-token defect is fixed in commits `e201ac9` (implementation), `768192e` (proof), and `43e5c06` (review). A newer CC unknown/unmapped model usage row clears older known-model context pressure across `/api/sessions`, `/messages/tail`/live runtime state, and browser `#ctxChip`. A CC assistant row without `message.usage` remains neutral and preserves the previous token.

## Evidence basis
- Focused and full local tests prove extraction, latest-log scanning, live-route clearing, session meta clearing, runtime snapshot clearing, no-usage neutrality, and known-model math preservation.
- Docker/browser proof in `browser-artifacts/cc-unknown-token-clear-19411/` shows mapped `claude-sonnet-4-5` usage projecting `tokens_in_context=4500` and browser `Ctx 98%`, then a later `claude-unmapped-future-9` usage projecting `token: null` in `/api/sessions` and `/messages/tail` and hiding/disabling `#ctxChip` while transcript rows remain visible.
- Clean-room review in `reviews/cc-unknown-token-clearing-cleanroom-review.md` audited all four token-state paths (`handle_messages_live`, `update_meta_counters`, `select_runtime_token`, `_find_latest_token_update`) and accepted the result with no blockers.

## Boundary
This proves Codoxear's deterministic log/session/browser mechanics for unknown CC model rows. It does not claim a context window for the unknown model, and it does not change real Claude provider/auth behavior.
