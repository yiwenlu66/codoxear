# EPISTEMIC

## Phenomenon
Codoxear's Claude Code context-pressure projection must avoid guessed values for unknown models. Current extraction conflates two states: no new token observation and a newer unsupported token observation.

## Current mechanism
`cc_token_update()` correctly returns `None` for unknown/unmapped CC models. `_extract_token_update()` then continues scanning older rows, so a known-model token before the unknown usage row can remain the selected latest token. Session/runtime selection preserves stored `session.token` when a poll/update yields `None`, so a later unsupported usage row cannot clear an older chip.

## Current claim
A known→unknown CC usage sequence can produce a stale context chip. The right product behavior is to clear/hide token pressure when the newest CC usage row cannot be mapped, while preserving existing tokens across assistant rows that have no usage object.

## Open implementation question
Represent the internal clear signal without leaking it into public API token fields, and ensure all token state mutation paths consume it: session meta refresh, live message polling, latest-log discovery, and message runtime snapshots.
