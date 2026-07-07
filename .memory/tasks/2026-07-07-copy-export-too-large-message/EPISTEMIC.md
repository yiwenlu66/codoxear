# Copy export too-large messaging epistemic model

## Phenomenon
Copy Conversation has a server-side transcript export cap. The guard is correct, but projecting its 413 as `copy failed: ...` makes a deliberate export-size refusal look like a clipboard/copy failure and leaks implementation terminology to the user.

## Accepted mechanism
`api()` throws non-OK responses as `Error` objects carrying `status` and `obj`. The oversized transcript export route supplies a discriminating shape: `status === 413`, JSON `max_bytes`, and an export/transcript-too-large error string.

`app_conversation_copy.js` maps only that shape to a specific copy-conversation toast naming the size limit: `Conversation too large to copy (max ...). Use search or copy a smaller range.` `copyConversation()` routes catch handling through `copyConversationFailureToast()`, which falls back to the previous generic `copy failed: ...` text for clipboard denial, network/auth/generic API errors, and unrelated 413s.

The server cap and route shape are unchanged. Active session and recovered/missing-session export guards still return 413 with `{error, max_bytes}`; unrelated 413s that lack `max_bytes` do not match the special projection.

## Evidence
- Frontend helper tests execute the conversation-copy module under Node and prove actual api-shaped 413/max_bytes/export-too-large errors produce a conversation-too-large message with `50 MiB`, while unrelated 413s, missing-limit 413s, and network errors return no special message (OPS 2026-07-07T15:32:00Z).
- App source-slice tests prove `copyConversation()` routes catch handling through `copyConversationFailureToast()` and that generic failures remain `copy failed: <message>` (OPS 2026-07-07T15:32:00Z).
- Message route tests prove active oversized transcript export still returns 413 plus `max_bytes`; missing recovered sessions retain the same cap behavior (OPS 2026-07-07T15:32:00Z).
- Docker/browser proof in `.memory/tasks/2026-07-07-copy-export-too-large-message/browser-artifacts/export-too-large-19464/` exercised the real browser Copy Conversation button against a Docker-only oversized log and lowered runtime export cap. The route returned `413`/`max_bytes=1024`, the browser toast was specific (`Conversation too large to copy (max 1 KiB)...`), generic `copy failed` wording was absent, and clipboard write was not reached (OPS 2026-07-07T15:40:00Z).
- Clean-room review `8e3f778c` accepted the slice with no blockers. It verified the three-gate discrimination mechanism, unchanged cap/route shape, generic failure preservation, lowered-cap proof validity, and absence of committed oversized/secret artifacts.

## Current claim
The copy-conversation oversized export slice is accepted: the UI now reports known transcript export-size refusals as conversation-too-large conditions with an actionable next step, while preserving server protection and generic copy failure behavior for other failures.

## Boundary
The browser proof uses a lowered Docker export cap to avoid committing a >50MiB log. Production-cap formatting is covered by unit tests with `max_bytes: 52428800`. This slice does not implement partial/ranged conversation copy; it only fixes the failure projection for whole-conversation export.
