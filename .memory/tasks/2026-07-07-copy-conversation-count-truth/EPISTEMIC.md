# Copy Conversation count truthfulness epistemic model

## Phenomenon
Copy Conversation reports how many messages were copied. Users interpret that count as the number of transcript messages present in the clipboard payload.

## Current mechanism
`copyConversation()` uses `/messages/export` raw `events.length` for the toast. `formatConversationForCopy(events)` filters the same events down to user/assistant rows with non-empty text before producing clipboard text. If raw export events contain system/tool rows or blank user/assistant rows, the toast overcounts.

## Target mechanism
The success toast derives its count from the same copyable-message parts used to format the clipboard payload. The copied text format, transcript-too-large error message, generic failure behavior, and other copy affordances remain unchanged.

## Live risks
- Duplicating the formatter predicate in app.js could drift again; counting must share formatter-owned selection logic.
- Changing the formatter return type could break existing callers/tests; preserve `formatConversationForCopy(events)` as a string or update callers deliberately.
- Browser proof must exercise the real Copy Conversation button and not only a helper VM.

## Current claim
This is a small truthfulness slice: a success message currently counts a proxy (raw export events) rather than the target (copyable sections). The fix should be local to conversation-copy formatting/wiring.
