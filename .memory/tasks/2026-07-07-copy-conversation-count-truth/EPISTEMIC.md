# Copy Conversation count truthfulness epistemic model

## Phenomenon
Copy Conversation reports how many transcript messages were copied. Users interpret the success toast as a count of the sections that actually reached the clipboard.

## Accepted mechanism
`app_conversation_copy.js` owns the copyable-message selection in `conversationCopyParts(events)`: only `user` and `assistant` events with non-empty trimmed text become formatted `## User` / `## Assistant` sections. `formatConversationForCopyResult(events)` derives both `text` and `messageCount` from that same parts array. `copyConversation()` copies `formatted.text` and toasts `formatted.messageCount`, with singular grammar for one copied message. `formatConversationForCopy(events)` remains a string-returning wrapper over the result helper, preserving the previous contract.

## Evidence
- Pre-fix discriminator failed as predicted: six raw export events with only two copyable sections produced `Copied 6 messages`, and a single copied message produced `Copied 1 messages` (OPS 2026-07-07T18:29:00Z).
- Post-fix source/VM and route-adjacent validation passed: focused conversation-copy tests, transcript export/copy tests, full local pytest, and diff check (OPS 2026-07-07T18:29:00Z).
- Docker/browser proof exercised the real Copy Conversation button and clipboard path on desktop and 390x844 mobile: 4 raw export events, 2 copied sections, exact clipboard payload, toast `Copied 2 messages`, no overflow, zero broker mutations, auth 401/200 (OPS 2026-07-07T18:38:00Z; `browser-artifacts/copy-count-19518/VERIFICATION-REPORT.md`).
- Clean-room review accepted the slice: no count/text drift path exists because one result object feeds both clipboard and toast; string helper and failure behavior are preserved; tests discriminate old failure modes (OPS 2026-07-07T18:42:00Z; `reviews/copy-conversation-count-cleanroom-review.md`).

## Ruled out
- Raw export event count is no longer a valid success-count source; it counts filtered blank/non-copyable rows and can overstate copied content.
- Duplicating a predicate in `app.js` is unnecessary and avoided; the count stays formatter-owned.
- Changing `formatConversationForCopy(events)` to return an object would have broken existing callers; the string contract remains.

## Residual boundaries
Deterministic Docker proof validates Codoxear UI/API mechanics, not live provider behavior. Browser proof covers blank user/assistant filtering; role filtering is covered by VM/source tests with system/tool rows.

## Current claim
The Copy Conversation success toast is accepted: it now counts copied sections rather than raw export rows while preserving clipboard text format and established failure handling.
