# Copy Conversation Count Truthfulness — Clean-Room Review

**Verdict: ACCEPT**

Branch: `recovery/product-gaps`
Commits: `cd691fb` (init), `a5e195a` (implementation), `abcbfd0` (browser proof)
Reviewer context: fresh, no prior involvement in this slice.

---

## 1. Count/text drift — NO DRIFT POSSIBLE

`conversationCopyParts(events)` returns a single array of formatted `## Role` parts. Both `formatConversationForCopy` (string) and `formatConversationForCopyResult` ({text, messageCount}) derive from this same array in the same call.

In `copyConversation()` (app.js:2189), one call to `formatConversationForCopyResult(events)` produces the `formatted` object. `formatted.text` goes to clipboard, `formatted.messageCount` goes to toast. They cannot diverge — same object, same function call, same underlying array.

The copyable-message predicate (role ∈ {user, assistant} AND text.trim() non-empty) is applied exactly once in `conversationCopyParts`. No second filtering path exists.

## 2. String contract and fail-loud — PRESERVED AND APPROPRIATE

`formatConversationForCopy(events)` still exists, still returns a string, delegates to `formatConversationForCopyResult(events).text`. Existing callers (if any) see identical output. Test `test_format_conversation_for_copy_preserves_existing_contract` verifies `resultText === text`.

The app.js guard (lines 414–421) checks all three exports:
- `formatConversationForCopy` (existing)
- `formatConversationForCopyResult` (new)
- `transcriptExportTooLargeCopyMessage` (existing)

Missing or partial → hard throw `"Codoxear conversation-copy helpers failed to load"`. Test `test_app_conversation_copy_guard_throws_for_missing_or_partial_helper` exercises missing, empty, partial (only format), partial (format+tooLarge but no result), and complete. The new `missing_result` case (format+tooLarge but no formatConversationForCopyResult) correctly throws.

`Object.freeze` on the exported module prevents monkey-patching. Verified by tests (`frozen: true`).

## 3. Test discrimination — DISCRIMINATES OLD BUG

`test_app_copy_conversation_success_toast_counts_copied_messages_not_raw_events`:
- 6 raw events (system, user, blank-assistant, tool, assistant, blank-user)
- Old code: `events.length` = 6 → "Copied 6 messages" → test FAILS
- New code: `messageCount` = 2 → "Copied 2 messages" → test PASSES
- Also verifies clipboard contains exactly 1 `## User` and 1 `## Assistant`, excludes system/tool text.

`test_app_copy_conversation_success_toast_uses_singular_message_grammar`:
- 1 event → "Copied 1 message" (singular)
- Old code would produce "Copied 1 messages" (wrong grammar)

`test_app_copy_conversation_failure_toast_preserves_generic_failures`:
- 413 transcript-too-large → specific message (unchanged)
- Generic error → "copy failed: denied" (unchanged)
- Null error → "copy failed: unknown error" (unchanged)

`test_format_conversation_for_copy_returns_empty_for_no_copyable_text`:
- Now also checks `messageCount = 0` for filtered-only events.

## 4. Browser proof — SUBSTANTIATES USER-VISIBLE CLAIM

Desktop and mobile evals both confirm:
- Export returned 4 raw events (2 user, 2 assistant — including blank text rows)
- Clipboard contained exactly 2 sections (1 user, 1 assistant with real text)
- Toast: `"Copied 2 messages"` (not "Copied 4 messages")
- `hasRawEventOvercountToast: false` — old bug would have shown "Copied 4 messages"
- No horizontal overflow on either viewport
- Clipboard text verified exact match to expected formatted output

Broker isolation: 549 state polls, 0 send/keys/shutdown calls — zero mutations.
Auth: 401 before login, 200 after.
Cleanup: exact-PID fake broker kill, exact-container sandbox stop, final port/container clean.

Sanitization: cookie value redacted, password placeholder in COMMANDS-RUN.md, server.log contains only metadata warnings, no secrets in any artifact.

Minor note: cleanup.log shows "fake broker still alive pid=52" after kill — process inside Docker wasn't reaped before the check, but `codoxear-docker-sandbox stop` removed the container. Final checks clean. Cosmetic only.

## 5. Category errors — NONE FOUND

**Count definition**: Copyable sections = user/assistant events with non-empty trimmed text. This matches what `conversationCopyParts` produces and what appears in clipboard as `## User`/`## Assistant` sections. Correct definition.

**Role filtering**: The server-side export (`_read_chat_export_events` → `_single_chat_event`) can produce events with various roles depending on backend (CC backend can produce system/tool roles). Client-side filtering in `conversationCopyParts` is defensively correct. Unit tests exercise system/tool roles; browser proof exercises blank-text filtering (the more subtle path).

**Blank rows**: `text.trim()` check prevents blank sections. Tested with `"   "`, `"\n\t"`, `null`, `0`, `false` text values.

**Toast race**: No race. `formatConversationForCopyResult(events)` returns synchronously. Both clipboard text and toast count come from the same synchronous result object.

**Locale/time artifacts**: `toLocaleString()` for timestamps is pre-existing behavior, not introduced by this slice.

**Mobile/browser proof proxy**: Real Chrome browser sessions with real `#copyConversationBtn` click, clipboard interception, and toast text capture. Not a proxy — exercises the actual UI flow.

## Residual boundaries

1. `formatConversationForCopy` wrapper at app.js:2169–2170 is dead code within app.js (no callers after `copyConversation()` switched to `formatConversationForCopyResult`). Harmless — preserves backward compatibility if future code needs string-only form.
2. Browser proof exercises text-filtering (blank user/assistant events) but not role-filtering (system/tool events) in the real browser. Role-filtering is covered by unit tests. Acceptable split.
3. The `copiedConversationToast(0)` path produces `"Copied 0 messages"` but is unreachable because `if (!formatted.text)` branches to "No conversation to copy" first. No issue, but if the guard were ever removed, "Copied 0 messages" would surface.
