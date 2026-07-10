## Why

Claude `AskUserQuestion` prompts intermittently never appear in the codoxear web UI. Investigation of session `broker-836953` (log `a0a2d71e-…jsonl`) proves the root cause is **upstream of codoxear**: when Claude emits a question object carrying only `header` + `options` and no `question` field, Claude Code's own input validator rejects the tool call (`InputValidationError: The required parameter questions[0].question is missing`) and the prompt never enters an interactive state. The JSONL shows the rejecting `tool_result` landing ~2 ms after the `tool_use` (lines 219/220 and 294/295, 2026-06-27). There is nothing interactive for codoxear to render, so the card never shows.

This was repeatedly misattributed to the already-fixed final-question-submit bug (df3f58c) and to the in-flight header-fallback patch. Those are display-side concerns; neither can make Claude accept a malformed tool call. The diagnosis and the limited codoxear-side hardening need to be recorded so the next "ask-user not visible" report is triaged in seconds, not hours.

## What Changes

- Document the confirmed root cause: a missing `questions[i].question` field causes Claude CLI to reject the `AskUserQuestion` call before it becomes interactive; codoxear receives only a `tool_use` immediately followed by an `InputValidationError` `tool_result`.
- Specify codoxear's required behavior for such records: it MUST NOT silently drop a recognizable rejected-ask-user exchange — it SHALL surface that the prompt was rejected upstream so the user understands why no answerable card appeared.
- Keep the existing header→`question` fallback (working-tree patch in `rollout_log.py` / `app.js`) as a display safeguard for records that *do* reach a renderable state, and pin it with a regression scenario.
- No change to the final-question-submit protocol or the cursor model (covered by the archived `fix-claude-ask-user-final-submit`).

## Capabilities

### New Capabilities
- `claude-askuser-render`: how codoxear translates Claude `AskUserQuestion` tool-call records from the rollout JSONL into web-UI prompt cards, including handling of records that Claude rejected upstream and records missing the `question` field.

### Modified Capabilities
<!-- none: no prior long-lived spec exists for this capability -->

## Impact

- Code: `codoxear/rollout_log.py` (`_claude_ask_user_questions`, interactive-event extraction in `_extract_chat_events`), `codoxear/static/app.js` (interactive-prompt render path), `tests/test_ask_user_normalize.py`.
- Behavior: no change to answering flow for well-formed prompts; adds a visible signal for upstream-rejected prompts and keeps the header fallback.
- Diagnostics: triage rule — for any "ask-user not visible" report, first grep the session JSONL for an `InputValidationError` `tool_result` matching the prompt's `tool_use_id`.
