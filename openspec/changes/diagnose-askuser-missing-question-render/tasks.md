# Tasks

## 1. Lock the diagnosis (already evidenced)

- [x] 1.1 Record in the change that the root cause for unrendered prompts in `broker-836953` is Claude CLI rejecting `AskUserQuestion` calls missing `questions[i].question` (`InputValidationError`), not a codoxear render bug — with the JSONL line/`tool_use_id` evidence.
- [x] 1.2 Add a triage note to repo docs/comment: for any "ask-user not visible" report, first grep the session JSONL for an `InputValidationError` `tool_result` matching the prompt's `tool_use_id` before editing codoxear code.

## 2. Display safeguard: header fallback

- [x] 2.1 In `codoxear/rollout_log.py::_claude_ask_user_questions`, keep/confirm the header→`question` fallback (replace the unconditional `if not question: continue` with: use `header` as displayed text when `question` is empty but `header` and options exist).
- [x] 2.2 In `codoxear/static/app.js`, confirm the interactive-prompt filter admits header-only questions (`q.question || q.header || options.length`) and renders `q.question || header || "(no question text)"`.

## 3. Surface upstream-rejected ask-user exchanges

- [x] 3.1 In `_extract_chat_events`, detect an `AskUserQuestion` `tool_use` whose matching `tool_result` (same `tool_use_id`) is an `InputValidationError` naming the missing `question` parameter, and mark that exchange as rejected.
- [x] 3.2 In `app.js`, render the rejected marker as a non-interactive notice ("prompt rejected by the agent — missing question text") instead of a silent/empty card.

## 4. Tests

- [x] 4.1 In `tests/test_ask_user_normalize.py`, add a case: header-only question object → emitted with displayed text from `header`, not dropped.
- [x] 4.2 Add a case: `AskUserQuestion` `tool_use` + paired `InputValidationError` `tool_result` → classified as rejected, not emitted as an answerable prompt.
- [x] 4.3 Add a regression case: well-formed single and multi-question prompts still emit correctly (no behavior change).

## 5. Verify

- [x] 5.1 Run `tests/test_ask_user_normalize.py` and confirm all cases pass.
- [x] 5.2 Replay session `broker-836953` log through `_extract_chat_events` and confirm: 4 well-formed prompts emit answerable, 2 header-only/rejected calls are surfaced as rejected notices (not blank).
