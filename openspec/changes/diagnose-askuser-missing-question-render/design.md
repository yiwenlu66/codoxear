## Context

Codoxear renders Claude `AskUserQuestion` prompts by parsing the agent's rollout JSONL (`codoxear/rollout_log.py::_claude_ask_user_questions`, emitted as an `interactive: "ask_user_question"` event in `_extract_chat_events`) and drawing a prompt card in `codoxear/static/app.js`. There is no separate live PTY/tmux channel for prompts — the card exists **only** if a renderable interactive event is produced from the log.

Confirmed evidence from session `broker-836953` (log `a0a2d71e-…jsonl`, 2026-06-27):
- 6 question objects across 5 `AskUserQuestion` calls.
- The 4 objects with a real `question` field were answered normally.
- 2 calls (lines 219, 294) carried only `header` + `options`, no `question`. Each was followed ~2 ms later by a `tool_result`: `InputValidationError: The required parameter questions[0].question is missing`.

So the unrendered prompts were **rejected by Claude Code's input validator before ever becoming interactive**. The codoxear server (running PID started Jun 16 07:35) already loads the working-tree header-fallback patch and its live API even synthesizes `question` from `header` — yet the user still saw nothing, because the underlying tool call never produced an answerable prompt in the agent.

Constraint: codoxear cannot retroactively make Claude accept a malformed call. Its only levers are (a) display anything that *does* reach a renderable state, and (b) make the rejected case observable instead of blank.

## Goals / Non-Goals

**Goals:**
- Pin the root cause in a spec + regression test so future "ask-user not visible" reports are triaged against the JSONL `InputValidationError`, not against codoxear render code.
- Keep the header→`question` display fallback for records that reach a renderable state.
- Ensure a rejected ask-user exchange is not presented as a silently empty/answerable card.

**Non-Goals:**
- Changing the final-question submit protocol or cursor model (archived `fix-claude-ask-user-final-submit`).
- Pi ask_user handling (separate `fix-pi-ask-user-cursor-events`).
- Fixing Claude itself or pre-validating tool input before Claude sees it — out of codoxear's control.

## Decisions

**D1 — Classify root cause as upstream rejection, not a render bug.**
The `tool_use`→`InputValidationError` `tool_result` pair (matched by `tool_use_id`, sub-second apart) is the signature. Rationale: the data pipeline (`_extract_chat_events` → live poll → `appendEvent`) was verified to carry interactive events end-to-end; the only records that fail are ones Claude rejected. Alternative considered: continue patching the renderer — rejected, because the renderer never received a renderable event for those calls.

**D2 — Keep header fallback as a display safeguard, not a fix for rejection.**
`_claude_ask_user_questions` currently does `if not question: continue` on HEAD, dropping header-only objects. The working-tree patch falls back to `header`. Keep the fallback: it helps any record that *does* reach the UI (e.g. partially-formed multi-question prompts where one object has a header). It does not and cannot un-reject the Claude call. Document this boundary explicitly so the fallback is not mistaken for the fix.

**D3 — Surface rejected ask-user exchanges.**
When extraction sees an `AskUserQuestion` `tool_use` whose matching `tool_result` is an input-validation error, emit/annotate so the UI can show "prompt was rejected by the agent" rather than nothing. Minimal form: detect the paired error during event assembly and attach a marker the frontend can render as a non-interactive notice. Alternative considered: leave silent — rejected, it is exactly the failure mode that cost hours of misattribution.

## Risks / Trade-offs

- [Over-eager rejection detection collapses a real prompt with a same-id error] → match strictly on `tool_use_id` equality AND error text naming the missing `question` parameter; do not infer from timing alone.
- [Header-fallback masks that Claude would reject the call] → the fallback only affects codoxear display; the spec scenario for D3 keeps the rejection observable, so the two do not conflict.
- [Frontend notice adds a new event shape] → keep it within the existing `interactive`/agent-error rendering paths; no new top-level message class unless needed.

## Migration Plan

No data migration. Changes are parser/renderer-local and additive. Rollback = revert the change; existing well-formed prompts are unaffected. Deploy requires restarting the codoxear server so it reloads `rollout_log.py`, and a hard browser reload to pick up `app.js` (long-lived tabs keep old in-memory JS — there is no version-drift auto-reload).

## Open Questions

- Should the rejected-prompt notice offer a one-click "ask me to retry" affordance, or remain informational only? Default: informational for this change; retry UX is a separate enhancement.
