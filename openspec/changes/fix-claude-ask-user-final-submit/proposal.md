## Why

The Claude `AskUserQuestion` browser answer flow shipped under `fix-reported-ux-issues` task 12 looks correct on a 1-question prompt and on the multi-question middle-stage tested at the time, but two real-world states fail:

1. **Single-select, multi-question, final question**: the user picks an option in the browser; the prompt card stays on screen indefinitely; the agent does not resume; the top-bar `✔ Submit` button stays visible.
2. **Cursor drift across questions**: the frontend computes the per-click move as `"\x1b[B".repeat(optIdx)` under the assumption that the TUI cursor begins each interaction on option 0 of the current question. Any event that desyncs that assumption — Tab navigation, Claude's auto-advance after a previous answer, browser polling that re-renders the prompt — silently shifts the actual cursor away from where the frontend thinks it is. The next click then sends an off-by-N move and selects the wrong option.

The cursor-drift failure was reproduced live on broker-169813 on 2026-05-31: probing the prompt with three `\t` keystrokes pushed the TUI cursor to question 4 (the last); a subsequent single Enter selected question 4's option 1 and submitted the entire prompt, marking questions 1-3 as `skipped` even though those questions had options the user intended to answer. The 4-question prompt collapsed into a 1-answer `tool_result` because the frontend's mental model of "cursor is at question 0" diverged from the actual TUI state.

The final-question failure is mechanically distinct: even with a correct cursor model, the frontend single-select handler sends `move + "\r"` for every question. In the multi-question TUI mode, Enter on a non-final question selects + auto-advances; on the final question, Enter behaves differently — the top-bar `✔ Submit` is the actual submit affordance, and the precise key sequence to trigger it has never been confirmed against a live session.

The shared root cause is the same as the original task 12.1 mistake: protocol assumptions were locked in without a live measurement against the Claude TUI in the exact final-question state. We are repeating the pattern unless the next fix is gated behind an explicit protocol-confirmation step before any code change.

## What Changes

- Replace the per-question `cursorIdx` (multi-select only) and the implicit "cursor starts at 0 every click" assumption (single-select) with a single prompt-level cursor model that tracks both the active question index (`promptCursorQIdx`) and the active option index (`promptCursorOptIdx`) across all clicks within one `AskUserQuestion` event.
- After each answered question, reset `promptCursorOptIdx = 0` to reflect Claude's auto-advance to the next question's option 0.
- Compute the per-click TUI move as `delta = optIdx - promptCursorOptIdx`, sending `\x1b[B` × delta or `\x1b[A` × |delta| as a separate awaited send before the action key. Commit `promptCursorOptIdx = optIdx` after the move resolves, before the action send, so a failed action does not leave a phantom cursor position.
- Distinguish the final-question submit path from the non-final advance path. In single-select, the final-question handler sends the confirmed Submit sequence (recorded during phase 1) instead of the bare `"\r"` used for non-final questions. In multi-select, the `Confirm selection` button branches identically.
- Gate every code change behind a phase-1 protocol probe that records, against a live Claude session, the exact key sequence required to trigger `✔ Submit` in each of: single-select n=1, single-select n≥2 final, multi-select n=1, multi-select n≥2 final. The recorded sequences live in `design.md` "Confirmed Submit Protocol" before any frontend code touches them.

## Capabilities

### Modified Capabilities
- `askuser-browser-answers`: extends the existing capability to (a) require a prompt-level cursor model that survives Claude auto-advance and frontend re-renders, and (b) require a final-question submit path distinct from the non-final advance path, with both paths gated by live-recorded protocol sequences.

### New Capabilities
<!-- None: this change refines an existing capability rather than introducing a new one. -->

## Impact

- Frontend:
  - `codoxear/static/app.js` (the `interactive-prompt` rendering block around 2657-2805): introduce prompt-level cursor state shared across question groups; replace single-select move calculation; branch single-select and multi-select on `qIdx === questions.length - 1` for the submit path; remove the stale "Enter auto-advances" comment.
- Tests:
  - `tests/test_ask_user_normalize.py`: extend with a stability test asserting that `_claude_ask_user_questions` preserves question order so the frontend can rely on `qIdx === questions.length - 1` to detect the final question.
  - `tests/test_broker_busy_state.py`: extend with a regression case asserting that an `AskUserQuestion` `tool_use` followed by a matching `tool_result` clears `busy` regardless of how many questions were in the prompt.
- OpenSpec:
  - `openspec/changes/fix-claude-ask-user-final-submit/`: this change.
  - `openspec/specs/askuser-browser-answers/spec.md` will be the long-lived spec once this change is applied (the `fix-reported-ux-issues` change introduced this capability but its spec lives only in the change directory; an apply step will need to copy or move it).
- Runtime: no broker / PTY changes; `codoxear-server` restart only after frontend changes are deployed.
- Backward compatibility: existing single-select on n=1 prompts already worked (verified in `fix-reported-ux-issues` task 12.8); the new code paths must not regress that case. Existing multi-select on n=1 was the same Confirm-button shape used for n≥2 middle questions and was verified in browser-validation section 4; this change only adds branching for n≥2 final, leaving the verified paths untouched.
