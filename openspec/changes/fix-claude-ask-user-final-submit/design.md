## Context

The `fix-reported-ux-issues` change (task 12) introduced a Claude `AskUserQuestion` browser answer flow under the assumption that the per-click TUI move could be computed as `"\x1b[B".repeat(optIdx)` because the cursor reliably opens on option 0 for each question, and that Enter on any question simultaneously confirms the option and advances to the next question. Live probing on broker-169813 on 2026-05-31 falsifies both assumptions in real-world states the original task did not cover.

```text
Failure 1 — cursor drift across the prompt
  T0  4-question single-select prompt opens; cursor at q[0] option 0.
  T1  user (or anything: Tab nav, Claude advance, frontend re-render)
      moves the TUI cursor to a different question or option.
  T2  user clicks an option in the browser.
  T3  frontend computes move = `"\x1b[B".repeat(optIdx)`, assuming
      the cursor is at q[?] option 0.
  T4  TUI applies that move from its actual cursor, lands on the
      wrong option (or the wrong question).
  T5  frontend sends `"\r"`. The wrong option is selected; the
      whole prompt may submit early with most questions skipped.

Failure 2 — final-question Enter does not submit the prompt
  T0  multi-question prompt; user has answered q[0..N-2]; cursor at q[N-1].
  T1  user clicks an option of q[N-1] in the browser.
  T2  frontend sends move + `"\r"`. q[N-1] gets selected.
  T3  Enter on the final question does not advance (no further
      question exists) and does not submit (`✔ Submit` is the
      submit affordance, separate from Enter).
  T4  prompt stays on screen indefinitely; agent does not resume.
```

The two failures are mechanically independent: failure 1 corrupts which option gets chosen; failure 2 corrupts whether the prompt closes. They share a root cause — the original implementation locked in protocol assumptions without measuring the live TUI in the exact failing states.

## Goals / Non-Goals

**Goals:**
- Make the per-click TUI move resilient to any state that desyncs the frontend cursor model (Tab nav, Claude advance, partial answer, re-render).
- Make the final question of a multi-question prompt submit on the user's click without manual intervention in the terminal.
- Cover all four shape combinations: single-select n=1, single-select n≥2 final, multi-select n=1, multi-select n≥2 final.
- Gate every code change behind a live-measured protocol record so we do not repeat the task 12.1 mistake of guessing a TUI sequence.

**Non-Goals:**
- Redesign the broker `_apply_rollout_obj_to_state` busy-state machine. The existing handling of `tool_use` + `tool_result` already closes the turn correctly; only the front-end submission path is broken.
- Add a freeform-answer text input for Claude prompts. Out of scope.
- Add a "skip this question" affordance. Out of scope.
- Cover Pi multi-select prompts. Pi multi-select is currently disabled in the browser (the existing note tells the user to answer in the terminal); leave that disabled.
- Translate or rewrite Claude's question text. Pass through verbatim, same as today.

## Confirmed TUI Protocol (broker-169813, 2026-05-31)

### Probe Run

| Field | Value |
|---|---|
| broker_id | broker-169813 |
| tmux window | codoxear:codoxear-0effa5 |
| Claude Code version | v2.1.156 |
| trigger prompt | "Use the AskUserQuestion tool to ask me three single-select questions about which programming language to pick for a new CLI project..." |
| tool_use_id | tooluse_IO4xmaql4fIt6hHKzsqgaP |
| n_questions | 3 |
| q[0] | header="Runtime priority", multiSelect=false, 3 options |
| q[1] | header="Distribution", multiSelect=false, 3 options |
| q[2] | header="Ecosystem need", multiSelect=false, 3 options |
| started | 2026-05-31 (initial probes) |
| extended | 2026-06-01 (phase-1 follow-through) |

The following are the facts established by direct probing.

```text
F1  Multi-question footer = "Enter to select · Tab/Arrow keys to navigate · Esc to cancel"
F2  Single-question footer = "Enter to select · ↑/↓ to navigate · Esc to cancel"
    -> the TUI distinguishes these two modes; protocol can differ.
F3  Tab (\t) in multi-question mode advances cursor q[i] -> q[i+1] WITHOUT selecting
    any option and WITHOUT toggling any checkbox. The footer's "Tab/Arrow keys to
    navigate" wording is literal: Tab is a navigation key, not a select key.
F4  Top-bar layout in multi-question mode:
       <-  ☐q0  ☐q1  ...  ☐q[N-1]  ✔ Submit  ->
    The ✔ Submit affordance is a distinct top-bar control, not a per-question control.
F5  In multi-question mode, Enter on q[i] for i < N-1 simultaneously selects the
    cursor's current option of q[i] and advances cursor to q[i+1] option 0.
F6  In multi-question mode, Enter on q[N-1] (the final question) selects the
    cursor's current option of q[N-1] and triggers submit ONLY when the prior
    questions are all answered. If any prior question is unanswered, the TUI
    submits the prompt with the unanswered ones marked "skipped" -- the original
    skipped-question failure observed on broker-169813.
    [VERIFY IN PHASE 1: this is the most likely interpretation but the precise
    semantics under "all prior answered" need a clean measurement.]
F7  RESOLVED (2026-06-01, broker-169813, tooluse_IO4xmaql4fIt6hHKzsqgaP):
    Single-select n>=2 with all prior answered, cursor on a valid option of q[N-1]:
      step 1: send "\r"   -> selects q[N-1]'s option, top-bar shows ☒ ☒ ☒,
                              TUI displays a "Review your answers" confirmation
                              screen with cursor pre-positioned on "1. Submit answers"
      step 2: send "\r"   -> dismisses the review, tool_result lands in JSONL
                              within ~1s, prompt card closes, agent resumes
    => <SUBMIT_SEQ_SS_NK> = ["\r", "\r"]   (two SEPARATE awaited sends; never merge.
                                            Decision 2 still applies: a merged "\r\r"
                                            would race against the review-screen render
                                            and could miss the second Enter.)
    Equivalent: the final-question handler sends ONE EXTRA "\r" compared to non-final.
F8  RESOLVED (2026-06-01, REVISED after live retest):
    Tab in multi-select on a non-final question advances to the next question
    directly (cursor lands on q[i+1] option 0 with NO auto-select). The
    "Next" affordance rendered under "Type something" is the visual label for
    that Tab behavior, NOT a separate control to navigate to.
    Tab in multi-select on the FINAL question opens the same "Review your
    answers" screen as bare Enter on single-select final.

    CRITICAL: a trailing Enter after Tab on a NON-FINAL question is HARMFUL.
    Live retest on broker-169813 (tooluse_yiuDu5cIJMYcvtqgBs76Y8) with
    capture-pane between every send showed: after Tab+Enter on q[0], the TUI
    landed on q[1] with opt 0 PRE-TOGGLED (`[✔] Red`). The user's first
    Space click then toggled opt 0 back OFF, silently losing their first
    selection on every non-first question. The original codoxear multi-select
    Confirm code (Tab+Enter for every question) was producing this exact
    failure -- it just looked like "selections randomly missing" rather than
    a stuck prompt, so it slipped past task 12.8 verification.

    Multi-select submit sequences:
      <SUBMIT_SEQ_MS_NONFINAL> = ["\t"]            (Tab only; NO trailing Enter)
      <SUBMIT_SEQ_MS_FINAL>    = ["\t", "\r"]      (Tab opens review, Enter submits)
      <SUBMIT_SEQ_MS_N1>       = ["\t", "\r"]      (n=1, same as final)
    Sends remain SEPARATE awaits per Decision 2.

    Single-select submit sequences (consolidated):
      <SUBMIT_SEQ_SS_NK> = ["\r", "\r"]   (n>=2 final: \r selects + opens review, \r submits)
      <SUBMIT_SEQ_SS_N1> = ["\r"]         (n=1: \r submits directly, NO review screen)


```

The protocol probe in phase 1 of `tasks.md` is the gate for any code change: the values for F7 and F8 must be recorded in this section before phase 2/3/4 begin.

## Decisions

### Decision 1: One cursor model per prompt, shared across all questions

The frontend SHALL track exactly one `(promptCursorQIdx, promptCursorOptIdx)` pair per `AskUserQuestion` event. Each click reads the pair, computes a delta against it, sends the move, then updates the pair. Reset behavior:

```text
on each successful answer:
   if isFinal:  do not reset (the prompt is closing)
   else:        promptCursorQIdx += 1
                promptCursorOptIdx = 0
                # because Claude auto-advances to q[i+1] option 0
```

Rationale: every cursor-drift failure observed on broker-169813 came from per-click state that pretended the cursor was at a fresh starting point. A single shared pair is the smallest model that survives Tab nav, Claude advance, and re-render.

Alternative considered — re-read the cursor position from the TUI before each click. Rejected because there is no read-side API for cursor position; the only signal is `tmux capture-pane`, which is server-side and not available to the browser. Modeling the cursor on the client side is the only available option.

### Decision 2: Move and action are always two awaited sends, never merged

A click that needs to move and then act SHALL issue the move as one `await sendSeq(...)` call and the action as a second `await sendSeq(...)` call. Merging them into a single string is forbidden.

Rationale: this is the same race that bit task 12.1 ("CRITICAL RACE found during live verification" in the original tasks.md) — a merged `move+action` string is processed against the pre-move cursor position by the TUI's input handling, toggling the wrong option. Splitting the sends forces the TUI to settle on the new cursor before the action key applies.

### Decision 3: Cursor commit happens after move, before action

After `await sendSeq(move)` resolves and before `await sendSeq(action)`, the frontend SHALL commit `promptCursorOptIdx = optIdx`. If the action send fails, the cursor model already reflects the actual TUI state.

Rationale: a failed action does not unmove the TUI — the move already happened. The commit-before-action ordering keeps the model consistent with reality even on partial failure, so the next click computes its delta from the right baseline. Without this, a transient `/keys` failure would leave the model believing the cursor was still at the old position; the next click would send a move that doubles up on the already-applied move and select an option two rows past the user's intent.

### Decision 4: Final-question submit is its own code path, branched on `qIdx === questions.length - 1`

Both the single-select option-button handler and the multi-select Confirm handler SHALL compute `const isFinal = qIdx === questions.length - 1` BEFORE any await, and branch on it for the post-selection action sequence.

```text
non-final single-select:    move + "\r"                              (selects + advances)
final   single-select:      move + <SUBMIT_SEQ_SS>                   (selects + submits)
non-final multi-select:     toggles via Space, then "\t" + "\r"      (advance per option)
final   multi-select:       toggles via Space, then <SUBMIT_SEQ_MS>  (submits)
```

`<SUBMIT_SEQ_SS>` and `<SUBMIT_SEQ_MS>` are filled in by phase 1 protocol probing and recorded in the "Confirmed TUI Protocol" section. Until they are recorded, the final-question branches MUST raise an explicit "submit failed; press ✔ Submit in the terminal" toast and not silently fall through to the non-final sequence.

Rationale: the non-final and final paths are mechanically distinct in the TUI (F5 vs F6/F7) and were conflated in the original implementation. Branching on `qIdx === questions.length - 1` is the smallest correct distinction — `questions.length` is already in scope at the call site.

### Decision 5: Protocol probe is a hard gate on all frontend changes

The phase-1 tasks SHALL be completed and the F7/F8 values recorded in `design.md` BEFORE phase 2 (cursor fix), phase 3 (single-select submit), and phase 4 (multi-select submit) make any code change to `app.js`. The phase ordering is non-optional.

Rationale: the only way the original task 12.1 fix shipped with a wrong final-question protocol was that no live probe was done in the final-question state before the code was written. Making the probe a hard gate is the structural fix to that pattern. Phase 6 (tests) and phase 7 (live verification) are additional defenses but they cannot substitute for measuring the protocol up front.

## Risks

- **Claude TUI version drift**: the protocol facts F1-F8 are measured against Claude Code v2.1.156. A future TUI version may change the Submit affordance position, the Tab semantics, or the auto-advance behavior. Mitigation: leave the recorded protocol in `design.md` with the version annotation; on any user report of a regression, the first diagnostic is to re-probe and compare. The shared-cursor model in Decision 1 is independent of which keys do what, so it should survive most TUI changes.
- **Browser re-render mid-prompt**: if the live poller re-renders the `interactive-prompt` card while the user is mid-answer, the prompt-level cursor state is in JS closure and would be lost. Mitigation: store `promptCursorQIdx` / `promptCursorOptIdx` on a DOM data attribute of the card (`data-cursor-q`, `data-cursor-opt`) so a re-render can restore it. This adds a small amount of state-marshalling code but is the only way to make the model survive re-render.
- **Concurrent prompts in different sessions**: a single browser tab may have multiple sessions open; each session can have its own prompt. The cursor state is per-prompt-card, not per-tab, so this is naturally isolated. No shared global state.
- **Protocol probe damages the user's real prompt**: a phase-1 probe sends keys to a live session; the keys may select wrong options or submit early. Mitigation: the phase-1 tasks must use a dedicated probe prompt the user explicitly triggers ("ask me three single-select questions about <topic>") rather than probing on a real working prompt. The broker-169813 probe on 2026-05-31 already destroyed one real 4-question prompt; we do not repeat that.
- **n=1 vs n>=2 protocol divergence**: F1 and F2 confirm the TUI uses different footers for these two modes. If the Submit key sequence also differs, the implementation needs separate `<SUBMIT_SEQ_SS_N1>` and `<SUBMIT_SEQ_SS_NK>` slots. Phase 1.9 covers this; the recorded values in "Confirmed TUI Protocol" must list both even if they turn out to be the same.

## Open Questions

These will be resolved by phase-1 probing and recorded in this document before any code is touched:

- F7: What exact key sequence triggers `✔ Submit` on a fully-answered final question, for each of (single-select n=1, single-select n≥2 final, multi-select n=1, multi-select n≥2 final)?
- F8: Does multi-select Tab in multi-question mode jump to a "Next" button between questions or directly to the next question's options?
- Does Claude's auto-advance always land cursor on option 0 of the next question, or on the same option index that was just selected? (Affects Decision 1's reset rule.)
- Does the TUI accept `\r` as the Submit trigger when q[N-1] is the only unanswered question and the cursor is on a valid option, or does it require an explicit `✔ Submit` selection? (Affects whether the single-select non-final path can be reused for the final path.)
- When a `tool_result` records a `skipped` answer for a question, does the broker close the turn cleanly or stay busy? (Affects whether `tests/test_broker_busy_state.py` needs a new case.)



