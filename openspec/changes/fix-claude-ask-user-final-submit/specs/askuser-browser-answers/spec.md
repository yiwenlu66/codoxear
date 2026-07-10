## MODIFIED Requirements

### Requirement: Browser single-select answers SHALL land on the option the user picked

When the user selects one option of a Claude single-select question in the browser, the key sequence sent to the TUI SHALL move the TUI selection to that exact option before confirming, **using a prompt-level cursor model that tracks the actual TUI cursor position across all clicks of the same prompt**, not a per-click assumption that the cursor starts on option 0. The model SHALL be updated after every send so that subsequent clicks compute their move as a delta against the most recent known cursor position.

#### Scenario: third option selected

- **WHEN** the user clicks the third option of a Claude single-select prompt in the browser
- **THEN** the TUI confirms the third option (not the first), as verified against a live Claude session

#### Scenario: cursor drift across questions does not corrupt the answer

- **WHEN** the TUI cursor has been advanced (by Tab, by Claude auto-advance after a previous answer, or by any other event) to a position other than option 0 of the current question
- **AND** the user then clicks an option in the browser
- **THEN** the frontend computes the move as a signed delta against the tracked cursor position and selects the option the user actually clicked, not the option that would have been at that index from option 0

#### Scenario: previously answered questions are not silently skipped

- **WHEN** the user answers all questions of a multi-question single-select prompt in order
- **THEN** the resulting `tool_result` content includes an answer for every question, with no question marked as `skipped`

### Requirement: Browser multi-select answers SHALL toggle each pick and submit once

For Claude questions where `allowMultiple` is true, the browser SHALL present a multi-select interaction that toggles each chosen option in the TUI and submits only once after the user confirms, rather than submitting on the first click. The set of options recorded by the TUI SHALL equal the set the user selected. The toggling SHALL use the same prompt-level cursor model as single-select so the cursor position remains accurate when the user toggles options out of order.

#### Scenario: three options selected then confirmed

- **WHEN** the user selects options 1, 2, and 3 of a Claude multi-select prompt and clicks confirm
- **THEN** the TUI records exactly options 1, 2, and 3 as the answer, submitted as a single response

#### Scenario: multi-select no longer submits on first click

- **WHEN** the user clicks one option of a Claude multi-select prompt
- **THEN** the prompt is not submitted, and the user can still toggle additional options before confirming

#### Scenario: out-of-order toggles preserve cursor accuracy

- **WHEN** the user toggles option 2 then toggles option 0 then toggles option 2 again within the same multi-select question
- **THEN** the TUI ends with option 2 unselected and option 0 selected, with the cursor model reflecting the actual TUI cursor for any subsequent click in the same prompt

## ADDED Requirements

### Requirement: Final-question answer SHALL trigger prompt submission

For multi-question Claude prompts (n>=2), the click that answers the final question SHALL trigger the TUI's submit affordance (the `✔ Submit` control) and not rely on the same key sequence used for non-final questions. The frontend SHALL distinguish the final-question code path by checking `qIdx === questions.length - 1` before any send, and SHALL send the submit-specific key sequence recorded in `design.md` "Confirmed TUI Protocol" when that condition is true.

#### Scenario: single-select final question submits

- **WHEN** the user clicks an option of the last question of a multi-question single-select prompt
- **THEN** within 3 seconds the prompt card disappears, the JSONL records a `tool_result` for the prompt's `tool_use_id`, the broker's `busy` flag flips to false, and the agent resumes work

#### Scenario: multi-select final question submits

- **WHEN** the user toggles options of the last question of a multi-question multi-select prompt and clicks the Confirm button
- **THEN** within 3 seconds the prompt card disappears, the JSONL records a `tool_result` listing exactly the toggled options, the broker's `busy` flag flips to false, and the agent resumes work

#### Scenario: single-question prompts continue to submit

- **WHEN** the user answers a Claude prompt that contains exactly one question (n=1, single-select or multi-select)
- **THEN** the prompt submits in the same way it did before this change, verified by the same observable evidence as the n>=2 final-question scenarios above

### Requirement: Submit failures SHALL surface a recoverable instruction

When a final-question submit send fails (network error, server restart, broker eviction), the frontend SHALL display a toast text instructing the user to press `✔ Submit` in the terminal directly, distinct from the generic non-final send-error toast.

#### Scenario: submit failure toast guides terminal recovery

- **WHEN** the `/keys` POST for a final-question submit returns a non-2xx status or raises a network error
- **THEN** the toast text reads "Submit failed; press ✔ Submit in the terminal" so the user can complete the answer without losing the in-progress prompt state

### Requirement: Cursor model SHALL survive prompt re-renders

The prompt-level cursor model SHALL be persisted in DOM data attributes (or equivalent durable storage tied to the prompt card) so that a transcript re-render mid-prompt does not reset the model. The handler entry point SHALL read the persisted values and fall back to defaults only when the persisted values are absent.

#### Scenario: re-render between clicks preserves cursor state

- **WHEN** the live poller forces a transcript re-render between two clicks of the same prompt
- **THEN** the second click computes its move using the cursor position established by the first click, not from option 0
