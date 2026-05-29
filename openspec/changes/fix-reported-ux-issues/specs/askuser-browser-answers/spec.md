## ADDED Requirements

### Requirement: Claude AskUserQuestion records SHALL be normalized to the shared question schema

The chat event parser SHALL normalize each Claude `AskUserQuestion` question to the same shape the Pi path already produces, so the frontend reads one schema regardless of backend. The normalized question SHALL carry `question`, `header` (when present), `options` as a list of `{label, description}` via the shared `_normalize_ask_user_options` helper, `allowMultiple` derived from the Claude `multiSelect` field, and `backend: "claude"`.

#### Scenario: multiSelect maps to allowMultiple

- **WHEN** a Claude session log contains an `AskUserQuestion` tool_use whose question has `{"multiSelect": true, "header": "Auth", "question": "Which?", "options": [{"label":"A","description":"x"}]}`
- **THEN** the emitted `ask_user_question` event's question has `allowMultiple == true`, `header == "Auth"`, `backend == "claude"`, and `options == [{"label":"A","description":"x"}]`

#### Scenario: single-select default

- **WHEN** a Claude question has `multiSelect` absent or `false`
- **THEN** the normalized question has `allowMultiple == false`

### Requirement: Browser single-select answers SHALL land on the option the user picked

When the user selects one option of a Claude single-select question in the browser, the key sequence sent to the TUI SHALL move the TUI selection to that exact option before confirming, without assuming the cursor starts on option 0. The mapping SHALL account for the confirmed Claude TUI navigation behavior (recorded during implementation).

#### Scenario: third option selected

- **WHEN** the user clicks the third option of a Claude single-select prompt in the browser
- **THEN** the TUI confirms the third option (not the first), as verified against a live Claude session

### Requirement: Browser multi-select answers SHALL toggle each pick and submit once

For Claude questions where `allowMultiple` is true, the browser SHALL present a multi-select interaction that toggles each chosen option in the TUI and submits only once after the user confirms, rather than submitting on the first click. The set of options recorded by the TUI SHALL equal the set the user selected.

#### Scenario: three options selected then confirmed

- **WHEN** the user selects options 1, 2, and 3 of a Claude multi-select prompt and clicks confirm
- **THEN** the TUI records exactly options 1, 2, and 3 as the answer, submitted as a single response

#### Scenario: multi-select no longer submits on first click

- **WHEN** the user clicks one option of a Claude multi-select prompt
- **THEN** the prompt is not submitted, and the user can still toggle additional options before confirming

### Requirement: The browser SHALL display the question header

When a Claude question carries a `header`, the web UI SHALL display it alongside the question text so multi-question prompts are distinguishable.

#### Scenario: header rendered

- **WHEN** a Claude prompt has questions with headers `"Auth"` and `"Storage"`
- **THEN** the rendered prompt shows each question's header
