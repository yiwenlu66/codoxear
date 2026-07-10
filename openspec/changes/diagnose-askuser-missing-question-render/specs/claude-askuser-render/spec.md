## ADDED Requirements

### Requirement: Render well-formed AskUserQuestion prompts
The system SHALL render an answerable prompt card in the web UI for every Claude `AskUserQuestion` tool-call record in the rollout JSONL whose question object contains a non-empty `question` field, preserving each question's options, header, and multi-select flag.

#### Scenario: Single well-formed question
- **WHEN** the rollout log contains an `AskUserQuestion` `tool_use` with `questions[0].question` non-empty and one or more `options`
- **THEN** the chat-event extractor emits an `interactive: "ask_user_question"` event carrying that question and its options
- **AND** the frontend renders a prompt card with one selectable option per option entry

#### Scenario: Multi-question prompt
- **WHEN** an `AskUserQuestion` `tool_use` contains two or more question objects each with a non-empty `question`
- **THEN** the emitted event's `questions` array preserves all of them in order
- **AND** the rendered card exposes each question group

### Requirement: Fall back to header when question text is absent but the call is otherwise renderable
The system SHALL, when a question object lacks a usable `question` field but provides a non-empty `header`, use the `header` text as the displayed question so the prompt remains answerable rather than being silently discarded.

#### Scenario: Header-only question object reaches the extractor
- **WHEN** a question object has empty/missing `question` but a non-empty `header` and at least one option
- **THEN** the extractor emits the question with its displayed text derived from `header`
- **AND** the question is not dropped from the event's `questions` array

### Requirement: Surface AskUserQuestion calls rejected by the agent backend
The system SHALL NOT silently produce an empty or answerable-looking UI for an `AskUserQuestion` tool call that the agent backend rejected with a validation error. When a `tool_use` for `AskUserQuestion` is immediately followed by a `tool_result` reporting an input-validation failure for that `tool_use_id`, the system SHALL make the rejection observable rather than presenting nothing.

#### Scenario: Missing required question field rejected upstream
- **WHEN** an `AskUserQuestion` `tool_use` omits the required `questions[i].question` field
- **AND** the agent backend writes a `tool_result` for that `tool_use_id` containing `InputValidationError` naming the missing `question` parameter
- **THEN** the system treats the exchange as a rejected prompt
- **AND** the UI does not present a silently empty prompt card with no explanation

#### Scenario: Diagnosis is traceable from the log
- **WHEN** a user reports that an ask-user prompt never appeared
- **THEN** the rejected-prompt case is identifiable by locating the `InputValidationError` `tool_result` whose `tool_use_id` matches the `AskUserQuestion` `tool_use`
