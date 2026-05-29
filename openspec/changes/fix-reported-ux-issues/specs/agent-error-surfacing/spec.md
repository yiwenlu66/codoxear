## ADDED Requirements

### Requirement: Backend log error records SHALL be parsed into a normalized agent_error event

The chat event parser SHALL recognize agent CLI error records in Codex, Pi, and Claude session logs and emit them as events of class `agent_error` with fields `type`, `message`, `ts`, and `source`. The recognition SHALL apply to both the full parser path (`_extract_chat_events`) and the cursor parser path (`_chat_events_for_record` and the helpers it feeds into `_read_chat_tail_page`, `_read_chat_history_page`, `_read_chat_live_delta`).

#### Scenario: Codex stream error appears as agent_error event

- **WHEN** a Codex rollout log contains a record `{"type": "event_msg", "payload": {"type": "stream_error", "message": "upstream error: 503"}}`
- **THEN** `_extract_chat_events` and `_read_chat_tail_page` both yield exactly one `agent_error` event whose `type` is `"stream_error"`, `message` is `"upstream error: 503"`, and `source` is `"codex"`

#### Scenario: Pi tool result with isError ends turn

- **WHEN** a Pi session log contains a `toolResult` record with `isError: true` and a non-empty error text
- **THEN** the parser yields one `agent_error` event with `source: "pi"` and the error text as `message`

#### Scenario: Claude system error event

- **WHEN** a Claude session log contains `{"type": "system", "subtype": "error", "message": "rate limit"}`
- **THEN** the parser yields one `agent_error` event with `source: "claude"`, `type: "error"`, `message: "rate limit"`

### Requirement: Terminal agent errors SHALL clear busy; auto-retried errors SHALL NOT

`broker._apply_rollout_obj_to_state` SHALL close the turn (`_close_turn_state`: `st.busy=false`, `st.turn_open=false`, pending calls cleared) for terminal agent errors. For Claude `api_error` records that schedule an automatic retry (a positive `retryInMs` with `retryAttempt < maxRetries`), the broker SHALL keep the turn open and treat the record as activity (`st.busy=true`, `st.last_turn_activity_ts=now`), because the CLI retries internally and the turn has not ended. Pi and Codex error records have no retry semantics and remain turn-terminal.

#### Scenario: Codex stream error clears busy

- **WHEN** the broker is in state `busy=true, turn_open=true` and observes a `stream_error` rollout record
- **THEN** after `_apply_rollout_obj_to_state` runs, `st.busy` is `false` and `st.turn_open` is `false`

#### Scenario: Claude auto-retried api_error keeps busy

- **WHEN** the broker observes a Claude `api_error` record with `retryInMs > 0` and `retryAttempt < maxRetries`
- **THEN** `st.busy` stays `true` and `st.turn_open` stays `true`, so the UI does not flap between working and idle while the CLI retries

#### Scenario: Claude terminal api_error clears busy

- **WHEN** the broker observes a Claude `api_error` record with no scheduled retry (no `retryInMs`) or with `retryAttempt >= maxRetries`
- **THEN** `st.busy` is `false` and `st.turn_open` is `false`

#### Scenario: Error after assistant_message still terminates turn

- **WHEN** the agent emits an `agent_message` followed by a `*_error` record without `task_complete`
- **THEN** the broker clears busy on the error record rather than waiting for an additional event

### Requirement: The browser SHALL render agent_error events as a distinct UI card

The web UI SHALL render `agent_error` events as a visually distinct card containing the event `type`, `message`, and a relative timestamp. The card SHALL be reachable from both the initial tail render and the live update path, and SHALL not be silently dropped when scrolling through history.

#### Scenario: Error card visible after refresh

- **WHEN** a session has an `agent_error` event in the most recent page of its log
- **THEN** opening the session in the browser shows the error card without requiring further interaction

#### Scenario: Live error appears without reload

- **WHEN** a new `agent_error` event is appended while the user is viewing the session
- **THEN** the live update path (`/messages/live`) delivers the event and the UI appends the card without a page reload
