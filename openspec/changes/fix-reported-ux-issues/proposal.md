## Why

Three user-reported issues hide what is actually happening inside Codoxear: the UI keeps showing "working" after the agent CLI hits an upstream error; `view file` returns a generic "file not found" even with a valid absolute path; and the existing Web Push channel cannot deliver notifications when Codoxear is reached over plain HTTP. The first two cause silent failures in everyday use, and the third forces users to either set up HTTPS or live without mobile alerts.

## What Changes

- Make `rollout_log` recognize agent error events (Codex `event_msg.payload.type` ending in `*_error` and Pi/Claude equivalents) and emit them as a new normalized event class.
- Make `broker._apply_rollout_obj_to_state` clear `busy` when an agent error event closes a turn so the UI no longer hangs on "working".
- Render the new error event in `static/app.js` as a visible error card with type, message, and timestamp.
- Replace `_resolve_client_file_path`'s silent `.resolve()` with a strict resolver that returns one of `not_found`, `dead_symlink`, `permission_denied`, `outside_allowed_root`, or `ok`.
- Update `/api/files/inspect` and `/api/sessions/<id>/file/read` to surface the resolver reason in the JSON error body, and update the file viewer UI to display it instead of the current generic message.
- Add an out-of-band notification channel abstraction (`NotificationChannel`) in `voice_push.py` and ship a Bark adapter as the first concrete channel so HTTP-only deployments can still receive mobile push.
- Add `tts_*`-style settings (`bark_enabled`, `bark_endpoint`, `bark_token`) and surface a per-device subscription type in `/api/notifications/subscription` so the existing settings UI can target Bark devices.
- Fix browser answering of Claude `AskUserQuestion` prompts: normalize the per-question schema in the parser (map `multiSelect`/`header` to the same shape Pi already uses), make the single-select cursor mapping reliable instead of assuming the TUI cursor starts on option 0, and implement multi-select submission (toggle then confirm) so the browser answer matches what the TUI records.

## Why (added scope)

A fourth user-reported issue: when answering a Claude `AskUserQuestion` prompt from the browser, the answer that lands in the TUI does not match what the user picked. Single-select option 3 lands on option 1, and multi-select picks collapse into a single wrong answer. Root cause: the frontend drives the TUI by simulating arrow keys (`"\x1b[B".repeat(optIdx) + "\r"`) under two false assumptions — that the cursor starts on option 0 and that one `Down` moves exactly one option — and it never implements Claude multi-select at all (the multi-select guard only covers Pi). The Claude question schema is also passed through unnormalized, so the frontend reads `q.allowMultiple` while Claude actually sends `multiSelect`, leaving the frontend unaware a question is multi-select.

## Capabilities

### New Capabilities
- `agent-error-surfacing`: Normalize agent CLI error events from backend logs into a UI-visible event so users see upstream/API failures instead of an indefinite "working" state.
- `file-resolution-diagnostics`: Resolve user-supplied file paths in the file viewer with explicit failure reasons (not found, dead symlink, permission denied) instead of a single generic 404.
- `out-of-band-push-channels`: Deliver push notifications through HTTP-callable third-party services (Bark first) so deployments without HTTPS can still receive mobile alerts.
- `askuser-browser-answers`: Answer Claude `AskUserQuestion` prompts from the browser such that the option(s) the user selects are the option(s) the TUI records, including multi-select.

### Modified Capabilities
<!-- None: openspec/specs/ is empty in this repo, so all capabilities are new. -->

## Impact

- Backend parsing:
  - `codoxear/rollout_log.py` (new error event extraction in `_chat_events_for_record` and `_extract_chat_events`)
  - `codoxear/pi_log.py`, `codoxear/claude_log.py` (per-backend error helpers)
  - `codoxear/broker.py` (`_apply_rollout_obj_to_state` clears `busy` on error)
- File resolution:
  - `codoxear/server.py` (`_resolve_client_file_path`, `/api/files/inspect`, `/api/sessions/<id>/file/read`, `/api/files/blob`, `/api/sessions/<id>/file/blob`)
- Notifications:
  - `codoxear/voice_push.py` (`NotificationChannel` interface, `BarkChannel`, `_send_push_notifications` dispatch)
  - `codoxear/server.py` (`/api/notifications/subscription` accepts a `kind` field; settings GET/POST add Bark fields)
  - `codoxear/static/app.js` (error event card, file viewer error reasons, Bark settings UI)
  - `codoxear/static/app.css` (styles for the error card and the new settings rows)
- AskUserQuestion browser answers:
  - `codoxear/rollout_log.py` (normalize Claude `AskUserQuestion` questions: map `multiSelect`->`allowMultiple`, carry `header`, normalize `options` via the shared helper, tag `backend: "claude"`)
  - `codoxear/static/app.js` (reliable single-select cursor mapping; Claude multi-select toggle+confirm UI; render `header`)
- Tests:
  - `tests/test_rollout_log.py`, `tests/test_message_route_source.py` (error event parsing)
  - new `tests/test_file_resolution.py` (resolver outcomes)
  - new `tests/test_bark_channel.py` (push dispatch with HTTP mock)
  - new `tests/test_ask_user_normalize.py` (Claude question normalization)
- Runtime: no broker/PTY changes; `codoxear-server` restart only.
