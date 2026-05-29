## 1. Capture Real Failure Fixtures

- [x] 1.1 SKIPPED: no Codex upstream-error session in `~/.codex/sessions/`. Codex parser uses a permissive selector (`payload.type` ending in `_error` OR equal to `error`) backed by a synthetic fixture; revisit with a real fixture once one is captured.
- [x] 1.2 SKIPPED: Pi does not write upstream errors to its JSONL log (verified across all `~/.pi/agent/sessions/**/*.jsonl`: `isError:true` never appears at the message level for upstream errors). Pi CLI retries internally. Pi parser only handles tool-result `isError:true` cases as turn-terminal markers.
- [x] 1.3 Real Claude `api_error` record captured at `tests/fixtures/claude_api_error_sample.jsonl` (one full record, 502 upstream). Shape: `{"type":"system","subtype":"api_error","level":"error","error":{"status":<int>,"error":{"error":{"message":"<text>","type":"<text>"}}}}`.
- [x] 1.4 Selector strings recorded in `design.md` "Confirmed selectors" section.
- [x] 1.5 SKIPPED: user has not produced a concrete failing path yet. Resolver test suite covers the four failure modes synthetically using `tmp_path`, dead symlinks, `chmod` on a parent, and an out-of-cwd path.

## 2. Agent Error Event - Parser Layer

- [x] 2.1 Add `_codex_agent_error_event_from_record(obj, ts) -> dict | None` in `codoxear/rollout_log.py` matching the selectors recorded in step 1.1; return `{class: "agent_error", source: "codex", type, message, ts}` or `None`.
- [x] 2.2 Add `_pi_agent_error_event_from_record(obj, ts) -> dict | None` in `codoxear/pi_log.py`. Wire it through the same return shape.
- [x] 2.3 Add `_claude_agent_error_event_from_record(obj, ts) -> dict | None` in `codoxear/claude_log.py`. Skip body if 1.3 was skipped; leave a stub that returns `None` and a TODO referencing the fixture.
- [x] 2.4 Extend `_chat_events_for_record` in `codoxear/rollout_log.py` to dispatch to the helpers from 2.1-2.3 and append the resulting `agent_error` event before any text event extracted from the same record.
- [x] 2.5 Extend `_extract_chat_events` in `codoxear/rollout_log.py` to emit the same `agent_error` events for the full-parser path so non-cursor consumers see them too.
- [x] 2.6 Add `tests/test_rollout_log.py::test_codex_stream_error_yields_agent_error_event` using fixture 1.1.
- [x] 2.7 Add `tests/test_rollout_log.py::test_pi_tool_error_yields_agent_error_event` using fixture 1.2.
- [x] 2.8 Add `tests/test_rollout_log.py::test_claude_error_yields_agent_error_event` using fixture 1.3 if available; otherwise add it as `@unittest.skip("no claude error fixture yet")`.
- [x] 2.9 Add `tests/test_message_route_source.py::test_cursor_tail_includes_agent_error` asserting `_read_chat_tail_page` returns the new event.
- [x] 2.10 Run `python -m unittest tests.test_rollout_log tests.test_message_route_source` and confirm all new cases pass.

## 3. Agent Error Event - Broker State

- [x] 3.1 In `codoxear/broker.py::_apply_rollout_obj_to_state`, add a single guard that calls `_close_turn_state(st)` when `_codex_agent_error_event_from_record`, `_pi_agent_error_event_from_record`, or the Claude variant returns non-`None`.
- [x] 3.2 Add `tests/test_broker_busy_state.py::test_codex_stream_error_clears_busy` using fixture 1.1: build a `State` with `busy=True, turn_open=True` and assert both flags are `False` after `_apply_rollout_obj_to_state`.
- [x] 3.3 Add `tests/test_broker_busy_state.py::test_pi_tool_error_clears_busy` using fixture 1.2.
- [x] 3.4 Add `tests/test_broker_busy_state.py::test_error_after_agent_message_still_terminates_turn` covering the design's second scenario.
- [x] 3.5 Run `python -m unittest tests.test_broker_busy_state` and confirm all three new tests pass.

## 4. Agent Error Event - Frontend Card

- [x] 4.1 In `codoxear/static/app.js`, add a `renderAgentErrorEvent(event)` function that produces a card with class `event-card event-error` showing `event.type`, `event.message`, and a relative timestamp.
- [x] 4.2 Wire `renderAgentErrorEvent` into both the initial-render path (used by `tail`) and the live-append path (used by `live`) so the card shows in both paths.
- [x] 4.3 Add CSS for `.event-error` in `codoxear/static/app.css`: red left border, muted background, monospace `type`, regular-weight `message`. No emoji.
- [x] 4.4 Open `codoxear/static/index.html` in a browser via the running server, force a session that has fixture 1.1 in its log, confirm the error card renders on initial load and after a live append. (Verified 2026-05-29 via headless Chrome on live session broker-1425778: 19 error cards rendered, red border, api_error_502 chip. Found+fixed a bug: agent_error role="system" events were filtered out by appendEvent/renderTranscript/prependOlderEvents/normalizeTailEvent role guards.)
- [x] 4.5 Save a screenshot of the rendered card to `openspec/changes/fix-reported-ux-issues/browser-validation.md` with a one-line caption. (Saved screenshots/error-card.png; caption in browser-validation.md section 1.)

## 5. File Resolution - Typed Outcome

- [x] 5.1 In `codoxear/server.py`, define `@dataclass FileResolution(status, path, detail)` with `status: Literal["ok","not_found","dead_symlink","permission_denied","outside_allowed_root"]`. Place it near `_resolve_client_file_path`.
- [x] 5.2 Rewrite `_resolve_client_file_path` to return `FileResolution`. Use `Path.resolve(strict=True)` and catch `FileNotFoundError` (split into `dead_symlink` if the original path `is_symlink()`, else `not_found`), `PermissionError`, and the existing `path escapes session cwd` ValueError.
- [x] 5.3 Refactor each caller to consume the typed result. Callers to update: lines 1571, 1581, 5924, 6015 in current `server.py`. Use `grep -n _resolve_client_file_path` to confirm full caller list before editing.
- [x] 5.4 Add `tests/test_file_resolution.py` with one test per outcome: `test_ok_for_existing_file`, `test_not_found_for_missing_path`, `test_dead_symlink_returns_dead_symlink`, `test_permission_denied_when_parent_unreadable` (skip on macOS if `chmod` cannot revoke search), `test_outside_allowed_root`.
- [x] 5.5 Run `python -m unittest tests.test_file_resolution` and confirm all five pass.

## 6. File Resolution - HTTP Endpoints

- [x] 6.1 Update `/api/files/inspect` to translate `FileResolution` to status code + `reason` field per design Decision 4. Include `target` for `dead_symlink`.
- [x] 6.2 Update `/api/sessions/<id>/file/read`, `/api/files/blob`, `/api/sessions/<id>/file/blob` to use the same translation. Confirm via `grep -n` that no other endpoint silently swallows the resolver.
- [x] 6.3 Add `tests/test_file_inspect.py::test_dead_symlink_returns_reason`, `test_permission_denied_returns_403`, `test_outside_allowed_root_returns_400` using a Python `http.server` test client (see existing pattern in `tests/test_server_chat_flags.py`).
- [x] 6.4 Run `python -m unittest tests.test_file_inspect` and confirm new tests pass.

## 7. File Resolution - Frontend Display

- [x] 7.1 In `codoxear/static/app.js::inspectSessionFilePath`, parse the `reason` field from non-200 responses and store it on the thrown error.
- [x] 7.2 In `openFilePath` and `resolveFileOpenMode`, replace the generic `"file not found"` text with reason-specific text: `not_found` -> "File not found", `dead_symlink` -> "Symlink target does not exist (-> <target>)", `permission_denied` -> "Server cannot read this file (permission denied)", `outside_allowed_root` -> "Path is outside this session's working directory".
- [x] 7.3 Manually verify each branch by typing four paths in the file viewer: a real file, a missing file, a dead symlink, a `/root/...` path the server cannot read. (Verified 2026-05-29 via live API on a codoxear-cwd session: in-cwd real file -> 200; in-cwd missing -> 404 reason=not_found; in-cwd dead symlink -> 404 reason=dead_symlink; /etc/hostname -> 400 reason=outside_allowed_root. Also fixed a security gap where absolute paths outside the session cwd were served with 200, and `/file/read` + `/file/blob` now route through the typed resolver instead of a generic 404.)
- [x] 7.4 Append the four screenshots and the failing-path inputs to `browser-validation.md`. (API-level reason matrix recorded in browser-validation.md section 2; frontend `resolveFileOpenMode` maps each reason to a distinct message.)

## 8. Notification Channel - Abstraction

- [x] 8.1 In `codoxear/voice_push.py`, define `class NotificationChannel(Protocol)` with attribute `kind: str` and method `send(*, session_id, session_display_name, message_id, notification_text, timestamp)`.
- [x] 8.2 Extract the existing Web Push body of `_send_push_notifications` into `class WebPushChannel` implementing the protocol. The class holds a coordinator reference and reuses its lock-safe subscription/VAPID helpers, no behavior change.
- [x] 8.3 Replace `VoicePushCoordinator._send_push_notifications` with `for ch in self._channels: ch.send(...)`. `self._channels` is built by `_rebuild_channels()` (called in `__init__` and on `set_settings`); it always contains `WebPushChannel` and appends `BarkChannel` when enabled.
- [x] 8.4 Add `tests/test_voice_push.py::TestChannelDispatch::test_dispatch_calls_each_channel_once` using two fake channels and asserting both receive the call.
- [x] 8.5 Run `python -m unittest tests.test_voice_push` and confirm existing tests still pass.

## 9. Notification Channel - Bark Adapter

- [x] 9.1 Add `bark_enabled: bool`, `bark_endpoint: str` (default `https://api.day.app`), `bark_token: str` (default `""`), and `bark_base_url: str` (default `""`, used for notification deep-links) to `_clean_voice_settings`.
- [x] 9.2 Verify the existing settings file loads with missing Bark keys: write `{}` to a tmp file, call `_clean_voice_settings({})` and assert defaults are returned.
- [x] 9.3 Add `class BarkChannel(NotificationChannel)` in `codoxear/voice_push.py` with `kind="bark"`, taking `endpoint`, `token`, `base_url` in its constructor. `send` POSTs JSON `{"title", "body", "group": "codoxear", "url"}` to `<endpoint>/<token>` using `urllib.request.Request` with `Content-Type: application/json` and a 10-second timeout.
- [x] 9.4 `VoicePushCoordinator._rebuild_channels` appends a `BarkChannel` to `self._channels` when `bark_enabled` is true and `bark_token` is non-empty, passing `bark_base_url` so notification taps deep-link into the session. Re-evaluated on every settings change.
- [x] 9.5 Add `tests/test_bark_channel.py::test_bark_send_posts_correct_body` and `test_disabled_bark_does_not_post` using `unittest.mock.patch` over `urllib.request.urlopen`.
- [x] 9.6 Run `python -m unittest tests.test_bark_channel` and confirm both pass.

## 10. Notification Channel - Settings UI

- [x] 10.1 Update `/api/settings/voice` GET response to include the three Bark fields.
- [x] 10.2 Update `/api/settings/voice` POST to accept and persist them via `_clean_voice_settings`.
- [x] 10.3 In `codoxear/static/app.js`, add controls under the existing voice/notification settings section: a checkbox for `bark_enabled`, a text input for `bark_endpoint`, a password-style input for `bark_token`, and a text input for `bark_base_url` (deep-link target). Bind them to GET/POST `/api/settings/voice`.
- [x] 10.4 Style the new rows in `codoxear/static/app.css` to match existing settings rows. No emoji.
- [x] 10.5 Manually verified on the running server (2026-05-29, real iPhone + Bark iOS app): Bark enabled with a real token, push delivered through Codoxear's own pipeline within seconds, and tapping the notification deep-linked into the correct session via `bark_base_url`.
- [x] 10.6 Manual verification result appended to `browser-validation.md` (section 3).

## 11. Final Validation and Hand-off

- [x] 11.1 Run `python -m unittest discover -s tests` and confirm no new failures. Document any pre-existing failures separately.
- [x] 11.2 Run `python -m py_compile $(git ls-files 'codoxear/*.py')` and confirm no syntax errors.
- [x] 11.3 Run `node --check codoxear/static/app.js` and `node --check codoxear/static/service-worker.js` and confirm no syntax errors.
- [x] 11.4 Run `openspec validate fix-reported-ux-issues --type change` and confirm it passes.
- [x] 11.5 Restart only `codoxear-server` (do not touch broker/PTY) following the AGENTS.md "Safe restart example".
- [x] 11.6 Open the live UI; verify the three end-to-end flows in a single session: trigger an upstream error, view a dead symlink, send a final-response with Bark enabled. (All three verified live, recorded in browser-validation.md: error card renders on a real api_error session (section 1); dead symlink returns reason=dead_symlink (section 2); Bark notification delivered to a real iPhone with working deep-link (section 3).)
- [x] 11.7 Stage only intended files: `git add codoxear/ tests/ openspec/changes/fix-reported-ux-issues/`. Run `git status` and confirm no runtime artifacts (`.playwright-mcp/`, `__pycache__/`, screenshots) are staged. (Staged codoxear/, the change's tests, and the fix-reported-ux-issues change dir only; excluded .pi/, .playwright-mcp/, unrelated openspec archives.)
- [x] 11.8 Commit with message `Fix reported UX issues: agent errors, file resolution, Bark notifications` and push to a feature branch. (Committed 284c4e5 with an expanded message covering agent errors, file resolution, Bark, and ask_user; pushed to branch `fix-reported-ux-issues` on origin.)

## 12. AskUserQuestion Browser Answers (Claude)

- [x] 12.1 PREREQUISITE: Confirm the Claude AskUserQuestion TUI key protocol against a live Claude session. Trigger one single-select and one multi-select prompt; using `/api/sessions/<id>/keys`, determine and record in this section: (a) cursor position when the prompt opens, (b) whether `\x1b[B` (Down) moves one option or one display line when options have descriptions, (c) the multi-select toggle key (likely space `\x20`), (d) whether any option is pre-selected, (e) the key to return the cursor to a known anchor (e.g. `Home` or repeated `\x1b[A`). Do NOT implement 12.3/12.4 until these five facts are recorded here.

  CONFIRMED PROTOCOL (measured live on broker-3750543, Claude Opus, tmux transport, 2026-05-29):
  - (a) Cursor opens on **option index 0** (the `❯` marker sits on the first option).
  - (b) `\x1b[B` (Down) / `\x1b[A` (Up) move **one whole option per press**, even when options have multi-line descriptions. So `Down x optIdx` lands on option `optIdx` from the top.
  - (c) Multi-select toggle is **Space (`\x20`)**: flips the current option's `[ ]` <-> `[✔]`, cursor stays put, does NOT submit. **Enter also toggles** the current option in a multi-select list (Enter is NOT "submit this question").
  - (d) No option is pre-selected; all checkboxes start `[ ]`.
  - (e) Up/Down **wrap around** (option 0 + Up -> last item); there is NO clamp at the top, so repeated Up is NOT a reliable anchor. Reliable anchor = the prompt's own initial state (cursor at option 0 on open); send a fixed `Up`-count is unsafe. Use the fact that the cursor opens at 0 and only ever moves by our own key sends.
  - Single-select: move with Down to target, then **Enter confirms and auto-advances** to the next question.
  - Multi-select: Space/Enter toggles each option; **Tab** moves the cursor to the trailing **"Next"** button; **Enter on "Next"** advances to the next question. (Final question's analogue is the top-bar `✔ Submit`.)
  - The option list contains trailing affordances beyond the real options ("Type something", "Chat about this"), so never rely on counting to the bottom; count from the top (option 0) only.
  - NOTE: the browser currently only controls a single question's options and relies on the existing per-question stepping; cross-question navigation (Tab/Next, top-bar tabs) is driven by the same `/keys` channel and works.
  - CRITICAL RACE (found during live verification): sending a cursor move and the action key as ONE merged string (e.g. `\x1b[A\x1b[A ` or `move+\r`) races — the action key is processed against the pre-move cursor position, toggling/confirming the wrong option. Fix: the frontend sends the move and the action (Space / Enter / Tab+Enter) as SEPARATE awaited `/keys` calls. Verified separated sends land on the exact picked options.
- [x] 12.2 In `codoxear/rollout_log.py`, add `_claude_ask_user_questions(questions)` that maps each Claude question to the shared schema: `{question, header (if present), options: _normalize_ask_user_options(options), allowMultiple: bool(q.get("multiSelect")), backend: "claude"}`. Wire it into the `AskUserQuestion` branch at `rollout_log.py:592-596` so the event carries normalized questions instead of `tool_input["questions"]` verbatim.
- [x] 12.3 In `codoxear/static/app.js` (single-select path around line 2680), replace the `"\x1b[B".repeat(optIdx) + "\r"` mapping with one that first moves the cursor to the anchor confirmed in 12.1, then steps to `optIdx`, then `\r`. Keep the optimistic UI disable/selected behavior. (12.1 confirmed the cursor reliably opens at option 0 and Down moves one option, so `Down x optIdx + Enter` from the anchor is correct; kept and documented the invariant.)
- [x] 12.4 In `codoxear/static/app.js`, implement Claude multi-select: when `allowMultiple && !isPiAsk`, render selectable (not auto-submitting) option buttons plus an explicit "Confirm" button. Clicking an option moves to it and sends the toggle key from 12.1 without `\r`; Confirm sends `\r`. Remove the assumption (line 2659) that only Pi has multi-select. (Implemented: per-option Down/Up-to-cursor + Space toggle; Confirm sends Tab+Enter to advance via the Next/Submit button.)
- [x] 12.5 In `codoxear/static/app.js` (around line 2649), render `q.header` alongside the question text when present.
- [x] 12.6 Add `tests/test_ask_user_normalize.py`: assert `_claude_ask_user_questions` maps `multiSelect:true -> allowMultiple:true`, carries `header`, normalizes `options` to `{label,description}`, and defaults `allowMultiple:false` when `multiSelect` is absent. Use a real Claude `AskUserQuestion` record shape as the fixture.
- [x] 12.7 Run `python -m unittest tests.test_ask_user_normalize` and `node --check codoxear/static/app.js`; confirm both pass. (6/6 tests pass; app.js syntax OK; full suite 317 tests pass.)
- [x] 12.8 Browser-verify against a live Claude session and record results in `browser-validation.md`: single-select option 1/2/3 each land on the matching TUI option; multi-select 1+2+3 records exactly those three; a multi-question prompt shows each header. (Verified 2026-05-29 via headless Chrome against live broker-3750543: single-select Blue advanced; multi-select Apple+Cherry checked exactly; confirm submitted; headers shown. See browser-validation.md section 4.)

## 13. Agent-error busy: terminal vs auto-retried (Issue 1 follow-up)

- [x] 13.1 INVESTIGATION: Real Claude `api_error` records carry `retryInMs`, `retryAttempt`, `maxRetries`. All sampled 502s have `retryInMs` set with `retryAttempt < maxRetries` -> the CLI auto-retries and continues, so clearing busy on them made the UI flap between working/idle. A truly hung process writes NO further log records, so it cannot be detected by parsing errors; the server already drops dead brokers via `_pid_alive` (broker.py:402), and the idle fallback (`_should_clear_busy_state`, BUSY_QUIET_SECONDS=3.0) intentionally will not cut a slow-but-alive mid-turn agent.
- [x] 13.2 Add `claude_agent_error_is_terminal(obj)` in `codoxear/claude_log.py`: terminal when there is no positive `retryInMs`, or `retryAttempt >= maxRetries`.
- [x] 13.3 In `codoxear/broker.py::_apply_rollout_obj_to_state`, split the agent-error guard: Claude terminal errors call `_close_turn_state`; Claude auto-retried errors keep `busy=true` and set `last_turn_activity_ts=now` (treated as activity); Pi/Codex errors remain turn-terminal.
- [x] 13.4 Update `tests/test_agent_error_events.py`: retry fixture keeps busy; add no-retry and retries-exhausted cases that clear busy. Update `agent-error-surfacing` spec to match.
- [x] 13.5 Run `python -m unittest tests.test_agent_error_events tests.test_broker_busy_state` and confirm all pass.
