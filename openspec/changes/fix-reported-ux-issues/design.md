## Context

Three independent failure modes were observed by the user while operating Codoxear in production:

```text
Issue 1: agent CLI hits an upstream/API error
  -> JSONL log gains a record with type ending in "*_error"
  -> rollout_log/_apply_rollout_obj_to_state has no case for it
  -> st.busy stays true forever
  -> UI keeps showing "working"

Issue 2: user opens "view file" in the UI with a valid absolute path
  -> server.py: _resolve_client_file_path -> Path.resolve() (strict=False)
  -> resolve() silently follows dead symlinks / unreadable parents
  -> caller calls path.exists() -> False
  -> UI shows generic "file not found" with no diagnosis

Issue 3: deployment is reachable only over plain HTTP
  -> browser refuses to register a service worker / PushManager.subscribe
  -> server-side webpush() never has any subscription to fire against
  -> mobile alerts never arrive
```

These problems live in three different subsystems (`broker.py` busy state machine, `server.py` file path resolution, `voice_push.py` push pipeline) but share one user-facing root cause: Codoxear silently drops information that the user expected to see.

## Goals / Non-Goals

**Goals:**
- Surface upstream agent errors in the UI as a distinct event class so the user sees an explanation instead of an indefinite "working" indicator.
- Make file-viewer errors carry a precise reason (`not_found`, `dead_symlink`, `permission_denied`, `outside_allowed_root`) so the user can act on the failure.
- Provide one HTTP-friendly push channel (Bark) so HTTP-only deployments can deliver mobile notifications without changing transport.
- Keep cursor APIs, byte offsets, and existing event ordering stable.
- Cover each behavior with a regression test that fails on the current code and passes after the change.

**Non-Goals:**
- Redesign the busy/turn state machine. Only add error event handling.
- Replace the existing Web Push pipeline. The Bark channel is added alongside it.
- Build a generic plugin system for arbitrary push services. Bark is concrete; the abstraction is internal and small.
- Map every possible filesystem error. Five outcomes cover the observed failure modes; everything else maps to `not_found` for now.
- Translate error event text. Pass through the backend's message verbatim.
- Implement Tailscale / HTTPS / TLS support. Out of scope for this change.

## Decisions

### Decision 1: Add a single new normalized event class for agent errors

Add `class: "agent_error"` to the cursor + full parser output. Each event carries `{ type, message, ts, source }`:

```text
type     = backend-specific error type (e.g. "stream_error", "api_error",
           "rate_limit_error", "request_error"). Pass through verbatim.
message  = human-readable error text from the log payload.
ts       = epoch seconds, taken from the record timestamp.
source   = "codex" | "pi" | "claude"
```

Rationale: keeps the existing event taxonomy (`text`, `interactive`, ...) clean, lets `app.js` render it with a single new card component, and keeps backend logic small.

Alternative considered: piggy-back error text on an existing `text` event with a flag. Rejected because the UI already uses `text` for assistant prose; an error needs its own visual treatment and accessibility role.

### Decision 2: Recognize errors per backend, normalize once

Each backend log uses a different shape for errors. Normalize at the parser layer, not at the renderer:

```text
codex   -> rollout_log: event_msg.payload.type matches "*_error"
                         OR payload.type == "stream_error"
                         OR payload contains an "error" sub-object
pi      -> pi_log: top-level type == "message" with role == "error"
                   OR a tool_result with isError == true that ends a turn
claude  -> claude_log: type == "system" with subtype == "error"
                       OR type == "result" with subtype == "error_*"
```

The exact selector strings are confirmed against a real failing log at implementation time and recorded in tasks.md; the design only commits to the **shape** of the dispatch (one `_agent_error_event_from_record` helper per backend, all returning the same dict).

Rationale: keeps backend specificity behind the parser boundary. The cursor APIs and the renderer see one event class regardless of backend.

Alternative considered: feature-detect at the renderer. Rejected because the renderer cannot reliably distinguish error tool results from ordinary tool results without knowing the backend.

### Decision 3: Error events close the turn and clear `busy`

In `broker._apply_rollout_obj_to_state`, treat a recognized agent error record the same way `task_complete` is treated today:

```text
recognized_error_event(obj) -> _close_turn_state(st)
```

Rationale: an upstream error is a terminal turn state from the user's point of view. Leaving `busy=true` is the bug.

Alternative considered: leave `busy=true` but surface the error. Rejected because the user has no way to tell whether the agent is recovering or stuck; the conservative answer is "turn ended, and here is why".

### Decision 4: Replace silent path resolution with an explicit outcome enum

Introduce a small typed result instead of returning a `Path`:

```python
@dataclass
class FileResolution:
    status: Literal["ok", "not_found", "dead_symlink",
                    "permission_denied", "outside_allowed_root"]
    path: Path | None
    detail: str   # short human-readable
```

`_resolve_client_file_path` uses `Path.resolve(strict=True)` and catches the specific exceptions:

```text
FileNotFoundError + p.is_symlink()         -> dead_symlink
FileNotFoundError                          -> not_found
PermissionError                            -> permission_denied
ValueError("path escapes session cwd")     -> outside_allowed_root
otherwise                                  -> ok
```

`/api/files/inspect` and `/api/sessions/<id>/file/read` translate the result:

```text
ok                    -> 200 + view payload (unchanged)
not_found             -> 404 { error: "...", reason: "not_found" }
dead_symlink          -> 404 { error: "...", reason: "dead_symlink", target: "<link target>" }
permission_denied     -> 403 { error: "...", reason: "permission_denied" }
outside_allowed_root  -> 400 { error: "...", reason: "outside_allowed_root" }
```

Rationale: the current code conflates four very different failures into one 404. The user reports symptoms that map to at least two of the new cases (dead symlinks under network-mount paths, server-side permission gaps).

Alternative considered: keep `Path` return type, add a sibling `_diagnose_file_path` helper called only on failure. Rejected because the success path becomes "resolve, then diagnose if it failed", which double-touches the filesystem.

### Decision 5: Push abstraction is `NotificationChannel`, with one concrete `BarkChannel`

```python
class NotificationChannel(Protocol):
    kind: str  # "webpush" | "bark"
    def send(self, *, session_id: str, session_display_name: str,
             message_id: str, notification_text: str,
             timestamp: float | None) -> None: ...
```

`VoicePushCoordinator._send_push_notifications` iterates over `self._channels` and calls `send` on each enabled channel. Existing Web Push logic is wrapped into `WebPushChannel` to satisfy the same protocol. `BarkChannel` is added.

Bark configuration lives in voice settings:

```text
bark_enabled: bool
bark_endpoint: str   # default "https://api.day.app"
bark_token: str      # the user's Bark device key
```

When `bark_enabled` is true and `bark_token` is non-empty, `BarkChannel.send` issues:

```text
POST <bark_endpoint>/<bark_token>
Content-Type: application/json
{
  "title": "<session_display_name>",
  "body":  "<notification_text>",
  "group": "codoxear",
  "url":   "<base_url>/#session=<session_id>"
}
```

Rationale: Bark's API is documented, free for personal use, works over HTTPS even when the Codoxear UI is HTTP, and does not require browser participation. One HTTP POST per notification keeps the new code surface tiny.

Alternatives considered:
- `ntfy.sh`: also viable. Defer to a later change if the user wants it; we'd plug a `NtfyChannel` into the same protocol.
- Telegram bot: harder to configure (bot creation, chat id), so not the first channel.
- Keep using only Web Push and require HTTPS: rejected because the user explicitly does not want to set up HTTPS now.

### Decision 6: Settings storage and migration

Add the three new fields to the existing `voice_settings.json`. `_clean_voice_settings` validates them. Existing files without these keys default to:

```json
{
  "bark_enabled": false,
  "bark_endpoint": "https://api.day.app",
  "bark_token": ""
}
```

No migration script needed; loaders treat missing keys as the defaults.

Rationale: settings already live in one JSON file. Adding three keys keeps the storage story simple.

### Decision 7: Test layering

```text
Unit tests:
  parser layer:    tests/test_rollout_log.py adds error event extraction cases
                   per backend, asserting shape and ordering through both
                   _extract_chat_events and the cursor path
  busy state:      tests/test_idle_heuristics.py (or new file) asserts
                   _apply_rollout_obj_to_state clears busy on error
  resolver:        tests/test_file_resolution.py covers all five outcomes
                   using tmp_path, dead symlinks, chmod, and an absolute
                   path outside the session cwd
  Bark dispatch:   tests/test_bark_channel.py uses a stub HTTP transport
                   and asserts the request body, URL, and that disabled
                   channels are skipped

Integration / endpoint tests:
  cursor APIs:     tail/live/history must include error events
  /api/files/*:    must return reason field on each non-ok outcome
  /api/notifications/subscription: accepts kind="bark" with token

Browser smoke (manual, recorded in tasks):
  trigger an upstream error -> error card visible, "working" cleared
  view a dead symlink path -> diagnostic shown
  trigger a final-response message -> Bark notification on phone
```

### Decision 8: Normalize Claude AskUserQuestion at the parser, fix cursor mapping at the renderer

Two layers are wrong today and both are fixed:

**Parser (normalize once).** The Pi path already runs `_pi_ask_user_event_from_tool_call` -> `_normalize_ask_user_options` and emits `{question, options:[{label,description}], backend:"pi", allowMultiple, allowFreeform, allowComment}`. The Claude path (`rollout_log.py:592`) passes `tool_input["questions"]` through verbatim. Claude's real per-question keys are `{header, multiSelect, options:[{label,description}], question}` (confirmed from a live log). So the frontend's `q.allowMultiple` is always `undefined` and it never learns a question is multi-select. Add a `_claude_ask_user_questions` normalizer that maps `multiSelect -> allowMultiple`, carries `header`, runs options through the shared helper, and tags `backend:"claude"`.

**Renderer (cursor mapping).** Today selecting option `optIdx` sends `"\x1b[B".repeat(optIdx) + "\r"` (`app.js:2680`). This assumes (a) the TUI cursor starts on option 0 and (b) one `Down` moves exactly one option. Both can be false: a multi-line option (label + description) or a pre-highlighted recommended option breaks the count. The fix must move to a known anchor first (e.g. enough `Up` presses, or `Home` if supported) and then step down, based on the **confirmed** Claude TUI navigation behavior.

**Multi-select.** Claude `multiSelect` questions currently fall through to the single-select path and submit on first click. Implement a toggle-then-confirm interaction: clicking an option moves to it and sends the toggle key (space, pending confirmation), and only an explicit confirm sends `\r`.

Rationale: keeping backend specificity behind the parser boundary mirrors Decision 2; the renderer stays backend-agnostic and reads one schema.

Alternative considered: send a structured answer to the broker instead of simulating keys. Rejected for this change because the broker drives the real CLI through a PTY and has no structured answer channel for AskUserQuestion; key simulation is the existing contract (see the prior "Fix Pi ask_user cursor events" commit).

**Prerequisite (must happen before the renderer fix):** the exact Claude TUI key protocol is not knowable from the codebase. Confirm against a live Claude session: (1) cursor position when the prompt opens, (2) whether `Down` moves one option or one display line when options have descriptions, (3) the multi-select toggle key, (4) whether any option is pre-selected. The renderer fix is implemented against these confirmed facts, not guesses.

## Risks / Trade-offs

- [Risk] Claude TUI navigation differs from the assumptions and the cursor still lands on the wrong option. -> Confirm the key protocol against a live session first (Decision 8 prerequisite); encode the confirmed behavior; verify each of single-select N=1/2/3 and multi-select in the browser before marking done.
- [Risk] Claude changes its AskUserQuestion TUI key bindings in a future CLI version. -> The normalizer and renderer are small and localized; document the confirmed key protocol in tasks.md so a future drift is a one-spot fix.

- [Risk] Backend log shapes for errors differ from the assumptions above and the dispatch misses real errors. -> Capture one real failing log per backend during implementation; record the matched selector in `tasks.md` and add a fixture before writing the helper.
- [Risk] Marking error events as turn-terminal masks recoverable retries that the CLI does internally. -> Only treat errors that the CLI itself surfaces to its log as terminal; ignore mid-stream warnings. Selector tightness is verified by tests.
- [Risk] `Path.resolve(strict=True)` differs across Python minor versions in symlink loop handling. -> Pin the dead-symlink branch to `OSError` subclasses we explicitly catch; add a test for symlink loops.
- [Risk] Switching the resolver return type breaks callers that assume `Path`. -> Keep `_resolve_client_file_path` return shape compatible by exposing a `path` attribute; refactor each caller in one commit per call site.
- [Risk] Bark API throttles or rate-limits with many subscribers. -> Bark is per-device; we POST once per subscription, not once per listener. Document the request volume in `voice_push.py`.
- [Risk] User shares one Bark token across phones; deleting the token affects all devices. -> Configuration is per-user, not per-device, in this iteration. Document the limitation and revisit if multiple users join.
- [Risk] Front-end error card and file-error display added to an already 8.7k-line `app.js`. -> Scope to two new render functions; avoid touching unrelated code paths.
- [Risk] Adding three settings keys without a migration breaks if `_clean_voice_settings` is strict. -> Verify in tests that an existing settings file without the new keys loads without error.

## Migration Plan

1. Land parser changes (Decision 1, 2) behind unit tests; no UI change yet.
2. Land broker change (Decision 3) with a new busy-state test; verify against an existing failing-session fixture.
3. Land resolver change (Decision 4); refactor each `/api/files/...` caller; update front-end error display.
4. Land `NotificationChannel` abstraction with `WebPushChannel` only (no behavior change), then add `BarkChannel` and settings UI.
5. Restart `codoxear-server`. Validate manually: trigger one upstream error, view a dead symlink, send a final-response message with Bark enabled.

Rollback: each step is its own commit; reverting any one keeps the rest working because the dispatch points are guarded by feature presence (settings disabled, no error events recognized, etc.).

## Open Questions

- The exact `payload.type` strings emitted by the user's Codex/Pi versions on upstream errors are unknown. The first task is to capture them from a real failing log and pin them in tests. If the strings differ from the design's guesses, Decision 2's selectors get refined, not the architecture.
- Should we also surface error events through `_extract_delivery_messages` so they trigger voice/push? Default for this change: no; voice/push fires only on `final_response`. Errors render in the UI only.

## Confirmed selectors (after fixture capture)

```
claude (CONFIRMED via tests/fixtures/claude_api_error_sample.jsonl):
   obj["type"] == "system"
   obj["subtype"] == "api_error"
   obj["level"] == "error"
   error_status   = obj["error"]["status"]                          # int (e.g. 502, 503, 429)
   error_message  = obj["error"]["error"]["error"]["message"]        # string
   error_type     = obj["error"]["error"]["error"]["type"]            # e.g. "upstream_error"
   error_ts       = obj["timestamp"]                                  # ISO 8601 string

   normalized event:
     {"class": "agent_error", "source": "claude",
      "type":  f"api_error_{error_status}",   # e.g. "api_error_502"
      "message": error_message,
      "ts": parse(error_ts) or fallback}

pi (NOT FOUND in any sampled log):
   Pi CLI does not surface upstream errors as JSONL events. The CLI retries
   internally. The Pi helper exists as a stub returning None, with a guard
   for the documented `toolResult` `isError:true` shape so that it activates
   automatically if Pi ever starts emitting these.

codex (NO SAMPLE AVAILABLE):
   Permissive selector based on Codex public-source naming conventions:
     obj["type"] == "event_msg"
     obj["payload"]["type"] in ("error", "stream_error", "api_error",
                                "request_error", "rate_limit_error")
     OR obj["payload"]["type"].endswith("_error")
   Captures error message from payload["message"] OR payload["error"]["message"].
   Tested against a synthetic fixture matching this shape.
```
