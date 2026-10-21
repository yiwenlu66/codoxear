# Undelivered prompt must not vanish

## User report

> my pi agent starts with a warning. this makes first prompt to pi sessions on
> codoxear cannot go through.
> 1. fix the warning in my local pi config.
> 2. in the future when prompt fails to go through for whatever reason,
>    possible to let the prompt stay in ui? e.g., in composer draft

## Root cause of the report (part 1, done)

`pi 0.82.1` runs `showDeprecationWarnings()` at startup: a non-empty
`~/.pi/agent/tools/` (other than pi-managed `fd`/`rg`) prints a warning and then
blocks on `process.stdin.once("data")` behind `Press any key to continue...`.
Codoxear's confirmed send writes the whole prompt + Enter into the PTY, so the
gate consumed it and no turn was ever opened. The blocking gate, not the
warning text, is the bug.

## Success criteria (part 2)

1. A prompt that the backend never recorded returns to the composer draft for
   the session it was sent to, with a toast, instead of disappearing.
2. No false restore: a prompt that did reach the backend never comes back.
   Non-turn submissions (slash commands) and busy-origin steering sends are
   never restored.
3. The verdict is authored once. The broker owns "the send latch settled with
   zero turn rows"; the client owns the text it submitted; the composer's
   draft writer (composer + draft-sync controller) owns restoring text.
4. Failure evidence is visible, never silent.
5. Behavioral tests: broker/server Python tests, client VM tests, and one
   Docker end-to-end run against a backend that swallows the first prompt.

## Contracts

### Verdict shape

```json
{"send_id": "s7-lz3k9a", "verdict": "undelivered", "ts": 1761023999.5}
```

### Producer (broker + sessiond)

`State.send_latch_id: str | None`, `State.send_verdict: dict | None`.

* Send handler arms `send_latch_id` from request `send_id` only when the send
  arrived while the backend was idle (`not prev_busy and not prev_turn_open`)
  and `text.lstrip()` does not start with `/`. Every send clears the previous
  `send_verdict`.
* `_mark_busy_state_idle`: when the existing `latch_settled_idle` condition
  holds and `send_latch_id` is set, publish
  `{"send_id": …, "verdict": "undelivered", "ts": now}` and clear
  `send_latch_id`. A send that produced any turn activity (log row or PTY
  "esc to interrupt" hint, both bump `last_turn_activity_ts`) never publishes.
* Inject-failure state restore restores both new fields.
* Control `state` response adds `"send_verdict": <copy|None>`.

### Server

`Session.send_verdict: dict | None`, mirrored from the broker state response in
`SessionControlCoordinator.get_state` through a validating parser, exposed on
the public session row as `send_verdict`. `/api/sessions/<id>/send` accepts an
optional `send_id` and forwards it into the broker `send` request.

### Client

`app_message_flow.js` keeps an outstanding-delivery map keyed by `send_id`
(armed only for idle-origin, non-control-command sends that render locally),
sends `send_id`, and reconciles against session rows on catalog updates. On a
match it drops the optimistic pending echo row, restores the text through the
injected `restoreUndeliveredDraft(sessionId, text)` seam (composer stays the
single textarea writer; restore only when the composer is empty), and toasts.
