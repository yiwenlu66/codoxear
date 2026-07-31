# Final blocker fixes — Docker/API proof on port 19234

Verdict: PASS on functional commits `2506938` and `b858bfd`.

Harness: real `codoxear.server` in Docker, fake Unix control sockets, and real backend-shaped JSONL logs. Host API client drove the same routes used by the browser. Container was removed after collection.

Certified claims:

1. **Claude Code split live polling:** `/api/sessions/cc-live/messages/tail` first returned the user row and a signed live cursor. After the harness appended a later `system/subtype:turn_duration` row, `/api/sessions/cc-live/messages/live?cursor=...` returned one assistant event with `message_class:error` and text `The backend completed this turn without producing a response.` This proves the public live route now uses backend-aware prior-turn context for CC split closes.
2. **Fresh interrupted-idle discovery:** A fresh registry discovery for `fresh-interrupt` used a fake broker socket reporting `busy=false, queue_len=0, interrupted_idle=true` over a non-final Codex log. `/api/sessions` listed the session with `busy:false`, proving fresh insertion preserves the interrupted-idle override after `reset_log_caches()`.

Artifacts:
- `final_blocker_server.py`: Docker harness setup script.
- `sessions.json`: public session listing showing `fresh-interrupt` busy false.
- `cc-tail-before.json`: initial tail response with only the CC user row and live cursor.
- `cc-live-after-close.json`: live response after appending CC turn close, containing the assistant no-response error.
- `SUMMARY.json`: concise combined verdict.
- `executor-report.md`: implementation/test report.

Boundary:
This is API-level browser-route evidence rather than a DOM screenshot. DOM rendering of assistant error rows is already covered by `cc-outcomes-19210`; this proof targets the previously missing split-live server route condition.
