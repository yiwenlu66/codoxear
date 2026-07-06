All five claims certified. Here is the final report.

## PASS/FAIL by user-facing claim

| # | Claim | Result |
|---|---|---|
| 1 | Attachment indicator truth | **PASS** |
| 2 | Deleted-session attachment cleanup | **PASS** |
| 3 | Mobile composer controls ≥ 44×44, no overflow | **PASS** |
| 4 | Transcript/no-answer projection | **PASS** |
| 5 | Idle projection after interrupt/resume | **PASS** (broker-state path; one log-only sub-variant not driven end-to-end) |

**Artifact path:** `/tmp/codoxear-cert-19200/cert-artifacts/` — report at `VERIFICATION-REPORT.md`, with screenshots, DOM/JSON summaries, before/after filesystem listings, API transcript evidence, and the harness sources (`fake_broker_harness.py`, `stateful_broker.py`, `browser_harness.js`).

### Key product facts observed (mechanism-grounded)
- **Indicator** (`#attachBadge`): projected from `max(localCount, serverPending?1:0)`; appears after `/inject_file`, survives reload (reads server `pending_attachment`), hides in **40 ms** after send (frontend mutates cached value pre-poll), hides after `/pending_attachment/clear`.
- **Cleanup**: `remove_session_uploads` removes only `<uploads>/<sid>`; e2e delete removed the dir, siblings survived, and the real-function symlink test showed the link unlinked while its outside target survived intact.
- **Mobile**: all four controls measured exactly 44×44 at 390×844 (desktop-pointer and touch-emulated); `scrollWidth == clientWidth == 390`, no overflow.
- **No-answer**: `/messages/tail` returned the assistant event `"The backend completed this turn without producing a response."` (`message_class: error`); control turn unaffected; browser DOM contains the text.
- **Idle/resume**: API `busy` went `false → true → false` and the sidebar dot went `idle → busy → idle` across interrupt→resume→complete — not falsely idle after resumed activity, not falsely busy after completion.

### Exact remaining gaps
1. **Claim 5 log-only variant**: the stale-interrupted-idle guard (`session_log_runtime`, broker keeps `interrupted_idle:true` while post-interrupt log activity arrives) was proven via the broker-state path (broker flips `busy:true` on resume, which is the primary mechanism). The log-only-without-broker-flip path is covered by the existing `tests/test_codex_no_response_projection.py` live-split suite but was not separately re-driven through a dedicated harness here. No evidence invented.
2. Screenshots are real captures but were not OCR-verified (image rendering unavailable to the agent); the DOM/JSON summaries are the primary evidence alongside each.

### Constraints honored
No code edits, no commits, no staged files; repo working tree clean at `fb42cfd`. Only the port-19200 container I started was stopped; pre-existing containers and the live 8743 server untouched; host live runtime never touched (isolation preflight passed).