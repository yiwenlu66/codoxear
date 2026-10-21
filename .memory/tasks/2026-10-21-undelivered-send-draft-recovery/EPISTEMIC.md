# EPISTEMIC — undelivered-send draft recovery

## Phenomenon

A codoxear prompt submitted to a Pi session vanished: the transcript showed the
optimistic user echo, the backend log had no user row, and the composer was
empty. Cause chain (confirmed by reading pi 0.82.1 `dist/migrations.js`):

1. `runMigrations()` collects deprecation warnings for a non-empty
   `~/.pi/agent/tools/`.
2. `showDeprecationWarnings()` prints them and blocks on
   `process.stdin.once("data")` ("Press any key to continue...").
3. Codoxear's confirmed send writes `text` + Enter into the PTY. The gate
   consumes it; the TUI has not started; nothing is ever recorded.
4. Server returns success because the PTY write succeeded. Client clears the
   composer and tombstones the server draft. The prompt is gone from the UI.

## The mechanism that matters

A confirmed send is a *bet*: PTY accept ≠ backend accepted. Codoxear already
detects the bet's failure in the broker (`send_latch_activity_ts` /
`latch_settled_idle` in `broker_turn_state._mark_busy_state_idle`) but spends
that observation only on the busy indicator (`interrupted_idle`). The prompt
text is never tied to the bet, so nothing can bring it back.

Fix: make the bet's settlement a per-send verdict (`undelivered`), carried by
the client-generated `send_id`, and let the composer's own draft writer put the
text back. The verdict is needed only for idle-origin, non-slash sends, where
"zero turn rows since the send" is unambiguous evidence that the backend did
not act on the submission.

## Ownership decisions

* **Broker** authors the verdict: it owns the quiet-window, the turn-aware
  latch, and the PTY. The client must not reimplement settle timing.
* **Client** owns the text: only it knows what the user typed. The broker
  publishes an id, not a copy of the prompt (no listing payload bloat, no
  second text store).
* **Composer / draft-sync** remains the only textarea + draft writer. The
  restore goes through `setDraftFromServer` + `noteDraftEdited`, not a new
  writer.
* Restore is refused when the composer already holds other text (never clobber
  a draft). The journal-backed ↑ recall remains the fallback; the toast names
  it.

## Discriminating predictions

| Case | Prediction |
| --- | --- |
| Backend swallows the prompt (modal/gate) | verdict `undelivered`, draft restored |
| Prompt reaches the log | user row bumps `last_turn_activity_ts` ⇒ no verdict |
| Backend shows "esc to interrupt" but writes the log late | PTY hint bumps activity ⇒ no verdict |
| Slash command (`/model`, `/effort`, `/reload`, `/new`) | never armed ⇒ no verdict |
| Send while busy (steering) | never armed ⇒ no verdict |
| Injection/PTY failure | `commit_unknown` path, composer already keeps text |

## Boundaries (accepted)

* Text is restored, not attachments (attachments were cleared on ack).
* A verdict that lands after a page reload has no matching client record and
  is ignored; the send journal + ↑ recall still holds the text.
* Queue-dispatched sends produce no client verdict (queue item identity).

## Verified (Docker, real Pi 0.82 + real browser)

`scripts/docker_undelivered_send_verify.sh` seeds `~/.pi/agent/tools/` in a
throwaway container HOME so the real Pi blocks on its deprecation gate, sends
the first prompt from the browser's Send button, and observes all four
scenarios. Measured: composer cleared 0.6 s (echo), verdict 3.3 s (quiet
window), draft restored 8.5 s (one session-list poll later). The negative
control (second send after the gate is gone) wrote a Pi user row and published
no verdict of its own, so the delivered path stays silent even when the log row
arrives late.

## Resolved

* The 3 s quiet window does not false-positive on a live backend: a real turn
  bumps activity (log row or "esc to interrupt" PTY hint) before the window
  can settle, demonstrated by scenario 4.
