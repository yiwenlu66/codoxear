# EPISTEMIC — rename projection hardening

## Phenomenon
Renaming a session "occasionally fails" (no clear trigger); the user now reports
it is **not** solved and that even a *newly created* session fails to take its
name. A rename is two writes: (1) the alias store — the sidebar/tmux display
name, which essentially never fails — and (2) projecting that name onto the
backend's own session title (Pi bridge / Codex app-server / CC `/rename`). Only
(2) fails. Symptom: sidebar name sticks, backend title (Pi footer, `pi --resume`
row) does not, plus a toast ("renamed — Pi title not updated").

## Root cause A — the stale-thread false alarm (FIXED, commit 0c4e8706)
The projection passed a server-snapshotted `thread_id` and the broker
hard-rejected any mismatch with its live `st.session_id` (`stale-thread`). The
snapshot lags the broker's live value while a transcript rebinds, so a rename in
that window was rejected and surfaced as a failure. Removed; the broker now names
its live conversation and reports the thread it named. This was real and is
fixed. It is **not** the failure the user still sees.

## Root cause B — the boot-window timeout (LIVE, EXPLAINS THE CURRENT REPORT)
A rename (and the default-name push) is projected **synchronously into the
running Pi process through the bridge socket**, and it is issued the moment the
session appears in the sidebar — which for a new session is *before* Pi has
finished booting. Measured on broker-1228953: broker start 15:25:29; bridge
socket created 15:25:36 (+7.0 s); Pi `session_start` / caps written 15:25:48.99
(+19.5 s). The failing push landed at +11.6 s.

Chain:
- `session_name_projection._push` → broker control `name`, server `sock_call`
  `timeout_s=2.0`.
- broker `_set_session_name` (Pi branch) → `pi_bridge_control.set_bridge_session_name`
  → `call_bridge(timeout_s=1.5)`. Pi's event loop is blocked during boot, so the
  socket accepts but the reply never comes in 1.5 s → `socket.timeout`.
- `set_bridge_session_name` has **no** exception handling (unlike
  `bridge_session_name`), so the timeout escapes to `broker_control.name_handler`'s
  `except` → the broker replies `{"error": "timed out"}` → the projection records
  `reason=error`. When the broker thread also misses the server's 2.0 s deadline,
  the server records `reason=transport / TimeoutError`. **Both observed failure
  shapes are the same underlying stall.**
- The client toasts for any reason outside {unsupported, unsupported-clear,
  pending, no-thread}; `error`, `transport`, and the benign backoff reply
  `retry-later` all render "renamed — Pi title not updated".

The alias store is written **before** the push (`alias_assign_default` /
`alias_set`), so the sidebar name is correct even when the push fails; the
failure is only the backend title + the toast.

## Root cause C — the write is in doubt, not failed (CONFIRMED BY REPRO)
`scripts/docker_session_name_stall_verify.sh` freezes the agent with `SIGSTOP`
(socket accepts, nothing is read) and shows the timeout is **not** a lost write:
the rename POST blocks 1.5 s, answers `ok:true` for the alias with
`name_propagation` = `reason=error, detail="timed out"`, and records a failure —
yet on `SIGCONT` the agent applies the name **with no further request**, because
the bytes were already in its socket receive queue. So the control plane treats
an *unacknowledged* write as a *failed* write. The three views then disagree:
the sidebar shows the new name, `session_names.json` says `failed`, and the agent
has the new title. Retries (15/60/300 s backoff, plus a re-push when the
transcript binds under a new key) eventually make the record agree, which is why
it reads as "random" and leaves no durable trace.

Corroborating: `discover: sock state call failed … TimeoutError` jumped from
~20/day to ~2200–2700/day on 11-21/11-22 across ~24 brokers — the same transient
broker-control stalls seen through the 0.5 s discovery budget, i.e. the machine
is under enough load that these budgets are routinely exceeded.

## Ruled out / deliberately unchanged
- **Stale-thread guard** — already removed (Root cause A); re-checking confirmed
  it is gone from broker/sessiond.
- **Alias store loss / empty map** — fixed earlier (e90ef501); the current
  `session_aliases.json` holds all 24 live names.
- **Permanent name loss** — none: every live session shows both an alias and a
  `backend_session_name`.
- **Silencing `unsupported`** — still action (old Pi bridge needs `/reload`).
- **`--name` at launch / deriving defaults on read** — rejected earlier
  (second producer / renumbering).

## Residual / open
- The 2.0 s server budget vs the 1.5 s bridge budget leaves almost no margin for
  scheduler/lock latency; the timeout shapes flip depending on which deadline
  fires first.
- `retry-later` is a *benign* outcome rendered as a failure toast.
- The boot-window race is not covered by any verifier (Docker's Pi boots too
  fast); it must be made deterministic (stall the Pi process) to be tested.
- Whether the Pi event-loop block is codoxear's own doing (two bridge copies
  loaded: the injected `pi_active_session_bridge.ts` and the global
  `codoxear-bridge.ts` loader) or third-party extensions is unproven.

## Current claim
The user-visible "rename fails" / "new session gets no name" is the projection
timing out against a busy Pi (boot is the most common busy window). The write is
*in doubt*, not failed: it reaches the agent's socket and is applied later, while
the broker reports a timeout and the client toasts a failure. `session_name_
projection` then holds a `failed` record that disagrees with both the sidebar and
the agent. Confidence: high — reproduced deterministically (Steps 1–3 above); the
timing and error strings match the live incident exactly. The precise blocking
extension at boot is not identified and is not required for the fix.

## Question that would most change the model
Does a name push issued during the Pi boot window reproducibly produce
`reason=error/transport (timed out)` in Docker (with the Pi process deliberately
stalled), and does gating the push on bridge/session readiness make it go away
without losing the eventual title? If instead the stall persists long after boot,
the model shifts from "boot race" to "a general Pi event-loop stall," and the fix
moves to bounded retry/readiness rather than readiness gating alone.
