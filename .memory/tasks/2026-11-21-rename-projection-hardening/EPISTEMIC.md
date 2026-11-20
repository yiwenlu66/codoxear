# EPISTEMIC — rename projection hardening

## Phenomenon
Renaming a session "occasionally fails" with no clear trigger. A rename is two
writes: (1) the alias store — the sidebar/tmux display name, which essentially
never fails since e90ef501 — and (2) projecting that name onto the backend's own
session title (Pi bridge / Codex app-server / CC `/rename`). Only (2) fails.
Symptom: sidebar name sticks, backend title (Pi tab, `pi --resume` row) does not,
plus a failure toast ("renamed — <backend> title not updated").

## Root cause (committed)
The alias is a **per-session** name that must follow the session's live
conversation across every transcript rebind (Pi first-message bind, `/new`,
resume, restore). The projection passes a **server-snapshotted** `thread_id`, and
the broker **hard-rejects** any mismatch with its live `st.session_id`
(`stale-thread`). The server's cached `thread_id` lags the broker's live value
during a rebind (both come from the sidecar, but asynchronously), so a rename
fired in that window is rejected and surfaced as a failure — even though naming
the broker's live conversation with the session's alias is exactly the intended,
correct operation.

The guard protected bookkeeping, not correctness: the actual name call always
targets the broker's live thread anyway (Pi bridge names the live Pi session;
Codex uses `st.session_id`, not the passed thread; CC injects into the live TUI).
The passed `thread_id` was used *only* for the rejection. It traded a harmless
redundant re-push for a hard failure + false user alarm.

Two structural facts made it undiagnosable: failures are **never logged** (the
`stderr` projection dep was unused and the applied record stored only
`state: failed`, no reason), and they **self-heal** on the next sidecar refresh
(reconcile re-runs under the now-converged thread key), erasing their own trace.

## Fix
- **A. Broker owns transcript identity.** Removed the `stale-thread` rejection in
  `broker_control.name_handler`, `broker._set_session_name`, and the sessiond
  twins. The broker names its live conversation and returns the `thread_id` it
  actually named in every success reply. Request shape unchanged (still carries
  `thread_id`) so rolling deploys stay wire-compatible.
- **B. Projection records reality.** `_push` stores the applied record under the
  broker-reported effective thread identity when present, so bookkeeping names
  the thread that was actually titled and convergence is a no-op, not a re-push.
- **C. Observability.** Failure records now carry `reason`/`detail`, and one
  stderr line is emitted per failure through the already-plumbed dep, so a
  genuine transport/unsupported failure is self-diagnosing instead of vanishing.

## Ruled out / deliberately unchanged
- **Silencing transport/unsupported toasts.** Not done. After A removes the
  `stale-thread` false alarm, the remaining failure reasons are honest: transport
  = broker genuinely unreachable, unsupported = old Pi bridge needing `/reload`.
  The client reason mapping stays as-is; these are actionable.
- **Dropping `thread_id` from the request.** Kept for deploy wire-compat
  (old broker + new server still parses it; new broker ignores the gate).
- **The pid guard** (`expected_pid`) on the Pi bridge stays: it prevents naming a
  different Pi process and is a correctness check, not the race.

## Residual
- Transport failure during lazy Pi-bridge (re)registration is still reported as a
  failure (and retried on backoff). It is now visible via (C); if it proves
  frequent, a transient class can be split later. No evidence it is frequent.
- CC mid-turn `/rename` safety is still gated by the idle/`pending` path
  (unchanged).

## Current claim
A rename issued while a Pi/Codex session is rebinding its transcript no longer
fails: the broker names the live conversation and the projection records it under
the real identity. Genuine backend-unavailable failures remain reported and are
now logged. Confidence: high — root cause is the thread guard, which is
redundant with the name-follows-session invariant; covered by broker + projection
behavioral tests.
