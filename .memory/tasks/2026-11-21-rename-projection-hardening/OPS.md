# OPS — rename projection hardening

## 2026-11-21 02:05 CST — investigation (live, read-only)
- session_names.json: 26 records, all applied/adopted, zero failed → failures
  self-heal and leave no trace. Projection never wrote to its stderr dep.
- Records keyed `pi:broker-<pid>` coexist with real-UUID keys for the same names
  (pi:broker-1347767 and pi:01a1fa1f-… both 'codox-naming-fail') → names are
  projected before/through the transcript bind, under shifting identities.
- Pi log for 01a1fa1f: session_info 'codoxear' (default) then
  'codox-naming-fail' 0.24s later, then again at 18:03:32 (re-push after the
  identity key changed).
- broker_control/sessiond_control name_handler and broker/sessiond
  _set_session_name rejected thread_id != st.session_id as `stale-thread`; the
  passed thread_id was used for nothing else (Pi bridge, Codex st.session_id, CC
  live TUI all target the live conversation). Client maps stale-thread to the
  generic "renamed — <backend> title not updated" toast.

## 2026-11-21 — implementation
- Removed stale-thread gate (broker_control, sessiond_control, broker, sessiond);
  success replies carry `thread_id` = live transcript named.
- Projection `_push` keys applied record by reply thread_id (fallback: requested).
- `_record_failure` stores reason/detail and logs one `session-name:` stderr line
  (deps.stderr = sys.stderr in session_manager_factories).
- Tests: broker race test (request thread-1, broker on thread-2 → ok, reports
  thread-2), projection race test (recorded under thread-2; convergence = no
  push), old-broker reply compat, failure reason+log. 3011 pass; 2 failures
  (test_hint_mode_coverage draftUndoBtn, test_pi_prompt_receipt) reproduce on the
  stashed tree → pre-existing, unrelated.
- Docker verifier step 5 added: raw broker `name` call with an outdated thread id
  must return ok + live thread_id and append the Pi session_info row.

## 2026-11-22 15:55 CST — residual failure located (user: "not solved; even new session fails")

Live evidence (read-only; deployed commit c3231557, service up since 11-21 02:46):
- `journalctl`: the *only* two `session-name:` lines in retention, both for the
  same new session `pi:broker-1228953`:
  - `15:25:41 name='codoxear' failed reason=error attempt=1 detail=timed out`
  - `15:49:49 name='codox-pagination-bug' failed reason=transport attempt=2
    detail=TimeoutError: timed out`
- `session_names.json`: no `failed` records now; `pi:broker-1228953` is
  `applied` ('codox-pagination-bug', ts 15:50:50) → failures self-heal.
- broker-1228953: start 15:25:29.4; bridge sock created 15:25:36.5 (+7.0s);
  caps (i.e. Pi `session_start`) written 15:25:48.99 (+19.5s). The failing push
  was at +11.6s — while Pi was still booting (session_start not yet fired).
- Across 24 live Pi sessions all aliases + `backend_session_name` are set;
  bridges answer `get_session_name` in 1–70 ms now (no persistent stall).
- `discover: sock state call failed … TimeoutError` exploded 11-21/11-22
  (2180 + 2735, vs ~20/day before) across ~24 brokers — same origin: transient
  broker-control stalls, 0.5 s budget.

Code path (confirmed by reading, not changed):
- `session_name_projection._push` → `sock_call({'cmd':'name'}, timeout_s=2.0)`.
- `broker_control.name_handler` → `set_session_name` → `broker._set_session_name`
  Pi branch → `pi_bridge_control.set_bridge_session_name` (no try/except) →
  `call_bridge(timeout_s=1.5)` raises `socket.timeout` on a stalled Pi event loop
  → escapes to `name_handler`'s `except` → reply `{"error":"timed out"}` →
  projection records `reason=error`, detail `timed out`.
- If the broker thread also misses the server's 2.0 s deadline, the server records
  `reason=transport / TimeoutError` instead — the two shapes are the same stall.
- Client (`app_session_lifecycle.namePropagationNotice`) toasts for any
  `ok !== true` reason outside {unsupported, unsupported-clear, pending,
  no-thread}; `error`, `transport`, and `retry-later` all read as
  "renamed — Pi title not updated".

## 2026-11-22 15:59 CST — mechanism

The projection pushes the backend title synchronously to the running Pi process
the moment the session appears in the sidebar (broker sidecar exists). For a new
session that is *before* Pi has finished booting (~19 s here, 30+ extensions):
the injected bridge's socket accepts (created at ~+3–7 s) but Pi's event loop is
blocked, so the 1.5 s write times out. The alias store is already written, so the
sidebar name is correct; the failure surfaces only as the backend-title toast.
A later reconcile/rename retry applies the title (self-heal), which is why
`session_names.json` shows no failed record and the incident is near-invisible
after the fact. `retry-later` (the benign backoff reply) is also rendered as a
failure toast, so a rename issued shortly after a poll-side failure can toast
even with no new transport attempt.

## 2026-11-22 16:20 CST — deterministic Docker repro

New verifier `scripts/docker_session_name_stall_verify.sh` (real Pi + broker +
server; no model call). It freezes the agent process with `SIGSTOP` so its
listening socket accepts but nothing is read or answered — the same condition a
blocked event loop produces during boot or work. Run (image
`codoxear-name-stall:c3231557-wt`) → PASS, artifacts
`/tmp/codoxear-name-stall-results.M6sq2b`:

```
STEP 1 ok: healthy agent -> rename propagated (applied)
STEP 2: agent pid 97 frozen (SIGSTOP); its listening socket still accepts
STEP 2: rename HTTP latency while agent frozen: 1.505191s
STEP 2: API says the rename succeeded (alias=during-stall) AND reports error timed out
STEP 2 ok: agent still has its old title: ["workspace", "healthy-name"]
STEP 2: recorded projection failures: {"pi:name-stall-session": {"attempts": 2,
        "detail": "timed out", "name": "during-stall", "reason": "error",
        "state": "failed", ...}}
STEP 3 ok: the name the API reported as failed appeared with no user action
STEP 3: codoxear still records the rename as failed: {"state": "failed",
        "name": "during-stall", "reason": "error", "detail": "timed out"}
```

Two distinct facts, one run:
1. **The request is delivered but reported failed.** The rename POST blocked the
   full 1.5 s bridge budget, answered `ok: true` for the alias while
   `name_propagation` said `reason=error, detail="timed out"`, and recorded a
   permanent-looking failure — yet the bytes had already reached the agent's
   socket. On `SIGCONT` the agent applied the name with no further request. This
   is an in-doubt write treated as a definite failure.
2. **The three views disagree.** After the incident the sidebar alias is the new
   name, `session_names.json` says `failed`, and the agent has the new title.
   Nothing reconciles them except a later poll's retry.

Client mapping confirmed by reading: `namePropagationNotice` (duplicated in
`static/app_session_edit.js` and `static/app_session_lifecycle.js`) returns
"renamed — Pi title not updated" for `reason=error`/`transport`, so this exact
repro is the toast the user sees.

Note: a plain transcript read did not re-trigger the push in the repro loop
(because a failed push's backoff is 15/60/300 s by attempt count, and because
`refresh_session_meta` returns early on an invalid sidecar read). The
read-route amplification is real but intermittent and was not made a verifier
assertion.

## 2026-11-23 — the "name flips back to the default" bug (root cause D, FIXED)

User report: creating a session with a chosen name sets it, then it reverts to
the cwd basename a few seconds later.

Evidence that the precondition is real: every live Pi session log contains the
cwd-basename `session_info` row *before* the chosen name, e.g.
`['codoxear','codox-pagination-bug','codox-pagination-bug']`,
`['dexgem-litellm-gateway','litellm-remove-fb',...]`. So codoxear assigns and
pushes its default before the user's rename arrives.

Unit-level repro (Harness): reconcile assigns default 'repo' + pushes it; the
broker echoes it into `backend_session_name`; the alias is then set to 'My
Required Name'; the next reconcile returns `reason="adopted", name="repo"` and
the alias store is back to 'repo'.

Root cause: in `SessionNameProjection._reconcile_locked_identity` the
"terminal title outranks an assigned default" rule was

    if _origin(record) == ORIGIN_DEFAULT and observed and observed != desired:

`desired` is the *new* alias. After the default is pushed, the backend's
`observed` title IS the default codoxear itself wrote, and the broker's echo
(`backend_session_name`) makes it visible to the server. A rename changes
`desired`, so `observed != desired` holds and the rule adopts codoxear's own
echoed default back over the user's name. The rule cannot tell a title the user
typed in the terminal from the default codoxear projected.

Fix: adopt only while the alias is still the default codoxear last projected --
require `desired == record_name` (equivalently `observed != record_name`). Once
the user renames in codoxear, `desired != record_name`, so the backend echo is
not treated as a terminal title and the rename is pushed. Comment updated.

Docker end-to-end repro `scripts/docker_session_name_flip_verify.sh` (real Pi +
broker + server; pre-created broker so the bridge is live and the default actually
reaches Pi):
- BEFORE the fix: final projection `{"state":"adopted","name":"workspace"}`, the
  alias flipped to the default on the very first poll after the rename.
- AFTER the fix: `{"state":"applied","name":"My Required Name"}`, alias stays the
  chosen name. Artifacts `/tmp/codoxear-name-flip-results.Y4EuzS` (pass) and
  `...ZQHU6U` (repro).

Regression test `test_the_echoed_default_is_not_mistaken_for_a_terminal_title`.
Related suites 37 passed; full suite 3014 passed, 2 pre-existing failures
(`test_hint_mode_coverage::draftUndoBtn`, `test_pi_prompt_receipt`) unchanged.

Follow-up (not done): `_handle_rename` returns the alias captured before the
reconcile, so a genuine adoption (terminal title over a default) can still make
the response disagree with the store. With this fix the normal rename path is
consistent; the response could be re-read after the projection for full honesty.
