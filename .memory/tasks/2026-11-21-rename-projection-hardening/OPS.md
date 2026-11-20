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
