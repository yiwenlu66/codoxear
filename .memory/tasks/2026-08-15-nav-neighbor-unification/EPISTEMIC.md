# Epistemic model — nav neighbor unification

## Claim (justified)
The navigation defect chain is fixed end-to-end:
1. `search_chat_logs_bounded(order="latest")` now passes the caller's order to
   the per-file scan (1d56655c); per-file `keep_latest` was already correct.
2. `GET /api/sessions/<id>/messages/neighbor` (ca77fed5) resolves exactly one
   positional neighbor with `same_log`, on top of the fixed wrapper.
3. One client owner `materializeNeighbor` (b5aec5e1): rendered→scroll;
   previous+same_log→bounded prepend; else detached window. Failure toasts
   distinct from boundary toasts (a1a435ae).
4. Cross-log reachability verified for codex/pi/cc rotated logs (e936511b) via
   the production cursor-target wiring (allowed paths from
   session_log_paths_for_search, mirrored from server.py).

Behavioral evidence (Docker, pi backend, OPS 2026-08-15): prev past the loaded
boundary landed on the true immediate neighbor (user msg 27) with 12→41 row
prepend; boundary toast at first message; cross-log prev loaded the rotated
log's detached window; zero page errors.

## Known non-defects / pre-existing behavior
- Next-direction at transcript bottom: trailing user rows whose rect.top can
  never reach the chat top re-match as "local targets" forever, so repeated
  next-clicks pulse the same row and the "At last user message" toast only
  fires once the last user row can sit at the viewport top. Pre-existing
  `loadedUserJumpTarget` geometry, untouched by this task.
- `refreshUserTotal` retries forever on persistent non-401 API errors
  (finally→syncButtons→refresh loop, spaced by request latency). Pre-existing.

## Pitfalls discovered
- Running tests with raw `python3 -m unittest` fakes failures: conftest.py
  installs assertContains/assertMatches aliases. Always use the pipx venv
  pytest.
- A VM-harness api stub returning `undefined` for the total-refresh URL turns
  the refreshUserTotal retry loop into an infinite microtask loop that hangs
  node (subprocess.run never returns). Stubs must always return a total.
- Synthetic pi assistant rows need a full `usage` object or pi's footer
  crashes on resume (`addUsageToTotals` reads usage.input).
- Commit ca77fed5 had clobbered the def line of
  test_messages_tail_returns_signed_live_and_history_cursors, silently folding
  its body into a neighbor test; restored in 506bd895.
- Static-source commits ship a rebuilt dist/app.bundle.js (repo convention);
  Docker verification serves the tracked bundle, not raw modules.
