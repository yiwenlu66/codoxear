# OPS — nav neighbor unification

2026-08-15 ~14:10 (prior run, 30-min timeout cut) Committed 1d56655c (wrapper
order-through fix + 3 behavioral tests in tests/test_transcript_export.py) and
ca77fed5 (neighbor endpoint + tests). Left client unification + toasts
uncommitted in working tree.

2026-08-15 15:0x (this run) Recovered state. Found and fixed:
- VM harness hang: api stub returned undefined for total-refresh URL →
  infinite refreshUserTotal retry loop kept node alive; pytest hung at
  test_local_user_target_scrolls_without_window_fetch. Fixed stub to always
  return {total, matches}.
- ca77fed5 defect: def line of
  test_messages_tail_returns_signed_live_and_history_cursors deleted, body
  folded into neighbor test. Restored → commit 506bd895 (24 tests collected
  again, was 23).

Commits this run:
- 506bd895 Restore tail-cursor test clobbered by neighbor test insertion
- b5aec5e1 Unify user-message navigation on the neighbor endpoint with one
  materialization owner (app_chat_navigation.js, VM tests, AGENTS.md
  state-authority section)
- a1a435ae Toast failed navigation distinctly from genuine boundaries
- e936511b Cover cross-log neighbor+window navigation for codex, pi, and cc
  rotated logs (tests/test_message_routes.py; fixtures validated by scratch
  run: all three backends discover rotated logs and extract user events)
- b5cf22da Rebuild app bundle with neighbor-endpoint navigation (npx esbuild,
  matches repo convention: prior nav commits all shipped the bundle)

Test runs:
- Full suite twice: `~/.local/share/pipx/venvs/codoxear/bin/python -m pytest -q
  tests/` → 1660 passed, 103 subtests, ~25s (before and after bundle commit).

Docker behavioral verification (bounded, pi backend, port 19643, image
codoxear-navverify:b5cf22da2153 built from git archive HEAD on
codoxear-iso3 base; container codoxear-navverify, removed by exact name):
- Bootstrap pi session + appended 40 synthetic user/assistant turns (usage
  object required on assistant rows; first attempt without usage crashed pi
  resume in addUsageToTotals).
- Browser (agent-browser session codoxear-navverify): login OK, session card
  rendered, initial window 12 user rows (user msg 28..39).
- Prediction: prev past boundary lands on user msg 27 (pre-fix code would
  land on the first message). Observed: prev clicks walked 34→28 locally,
  then boundary click prepended (12→41 rows) and pulsed user msg 27; further
  clicks walked 26,25,... Correct.
- Boundary: at top, prev → toast "At first user message", no pulse.
- Next-after-last: direct fetch of neighbor endpoint returned
  {neighbor:null,same_log:false} (status 200). UI toast not reachable at the
  bottom due to pre-existing local-target geometry (rows that cannot scroll
  to top re-match); recorded as pre-existing, out of scope.
- Cross-log: wrote rotated pi log (same session id, older mtime, 3 archived
  turns) into container sessions dir; prev from current log's first user
  message loaded detached window of the rotated log (DOM = 3 archived user
  rows), no failure toast, zero page errors.
- Artifacts: /tmp/codoxear-navverify.YzMHOl (build log, screenshot
  navverify-crosslog.png). Container and browser session closed.

Anomalies preserved:
- tests/test_transcript_export.py assertContains "failures" under raw
  unittest are an artifact of missing conftest aliases; pytest is the runner.
- docker/iso-broker-16447.json is a stale untracked artifact from a previous
  session's docker run (predates this task); left in place.

2026-08-15 (audit follow-up run) Independent audit (opus-4-8) confirmed the
four fixes at mechanism level; two follow-ups implemented:
- 4eddff37 Neighbor endpoint serves launch-payload sessions (mirrors
  handle_messages_search's missing-session path) and returns 200
  {neighbor:null,same_log:false} when a known session has no readable log
  (was 404 with boundary-shaped body). Tests:
  test_messages_neighbor_missing_session_uses_launch_payload,
  test_messages_neighbor_without_transcript_log_is_boundary_not_404.
- 0a81b927 fetchNeighbor distinguishes {stale} (selection/poll-generation
  moved mid-flight -> silent) from {error} (genuine failure -> toast). VM
  test test_stale_session_switch_is_silent_not_a_failure_toast.
- 871c1a34 bundle rebuild; 935336ff issue #52 in
  .memory/project/ISSUES_TRACKER.md records the pre-existing next-direction
  geometry gate (out of scope; VM tests stub that geometry).
Full suite: 1663 passed + 103 subtests (was 1660; +3 new tests). No Docker
re-verification: endpoint logic handler-tested, toast race VM-tested.
