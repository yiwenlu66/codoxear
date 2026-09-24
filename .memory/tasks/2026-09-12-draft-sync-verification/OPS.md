# OPS (append-only)

- 2026-09-12: Environment preflight: codoxear-iso3:latest present; sudo docker ok;
  npx esbuild 0.28.2 cached; agent-browser 0.15.1 installed. All existing docker_verify
  style scripts build from `git archive <commit>` — uncommitted feature requires a
  working-tree build context (scripts/docker_draft_verify.sh does this; bundle rebuilt
  into the temp context only, checkout untouched).
- 2026-09-12: Baseline `scripts/docker_verify.sh` (no arg → HEAD 72706108) → PASS.
  All 12 checks true (app_bootstrapped, session_cards_rendered, no_page_errors, ...).
  Container + temp root auto-removed; artifacts /tmp/codoxear-docker-verify-results.sqkJJl.
- 2026-09-12: Prediction before UI run (from code reading of app_draft_sync.js):
  deletion has no tombstone (POST empty → ts 0, row 0), so after B sends/clears,
  A keeps its draft locally and A's next reconcile (reload/re-select) RE-PUSHES the
  draft to the server (resurrection). Task expects "A does not keep a stale draft" —
  predicted FAIL-by-design; to be confirmed behaviorally and reported as finding.
- 2026-09-12 run1 FAIL (harness): CODEX_WEB_PORT="$port}" typo → server ValueError in container.
  Fixed. Image build itself succeeded from working tree + in-context esbuild bundle.
- 2026-09-12 run2 FAIL (harness, 3 bugs in my script, all fixed):
  (a) report heredocs bound artifacts as str not Path → read_json returned read_error
      dicts → crash in row_of; (b) oversize POSTs sent the file PATH as body
      (--data "$file" instead of --data @"$file") → misleading 400 "invalid json body"
      — NOT server behavior; (c) grep -q true matched "success":true always → replaced
      with python result check. Raw API observations from run2 still valid:
      initial GET {ok:true,text:"",updated_ts:0.0}; post/get/delete all 200;
      unauth 401; badtype 400; unknown session 404; sessions row draft_updated_ts
      = 1790243662.024023 after B's POST, row after delete to be re-verified.
- 2026-09-12 run3 (artifacts U4jnd2): all substantive behaviors PASS; gate false-negatives
  were my check exprs (`x or -1` when x==0.0) and a selection-eval quoting bug. Raw proof:
  ui-b-05-server-after-send.json = {ok,text:"",updated_ts:0.0}; row draft_updated_ts 0.0;
  B composer ""; cross-context + live-pull + debounced upload all true; no page errors.
  A-stale observation: A kept beta while open; A reload re-pushed draft (ts 1790244056.2);
  B re-pulled resurrected draft. Fixed gate exprs; run4 = clean confirmation run.
- 2026-09-12 run5 (artifacts Uu4JDw): PASS — api 17/17, ui 9/9, report.json pass=true.
  Evidence: cross-context appearance (B fresh profile, #msg = alpha on select), live
  pull-if-clean (B got beta while open), B's composer send cleared server draft
  ({text:"",ts:0}; row 0.0), write-through cache confirmed in both profiles
  (localStorage keys codexweb.draft.<sid>[.server_ts]). A-stale finding reproduced
  (A kept beta; A reload re-pushed at ts 1790244744.6; B re-pulled).
- 2026-09-12 final: .venv/bin/python3 -m pytest tests/ -q → 1942 passed,
  112 subtests passed, 0 failed. No leftover containers/daemons; one-off image
  codoxear-draft-verify:worktree removed; repo bundle untouched; nothing staged.

## Run 4 (re-verification after tombstone fix) — 2026-09-12
- Tombstone fix now in working tree: DraftStore.set/tombstone write {text:"", updated_ts: server wall clock}; rows carry tombstone ts; app_draft_sync.js stores send-clear tombstone ts as companion; applyServerDraftState discards pending debounced echo.
- Extended scripts/docker_draft_verify.sh: tombstone API expectations (delete ts>0, GET/row keep tombstone), oversize-blank (300KiB spaces) never 413s, never-drafted row 0.0 pre-draft check, UI regression scenario (A open + B sends + A clears via pull-if-clean + no re-POST, verified by tombstone ts stability), server-only in-container restart via exact recorded PID, A/B reload+reselect after restart must not re-push.
- Started: bash scripts/docker_draft_verify.sh in tmux pane (port 19673, password docker-draft-verify-password).
- Run 4a failed on harness bug only: neverdrafted_row_zero used `.get(x) or -1` and 0.0 is falsy (row was correct at 0.0). Fixed check (row_ts helper, None-guard).
- Run 4b (artifacts /tmp/codoxear-draft-verify-results.PhQPzD): PASS. API 21/21 (delete→tombstone ts 1790246753.1002 > beta ts; GET/row keep tombstone; 300KiB spaces → 200 tombstone ts, never 413; never-drafted row 0.0). UI 19/19:
  - Regression: B sent A's beta via real UI → server tombstone {text:"", ts 1790246773.5066657} > beta ts 1790246769.1186; B composer "" for all 11 samples/10s and server ts unchanged; A (still open) self-cleared in 12s (~2 visible 5s sessions polls) via pull-if-clean, draft key removed, companion == tombstone ts; settled GET, post-A-reload GET, post-restart GET, post-A/B-reselect GETs all byte-identical tombstone 1790246773.5066657 → no client ever re-POSTed the sent text; no page errors.
  - Persistence: server-only in-container restart (exact PID kill; broker+Pi untouched) → GET {text:"", ts 1790246773.5066657} (same, not 0); row same ts; session_drafts.json on disk holds the tombstone; A and B reload+reselect → composer "", server unchanged.
- Full suite: .venv/bin/python3 -m pytest tests/ -q → 1953 passed, 112 subtests passed in 45.40s.
- Cleanup verified: no codoxear-draft-verify container/image remains; agent-browser contexts closed; verifier tmux session ended.
