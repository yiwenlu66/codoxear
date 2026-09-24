# EPISTEMIC — current model of the draft-sync feature under verification

## Model: tombstoned deletion (run4 verified, Dockerized working tree, artifacts /tmp/codoxear-draft-verify-results.PhQPzD)

Draft deletion is a first-class write: POST "" (or blank) stores
`{text:"", updated_ts:<server wall clock>}`; GET, sessions rows
(`draft_updated_ts`), and `session_drafts.json` all carry the tombstone ts.
Only a never-drafted session reports 0/0.0. Client companion ts
(`codexweb.draft.<sid>.server_ts`) is compared against the tombstone ts in the
same last-writer-wins channel as edits, so a clear outranks any earlier edit
and never resurrects.

## Verified behaviors (all behavioral, run4b)
- API: never-drafted GET {ok,text:"",ts:0} and row 0.0; POST → ts>0; second
  client overwrites; row tracks latest ts; delete → tombstone ts (1790246753.10
  > beta ts); GET/row keep tombstone after delete; whitespace deletes as
  tombstone; 300KiB-of-spaces blank → 200 tombstone (blank clears bypass the
  256KiB cap — never 413); 256KiB+1 → 413; exact 256KiB → 200; non-string →
  400; unknown session → 404; unauth → 401.
- Regression case (the run3 failure, now fixed): A left open + B sends A's
  draft through the real UI → server tombstone ts 1790246773.5066657; B's
  composer stays "" (11 samples/10s) and never re-pushes (ts unchanged); A
  self-clears via pull-if-clean 12s after the tombstone lands (~2 visible 5s
  sessions polls), draft key removed, companion == tombstone ts; the tombstone
  ts never advances afterwards (settled, post-A-reload, post-restart,
  post-A/B-reselect GETs all byte-identical) → no client re-POSTs the sent
  text. No page errors in either context.
- Persistence: server-only in-container restart (exact-PID kill, broker+Pi
  alive) reloads the tombstone from session_drafts.json: GET returns the same
  ts (not 0), row same ts, and A/B reload+re-select do not re-push.
- Full suite: 1953 passed, 112 subtests (host .venv).

## Ruled out / resolved
- Run3's "A reload re-pushes deleted draft" design failure: resolved by the
  tombstone model (delete returns ts>0, clients store it as companion; empty
  server text is applied, ts-0 is never a deletion signal).
- Run4a's only failure was a harness falsy-zero bug (`0.0 or -1`), not an
  implementation bug.

## Harness state
- scripts/docker_draft_verify.sh now covers tombstone API contract, oversize
  blank, never-drafted row, the full regression scenario (ts-stability proof
  of no re-POST), and a server-only restart phase (restartable background
  server with recorded PID; container stays alive via `sleep infinity`).
- Sessions poll is 5s visible → "a poll cycle or two" ≈ 5–15s; observed 12s.
