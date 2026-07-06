# Interruption-outcome fix — verification report (FIXED)

Verdict: **PASS** — proven end-to-end through the real Docker server API and a
real browser against deterministic synthetic logs at HEAD `365164b`.

The product invariant — *every sent turn must persistently render one of
answer / error / no-answer / interruption* — now holds for interrupted turns.
At the prior HEAD (`55896d5`, the defect) every interrupted turn rendered only
the user row (indistinguishable from an ignored prompt) and discarded Pi
partial text. The committed fix (`365164b`) makes every interrupted turn
persistently render an assistant error-class row whose canonical text is
`"The backend turn was interrupted before completion."` (with the Pi partial
output appended under a clear label). The word `interrupted` makes the row
searchable and keeps it distinct from the generic no-response completion text.

This is a read-only certification: no source or tests were modified, nothing
was staged or committed. All artifacts live under this directory.

## HEAD / scope

- Repo: `/home/yiwen/codex-web-product-recovery`, branch `recovery/product-gaps`.
- HEAD: `365164b85587a9c4fdc9cc88527c3eee1b48fd03` ("Render interrupted turn outcomes").
- `git status --short`: only the untracked artifact directory
  `…/interruption-outcome-fixed-19272/`. **No source/tests modified, nothing
  staged** (`git diff --cached --name-only` empty).

## Mechanism under test (the fix)

- `codoxear/rollout_events.py::_build_interrupted_event` — new helper producing
  `{role:"assistant", message_class:"error", text:_INTERRUPTED_TEXT, …}` where
  `_INTERRUPTED_TEXT = "The backend turn was interrupted before completion."`.
  When `partial_text` is present it appends
  `"\n\nPartial output before interruption:\n<partial>"`.
- `codoxear/agent_backend.py` (Pi, `~621`): an aborted assistant turn previously
  returned `None` (suppression root cause); now returns
  `_build_interrupted_event(row, partial_text=pi_assistant_text(row))`.
- `codoxear/agent_backend.py` (Codex, `~323`): `event_msg` `turn_aborted` had no
  branch (projected nothing); now returns `_build_interrupted_event(row)`.

Because all transcript surfaces (tail, history, search, export, browser render)
funnel through the same per-backend `chat_event_from_log_row`, the fix is
visible everywhere the defect was.

## Environment / isolation

- Docker image `codoxear-sandbox:latest` (built from `docker/sandbox.Dockerfile`).
- Container `codoxear-sandbox-19272`, port `127.0.0.1:19272->19272/tcp`.
- Throwaway HOME `/tmp/codoxear-docker-sandbox-19272/home` (bind-mounted into
  the container as `/home/tester`). The sandbox preflight isolation guard passed
  (`scripts/codoxear-docker-sandbox preflight`); no host live runtime
  (`~/.local/share/codoxear`) was touched. Synthetic logs/sidecars/sockets only;
  no backend credentials or live CLI sessions.

## Synthetic sessions (created inside the container)

`fake_interrupt_sessions.py` writes, under the container
`/home/tester/.local/share/codoxear`, three synthetic sessions. Each gets a
deterministic `.jsonl` log, a `.json` sidecar (broker/codex PID = the helper's
own PID, which stays alive), and a live broker control `.sock` that answers
`{"cmd":"state"}` (required by `session_discovery.discover_sessions`). The log
rows are identical in shape to the committed defect proof's inputs, which are
known to trigger the abort path:

| session | backend | log rows |
|---|---|---|
| `interrupt-pi-empty` | pi | `message` user; `message` assistant `stopReason:"aborted"`, `content:[]` |
| `interrupt-pi-partial` | pi | `message` user; `message` assistant `stopReason:"aborted"`, `content:[{text:"I was halfway through the answer"}]` |
| `interrupt-codex-abort` | codex | `event_msg` `user_message`; `event_msg` `turn_aborted` |

## Exact commands

```bash
# 0. focused unit tests (host)
python3 -m pytest -q tests/test_codex_no_response_projection.py \
  tests/test_transcript_export.py tests/test_server_chat_flags.py \
  tests/test_idle_heuristics.py tests/test_message_routes.py
#   -> 124 passed in 1.79s

# 1. build + start the real server in Docker on port 19272
CODOXEAR_DOCKER_PORT=19272 scripts/codoxear-docker-sandbox build
CODOXEAR_DOCKER_PORT=19272 scripts/codoxear-docker-sandbox start
#   container=codoxear-sandbox-19272  url=http://127.0.0.1:19272/

# 2. seed the three synthetic sessions inside the container (helper stays alive)
docker cp …/interruption-outcome-fixed-19272/fake_interrupt_sessions.py \
  codoxear-sandbox-19272:/tmp/fake_interrupt_sessions.py
docker exec -u tester codoxear-sandbox-19272 sh -lc \
  'cd /workspace && nohup python3 /tmp/fake_interrupt_sessions.py >/tmp/fake.log 2>&1 &'

# 3. real-server API proof (login, sessions, tail, search, history)
python3 …/interruption-outcome-fixed-19272/api_probe.py \
  http://127.0.0.1:19272 …/interruption-outcome-fixed-19272/api test-password

# 4. browser proof (agent-browser, ephemeral headless)
AGENT_BROWSER_SESSION=interrupt19272 agent-browser open http://127.0.0.1:19272/
#   fill password -> Login -> open #session=interrupt-{pi-empty,pi-partial,codex-abort}
#   eval DOM (.msg-row) + screenshots + reload

# 5. fresh-server persistence proof (second python -m codoxear.server, port 19273)
docker exec -u tester -d codoxear-sandbox-19272 sh -lc \
  'CODEX_WEB_PASSWORD=test-password CODEX_WEB_HOST=127.0.0.1 CODEX_WEB_PORT=19273 \
   PYTHONPATH=/workspace python3 -m codoxear.server'
#   query 127.0.0.1:19273 from inside the container; then kill <exact pid>

# 6. container-scoped cleanup
CODOXEAR_DOCKER_PORT=19272 scripts/codoxear-docker-sandbox stop
```

## Observations — API (real server, port 19272)

Raw JSON for every call is under `api/`. `api/SUMMARY.json` is the rollup.
`/api/me` returned `401` pre-login (auth gate intact). After login all three
sessions were discovered (`api/02-sessions.json`). Per scenario:

| scenario | tail roles | user prompt in tail | assistant interruption row | row class | search `interrupted` | partial preserved | search `halfway through` |
|---|---|---|---|---|---|---|---|
| `interrupt-pi-empty` | `["user","assistant"]` | yes | yes | `error` | `match_count=1` | n/a | n/a |
| `interrupt-pi-partial` | `["user","assistant"]` | yes | yes | `error` | `match_count=1` | yes | `match_count=1` |
| `interrupt-codex-abort` | `["user","assistant"]` | yes | yes | `error` | `match_count=1` | n/a | n/a |

Each assistant row text matches the canonical constant; the Pi-partial row text
is exactly
`"The backend turn was interrupted before completion.\n\nPartial output before
interruption:\nI was halfway through the answer"`. Every matched search row also
returned a usable `load_cursor`, and loading it via `/messages/history` (see
`api/history-*.json`) re-materialised the same row in every scenario — proving
the row is durable through the history cursor path, not just the tail.

Compare to the defect (HEAD `55896d5`, `interruption-outcome-defect/`): every
scenario there had tail roles `["user"]` only and every `interrupted` search
returned `match_count=0`. The fix flips both.

## Observations — browser (real Chrome via agent-browser)

DOM evidence in `browser/browser-dom-summary.json`; screenshots
`browser/{pi-empty-abort,pi-partial-abort,codex-abort}.png`. After login the UI
listed all three sessions and, for each selected session, rendered exactly two
`.msg-row`s: a `.msg user` row with the prompt and a `.msg assistant error` row
with the interruption text. For `interrupt-pi-partial` the assistant row's
visible text included the preserved partial output:

```
The backend turn was interrupted before completion.

Partial output before interruption:
I was halfway through the answer
```

The `error` class is applied to the assistant bubble (`msg assistant error`),
so the row carries error styling, not user-only silence.

## Persistence / rehydration (the volatile-state boundary)

The fix is provably not volatile. Two independent proofs:

1. **Browser reload.** A full navigation away and back to
   `#session=interrupt-pi-partial` re-rendered `roles: ["user","assistant"]`
   with `partialPreservedAfterReload: true` (`browser-dom-summary.json` →
   `reload_persistence`). The browser re-fetches `/messages/tail`, which the
   server re-reads from the log on every call (no message-event cache).

2. **Fresh server process (zero shared memory).** A *second*
   `python3 -m codoxear.server` (PID 3713) was started inside the same container
   on port `19273`, sharing only the persistent HOME (logs + sidecars + the live
   helper sockets). It has no shared address space with the first server (PID 1).
   It discovered all three sessions from disk (`api/fresh-sessions.json`) and
   projected the identical interruption rows: `interrupt-pi-partial` tail =
   `["user","assistant"]`, class `error`, partial text preserved; both
   `interrupted` searches returned `match_count=1`
   (`api/fresh-tail-*.json`, `api/fresh-search-*.json`). The second server was
   then killed by exact PID (`kill 3713`); port 19273 confirmed closed; the
   original 19272 server remained healthy. This proves the rows rehydrate from
   log/sidecar, not from any in-process or browser state.

## Conclusion

PASS. Across the real Docker server API and the real browser, all three
required scenarios now render a persistent, error-classed assistant
interruption row (searchable via `interrupted`), and the Pi partial-abort
preserves and surfaces the streamed partial text. A fresh server process with
no shared memory rehydrates the same rows from the persistent logs/sidecars.
The committed tests for this change pass (124/124 focused). The defect's
user-only silence is gone.

## Boundaries

- **Proof medium is deterministic synthetic logs**, not a live CLI Stop-click.
  This is an epistemic strength, not a gap: the suppression the defect
  certified was entirely in the backend normalizer
  (`chat_event_from_log_row` → `None`/no-branch), and every transcript surface
  — API and browser — consumes that normalizer. The synthetic rows are the same
  shape real backends emit for an abort (`stopReason:"aborted"` for Pi;
  `event_msg` `turn_aborted` for Codex), so they exercise the fixed code path.
- **Codex "missing session metadata" log warnings** (`server-log-excerpt.txt`)
  are benign: codex discovery attempts to read a `session_meta` row from the
  synthetic log for the thread id, fails, logs a warning, and falls back to the
  sidecar `session_id` — exactly the graceful-degradation path production uses.
  Discovery, tail, and search are unaffected (all green above).
- **No source/tests changed; nothing staged.** Only the untracked artifact
  directory under `.memory/tasks/…/interruption-outcome-fixed-19272/` was added.
- The defect artifact directory (`interruption-outcome-defect/`) was left
  untouched; a fresh script (`api_probe.py`, `fake_interrupt_sessions.py`) was
  written here rather than reusing the failing proof in place, per the
  no-overwrite constraint.

## Artifacts

- `fake_interrupt_sessions.py` — seeds the 3 synthetic sessions inside Docker.
- `api_probe.py` — real-server API driver (login, sessions, tail, search, history).
- `api/` — raw JSON for every API call + `SUMMARY.json` rollup; `api/fresh-*` =
  the fresh-server-process persistence evidence.
- `browser/browser-dom-summary.json` + `browser/*.png` — DOM and screenshots
  per session, incl. reload-persistence.
- `container-state.txt`, `server-log-excerpt.txt` — container/socket/log layout
  and server discovery log.
