# Verification Report — log-only stale `interrupted_idle` projection

**Target commit:** `f278474` (Codoxear `recovery/product-gaps`).
**Verdict: FAIL** — the stale `interrupted_idle` override is **not** cleared in the
real end-to-end listing path. The product projects `busy=false` for a session
whose log is non-idle and whose same-log turn has resumed, as long as the broker
control socket keeps reporting `interrupted_idle=true`.

The unit tests (`tests/test_stale_interrupted_idle.py`) pass because they mock
`prune_dead_sessions` to a no-op. The real `/api/sessions` path does not mock
it, and `prune` re-baselines the override *before* the log watcher runs,
defeating the guard.

## Scope and isolation

- Ran in Docker image `codoxear-stale-cert:latest` (built from
  `docker/sandbox.Dockerfile`). The repo was mounted **read-only** at
  `/workspace`; no product file was edited, staged, or committed by this run.
- App dir / fake socket / rollout log lived only inside the container
  (`/home/tester/.local/share/codoxear`) and the mounted artifacts volume.
- Port `13790` (not `8743`). Server `stderr`/`stdout` logs: 0 lines, no
  discovery/coercion warnings — the session was discovered and projected cleanly.
- The interrupted_idle-relevant modules
  (`session_runtime.py`, `session_prune.py`, `session_log_runtime.py`,
  `session_list.py`, `session_control.py`, `session_discovery*.py`,
  `rollout_idle.py`, `session_listing.py`) are byte-identical across
  `f278474`, HEAD, and the working tree, so the result applies to `f278474`.

## Harness

`run_cert.py` (in this directory) drives the **real** `codoxear.server`:

- A fake Unix control socket `socks/broker-1.sock` implements the broker control
  protocol (`{"cmd":"state"}` → `{"busy":false,"queue_len":0,"interrupted_idle":true,"token":null}`,
  `{"cmd":"tail"}` → `{"tail":""}`). The state response is **always** the stale
  interrupted-idle state — the faithful "broker keeps reporting
  interrupted_idle:true" scenario.
- A real Codex-style JSONL rollout log (`session_meta`, `event_msg`, `response_item`
  rows — same shapes as the unit test) is appended across phases.
- Sidecar `broker-1.json` advertises `agent_backend=codex`, real `log_path`,
  `owner=terminal`, nonexistent broker/codex PIDs (the live log keeps the
  session discoverable; the socket stays responsive so `prune` never drops it).
- HTTP client authenticates via `POST /api/login`, then polls `GET /api/sessions`.

## End-to-end observations (`api-snapshots.json`)

`interrupted_idle` is **stripped** from the public `/api/sessions` row
(verified: the key is absent from every `session_row` snapshot — it is in
`_PRIVATE_LISTING_KEYS`). The user-visible signal is therefore `busy`. The fake
socket's `state` handler was invoked 17 / 38 / 62 times across the three phases,
proving `prune_dead_sessions` actively re-read the socket on every poll.

| Phase | Log content (tail) | `_compute_idle_from_log` | `/api/sessions` `busy` | Expected |
|-------|--------------------|--------------------------|------------------------|----------|
| 1 — interrupted turn (non-final assistant `response_item`) | non-idle | `False` | `false` ×3 | `false` ✓ (override masks; correct immediate-interrupt projection) |
| 2 — append `user_message` starting a new turn on the **same** log | non-idle (open user turn) | `False` | **`false` ×5** | `true` if override cleared → **FAIL** |
| 3 — append `task_complete` | idle | `True` | `false` ×5 | `false` ✓ |

Log sizes: phase 1 = 312 B → phase 2 = 404 B → phase 3 = 476 B.

**Phase 2 is the discriminator.** The log is non-idle and a new turn has
resumed, yet `/api/sessions` projects `busy=false`. The stale override was not
cleared.

## Mechanism (why the guard does not fire in the real path)

`SessionListCoordinator.list_sessions()` (`codoxear/session_list.py`) runs, on
**every** `/api/sessions` call, in this order:

1. `discover_existing_if_stale()` (TTL-gated).
2. `prune_dead_sessions()` → `SessionPruneCoordinator.refresh_session_state()`
   (`codoxear/session_prune.py`) → reads `{"cmd":"state"}` from the socket and
   calls `set_session_interrupted_idle(session, interrupted_idle_from_socket)`.
3. `update_meta_counters()` — the log watcher
   (`SessionLogRuntimeCoordinator.update_meta_counters`,
   `codoxear/session_log_runtime.py`) that is supposed to clear the stale
   override when it sees post-baseline activity.

`set_session_interrupted_idle(session, True)` (`codoxear/session_runtime.py`)
unconditionally re-records `interrupted_idle_log_off = log_path.stat().st_size`
— the **current** log size. Because step 2 runs after the post-interrupt
activity has already been appended, the baseline is moved up to **include** that
activity. Then step 3's guard

```python
if interrupted_idle_active and 0 < interrupted_idle_baseline <= size:
    if offset < interrupted_idle_baseline:
        offset = interrupted_idle_baseline   # advance cursor past pre-baseline bytes
    post_baseline = True
```

advances the read cursor to the (now full) baseline, so the `while offset < size`
loop sees no post-baseline content, `clear_interrupted_idle` stays `False`, and
`session.interrupted_idle` is left `True`.

Downstream, `build_runtime_enriched_session_rows` computes
`broker.allows_interrupted_idle_override` `(= not busy and queue_len==0 and
interrupted_idle)` as `True`, and `resolve_runtime_status` returns
`busy = not (log_idle or override) = not (False or True) = False`.

### In-process confirmation (`in-process-diagnostic.json`)

Two conditions were reproduced against copies of the same log using the **real**
coordinators:

- **Condition A — `update_meta_counters` alone** (baseline captured *before*
  the append, no `prune` re-baseline, i.e. the unit-test condition):
  `interrupted_idle_after = False`. The guard fires and clears the override.
  This is exactly why `tests/test_stale_interrupted_idle.py` passes.
- **Condition B — real listing order** (`prune` re-baselines to the *current*
  log size, *then* `update_meta_counters`): `interrupted_idle_after = True`,
  `meta_log_off_after = interrupted_idle_log_off = 568` (full size). The watcher
  advanced past all content and never saw post-baseline activity. Override
  survives.

> Note: `compute_idle_pre_append.idle = true` in the diagnostic is a
> snapshot-timing artifact — that field read `LOG` after phase 3 had already
> appended `task_complete`. A standalone check of `_compute_idle_from_log` on the
> pure phase-1 log returns `False` (non-idle), confirming the parser is correct;
> the bug is in the override-clearing path, not the idle parser.

## What would fix it (not applied — no code edits per task)

The override-clearing authority is split. Either the stale boolean must not be
re-baselined by `prune` on every poll while the log watcher is responsible for
clearing it, or `update_meta_counters` must clear based on the watcher's own
monotonic cursor rather than the socket-supplied baseline. Any fix belongs to a
separate change with its own tests; this run made none.

## Boundary

- Browser sidebar proof was not produced. The sidebar dot is a pure projection
  of the `busy` boolean from `/api/sessions`, so an API-level `busy=false` in
  phase 2 entails a gray sidebar dot; a browser run would not change the
  verdict, only re-render the same false projection.
- No real backend CLI was involved (by design — the cert targets the
  listing/projection path, which is backend-agnostic). The fake socket
  faithfully returns the stale state the task specifies.

## Artifacts (`/tmp/codoxear-stale-cert/`)

- `run_cert.py` — harness source.
- `broker-1.sidecar.json` — discovery sidecar.
- `rollout-broker-1.final.jsonl` — full rollout log after all phases.
- `api-snapshots.json` — `/api/sessions` row + busy samples per phase.
- `in-process-diagnostic.json` — condition A vs B reproduction.
- `server.stderr.log` / `server.stdout.log` — clean (0 lines).
- `VERDICT.txt` — `FAIL`.
