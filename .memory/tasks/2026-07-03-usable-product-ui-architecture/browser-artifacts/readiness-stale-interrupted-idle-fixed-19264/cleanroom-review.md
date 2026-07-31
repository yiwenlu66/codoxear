I have completed the clean-room review across all six audit dimensions. Let me consolidate the verdict.

The working tree is clean (I made no edits — only reads and read-only test runs). Every finding is grounded in source at HEAD `2d76dcc` and machine-checked artifacts.

## Review verdict: **PASS**

### Q1 — Mechanism (authority split correct)
`SessionReadinessCoordinator.runtime_status_from_state_and_log` (`session_readiness.py:28-56`) reads the stored `session.interrupted_idle` under lock and builds the `BrokerRuntimeState` via `broker_runtime_state_with_session_idle_authority(state, session_interrupted_idle=...)` (`session_runtime.py:424-451`). That helper calls raw `broker_runtime_state(state)` first — so `busy`/`queue_len` remain authoritative **and** a corrupt raw `interrupted_idle` still raises — then substitutes the override with the stored flag. Listing uses the same stored flag (`session_listing.py:338` `interrupted_idle=bool(s.interrupted_idle)`, consumed at `session_runtime.py:382`). Readiness and listing now share one interrupted-idle source of truth while both preserve raw broker busy/queue_len. ✔

### Q2 — User-visible state (contradiction closed)
All five readiness surfaces funnel through `runtime_status_from_state_and_log` → `session_runtime_readiness(remote_ready)` with no bypass: send (`session_send.py:47`), immediate enqueue promotion (`session_queue.py:111`), drain-sweep promotion (`session_manager_core_methods.py:153`), attachment (`session_readiness.py:146`), unattended (`unattended_sweep.py:99`). The raw-authority function `broker_allows_interrupted_idle_override` has **no production caller** (tests only). Suppression is monotonic inside a stale-true window (`set_session_interrupted_idle` keeps the stored flag False while `interrupted_idle_suppressed`), so the dangerous direction — listing busy while a send/promotion delivers — is structurally impossible once suppressed, and before suppression both surfaces agree (idle). ✔

### Q3 — Evidence (discriminates the defect, machine-checkable)
The fake brokers `stale_broker_send.py`/`stale_broker_q.py` are **byte-identical** between the defect (19260) and fixed (19264) runs; only ports/paths/labels, an added discovery-wait, and richer call-log accounting differ. Same raw precondition `{"busy":false,"queue_len":0,"interrupted_idle":true}`; only HEAD changed. I independently verified every decisive artifact:
- `phase4-send-result.json`: HTTP **409**, sends:0 — inverts defect (HTTP 200 + 1 `cmd:send`).
- `phase5`/`phaseQ5-enqueue-result.json`: HTTP 200 `queued:true, queue_len:1`, sends:0; queue GET item retained `sending:false` — inverts defect (`queued:false` + 1 send + queue emptied).
- `broker1-call-log.jsonl`: 195 lines, all `state`, 0 send/keys. `broker2-call-log.jsonl`: 375 lines, all `state`, 0 sends.
- `phase2`/`phaseQ2-polls.json`: `busy:true` every poll while raw `interrupted_idle:true`.
- Browser: statedot 2/2 busy, attach disabled with busy title, queue badge `1`.
- I reproduced the recorded `139 passed, 26 subtests`. ✔

### Q4 — Boundaries (correctly calibrated)
"Attachment affordance browser-proven" is accurate: the disabled attach button traces to the `currentRunning` (busy) branch at `app.js:6560-6562`, confirmed disabled in DOM — the affordance reflects the corrected busy projection. The server-side attachment-injection gate and unattended injection inherit the fix by the shared `runtime_status_from_state_and_log`, and memory states plainly they were **not separately live-proven**. No overclaim. ✔

### Q5 — Regression (immediate-interrupt preserved)
With stored `interrupted_idle=True` unsuppressed, `resolve_runtime_status` keeps the override active → `busy=False, remote_ready=True`. `test_valid_interrupted_idle_still_ready_for_immediate_interrupted_tail` plus the 3 other new tests pass (4/4). The amended snapshot tests now set the stored flag to mirror what `get_state` records — a correct adaptation to the moved authority, not a weakening. ✔

### Q6 — Category check (no missed contradiction)
Single source of truth (stored flag) is consumed identically by listing and all readiness surfaces. Considered and dismissed as out-of-scope non-blockers: watcher-liveness (both surfaces stay consistent), stored-vs-raw busy polling lag (readiness is always at-least-as-fresh via its own `get_state`), and a sub-poll TOCTOU during the exact suppression transition (ordinary snapshot lag shared with listing, self-correcting — not the deterministic every-poll defect). ✔

### Residual boundaries (not blockers)
1. Server-side attachment-injection and unattended-injection gates inherit the fix by shared function but were not separately API/live-proven (memory calibrates this correctly).
2. A sub-poll TOCTOU during the exact suppression transition is ordinary snapshot lag, self-correcting; the deterministic divergence is closed.
3. `_runtime_broker_state` import at `server.py:199` is now unused (pre-existing dead import, harmless, out of scope).

`git status --short`: *(empty — clean working tree)*
