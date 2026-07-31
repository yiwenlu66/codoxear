BLOCKER.

`git status --short`: clean / no output.

## Blocker: stale interrupted-idle can still project false idle

### Mechanism
`8e5bae8` fixed the stale `interrupted_idle=true` path when the stale broker state is applied through `set_session_interrupted_idle(...)`.

But discovery still bypasses that fixed path:

- `codoxear/session_discovery_registry.py:118-120` directly does:
  - `previous.interrupted_idle = True`
  - `previous.interrupted_idle_log_off = registration.meta_log_off`
- That re-baselines the interrupt offset to the current log size.
- If resumed log activity arrives and the first following `/api/sessions` poll runs discovery before `update_meta_counters()`, the watcher skips the resumed user row as “pre-baseline” and never suppresses the stale override.
- Public `/api/sessions` then reports `busy=false` while the log contains an open resumed turn.

The committed fixed evidence covers the timing where the first post-resume poll skips discovery and prune/update clears the stale override. It does not cover the discovery-first timing.

### User-visible failure
The sidebar/listing can show the session idle/gray and sendable while the transcript log is non-idle after a resumed same-log turn. That contradicts the binary busy/idle contract.

### Reproduction I ran
I reran the existing Docker stale-idle harness against current HEAD, then inserted one timing discriminator: wait past `CODEX_WEB_DISCOVER_MIN_INTERVAL_SECONDS=0.2` after appending the resumed `user_message`, before the first phase-2 poll.

Result:

```text
phase1_busy: [false, false, false]
phase2_busy: [false, false, false, false, false]   <-- blocker
phase3_busy: [false, false, false, false, false]
verdict=FAIL
```

Artifacts from my repro: `/tmp/codoxear-review-race-vBkk4q/artifacts`.

### Minimal fix target
Patch `SessionDiscoveryRegistryCoordinator.upsert_registration()` to use the same suppression/baseline semantics as `set_session_interrupted_idle(...)`, rather than direct assignment on refresh. Add a regression where discovery runs before `update_meta_counters()` after post-interrupt log activity.

CC transcript outcomes and mobile dpad evidence held under review; I did not find a blocker there.