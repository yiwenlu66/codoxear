# Stale interrupted-idle discovery-refresh race — failing evidence

Verdict: FAIL on HEAD `711dd5f`.

Mechanism:
- The earlier stale-interrupted-idle fix routed broker/prune refresh through `set_session_interrupted_idle()`, which preserves the original interrupt baseline and suppresses stale true after post-interrupt log activity.
- Discovery refresh still directly assigned `previous.interrupted_idle = True` and `previous.interrupted_idle_log_off = registration.meta_log_off` in `SessionDiscoveryRegistryCoordinator.upsert_registration()`.
- If a resumed same-log user row is appended and the first `/api/sessions` poll runs discovery before log-counter update, the direct assignment re-baselines the interrupt offset to the current log size. The watcher then skips the resumed user row as pre-baseline and public listing reports idle while the log is non-idle.

Observed failure:
- Critic inserted a timing discriminator: wait beyond `CODEX_WEB_DISCOVER_MIN_INTERVAL_SECONDS=0.2` after appending the resumed `user_message`, before the first phase-2 poll.
- `api-snapshots.json` shows phase 1 interrupted idle stays `busy=false`; phase 2 resumed turn remains `busy=false` across repeated polls; phase 3 completion remains `busy=false`.
- `VERDICT.txt` records `verdict=FAIL` with `phase2_busy: [false, false, false, false, false]`.

Artifacts:
- `api-snapshots.json`: public API snapshots across phases.
- `broker-1.sidecar.json`: stale sidecar/control metadata.
- `rollout-broker-1.final.jsonl`: final synthetic log.
- `in-process-diagnostic.json`: diagnostic state.
- `critic-report.md`: clean-room blocker report.

Implication:
The stale-interrupted-idle invariant must be owned by one helper path. Discovery must not set the flag/baseline directly.
