# Stale interrupted-idle discovery-refresh race — fixed evidence

Verdict: PASS on functional commit `f5b4710`.

Mechanism fixed:
- Discovery refresh for an already-tracked session now routes `registration.interrupted_idle` through `set_session_interrupted_idle(previous, ...)`, the same helper used by broker/prune refresh.
- That helper preserves the interrupt-time log baseline across repeated stale true reports, respects suppression after post-interrupt activity, and clears suppression when the broker reports false.
- The removed direct assignment had re-baselined `interrupted_idle_log_off` to discovery's current `meta_log_off`, hiding resumed same-log user activity from the watcher when discovery ran before log-counter update.

Docker/API discriminator:
- Harness: real `codoxear.server` inside Docker, fake broker socket always reporting `busy=false`, `queue_len=0`, `interrupted_idle=true`, and a real Codex-shaped JSONL log.
- Timing discriminator: after appending the resumed same-log `user_message`, the harness sleeps 0.35s, beyond `CODEX_WEB_DISCOVER_MIN_INTERVAL_SECONDS=0.2`, before the first phase-2 `/api/sessions` poll. This forces the discovery-refresh-before-counter timing that failed on `711dd5f`.
- Result: `VERDICT.txt` and `api-snapshots.json` show phase 1 busy `[false,false,false]`, phase 2 busy `[true,true,true,true,true]`, and phase 3 busy `[false,false,false,false,false]`.

Artifacts:
- `run_cert_discovery_wait.py`: harness with the discovery-interval wait.
- `docker-output.json`: printed verdict and busy arrays.
- `api-snapshots.json`: public `/api/sessions` evidence.
- `VERDICT.txt`: concise verdict.
- `rollout-broker-1.final.jsonl` and `broker-1.sidecar.json`: synthetic log and sidecar.
- `executor-report.md`: implementation/regression report.

Boundary:
- `in-process-diagnostic.json` is retained from the legacy harness but still contains an artificial manual re-baseline condition. The decisive evidence for this race is the public API phase-2 busy array under forced discovery-first timing plus the unit regression in `tests/test_stale_interrupted_idle.py`.
