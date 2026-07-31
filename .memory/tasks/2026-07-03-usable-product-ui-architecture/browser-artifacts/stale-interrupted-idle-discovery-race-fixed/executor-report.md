All work is complete. Here is the acceptance report.

---

## Summary

**PASS.** The stale `interrupted_idle` discovery-refresh race is fixed.

**Root cause:** `SessionDiscoveryRegistryCoordinator.upsert_registration()` directly assigned `previous.interrupted_idle_log_off = registration.meta_log_off` for already-tracked sessions. Since discovery captures `meta_log_off` as the *current* log size, this re-baselined the interrupt override forward past any post-interrupt resumed activity. When discovery ran before `update_meta_counters()`, the log watcher advanced its read cursor to that baseline and never saw the resumed user row, so it never cleared the stale override — `/api/sessions` reported `busy=false` against a non-idle log.

**Fix (1 file, `codoxear/session_discovery_registry.py`):** Replaced the direct assignment block in the existing-session refresh branch with `set_session_interrupted_idle(previous, registration.interrupted_idle)` — the same helper the broker and prune paths already use. This helper (1) preserves an existing baseline when `interrupted_idle` is already true (does not move it forward), (2) records a fresh baseline (`log_path.stat().st_size`) only for a genuinely new interrupt, (3) respects stale-true suppression, and (4) clears suppression when the broker reports false.

**Point 2 (new-session construction):** Left unchanged. For new sessions, `reset_log_caches` is always called immediately after construction and unconditionally sets `interrupted_idle=False`/`interrupted_idle_log_off=0`/`interrupted_idle_suppressed=False`. The constructor's `interrupted_idle`/`interrupted_idle_log_off` values are therefore dead code with no staleness path; the baseline is established fresh on the next prune/refresh poll. No second authority introduced.

**Regression:** `test_discovery_refresh_before_counters_does_not_rebaseline_past_resumed_activity` — builds a `DiscoveryRegistration` with stale `interrupted_idle=True` and `meta_log_off`=post-resume log size, calls `upsert_registration` (discovery refresh) *before* `update_meta_counters`, then verifies the watcher clears/suppresses and the listing projects `busy=True`. Failed on HEAD (baseline moved 312→404); passes with fix. Two companion tests cover broker-false clears suppression and suppressed-stale-true stays cleared through discovery refresh.

**Docker harness with critic timing discriminator:** Ran the existing `run_cert.py` adapted with the critic's discriminator (`time.sleep(0.3)` past `DISCOVER_MIN_INTERVAL_SECONDS=0.2` after appending resumed `user_message`, before first phase-2 poll). HEAD: `verdict=FAIL`, `phase2_busy=[false×5]`. Fix: `verdict=PASS`, `phase2_busy=[true×5]`. Container cleaned up; port 13790 used (not 8743).