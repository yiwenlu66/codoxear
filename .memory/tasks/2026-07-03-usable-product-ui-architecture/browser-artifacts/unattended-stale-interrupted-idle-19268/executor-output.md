## Result: PASS

The unattended injection path does **not** send while `/api/sessions`/sidebar report busy, even when the raw broker socket keeps returning `{busy:false, queue_len:0, interrupted_idle:true}`. The last boundary is closed.

**Mechanism.** The unattended sweep (`UnattendedSweepCoordinator.sweep`) gates each enabled session through readiness at `session_manager_factories.py:383`, which is wired to `manager._runtime_status_from_state_and_log` — the *same* basis as direct-send/queue/attachment readiness. That method builds its `BrokerRuntimeState` via `broker_runtime_state_with_session_idle_authority`, taking the interrupted-idle override from the stored, suppression-aware `Session.interrupted_idle`, not the raw broker value. So the fix proven for send/queue extends to unattended with no separate code path.

**Why the discriminator is decisive, not coincidental.** I constructed a log where the unattended *tail* gate passes by design — `_compute_idle_from_log` returns False (busy: latest `agent_reasoning`) while `_last_chat_role_ts_from_tail(final_assistant_only=True)` returns `("assistant", task_complete_ts)` (since `agent_reasoning` is not a chat role). I verified this prediction locally *before* the live run. With the tail gate passing and cooldown not tripped (never injected), the readiness gate is the **sole** thing that can block injection. Under the defect shape the first sweep would inject and decrement `remaining_injections` 1→0 / disable; neither happened across ~5 sweeps.

**Exact evidence.**
- Raw broker stale throughout: phases A/B/D all `{"busy":false,"queue_len":0,"interrupted_idle":true}`.
- Listing busy after the post-interrupt append (poll 1) and held busy through the sweep window.
- Unattended enabled via real API: `POST .../unattended` → HTTP 200 `enabled:true,remaining_injections:1,cooldown_minutes:1`.
- Call log across the 12s sweep window: **208 → 254 calls, 100% `cmd:state`, sends=0, keys=0** (+46 `cmd:state` confirms the sweep actively polled the broker each cycle).
- `GET .../unattended` final: `enabled:true,remaining_injections:1` — not decremented, not disabled.
- Browser DOM: `stateDot busy`, `badge unattended`, attach button disabled ("Wait for the current response to finish…"); screenshot saved.
- Focused validation: **66 passed, 4 subtests passed**.

Container removed via `codoxear-docker-sandbox stop` (port 19268 free); no host live dirs touched; no `pkill`/pattern cleanup. `git status --short` shows only the new untracked artifact directory — no staged or modified tracked files.
