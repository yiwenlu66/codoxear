# Post-log-bound backend death recovery — architecture audit

**Verdict: BLOCKERS** (design abstraction is correct; the in-progress implementation must not be accepted as-is)

The planned abstraction is the right one. The working-tree realization has two blocker-grade defects and several material concerns. I reviewed the design against the code now in the working tree (`post_log_recovery.py` + diffs to `broker.py`, `launch_ledger.py`, `message_routes.py`, `session_control.py`, `session_discovery.py`, `session_manager_factories.py`, `session_prune.py`). No files were edited/staged/committed.

---

## Answers to the audit questions

### Q1 — Is ledger-backed recovery the right abstraction? **PASS**

Yes. The two alternatives fail on mechanism:

- **Preserve dead sessions in the active map**: the session map is the live control plane — every entry is keyed to a Unix socket used for `state`/`tail`/`send`. A dead entry has no socket, so it would need a synthetic "dead" pseudo-state, which contradicts the codified binary busy/idle invariant (EPISTEMIC: "busy/idle is binary… duplicated authority is the risk"). It is also memory-only, so it evaporates on server restart — exactly the failure the proof (`f35bae5`) demonstrates.
- **Write synthetic events into backend logs**: the backend owns its log as source of truth; mutation corrupts provenance and is explicitly forbidden (AGENTS.md; EPISTEMIC "do not mutate backend logs").
- **Ledger record pointing at the orphaned log, lifecycle error appended at render**: durable (survives restart), append-only, and reuses the existing launch-attempt ledger that already carries pre-log failures. Correct. `launch_attempt_transcript_payload` synthesizing the error row while reading the real log through `_extract_positioned_chat_events` (no mutation) is the right seam.

### Q2 — Is the boundary condition correct? **PASS, with edges**

The condition — web-owned + bound log exists + both pids dead / socket definitively stale + `_compute_idle_from_log(log) is False` — is correct and, importantly, reuses the single busy/idle authority rather than inventing a new signal. `log_needs_post_log_bound_recovery` (`post_log_recovery.py`) implements it.

Edges that matter:
- **False negative (indeterminate idle):** `idle is None` returns `False` → a genuinely incomplete-but-ambiguous log dies silently. Only `idle is False` triggers recovery.
- **False negative (non-refused socket error):** discovery's post-log path (`session_discovery.py:~295`) requires `sock_error_definitely_stale`. A dead session whose stale socket yields a non-ECONNREFUSED error defers recording until the sidecar is cleaned.
- **Completed-then-dead is out of scope by design (correct), but confirm the product intent:** a web session that *completes* its turn (idle `True`) then dies produces no recovery row, and its completed transcript is not re-openable after reload. This is the intended non-idle boundary, but it sits under "every recorded prompt must produce a durable browser-visible result." Confirm this boundary is acceptable rather than silently narrowing the promise.
- **No false positive from Pi `length`:** a `length` continuation is non-idle by the closed length-semantics rule, so a death mid-continuation correctly triggers recovery.

### Q3 — What identity should the recovery row use? **CONCERN (near-blocker): the paths disagree**

Correct identity = the **routing id used while the session was alive = the sock stem `broker-<pid>`** (the sessions-map key and the URL id the frontend holds), with **thread_id = backend rollout id** preserved separately.

Three of four paths do this: `session_discovery.py` (`session_id = sock.stem`, `thread_id = meta["session_id"]`), `session_prune.py` (`session_id = sid` map key, `thread_id = session.thread_id`), `session_control.py` (`session_id` = URL id, `thread_id = session.thread_id`).

The **broker-exit path is the outlier and it is the common death case.** `broker.py:982-983` records `session_id=st2.session_id, thread_id=st2.session_id`. For web-owned sessions `st.session_id` is unconditionally overwritten to the rollout-derived id at log bind (`broker.py:451`), while the routing id stays `broker-<pid>` (`broker.py:871`). So a normal broker-exit death produces a recovery row keyed by the **rollout id**, diverging from the other three paths and from the id the user was viewing. The row is still self-consistently reachable (route lookup uses `launch_attempt_route_id`), so the durable-result invariant holds, but:
- The same session gets a **different identity depending on whether the broker ran its exit guard vs. was SIGKILLed** (discovery path).
- It **conflates session_id with thread_id**, which the design lists as distinct preserved fields.

Fix: broker path should pass `session_id=st2.sock_path.stem` and `thread_id=st2.session_id`.

### Q4 — Which surfaces must be synchronized? **BLOCKER — real authority split**

Required set: sidebar/listing, tail, search, history, live, export, send/queue/attach/unattended, delete/dismiss, restart rediscovery. Current state:

| Surface | State | Mechanism |
|---|---|---|
| sidebar/listing | ✅ | `launch_attempt_row` → `build_launch_attempt_rows` (`session_list.py`), deduped vs active by launch_id/spawn_nonce |
| tail | ✅ | `handle_messages_tail` falls back to `launch_attempt_transcript_for_session_id` (`message_routes.py:227-233`) |
| **search** | ❌ | `handle_messages_search` returns 404 when session gone; **`_search_launch_payload_events` (`message_routes.py:79`) is dead code — defined, never called** |
| **history** | ❌ | `handle_messages_history` returns 404 (masked because payload sets `has_older:false`, but the route is still inconsistent) |
| **live** | ❌ | `handle_messages_live` returns 404; a watched bound session that dies hits this 404 → `clearSelectedSessionAfterRemoval` (`app.js:3773`) **deselects** instead of transitioning to the recovery transcript |
| **export** | ❌ | `handle_messages_export` returns 404; the **Copy conversation button is enabled whenever a session is selected** (`app.js:2018/2031`) → user gets a "copy failed" toast over a visibly-rendered transcript |
| send/queue/attach/unattended | ✅ | row carries `launch_state:"failed"` → `sessionLaunchFailed` true (`app_session_helpers.js:12`) blocks all input paths |
| delete/dismiss | ✅ (minor) | `hidden_failure_ids` keyed by session_id; discovery path 3 (`:258`) re-`unhide`s on death, so a SIGKILL+persistent-sidecar case can take a second dismiss |
| restart rediscovery | ✅ | ledger persists on disk; broker-exit records before unlink; discovery/prune cover SIGKILL |

Only `tail` honors recovery. The dead `_search_launch_payload_events` helper is direct evidence the search wiring was intended but not completed. The project's own precedent treats this exact class — a rendered synthetic row that is not searchable — as blocker-grade (transcript-search fix `06930c9`/`bb7d38d`, accepted by critic `0818468c`). **Fix: `search`, `history`, `live`, and `export` must each honor the same recovery payload as `tail` (route the launch-attempt transcript through the same event stream, including `_search_launch_payload_events` for search).** Making `live` return the recovery payload (with `live_cursor:null`) additionally fixes the deselect-on-death discontinuity.

### Q5 — Exact acceptance evidence (user perspective)

Isolated Docker, real Codoxear server + real browser DOM (not just API), one matrix per drop path — **broker exit after bind**, **SIGKILL → stale-sidecar discovery**, **prune of dead session**, **session-control dead-drop**:

1. **Listing:** recovery row appears in the sidebar with the *routing* id (`broker-<pid>`) and a failed/recovery badge; `busy:false`.
2. **Tail:** selecting it renders the preserved user prompt (from the real log) plus one assistant error row `The backend process stopped before completing this turn.`
3. **Search:** `/messages/search?q=<prompt phrase>` and `?q=stopped before completing` both return `match_count≥1` with cursors; history-load at the cursor rehydrates the same rows.
4. **Export:** Copy conversation returns the transcript including the lifecycle error row (no "copy failed").
5. **Live continuity:** a session watched live at death transitions in place to the recovery transcript (no forced deselect), or an explicit accepted deselect+reselect is documented.
6. **Input safety:** send/queue/attach/unattended all blocked; composer disabled with accurate copy.
7. **Restart:** fresh second server against the same app dir re-projects the identical recovery row + transcript.
8. **Negative control:** a session that *completes* its turn then is killed produces **no** recovery row and **no** false "stopped" error.
9. **Identity control:** broker-exit death and SIGKILL-discovery death of the same launch produce the **same** `session_id`.
10. Unit tests (see below).

### Q6 — Blocker-level concerns before code is accepted

1. **Route authority split + dead code** (Q4) — search/history/live/export lack the recovery fallback; `_search_launch_payload_events` never wired.
2. **Identity divergence** (Q3) — broker-exit path keys the row by rollout id, not the routing id used by the other three paths and the frontend.
3. **No tests** — zero coverage across 8 touched modules (see acceptance-report `testsAddedOrUpdated: []`). Required by the acceptance contract.
4. **Unbounded transcript read** — `launch_ledger._post_log_bound_transcript_payload` reads the entire log (`max_bytes=max(size,1)`, `has_older:false`, no pagination, no cap) while `export` caps at `transcript_export_max_bytes` with 413. A large incomplete log yields an unbounded single payload.

---

## Concerns (non-blocking, mechanism noted)

- **Badge/copy imprecision:** the row uses `launch_state:"failed"`, so the UI shows the generic "failed" badge and "Failed launch cannot receive messages" (`app.js:2945` + composer sync). Accurate that input is blocked, but "launch failed" is wrong for a session that ran and produced a transcript. Consider distinct copy for post-log death.
- **Dismissed-session resurrection:** discovery path 3 (`session_discovery.py:258`) sets `unhide_session=True`, so a previously-dismissed session whose broker later dies with a non-idle log reappears. Consistent with the pre-existing pre-log path (`:246`), so not newly introduced, but worth a decision.
- **Boundary edges from Q2** (indeterminate idle; non-refused stale socket) — document or handle.

## Exact files/functions to inspect for the fixes

- `codoxear/message_routes.py`: `handle_messages_search`, `handle_messages_history`, `handle_messages_live`, `handle_messages_export` (all `if not s: 404` with no fallback); wire/replace `_search_launch_payload_events`.
- `codoxear/broker.py:982-983` (and context of `_register_from_log` `st.session_id = sid` at `:451`, sock stem at `:871`): identity fix.
- `codoxear/launch_ledger.py`: `_post_log_bound_transcript_payload` (unbounded read), `launch_attempt_row` (`launch_state:"failed"` copy), `latest_launch_attempt_for_session_id`/`launch_attempt_route_id` (identity lookup).
- `codoxear/post_log_recovery.py`: `log_needs_post_log_bound_recovery` (idle==None edge).
- `codoxear/session_discovery.py:~258,~295`: stale/dead post-log paths (unhide, definitely-stale gate).

## What is already correct (do not re-litigate)

- Ledger abstraction, no-log-mutation, binary busy/idle preservation (Q1).
- Four drop paths present and idempotent: broker-exit guarded by `prelog_failure_recorded` + sidecar unlink; discovery unlinks sidecar (no re-record); prune/control guard on `existing_launch_failed` (launch_id dedup via `read_launch_attempts`).
- Listing dedup suppresses the transient active+failed window by launch_id/spawn_nonce.
- Factory wiring is complete (`session_manager_factories.py` supplies `compute_idle_from_log`/`latest_launch_attempt`/`record_launch_attempt`/`stderr` at lines 149/163/193/202). Code parses and imports cleanly.