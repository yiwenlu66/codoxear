# Phase 7 backend audit

## Decision: **GO**, narrowly scoped to runtime-state integrity

The audit found a live ordering defect, not merely large files. A cursor-relative message poll can overwrite the shared session registry with older model/provider/effort or token evidence after a newer projection has already been established. The backend refactor phase is therefore justified, but only around the log-derived session projection and the plumbing that makes this defect easy to reproduce. This decision does **not** authorize a general rewrite of `server.py`, `broker.py`, or every size outlier.

### Demonstrated live defect

I exercised the production `handle_messages_live` function with a real temporary Pi JSONL log larger than the route's 2 MiB read bound:

- the log began with `model_change -> old-model`;
- enough valid JSONL was inserted to put `model_change -> new-model` beyond the first bounded read;
- the shared `Session` began at the correct full-log projection, `new-model`;
- one valid poll from cursor `0` processed only the older bounded prefix;
- afterward the shared `Session.model` had regressed to `old-model`, while the returned cursor was still before EOF.

Observed probe output:

```text
before new-provider new-model bytes 2228582 poll_bound 2097152
after_one_stale_poll old-provider old-model next_cursor_before_eof True
```

The causal chain is direct:

1. Every poll refreshes current metadata before reading its cursor window (`codoxear/message_routes.py:1164-1166`).
2. The poll reads at most `LIVE_POLL_READ_MAX_BYTES` from that potentially old cursor (`codoxear/message_routes.py:1226-1231`; bound declared at `codoxear/message_routes.py:22-25`).
3. It passes the cursor-relative batch to `manager.mark_log_delta(..., new_off=next_after)` (`codoxear/message_routes.py:1255-1256`).
4. `SessionLogRuntimeCoordinator.mark_log_delta` accepts `new_off` but never compares or records it (`codoxear/session_log_runtime.py:188-238`). It blindly assigns any model/provider/effort found in the batch (`codoxear/session_log_runtime.py:232-237`).
5. The server is threaded (`codoxear/server_main.py:11-12`), and the same route is used by independent browser tabs, SSE/poll reconnects, and fallback polling. A late request from an older cursor can therefore win after a request that observed newer log state.

A second direct probe called `mark_log_delta` with `new_off=200` and then with `new_off=100`; the second call regressed `new-model` to `old-model`. The method has no log-identity or monotonic-offset guard. This is a **LIVE** defect: the public shared session state is order-dependent on request completion rather than log order.

Token state has the same bug class. Both live-delivery paths assign `Session.token` directly (`codoxear/message_routes.py:376-378`, `codoxear/message_routes.py:1257-1259`), outside the registry lock and independently of the background log projection that also writes it under lock (`codoxear/session_log_runtime.py:126-142`). A direct assignment probe showed an older observation replacing a newer token. The current tests establish source priority (`tests/test_session_runtime.py:252-277`) and that a live delta was marked (`tests/test_message_routes.py:465-507`), but they do not establish monotonic ordering across two clients or two cursor windows.

---

## Scope and classification

- **LIVE** — exercised behavior is wrong, or the code has a concrete, reachable bug-class mechanism that can change runtime behavior.
- **LATENT-STRUCTURAL** — cohesion or maintenance hazard with no demonstrated behavioral failure in this audit.
- **NO FINDING** — multiple layers exist, but the ownership/reconciliation rule is explicit and consistently routed in the inspected paths.

I read `codoxear/server.py`, `codoxear/message_routes.py`, and `codoxear/broker.py` in full. I also inspected the requested secondary outliers and the supporting control flow needed to adjudicate ownership: `voice_push.py`, `session_store.py`, `launch_config.py`, `session_runtime.py`, `session_manager_factories.py`, `cc_log.py`, `session_listing.py`, `session_log_runtime.py`, `session_refresh.py`, `session_discovery.py`, `session_discovery_registry.py`, `session_readiness.py`, `session_control.py`, `session_prune.py`, `session_input.py`, `server_route_deps.py`, `server_handler.py`, `agent_backend/*`, `pi_log.py`, `rollout_log.py`, `rollout_jsonl.py`, and the relevant tests/history.

## Signature 1 — pass-through plumbing

### Finding 1.1 — `SessionManager` is a generic forwarding facade

**Evidence**

- `SessionManager` exposes generic `*args, **kwargs` pass-through methods for its core surface (`codoxear/server.py:948-1015`) and nearly every coordinator method (`codoxear/server.py:1086-1303`).
- Each call asks a factory for a coordinator and then forwards the whole argument set unchanged, for example queue/readiness/log paths at `codoxear/server.py:1155-1204` and `codoxear/server.py:1251-1267`.
- The facade contains 96 methods with both a vararg and kwarg signature. This is evidence about signature erasure, not a numeric refactor target.

**Mechanism**

The public manager surface no longer tells callers or type tooling what a method actually accepts. A parameter rename or missing argument is deferred until the eventual leaf call. This is the Python analogue of forwarding an options bag through a composition layer: the middle layer publishes no contract of its own.

**Classification: LATENT-STRUCTURAL.** I found no current argument-mismatch failure attributable to these forwards. The risk is real but remains a maintenance hazard in this audit.

### Finding 1.2 — one global capability bag feeds every manager subsystem

**Evidence**

- `SessionManagerFactoryCaps` mixes log parsing, cleaning, process control, launch configuration, queue timing, tmux, unattended settings, filesystem paths, exception types, and test seams in one object (`codoxear/session_manager_factories.py:45-129`).
- `session_manager_factory_caps(server)` populates that object from the entire `server` module (`codoxear/session_manager_factories.py:132-217`).
- Every coordinator access reconstructs the full capability object before selecting a subsystem-specific subset (`codoxear/server.py:1017-1084`).
- The eventual coordinator constructors are explicit — for example readiness at `codoxear/session_manager_factories.py:353-364` and send at `codoxear/session_manager_factories.py:601-623` — so this is not an unchanged bag all the way to the leaf. The leak is the `server module -> mega-caps -> focused constructor` middle layer.
- Coordinators are reconstructed on demand. Queue-sweep state consequently lives back on `manager` through getter/setter lambdas rather than on the coordinator (`codoxear/session_manager_factories.py:391-407`).

**Mechanism**

All manager subsystems depend indirectly on a service locator whose contract is the union of every subsystem. A new dependency can be added to the global bag without making the owning subsystem boundary clearer, and coordinator identity cannot safely carry state because each access returns a fresh object. The design pushes state back into the manager and makes the nominal coordinators transient method bundles.

**Classification: LATENT-STRUCTURAL.** The construction is on live request/background paths, but I did not benchmark a material performance effect or demonstrate a current identity bug. It is a clear recurrence risk for hidden state ownership, not the GO trigger by itself.

### Finding 1.3 — route wiring uses a whole-module service locator, but narrows at the leaf

**Evidence**

- `ServerRouteDepsFactory` holds `server: Any` (`codoxear/server_route_deps.py:27-29`) and reads arbitrary module attributes while building typed route-specific dependency records (`codoxear/server_route_deps.py:52-243`).
- `make_server_handler` captures the whole server module and recreates a route-dependency factory for route calls (`codoxear/server_handler.py:318-345`; `codoxear/server.py:1400-1409`).

**Mechanism**

The composition root remains a global service locator, but the route handlers themselves receive focused dataclasses such as `MessageRouteDeps`. That leaf contract prevents the exact unrestricted propagation seen in the pre-refactor frontend.

**Classification: LATENT-STRUCTURAL.** Narrowing the composition root would improve ownership, but the inspected route contracts already stop arbitrary server state from reaching every handler. No standalone route-wiring phase is justified before the live projection defect.

## Signature 2 — split ownership of authoritative values

### Finding 2.1 — log-derived model/provider/effort projection can move backward

**Evidence**

- `Session` stores model/provider/effort and one full-log revision marker (`codoxear/session_model.py:28-30`, `codoxear/session_model.py:49-53`). It has no cursor-relative projection offset.
- Sidecar/log reconciliation computes settings and refresh writes them (`codoxear/session_runtime.py:257-327`; `codoxear/session_refresh.py:111-141`).
- Session-list backfill is another writer with a full-log revision guard (`codoxear/session_runtime.py:345-370`).
- Live poll/SSE delivery is a third writer through `mark_log_delta` (`codoxear/message_routes.py:374-378`, `codoxear/message_routes.py:1255-1259`; `codoxear/session_log_runtime.py:188-238`). That writer ignores the supplied `new_off` and has no expected-log-path check.
- The production poll probe above demonstrated the shared state regression.
- Repository history corroborates the bug class: commit `b5ca5cb7` describes sidebar model oscillation from two writers with different priority rules, and `6cc980e6` reverted its first fix because another assumed writer did not actually write back. The current source documents authority locally, but no single commit boundary enforces it across all writers.

**Mechanism**

The system has sensible source-priority rules — live bridge, log evidence, launch baseline — but lacks one ordered commit mechanism. Source priority answers “which feed wins”; it does not answer “is this observation newer than the value already stored?” Cursor-relative requests can replay old evidence after a full-log refresh or a newer request and overwrite the registry.

**Classification: LIVE.** This is the demonstrated GO condition.

### Finding 2.2 — token mutation has three writers and no ordered commit boundary

**Evidence**

- The background log runtime writes `Session.token` under the manager lock and advances `meta_log_off` (`codoxear/session_log_runtime.py:126-142`, `codoxear/session_log_runtime.py:184-186`).
- Poll and SSE write the same field directly without that lock (`codoxear/message_routes.py:376-378`, `codoxear/message_routes.py:1257-1259`).
- Broker state may populate the token only when no log is bound (`codoxear/session_control.py:101-114`; the same rule is duplicated at `codoxear/session_prune.py:45-57`).
- `select_runtime_token` defines feed priority for one response (`codoxear/session_runtime.py:671-689`) but does not own mutation of the shared session cache.

**Mechanism**

A cursor batch is allowed to update the shared cache without proving that its log identity/offset is at least as new as the stored observation. An old token or explicit clear can therefore race a newer token. The background scanner cannot always repair an old non-null cache because its fallback full-log search is gated on `session.token is None` (`codoxear/session_log_runtime.py:126-128`).

**Classification: LIVE bug-class risk.** I directly demonstrated the stale assignment mechanism. I did not run a real two-browser race, so I am not claiming a production incident was observed.

### Finding 2.3 — busy/readiness is layered, but its authority is declared

**Evidence**

- Broker turn state owns the raw reducer and updates `State.busy` from PTY/log evidence (`codoxear/broker_turn_state.py:85-109`, `codoxear/broker_turn_state.py:175-205`, `codoxear/broker_turn_state.py:210-459`).
- Server-side `Session.busy` and `Session.queue_len` are cached raw broker projections updated after control responses (`codoxear/session_control.py:101-108`, `codoxear/session_prune.py:45-50`, `codoxear/session_input.py:67-78`).
- `resolve_runtime_status` owns the public semantic `busy`/readiness value by reconciling broker state, a bound log's idle state, the confirmed-send boundary, queue state, and interrupted-idle override (`codoxear/session_runtime.py:600-648`).
- The exceptional split for interrupted-idle is documented and centralized (`codoxear/session_runtime.py:507-536`, `codoxear/session_runtime.py:552-597`). Message responses route through that resolver (`codoxear/server_route_deps.py:31-50`).
- The listing distinguishes `state_busy`/`broker_queue_len` from the local persisted `queue_len` before producing the public row (`codoxear/session_listing.py:367-382`, `codoxear/session_listing.py:429-440`).

**Mechanism**

There are multiple representations, but they are not competing writers to one semantic value: raw broker state is evidence; `resolve_runtime_status` is the policy owner. This is the kind of explicit reconciliation the frontend refactor required.

**Classification: NO FINDING** in the inspected paths. Renaming raw cached fields could improve readability, but that is not grounds for a refactor phase.

### Finding 2.4 — persistent session UI state has a declared store owner

**Evidence**

- `SessionStore` owns aliases, sidebar metadata, hidden sessions, files, queues, attachments, commit-unknown sends, recent cwd state, and unattended configuration (`codoxear/session_store.py:126-176`).
- `SessionManager` exposes these through store-backed descriptors rather than separate copies (`codoxear/server.py:913-943`).

**Classification: NO FINDING** for split value ownership. `SessionStore` is broad, but the inspected mutation path deliberately centralizes persistence rather than duplicating it.

## Signature 3 — god objects/modules

### Finding 3.1 — `server.py` is a service-locator/composition god module

**Evidence**

- It exports configuration into module globals (`codoxear/server.py:245-246`), owns unrelated runtime caches (`codoxear/server.py:266-273`), wraps route/file/git/launch/session helpers, defines the manager facade (`codoxear/server.py:907-1374`), and initializes global runtime objects (`codoxear/server.py:1382-1409`).
- Manager factories and route factories consume the module itself rather than a stable composition object (`codoxear/session_manager_factories.py:132-217`; `codoxear/server_route_deps.py:27-29`).

**Mechanism**

Prior extractions moved implementations out but retained `server.py` as the namespace through which unrelated consumers obtain them. It is now more facade than implementation, yet changing a dependency still crosses this central module and the mega-capability records.

**Classification: LATENT-STRUCTURAL.** The actionable part is the manager wiring described below; line count is not a reason to split the rest. A “shrink server.py” project is explicitly not recommended.

### Finding 3.2 — `message_routes.py` combines several message-domain workflows; the live transport duplication is the risky seam

**Evidence**

- The module owns failed-launch transcript projection/search, cursor attachment, full search, neighbor navigation, detached windows, tail/history paging, poll delivery, and SSE delivery (`codoxear/message_routes.py:77-333`, `codoxear/message_routes.py:400-562`, `codoxear/message_routes.py:595-1281`).
- A shared `_live_payload_from_records` exists for SSE (`codoxear/message_routes.py:343-397`), but polling reimplements the same extraction, token, session-mutation, notification, cursor, and payload pipeline (`codoxear/message_routes.py:1238-1281`) instead of calling it.
- The duplication has already created path-specific regression surface: the live-poll test records that dropping `max_bytes` broke bound-cursor polling while tail tests remained green (`tests/test_message_routes.py:465-507`).

**Mechanism**

Search/navigation/paging are related transcript query responsibilities and do not need to be split solely because the file is large. Poll and SSE, however, are two transports for the same delta semantics. Maintaining separate projections lets mutation ordering, cursor behavior, token clears, and backend-specific turn context drift.

**Classification: LIVE bug-class risk** for the duplicated poll/SSE projection; **LATENT-STRUCTURAL** for the rest of the module. Extract only the shared live-delta projection/commit boundary.

### Finding 3.3 — `Broker` is large but remains one coupled lifecycle

**Evidence**

- `Broker` owns launch, PTY forwarding, log discovery/binding, log reduction, control socket, metadata, backend live settings, failure recovery, and teardown (`codoxear/broker.py:315-1243`).
- Its largest orchestration method explicitly describes one lifecycle and its state effects (`codoxear/broker.py:952-968`).
- Core policies already live in focused modules (`broker_turn_state`, `broker_log_binding`, `broker_log_watcher`, `broker_control`, `broker_metadata`, `codex_live_control`). The class mostly sequences those pieces.

**Mechanism**

The broker has cohesion pressure — especially log discovery at `codoxear/broker.py:475-613` and process lifecycle at `codoxear/broker.py:952-1243` — but both operate on one lock/state/process lifetime. Splitting them without a stronger boundary would add cross-owner state rather than remove it.

**Classification: LATENT-STRUCTURAL, not a demonstrated god-object defect.** No broad broker decomposition is authorized by this audit.

### Secondary outlier sampling

| Module | Finding | Classification |
| --- | --- | --- |
| `voice_push.py` | One coordinator owns a coupled listener/task/generation/playback/push-delivery state machine. Persistence/projection helpers are already extracted. No second authoritative writer was found. | No refactor evidence from this audit. |
| `session_store.py` | Broad persistence owner, but breadth is deliberate and manager descriptors point back to it. | LATENT size/cohesion pressure only. |
| `launch_config.py` | Backend-specific readers and validation all serve one launch/defaults transaction and dispatch to `AgentBackend`. | Cohesive enough; no phase justified. |
| `session_runtime.py` | Mixes JSONL boundaries, run-settings authority, listing enrichment, busy/readiness, and token selection. The log-derived projection portion participates in the live defect. | LIVE for projection ownership; otherwise LATENT. |
| `session_manager_factories.py` | Wiring composition plus the global capability bag. | LATENT-STRUCTURAL; scoped manager-wiring phase justified after the live fix. |
| `cc_log.py` | Cohesive Claude-log parser, though it duplicates a low-level reverse JSONL iterator. | LATENT duplication only. |
| `session_listing.py` | Cohesive staging/public projection/sort pipeline. | No split evidence. |

## Signature 4 — duplicated backend-adapter logic

### Finding 4.1 — incremental settings parsing bypasses the existing adapter authority

**Evidence**

- Full-log settings extraction dispatches through `AgentBackend.read_run_settings_from_log` (`codoxear/session_log_metadata.py:126-146`; backend implementations in `codoxear/agent_backend/codex.py:188-215`, `pi.py:103-113`, and `cc.py:78-88`).
- Incremental live projection reimplements a second backend switch in `SessionLogRuntimeCoordinator.mark_log_delta`: Pi scans `model_change`/`thinking_level_change`, CC scans assistant model, and Codex scans `turn_context` (`codoxear/session_log_runtime.py:194-225`).
- The full-log path has a revision guard; the incremental path ignores its `new_off`. These duplicated implementations therefore have different ordering semantics.

**Mechanism**

Adding or changing a backend setting requires updating both the adapter/full-log reader and an unrelated coordinator switch. The two paths already disagree on what constitutes a newer observation, which is the mechanism behind the reproduced regression.

**Classification: LIVE.** Consolidating incremental and full-log setting projection behind one backend-aware observation type is part of the first recommended phase.

### Finding 4.2 — Pi and Claude duplicate reverse JSONL iteration

**Evidence**

- Near-identical bounded reverse-object iterators exist at `codoxear/pi_log.py:156-193` and `codoxear/cc_log.py:307-344`.
- A shared JSONL reverse-record primitive already exists at `codoxear/rollout_jsonl.py:133-175`; the simpler shared object iterator lacks the `before` boundary (`codoxear/rollout_jsonl.py:99-130`).
- Pi also has a complete-record offset helper (`codoxear/pi_log.py:196-218`) alongside the generic equivalent in `codoxear/session_runtime.py:77-134`.

**Mechanism**

Partial trailing records, byte boundaries, and malformed JSON handling are correctness-sensitive. Three copies can diverge when one is fixed.

**Classification: LATENT-STRUCTURAL.** No differing result was demonstrated. Reuse the existing `rollout_jsonl` domain primitive if this code is touched by the live-projection phase; do not create another generic helper solely to reduce lines.

### Finding 4.3 — backend launch normalization is already shared appropriately

**Evidence**

- `AgentBackend` defines launch defaults projection, request normalization, launch/resume args, sessiond args, and environment construction (`codoxear/agent_backend/base.py:31-265`).
- Codex/Pi/Claude override the backend-specific semantics in focused modules (`codoxear/agent_backend/codex.py`, `pi.py`, `cc.py`).
- `launch_config.py` delegates request options to that adapter (`codoxear/launch_config.py:631-667`).

**Classification: NO FINDING.** Source-specific config readers are legitimate adapter behavior, not accidental duplication.

### Adjacent duplication not counted as backend-adapter evidence

`broker.py` and `sessiond.py` duplicate PTY helpers, process-group teardown, socket-server loops, metadata writing, and log-watcher structure (`codoxear/broker.py:240-291`, `codoxear/broker.py:404-427`, `codoxear/broker.py:782-917`; `codoxear/sessiond.py:83-109`, `codoxear/sessiond.py:162-321`). Their product contracts differ — foreground-terminal broker versus deliberately headless helper — so a mechanical merger is high risk. I found no behavior defect in this bounded audit that requires unifying them. Treat this as a future evidence gate, not part of the approved phase list.

---

## Scoped backend refactor phases, ordered by risk reduction

### Backend Phase 1 — one monotonic log-derived session projection owner

**Goal:** fix the demonstrated state regression before moving architecture around it.

1. Define one backend-aware log observation/result for the shared registry fields derived from JSONL: token (including explicit clear), model provider, model, reasoning effort, last conversation timestamp, and cache invalidation.
2. Give one coordinator the only write method for those fields. A commit must carry **log identity plus byte range/revision** and be rejected if it belongs to a prior binding or ends before the last committed observation. Offset comparison without log identity is insufficient because `/new` and resume/rebind reset offsets.
3. Make the commit atomic under the session registry lock. Route handlers may project cursor-relative response data, but they must not assign `Session.token`, `Session.model`, `Session.model_provider`, or `Session.reasoning_effort` directly.
4. Route full-log refresh/backfill and incremental poll/SSE observations through one documented authority hierarchy. Preserve the intentional live-bridge-versus-log rules; the change is ordered commitment, not a new priority policy.
5. Add behavioral regressions for:
   - newer delta followed by older delta;
   - a stale cursor in a log larger than the live read bound;
   - poll and SSE clients completing out of order;
   - an old token update/clear arriving after a newer token;
   - same-path truncation and different-log rebind, where an offset reset is legitimate.

**Risk reduced:** actual user-visible model/token regression and future multi-feed authority bugs.

### Backend Phase 2 — one live-delta projection for SSE and polling

**Goal:** make transport choice unable to change transcript semantics or registry mutations.

1. Extract the record-window-to-delta operation currently implemented once in `_live_payload_from_records` and again inside `handle_messages_live`.
2. Both transports must consume the same result for positioned events, carried Claude pending-tool state, no-response injection context, meta counters, token observation, notification text, cursor, and the Phase 1 ordered registry commit.
3. Keep HTTP/SSE framing, heartbeat, reconnect, and status-code handling in their respective transport functions.
4. Add parity tests that feed identical records/cursors through poll and SSE and compare the normalized message payload, including token clear and cross-window turn-close cases.

**Risk reduced:** one transport being fixed while the fallback transport retains stale or divergent behavior.

### Backend Phase 3 — retain manager coordinators and narrow their contracts

**Goal:** remove the Python options-bag/service-locator pattern after correctness has one owner.

1. Build and retain the coordinator graph once during manager initialization, rather than reconstructing coordinators and `SessionManagerFactoryCaps` on every method call.
2. Replace the global union capability record with focused dependency records per coordinator (or direct explicit constructor arguments). Preserve constructor injection; it is a real test seam.
3. Give public `SessionManager` methods honest signatures instead of blanket `*args, **kwargs` forwarding. Internal leaf relays that add no contract should be removed or made explicit.
4. Keep registry/store state in their current declared owners; do not migrate it merely to make the facade smaller.
5. Verify background-loop coordinator identity/state and all route call paths behaviorally. The existing manager-factory test (`tests/test_session_manager_factories.py`) proves only that the mega-cap captures live server values; replacement tests must prove focused wiring and retained identity behavior.

**Risk reduced:** hidden dependency drift, accidental coordinator-local state loss, and signature mismatches. This phase is lower priority because no current failure was demonstrated.

## Explicit non-goals

- No target line counts or module counts.
- No wholesale `server.py` split.
- No broad `Broker` rewrite or `broker.py`/`sessiond.py` merger.
- No `SessionStore`, voice, launch-config, or backend-adapter rewrite without new behavioral evidence.
- No behavior-policy changes to busy/readiness, bridge-vs-log settings priority, queue semantics, transcript cursors, or locked frontend branches.

## Audit boundary and residual risk

- This was a bounded architecture audit, not an exhaustive backend review. I did not audit every route, persistence format, auth/security path, git/file operation, voice delivery failure mode, or every backend-native log row.
- I did not run against the live deployment. The stale-model observation used the production route/coordinator code in an isolated in-process temporary-log probe. It establishes the mutation mechanism and result, but not how frequently users encounter it.
- I did not run a real concurrent two-browser race or benchmark the cost of rebuilding manager coordinators.
- The poll/SSE parity review focused on record projection and shared-state mutation; it did not prove every HTTP framing/reconnect edge is equivalent.
- No product code or tests were changed. Only this requested audit report was created.
