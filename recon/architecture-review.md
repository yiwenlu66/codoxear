# Codoxear Architecture Review

**Branch:** `develop`  
**Date:** 2026-06-12  
**Scope:** server.py, broker.py, sessiond.py, rollout_log.py, pi_log.py, agent_backend.py, util.py, voice_push.py  
**Constraints:** Read-only. No process interaction, no file edits.

---

## 1. System Understanding

### 1.1 Component Map and Responsibilities

| Component | Lines | Responsibility |
|-----------|-------|----------------|
| `server.py` | 7535 | HTTP API, session discovery, state persistence, harness/queue sweep loops, file/git viewer, voice push, all HTTP routing |
| `broker.py` | 1750 | PTY wrapper, busy-state machine, log discovery, Unix socket control server |
| `sessiond.py` | 568 | Headless PTY wrapper, simpler busy heuristic, pending-log placeholder |
| `rollout_log.py` | 1102 | JSONL parsing, chat-event extraction, idle/busy heuristic from log, token snapshots, delivery message extraction |
| `pi_log.py` | 477 | Pi-specific session header parsing, assistant/user text extraction, final-turn detection, context-token math |
| `agent_backend.py` | 79 | Backend registry dataclass: bin path, home dir, sessions dir per backend |
| `util.py` | 645 | Shared log-scan, path matching, process-tree FD enumeration, launch-attempt ledger |
| `voice_push.py` | 1447 | TTS, web-push notifications, HLS streaming coordinator |

### 1.2 Data Flow

```
Terminal/Web
  └─► broker.py (PTY fork, env injection)
        ├── writes: socks/*.sock (Unix socket)
        ├── writes: socks/*.json (metadata sidecar)
        ├── reads:  ~/.codex/sessions/**/*.jsonl  OR  ~/.pi/agent/sessions/**/*.jsonl
        └── drives: busy/idle state machine from PTY text + JSONL log

Server (single process)
  └─► SessionManager
        ├── discovers: socks/*.sock + *.json
        ├── polls: broker sockets for {busy, queue_len, token}
        ├── reads: backend JSONL logs for chat events, idle state, token counters
        ├── persists: ~10 JSON state files under APP_DIR
        └── runs: 3 background sweep threads (harness, queue, voice-push)

Browser
  └─► polls /api/sessions, /api/sessions/<id>/messages
  └─► sends via /api/sessions/<id>/send or /enqueue
```

### 1.3 Sources of Truth

| Data | Authoritative Source | Consumers |
|------|---------------------|-----------|
| Session transcript | Backend JSONL log file | server (read), rollout_log (parse), broker (busy state), UI (via API) |
| Broker liveness | Unix socket + PID checks | server discovery, server prune |
| Busy/idle state | **Dual**: broker PTY heuristic AND server log-scan heuristic | server list_sessions, harness sweep, queue sweep |
| Session metadata | `socks/*.json` sidecar | server discovery, server refresh |
| UI sidebar state | `session_sidebar.json` | server (read/write), UI (via API) |
| Queue state | `session_queues.json` | server (read/write), queue sweep |
| Harness config | `harness.json` | server (read/write), harness sweep |
| File history | `session_files.json` | server (read/write), UI file picker |
| Launch attempts | `session_launches.jsonl` | server (read/write), broker (write) |
| Session aliases | `session_aliases.json` | server (read/write) |
| Hidden sessions | `hidden_sessions.json` | server (read/write) |
| Recent cwds | `recent_cwds.json` | server (read/write) |
| Voice settings | `voice_settings.json` | voice_push (read/write) |
| Voice delivery | `voice_delivery_ledger.json` | voice_push (read/write) |

---

## 2. Findings

### Finding 1: Massive God-Module in `server.py` (7535 lines)

**Evidence:** `server.py` contains:
- All HTTP routing (Handler class ~2400 lines of `do_GET`/`do_POST`)
- SessionManager (~1800 lines) with session discovery, harness sweep, queue sweep, voice-push scan, file history, sidebar meta, aliases, hidden sessions, recent cwds
- ~100 module-level constants and env-var parsing
- ~50 standalone utility functions (file kind detection, PNG CRC repair, video preview generation, git operations, file search scoring, path resolution)
- Launch attempt transcript rendering
- Cookie/HMAC auth

**Impact:** Every change touches this file. Merge conflicts are near-guaranteed. Cognitive overhead for any reviewer is extreme. Functions like `_repair_png_crc`, `_ensure_video_preview`, `_file_search_score`, and the entire git-viewer layer have zero coupling to session management.

**Recommendation:** Extract into coherent modules:
1. `codoxear/auth.py` — cookie/HMAC, password
2. `codoxear/file_viewer.py` — file read, kind detection, PNG repair, video preview
3. `codoxear/git_viewer.py` — diff, changed files, file versions, worktrees
4. `codoxear/session_manager.py` — SessionManager class (already self-contained behind a lock)
5. `codoxear/launch.py` — launch attempt ledger, transcript rendering, spawn logic
6. `codoxear/api.py` or keep `server.py` — HTTP handler, routing, static serving

**Risk:** Moderate. These are mechanical extractions. The main risk is the module-level singleton `MANAGER = SessionManager()` and the ~30 module-level constants that several functions reference. Preserve those as module-level or inject them.

**Validation:** `python3 -m pip install -e .` + all existing tests pass. Count imports per module to verify coupling reduction.

---

### Finding 2: Duplicated Functions Across broker.py, sessiond.py, server.py

**Evidence:** The following functions are defined in multiple files with near-identical implementations:

| Function | Locations | Notes |
|----------|-----------|-------|
| `_pid_alive` | server.py:639, broker.py:351 | Identical semantics |
| `_process_group_alive` | server.py:652, broker.py:366, sessiond.py:91 | Identical |
| `_paths_match` | broker.py:493, util.py:164 | Identical |
| `_write_all` | broker.py:463, sessiond.py:69 | Identical |
| `_inject` | broker.py:472, sessiond.py:78 | Identical |
| `_set_winsize` | broker.py:482, sessiond.py:57 | Both delegate to pty_util |
| `_encode_enter` | broker.py:456, sessiond.py:61 | Different default env var |
| `_seq_bytes` | broker.py:451, sessiond.py:65 | Both delegate to pty_util |
| `_read_jsonl_from_offset` | broker.py:561, sessiond.py:87, server.py:2105, util.py:read_jsonl_from_offset | Four copies |
| `_session_id_from_rollout_path` | broker.py method, server.py standalone | Identical regex |

**Impact:** Bug fixes must be applied in multiple places. The `_read_jsonl_from_offset` in broker.py is a simpler version that lacks the safe-newline logic of util.py's version — this is a latent correctness divergence.

**Recommendation:**
- Move `_pid_alive`, `_process_group_alive`, `_terminate_process`, `_terminate_process_group` into `util.py`
- Move `_write_all`, `_inject`, `_set_winsize` into `pty_util.py` (which already exists)
- Eliminate all local `_read_jsonl_from_offset` copies — use `util.read_jsonl_from_offset` everywhere
- Move `_session_id_from_rollout_path` regex into `util.py`

**Risk:** Low. These are pure functions with no state. Import path changes only.

**Validation:** `grep -rn 'def _pid_alive\|def _process_group_alive\|def _write_all\|def _inject' codoxear/` should yield exactly one definition per function.

---

### Finding 3: Competing Busy/Idle Authorities

**Evidence:**
- **Broker busy state** (broker.py `State.busy`): Driven by PTY text heuristic (`_update_busy_from_pty_text`) + JSONL log state machine (`_apply_rollout_obj_to_state`). Reports via socket `cmd:state`.
- **Server idle-from-log** (rollout_log.py `_compute_idle_from_log`): Independent scan of the JSONL tail. Used in `list_sessions()` and `idle_from_log_path()`.
- **Server list_sessions** discards `state_busy` from the broker socket and replaces it with `not idle_from_log()` (server.py ~line 4190). The broker's PTY-based busy signal is ignored for the sidebar.

The broker maintains a sophisticated turn-level state machine (`turn_open`, `turn_has_completion_candidate`, `pending_calls`, `last_interrupt_hint_ts`) that the server never uses. The server's `_compute_idle_from_log` is a simpler last-signal heuristic that scans the JSONL tail.

**Impact:**
- The harness sweep uses `get_state()` (broker socket) for `busy` but also calls `_queue_len`. The queue sweep calls `_queue_remote_ready` which checks BOTH broker `busy` AND `idle_from_log`. These are two different answers to the same question.
- Potential for the broker to report "busy" while the server reports "idle" or vice versa, creating inconsistent queue/harness behavior.

**Recommendation:** Define one authoritative busy/idle signal. Options:
1. **Broker is authority** (preferred): Server trusts the broker socket for busy/idle, using log-scan only as a fallback when the socket is unresponsive. The broker already has the most information (PTY + log + pending calls).
2. **Log is authority**: Remove the broker's turn state machine and use log-only idle detection. Simpler but loses the PTY "esc to interrupt" signal.

The current hybrid where the server overrides the broker's busy with its own log scan is the worst of both worlds.

**Risk:** Medium. Changing idle semantics affects harness injection timing, queue drain timing, and sidebar "busy" display.

**Validation:** Create a test that sends a message, verifies busy=true from both broker and server, waits for turn_complete, verifies busy=false from both. Currently no such end-to-end test exists.

---

### Finding 4: sessiond.py Is an Under-Maintained Parallel Implementation

**Evidence:**
- `sessiond.py` has its own `State` class (line 104) with a subset of broker.py's `State` fields (no `turn_open`, no `pending_calls`, no `token`, no `interrupt_hint_tail`).
- Its `_log_watcher` has a crude busy heuristic: `user_message` → busy, `token_count.total_token_usage` → idle. This ignores agent_message, reasoning, tool calls, errors, turn_aborted.
- It has its own `_pty_reader` that only handles `\x1b[6n`, `\x1b[18t`, `\x1b[14t` terminal queries vs. broker's comprehensive list.
- It creates a `ROOT_REPO_DIR` with `git init` for Codex sessions — behavior not documented in AGENTS.md.
- It lacks the broker's shell-startup watchdog, detach-trigger detection, Pi active-session-marker bridge, and launch-attempt recording.

**Impact:** Web-owned sessions launched through sessiond would have degraded busy detection, missing terminal query responses, and no launch failure recording. However, web sessions appear to always use `broker.py` (the server's `spawn_web_session` runs `python3 -m codoxear.broker`), so sessiond may be partially dead code for the primary use case.

**Recommendation:**
1. Clarify sessiond's role. If it is only for third-party headless integrations, document the reduced fidelity.
2. If it should match broker behavior, extract the shared state machine and terminal query handling into a common module.
3. If sessiond is vestigial, deprecate it.

**Risk:** Low for deprecation. Medium if attempting unification (risk of breaking headless workflows that may exist undocumented).

---

### Finding 5: 10+ Flat JSON State Files with No Unified Lifecycle

**Evidence:** The server persists at least 11 distinct JSON files under `APP_DIR`:
- `session_sidebar.json`, `session_files.json`, `session_queues.json`, `harness.json`, `session_aliases.json`, `hidden_sessions.json`, `recent_cwds.json`, `voice_settings.json`, `push_subscriptions.json`, `voice_delivery_ledger.json`, `session_launches.jsonl`

Each has its own `_load_*()` / `_save_*()` pair inside SessionManager. Every save writes the entire file atomically (write tmp, rename). There is no coordination: a crash between saving aliases and saving sidebar_meta leaves inconsistent state.

**Impact:**
- The `_clear_deleted_session_state` method (server.py ~line 3176) must remember to clean up every file — and it does, but adding a new file requires touching this method.
- Per-session data is scattered: to understand all state for session X, you must read aliases[X], sidebar_meta[X], harness[X], files[sid:X], queues[X], hidden_sessions contains X?, and launch_attempts for X.

**Recommendation:**
1. Short-term: Extract a `StateStore` class that encapsulates all per-session persistent state behind a single interface (`get(session_id)`, `set(session_id, field, value)`, `delete(session_id)`).
2. Longer-term: Consider SQLite for atomic multi-table writes, but only if the JSON-file approach becomes a measured problem.

**Risk:** Low-medium. This is a refactor of internal persistence, invisible to the API contract.

---

### Finding 6: rollout_log.py Has Two Parallel Chat-Event Extraction Paths

**Evidence:**
- `_single_chat_event(obj)` (rollout_log.py ~line 107): Extracts one event from one JSONL record. Used by `_read_chat_page_reverse`.
- `_extract_chat_events(objs)` (rollout_log.py ~line 316): Extracts events from a batch. Also counts thinking/tools/system and tracks turn_start/turn_end. Used by `_read_chat_live_delta`, `_analyze_log_chunk`, `_read_chat_tail_snapshot`.

Both contain the same type-dispatch logic for `event_msg`, `response_item`, and `message` types but with different code paths. For example, `_single_chat_event` computes `message_class` for Pi assistant messages and for Codex response_items, while `_extract_chat_events` does the same computation inline with different variable names.

The batch version also contains a duplicate `event_ts` and `text_message_id` as local functions defined inside `_extract_chat_events` (despite identical module-level versions existing at lines 47 and 89).

**Impact:** A bug fix to chat-event extraction must be applied to both paths. The `message_id` computation must match between the two for cursor-based pagination to work correctly.

**Recommendation:** Make `_extract_chat_events` call `_single_chat_event` per record, then accumulate the metadata (thinking counts, turn flags) separately. This eliminates the duplicated type-dispatch.

**Risk:** Low. The logic is already identical in intent; the refactor mechanically deduplicates it.

**Validation:** Existing test `test_chat_scrollback_source.py`, `test_chat_transcript_runtime.py`, `test_message_index.py` should continue to pass.

---

### Finding 7: Backend Abstraction is Incomplete — Pi Logic Leaks Everywhere

**Evidence:** `agent_backend.py` defines a clean `AgentBackend` dataclass with `cli_bin()`, `home()`, `sessions_dir()`. But backend-specific behavior is scattered:

- **broker.py**: `if AGENT_BACKEND == "pi":` appears ~15 times for Pi session args, bridge extension, active-session marker, session-dir naming
- **server.py**: Separate code paths for `codex` vs `pi` launch defaults, resume candidates, run settings. `PI_HOME`, `PI_SESSIONS_DIR`, `PI_SETTINGS_PATH`, `PI_MODELS_PATH`, `PI_AUTH_PATH` are module-level constants.
- **rollout_log.py**: Calls `pi_user_text`, `pi_assistant_text`, etc. inline in the type-dispatch for `type == "message"` records.
- **util.py**: `_is_codex_rollout_log_path` vs `_is_pi_session_log_path` — two separate path-matching functions.

The `AgentBackend` dataclass only covers filesystem conventions. It does not abstract:
- Log record format / chat-event extraction
- Session ID discovery from log
- Busy/idle state machine transitions
- Launch argument construction
- Resume semantics

**Impact:** Adding Claude Code (`cc`) as a third backend (per PROMPT.md workstream 4) would require touching 10+ files with `if backend == "cc":` branches.

**Recommendation:** Extend `AgentBackend` (or a parallel `BackendAdapter` protocol) with methods for:
1. `extract_chat_event(obj) → ChatEvent | None`
2. `is_session_log_path(path) → bool`
3. `session_id_from_log(path) → str | None`
4. `launch_args(config) → list[str]`
5. `is_busy(obj) → BusySignal`

Each backend registers its implementation. The rollout_log, broker, and server call through the abstraction.

**Risk:** Medium. This is a larger refactor that touches the core data path. Must be incremental.

---

### Finding 8: Harness Feature Naming is Inaccurate

**Evidence:**
- The feature is called "harness" in code (`harness.json`, `harness_enabled`, `_harness_sweep`, `HARNESS_PATH`).
- The actual mechanism (server.py `_harness_sweep`): when a session is idle for `cooldown_minutes` and the last chat role was "assistant", inject a predefined prompt via `self.send()`. The prompt text (`HARNESS_PROMPT_PREFIX`) describes "Unattended-mode instructions."
- The UI (not reviewed in detail) presumably exposes "Harness mode" terminology.

**Impact:** "Harness" suggests test scaffolding or instrumentation. The actual behavior is "unattended continuation" — idle-triggered prompt injection. The PROMPT.md explicitly flags this for renaming.

**Recommendation:** Rename to `unattended` throughout:
- `harness.json` → `unattended.json` (with migration/alias)
- `harness_enabled` → `unattended_enabled`
- `_harness_sweep` → `_unattended_sweep`
- `HARNESS_*` constants → `UNATTENDED_*`
- API fields: `harness_enabled` → `unattended_enabled`, etc.

Decide upfront: do internal names change immediately (clean break) or alias for one release? Given the small user base, a clean break with a migration script for the JSON file is simplest.

**Risk:** Low (naming only). The migration for `harness.json` → `unattended.json` needs a fallback read of the old path.

---

### Finding 9: Server `list_sessions()` Does Too Much Synchronous Work

**Evidence:** `list_sessions()` (server.py ~line 4148) on each call:
1. `_discover_existing_if_stale()` — scans `socks/*.sock`, reads each `.json`, calls `_sock_call(state)` per socket
2. `_prune_dead_sessions()` — calls `_refresh_session_state()` per session (another socket round-trip)
3. `_update_meta_counters()` — reads JSONL logs for all sessions, up to 16 chunks of 256KB each
4. Inside the lock: reads run settings from log, reads `_last_conversation_ts_from_tail`, computes `_current_git_branch` per session
5. After the lock: calls `idle_from_log_path` per session, calls `_maybe_drain_session_queue` for busy sessions with queues
6. Reads launch attempts and renders failure rows

For N sessions, this is O(N) socket calls + O(N) log reads + O(N) `git rev-parse` subprocess calls. With 10 sessions, this is ~30+ socket round-trips and ~10 git subprocesses per poll.

**Impact:** The UI polls `/api/sessions` regularly. On mobile/slow networks, the latency of this call directly impacts perceived responsiveness. The `_record_metric("api_sessions_ms", dt_ms)` metric tracks this.

**Recommendation:**
1. Cache `_current_git_branch` with a TTL (e.g., 5s). Git branch rarely changes sub-second.
2. Move `_update_meta_counters` to a background thread (like harness/queue sweep) instead of computing it synchronously on API calls.
3. Consider returning stale data for most fields and only refreshing the active session on demand.

**Risk:** Low for caching. Medium for background-thread approach (need to ensure the UI still gets fresh data when the user explicitly refreshes).

---

### Finding 10: Broker-Server Metadata Refresh is Expensive and Redundant

**Evidence:** `refresh_session_meta()` (server.py ~line 4262):
1. Reads the `.json` sidecar
2. Parses it
3. Checks if log_path changed
4. If detached, does another socket call (`cmd:tail`)
5. Falls back to `/proc` FD scanning for log discovery
6. If Codex, reads session_meta from the log, checks for subagent, resolves main thread log
7. Reads run settings from log
8. Updates session state

This is called from `list_sessions()` (indirectly via `_discover_existing`), from every `do_GET` endpoint that takes a session_id, and from the voice-push scan loop. The `.json` sidecar is the broker's output; rereading it from the server on every API call is redundant when nothing has changed.

**Recommendation:** Add a stat-based freshness check: only re-parse the `.json` sidecar if its mtime has changed since the last read. The broker already writes atomically.

**Risk:** Very low. Pure optimization.

---

## 3. Recommended Refactor Order

Priority is based on: (1) invariant preserved, (2) risk, (3) downstream enablement.

| Order | Finding | Effort | Enables |
|-------|---------|--------|---------|
| 1 | F2: Deduplicate functions into util.py/pty_util.py | Small | Reduces noise for all subsequent work |
| 2 | F6: Unify chat-event extraction in rollout_log.py | Small | Cleaner base for F7 |
| 3 | F8: Rename harness → unattended | Small | Naming clarity before new features |
| 4 | F1: Extract server.py into modules | Medium | Reduces merge conflicts, enables parallel work |
| 5 | F7: Extend backend abstraction for cc support | Medium | Required for Claude Code backend |
| 6 | F3: Unify busy/idle authority | Medium | Fixes semantic confusion in queue/harness |
| 7 | F5: Unify state file persistence | Medium | Reduces per-file maintenance burden |
| 8 | F9+F10: Optimize list_sessions / refresh_session_meta | Small | Performance |
| 9 | F4: Clarify/deprecate sessiond.py | Small | Remove dead code |

---

## 4. Risks and Constraints

### 4.1 Live Sessions Must Not Break
The server is stateless w.r.t. transcript data — sessions survive server restarts. However, the harness, queue, and voice-push sweep threads hold in-memory state that is periodically flushed. A crash during refactoring that corrupts the JSON state files would lose queue items and harness configs. Mitigation: always write atomically (current pattern is correct).

### 4.2 Backend Abstraction Must Remain Incremental
The Pi backend has ~15 special cases in broker.py. A clean abstraction cannot be done in one pass. The recommended approach is:
1. Define the interface (BackendAdapter protocol)
2. Implement for Codex first (extract existing code)
3. Implement for Pi (extract existing code)
4. Implement for cc (new code)
5. Remove `if backend == "xxx":` branches

### 4.3 Broker State Machine Must Not Regress
The broker's busy/idle state machine is the most complex piece of logic (broker.py lines 610-780). Any change to idle semantics must preserve:
- `user_message` → busy
- `turn_complete` / `task_complete` → idle
- `agent_reasoning` / `function_call` → keeps busy
- `turn_aborted` → idle
- PTY "esc to interrupt" hint → busy
- Quiet timeout (`BUSY_QUIET_SECONDS`) → idle

Test coverage: `test_broker_busy_state.py` exists. Verify it covers all transitions.

### 4.4 Product Philosophy Constraints
- **No sidebar nesting:** The flat session list with priority-based sorting is the correct abstraction. Do not add folders, tags, or hierarchies.
- **Deliberate chat detail omission:** The chat view shows user messages and assistant final responses. Thinking, tool calls, and system messages are counted but not displayed. This is intentional.
- **Shared broker:** CLI and web sessions use the same broker binary. Do not create a web-only session management path.

---

## 5. Validation Checks

| Check | Purpose | Command |
|-------|---------|---------|
| No duplicate function definitions | F2 resolved | `grep -rn 'def _pid_alive\|def _process_group_alive\|def _write_all\|def _inject' codoxear/ \| wc -l` should be ≤ 4 (one each) |
| All tests pass | No regression | `python3 -m pytest tests/ -x` |
| server.py line count < 3000 | F1 progress | `wc -l codoxear/server.py` |
| Backend dispatch is polymorphic | F7 progress | `grep -rn 'if.*backend.*==.*"pi"\|if.*backend.*==.*"codex"\|AGENT_BACKEND ==' codoxear/ \| wc -l` should decrease |
| Harness references removed | F8 resolved | `grep -rni 'harness' codoxear/ \| grep -v 'unattended' \| wc -l` should be 0 |
| No competing busy definitions | F3 resolved | Server `list_sessions` uses broker busy without override |
| State files have unified lifecycle | F5 progress | Single `StateStore` class manages all per-session persistence |

---

## 6. Architecture Diagram (Current)

```
┌──────────────────────────────────────────────────────────────────────┐
│                         codoxear/server.py (7535 LOC)                │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  ┌───────────┐  │
│  │ HTTP Handler │  │ SessionMgr   │  │ File/Git   │  │ Auth/     │  │
│  │ (routing)    │  │ (discovery,  │  │ Viewer     │  │ Cookie    │  │
│  │              │  │  sweep loops,│  │            │  │           │  │
│  │              │  │  10 state    │  │            │  │           │  │
│  │              │  │  files)      │  │            │  │           │  │
│  └──────────────┘  └──────────────┘  └────────────┘  └───────────┘  │
└─────────────┬───────────────┬────────────────────────────────────────┘
              │               │
     Unix socket calls    JSONL log reads
              │               │
              ▼               ▼
    ┌──────────────┐  ┌───────────────────┐
    │ broker.py    │  │ rollout_log.py    │
    │ (PTY, state  │  │ (parse, extract,  │
    │  machine,    │  │  idle detect)     │
    │  sock server)│  │                   │
    └──────┬───────┘  │   pi_log.py       │
           │          │   (Pi-specific)    │
           │          └───────────────────┘
           │
     PTY fork/exec
           │
           ▼
    ┌──────────────┐
    │ codex / pi   │
    │ CLI process  │
    │ (writes JSONL│
    │  log files)  │
    └──────────────┘
```

The `sessiond.py` is a parallel, simpler implementation of broker.py for headless use, sharing almost no code with it.

---

## 7. Summary of Key Issues by Severity

### Critical (blocks new backend addition)
- **F7:** Backend abstraction doesn't cover log format, busy detection, or launch args. Adding cc requires touching 10+ files.

### High (architectural confusion)
- **F3:** Two competing busy/idle authorities with different semantics
- **F1:** 7535-line god module makes all changes risky

### Medium (tech debt)
- **F2:** ~10 duplicated functions across 3 files
- **F6:** Two parallel chat-event extraction paths
- **F5:** 10+ flat state files with no unified lifecycle

### Low (naming / optimization)
- **F8:** "Harness" naming is inaccurate
- **F9/F10:** list_sessions does too much synchronous work
- **F4:** sessiond.py is an under-maintained parallel implementation
