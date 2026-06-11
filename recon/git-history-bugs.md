# Codoxear Bug/Regression Ledger — Git History Mining

Generated: 2026-06-12  
Source: 200+ commits on `develop` (current HEAD `b003da6`), plus branches.  
Test suite status: **359 passed, 0 failed** (at HEAD).

---

## 1. Cluster: Chat Scrollback / Cursor / Cache Staleness

**Pattern:** The UI's paginated scrollback state machine has been fixed ≥7 times across distinct bugs. Each fix touched `app.js` scrollback/cursor/cache logic.

| Commit | Date | Title | Root Cause |
|--------|------|-------|------------|
| `455c061` | 03-13 | Fix desktop chat scrollback loading | Initial scrollback pagination broken on desktop |
| `788ba34` | 03-13 | Fix jump to latest after loading older messages | Jump button didn't reset after older-load |
| `e051e37` | 04-02 | Fix stale chat cache after /new | Cache from previous session leaked into `/new` |
| `9f0e4e6` | 04-03 | Fix scrollback older-load races | Concurrent older-load requests clobbered each other |
| `dd8164d` | 04-12 | Fix scrollback pagination cursor drift | `historyCursor` drifted when messages arrived during scroll-up |
| `b2a891e` | 05-30 | Keep scrollback history cursor aligned | History cursor misaligned after log-anchored cursors refactor |
| `988bc54` | 06-05 | Fix transcript renewal and scrollback range state | Transcript renewal didn't reset scrollback range |

**Mechanism:** Scrollback has 3 cursors (`liveCursor`, `historyCursor`, `backfillToken`) plus a `sessionTailCache` per session. Each bug was a state-synchronization failure between these, especially during:
- Session switching
- `/new` session creation
- Transcript renewal (log rebinding)
- Concurrent poll + older-load

**Current test coverage:** `test_chat_scrollback_source.py` (9 tests, source-pattern checks), `test_chat_transcript_runtime.py` (node VM integration tests for transcript renewal).

**Gap:** No test exercises *concurrent* older-load + live tail append. The race in `9f0e4e6` was fixed by incrementing `backfillToken` to cancel stale fetches, but there's no regression test that simulates interleaved responses.

**Pressure test:**
1. Open a session mid-turn, scroll up to trigger older-load, then send a message while older-load is in-flight.
2. Create `/new`, wait for pending_bind, switch away, switch back—verify no stale messages from previous session appear.
3. Resume a session (transcript renewal) while scrolled up—verify cursor doesn't jump or duplicate.

---

## 2. Cluster: Transcript Identity / Renewal / Binding

**Pattern:** When a session's rollout log changes (resume, `/new`, Pi session switch), the UI must rebind its transcript. This state machine has been patched ≥6 times.

| Commit | Date | Title | Root Cause |
|--------|------|-------|------------|
| `dd4170f` | 04-23 | Fix /new transcript renewal state | `/new` didn't reset `activeTranscriptState` |
| `f391a6b` | 05-24 | Clear active transcript on pending bind | Stale transcript lingered during rebind |
| `eda5c97` | 05-26 | Preserve /new pending transcript detach | `/new` detach was prematurely cancelled |
| `79020e9` | 04-24 | Reject duplicate live transcript resumes | Duplicate resume socket calls caused double bind |
| `988bc54` | 06-05 | Fix transcript renewal and scrollback range state | Renewal didn't flush scrollback cursors |
| `0fe223c` | 03-31 | Fix Pi session log binding and switching | Pi sessions didn't rebind when log_path changed |

**Mechanism:** The `sessionTranscriptSlots` Map tracks `{ state, threadId, logPath, key, epoch, ignoredKey }` per session. `beginTranscriptRenewal()` sets state to `pending_bind` and records an `ignoredKey` so stale `bound` responses for the old log are rejected. A single timing window can break this: if the session-list poll returns an old binding *after* renewal but *before* the new binding arrives.

**Current test coverage:** `test_message_transcript_state.py` (server-side identity), `test_chat_transcript_runtime.py` (node VM tests for the JS renewal logic).

**Gap:** The JS tests don't exercise the full poll loop—they call functions directly. The `ignoredStaleBound` guard was added in `dd4170f` but there's no test for the case where *three* different bindings arrive (old, intermediate, new).

**Pressure test:**
1. Launch a `/new` session, then immediately resume an old thread before the new session's log appears.
2. Kill and restart the broker for a session—verify the UI rebinds to the new log without showing stale messages.
3. Pi: switch active session in the terminal while the UI is open—verify the UI picks up the new log.

---

## 3. Cluster: Idle Detection / Busy State

**Pattern:** The idle/busy signal is the foundation for queue injection and harness timing. It has been redesigned 4+ times.

| Commit | Date | Title | Root Cause |
|--------|------|-------|------------|
| `4294323` | 02-26 | Busy: wait for explicit turn completion | Assistant messages were falsely signaling idle |
| `d1d9911` | 03-05 | Fix idle detection after off-turn assistant activity | Commentary after `task_complete` reopened busy |
| `b7b062a` | 05-13 | Trust transcript idle over stale broker busy | Broker busy bit was stale; log was authoritative |
| `75a08dd` | 04-23 | Fix busy indicator on session switch | Previous session's busy state leaked into new |

**Mechanism:** `_compute_idle_from_log()` in `rollout_log.py` scans the tail of the JSONL log for terminal signals (`task_complete`, `turn_complete`, `turn_aborted`, `error`, `end_turn`). State: `saw_terminal_signal` + `idle = True`, toggled by user messages, tool calls, and response items. The server then overrides the broker's `busy` bit using this log-derived idle state.

**Current test coverage:** `test_idle_heuristics.py` (14 cases covering all turn shapes), `test_sessions_pending_log_idle.py` (7 cases for the override), `test_server_chat_flags.py` (turn flag extraction), `test_broker_busy_state.py`.

**Good coverage but notable gap:** No test for the *timing* of idle detection—specifically, the scenario where the log writes `task_complete` but the agent immediately starts a new turn before the sweep runs. The `a4c0854` (require 10s consecutive idle) fix addressed this for queue injection, but harness injection uses a separate cooldown.

**Pressure test:**
1. Agent completes a turn, immediately starts a new one (e.g., multi-step task)—verify queue doesn't inject mid-turn.
2. Agent emits `task_complete` + off-turn commentary—verify busy indicator goes back to busy then idle.
3. Kill the backend process mid-turn—verify session shows idle (not stuck busy).

---

## 4. Cluster: Queue / Streaming Semantics

**Pattern:** Server-side queue injection has been through 5+ iterations of its inject/drain/idle-guard logic.

| Commit | Date | Title | Root Cause |
|--------|------|-------|------------|
| `6be7734` | 02-23 | Queue: inject only when log idle | Queue injected during tool execution |
| `a4c0854` | 02-25 | Queue: require 10s consecutive idle | Brief idle gaps caused premature injection |
| `30063f9` | 03-10 | Unify queued sends under the server queue | Multiple queue paths (UI + server) conflicted |
| `70e503c` | 04-22 | Fix queued message controls and semantics | UI queue badge and controls desynced from server |
| `e72d030` | 03-11 | Fix queued delete session route match | DELETE on a session with queued items hit wrong route |

**Mechanism:** `_queue_sweep()` runs on a timer, checks `busy=False` + `queue_len=0` from broker, then checks `idle_from_log()`, then requires `queue_idle_since` to be ≥ `QUEUE_IDLE_GRACE_SECONDS` (10s) old. Items are popped from `self._queues[sid]` and sent via `self.send()`.

**Current test coverage:** `test_queue_sweep_idle_guard.py` (4 cases: skip when not idle, grace period, inject when idle, dedup by item_id), `test_server_queue_persistence.py`.

**Gap:** The `queue_sending_item_id` field (used to prevent duplicate sends) is tested only indirectly. No test for the race where `send()` succeeds but the broker dies before acknowledging—does the item get re-queued or lost?

**Pressure test:**
1. Queue 3 messages, kill the broker after the first is sent—verify remaining 2 are still queued.
2. Queue a message, then manually send via terminal—verify the queued message doesn't inject on top.
3. Queue a message with the session busy, then stop the agent—does the queued message eventually drain?

---

## 5. Cluster: Shell Startup / Web Session Launch

**Pattern:** Web-owned sessions launch via a login shell that must pass through shell startup prompts (zsh updater, oh-my-zsh, etc.) before exec'ing the agent. This was a 4-commit saga with a revert.

| Commit | Date | Title | Root Cause |
|--------|------|-------|------------|
| `b248a90` | 04-28 | Handle shell startup prompts in web sessions | zsh/oh-my-zsh prompts blocked agent launch |
| `b9abf4b` | 05-12 | Suppress zsh updater prompts in web sessions | Tried to suppress via env vars |
| `0f2d425` | 05-12 | **Revert** "Suppress zsh updater prompts" | Env var approach broke normal shell startup |
| `613531f` | 05-12 | Advance shell startup prompts before agent exec | PTY marker-based shell readiness detection |
| `e6b486e` | 05-12 | Run web startup without PTY input | Deferred stdin redirect to avoid shell reading from PTY |

**Mechanism:** `_agent_shell_command()` generates a Python-in-shell bootstrap that writes a `SHELL_PRE_EXEC_MARKER` to the PTY, then `_observe_shell_pre_exec_marker()` waits for it before proceeding. The broker restores the PTY slave fd mapping before exec'ing the agent.

**Current test coverage:** `test_broker_fail_closed.py` (tests the command generation, marker observation, login shell argv, teardown). 

**Gap:** The tests mock the PTY and don't exercise a real zsh/bash startup. A Codoxear instance launched in a Docker container with zsh + oh-my-zsh would be the true regression test.

**Pressure test (Docker):**
1. Install zsh + oh-my-zsh in the Docker sandbox, create a web session—verify it launches without hanging.
2. Set `SHELL=/bin/zsh` and create a web session with a non-existent cwd—verify error handling.
3. Rapidly create + delete 3 sessions—verify all broker processes are cleaned up.

---

## 6. Cluster: Rollout Log Discovery / Binding

**Pattern:** Finding the correct rollout log for a session has been rewritten 4 times across different strategies.

| Commit | Date | Title | Strategy |
|--------|------|-------|----------|
| `62d760d` | 02-04 | server: discover rollout log from trace | strace-based |
| `2d2d09d` | 02-13 | Remove strace; track rollout via /proc | /proc fd scanning |
| `ef3723a` | 02-19 | Unify rollout log discovery | Consolidated /proc scanning |
| `557a3a1` | 02-22 | Correlate rollout logs via /proc open fds | Writable-fd preference |
| `c90b587` | 03-18 | Fix macOS rollout log discovery | lsof fallback for macOS |
| `d1a6d6c` | 05-13 | Bind sessions from open rollout logs | Use proc discovery during session registration |

Related fixes:
- `6d9ef3a` Fix web sessions dying under ptrace_scope=1
- `9f22c8b` Fix web sessions dying on /proc fd PermissionError
- `5168919` Fix pending session rollout selection
- `0d2b8ec` Fix JSONL tail reader for truncated UTF-8
- `f79eed8` Fix oversized rollout JSONL reads

**Mechanism:** `proc_find_open_rollout_log()` walks `/proc/<pid>/task/<tid>/children` recursively, reads `/proc/<pid>/fd/*` symlinks + `/proc/<pid>/fdinfo/*` flags to find writable rollout logs matching the session cwd. Subagent logs are filtered via `is_subagent_session_meta()`.

**Current test coverage:** `test_broker_proc_rollout.py` (7 cases with fake /proc trees), `test_session_log.py` (classification, find, Pi variants).

**Gap:** Tests use synthetic /proc trees but don't cover permission errors (ptrace_scope=1 was a real production issue). Also no test for the race where a log file is created but empty when scanned (the `read_session_meta_payload` wait-with-timeout covers this but relies on polling).

**Pressure test (Docker):**
1. Launch a web session, verify log_path appears in session metadata within 5s.
2. Launch a session, then resume it—verify the UI switches to the new rollout log.
3. Set `/proc/sys/kernel/yama/ptrace_scope=1` (if container allows), launch a web session—verify discovery falls back gracefully.

---

## 7. Cluster: File Viewer / Editor Ergonomics

**Pattern:** The file viewer/editor has been fixed 6+ times for races, stale state, and input handling.

| Commit | Date | Title | Root Cause |
|--------|------|-------|------------|
| `6114bad` | 04-07 | Fix file viewer load races | Concurrent file opens clobbered each other |
| `b6a59cc` | 03-31 | Fix file viewer editor and stale assets | Monaco loaded stale cached assets |
| `6d53562` | 04-24 | Disable Monaco textarea mirroring | Accessibility mirror div caused input lag |
| `b0a1953` | 05-12 | Own file editor delete events | Backspace/Delete not working in Monaco on touch |
| `d2a6e16` | 03-17 | Fix file picker candidate contamination | Chat-extracted candidates leaked across sessions |
| `f7a7e36` | 04-03 | Scope file picker state to sessions | File selections weren't per-session |

**Mechanism:** `fileOpenRequestId` is incremented on each file open; stale responses with old IDs are discarded. Monaco editor is loaded via a web worker with `accessibilitySupport: "off"`. Delete key events are intercepted and forwarded as Monaco commands.

**Current test coverage:** `test_file_viewer_source.py` (source pattern checks for Monaco config, open-request race guard, touch controls), `test_file_list.py`, `test_file_picker_session_state.py`.

**Gap:** No integration test that actually loads Monaco in a browser context. The file-open race guard relies on `fileOpenRequestId` matching, but the test only checks the JS source for the pattern—it doesn't simulate the interleaved async sequence.

**Pressure test:**
1. Rapidly click 5 different files in the picker—verify the final displayed file is the last clicked.
2. Open a large file (>1MB), then immediately open a small file—verify no content from the large file appears.
3. Edit a file, switch sessions, switch back—verify edit state is preserved.
4. On a touch device (or touch emulation), verify Backspace and Delete work in the editor.

---

## 8. Cluster: Harness / Unattended Mode

**Pattern:** The harness auto-injects prompts into idle sessions, with deduplication per thread.

| Commit | Date | Title | Root Cause |
|--------|------|-------|------------|
| `4dfc8b8` | 03-01 | Keep harness thread alive on session timeouts | `get_state()` timeout killed the sweep loop |
| `42116b0` | 02-25 | Keep harness count editable while typing | UI clobbered draft values on poll updates |
| `b626c02` | 02-20 | Tune harness controls and fast-tier launch | Injection timing too aggressive |

**Mechanism:** `_harness_sweep()` iterates sessions with `enabled: True`, checks `get_state()` (busy/queue_len), checks `_last_chat_role_ts_from_tail()` for cooldown, then calls `send()`. Per-thread dedup via `_harness_last_injected_scope["thread:<id>"]`.

**Current test coverage:** `test_harness_sweep.py` (5 cases: thread dedup, distinct threads, timeouts, cooldown, injection count), `test_harness_input_source.py` (UI source patterns).

**Gap:** No test for `remaining_injections` going negative (off-by-one). No test for the case where the same thread appears under 3 different session IDs (only 2 tested).

**Pressure test:**
1. Enable harness on 2 sessions sharing the same thread—verify only 1 injection.
2. Set `remaining_injections=1`, let it fire, then re-enable—verify counter behavior.
3. Kill the broker socket during a harness sweep—verify the sweep continues to other sessions.

---

## 9. Cluster: Message Deduplication

**Pattern:** Assistant messages were being duplicated in the chat UI due to overlapping log reads.

| Commit | Date | Title | Root Cause |
|--------|------|-------|------------|
| `bf20fc8` | 02-07 | Server: dedupe assistant duplicates by timestamp bucket | Timestamp-based dedup was too precise |
| `8db98f6` | 02-07 | Server: dedupe assistant stretch duplicates | Stretch-based dedup for assistant blocks |
| `b350b29` | 02-05 | UI: dedupe adjacent assistant messages | Client-side dedup as fallback |
| `ccc548e` | 03-26 | Deduplicate final response notifications | Notification dedup for voice push |
| `042d33b` | 03-26 | Deduplicate duplicate final-response notifications | Second pass at notification dedup |

**Mechanism:** Server-side dedup uses a per-session `seen` set keyed by `(role, timestamp_ms, text)`. For assistant messages, the `assistant_stretch_seen` set additionally dedupes within the current assistant block (reset on user message). Client-side dedup in `renderEvents()` skips adjacent identical assistant texts.

**Current test coverage:** Partially tested via `test_message_index.py` (event dedup), `test_server_chat_flags.py`.

**Gap:** No direct unit test for the `assistant_stretch_seen` dedup path or the `042d33b` notification-level dedup. The double-dedup commit (`042d33b` → `ccc548e`) suggests the first fix was incomplete.

**Pressure test:**
1. Agent emits the same response text in two consecutive `response_item` events with timestamps 0.5s apart—verify only one appears in chat.
2. Long assistant response with streaming—verify no duplicate chunks.
3. Rapidly poll `/api/sessions/<id>/messages` while agent is responding—verify no duplicate events in successive responses.

---

## 10. Cluster: JSONL Tail Reading

**Pattern:** Two distinct bugs in the JSONL reader that are easily reproducible with large/concurrent logs.

| Commit | Date | Title | Root Cause |
|--------|------|-------|------------|
| `0d2b8ec` | 02-20 | Fix JSONL tail reader for truncated UTF-8 | Reader advanced past incomplete multibyte sequences |
| `f79eed8` | 04-07 | Fix oversized rollout JSONL reads | `max_bytes` read didn't find a newline; entire record missed |

**Mechanism:** `read_jsonl_from_offset()` now:
1. Reads `max_bytes` from the offset
2. If no `\n` found in the initial read, continues reading in 64KB chunks until one is found
3. Only parses up to the last `\n` to avoid truncated UTF-8 or partial JSON

**Current test coverage:** `test_read_jsonl_from_offset.py` (tests for basic read, truncated UTF-8, oversized single line).

**Pressure test:**
1. Write a JSONL file with a single 2MB line, read with `max_bytes=65536`—verify it's still parsed.
2. Append to a JSONL file from another process while reading—verify no partial JSON errors.
3. Write a line containing emoji (4-byte UTF-8), split at exactly the 3rd byte boundary—verify no decode error.

---

## 11. Isolated But Notable Bugs

| Commit | Date | Title | Notes |
|--------|------|-------|-------|
| `13481da` | 03-07 | Fix tmux input for web-owned brokers | `send-keys` target was wrong for tmux transport |
| `37c1c84` | 04-28 | Handle tmux no-server launch errors | `tmux new-session` fails if no tmux server running |
| `a9b3747` | 03-07 | Handle broker socket peer disconnects | `recv()` returned empty bytes, causing JSON parse error |
| `eaafae6` | 05-15 | Disable Codex startup update checks | `codex --skip-update-check` needed for headless launch |
| `f3acfef` | 06-03 | Disable Codex goals for web sessions | Goals prompt blocked web session startup |
| `3cb6bea` | 05-20 | Make web login cookies permanent | Cookies expired on browser close |
| `b003da6` | 06-12 | fix: repair Docker baseline failures | Legacy cwd file-history state survived session deletion |

---

## Summary: Top Regression Risk Areas

| Area | Bug Count | Last Fix | Risk Level |
|------|-----------|----------|------------|
| Chat scrollback / cursors | 7 | 06-05 | 🔴 High — still actively being fixed |
| Transcript renewal / binding | 6 | 06-05 | 🔴 High — tightly coupled to scrollback |
| Idle detection | 4 | 05-13 | 🟡 Medium — well-tested but timing-sensitive |
| Queue injection | 5 | 04-22 | 🟡 Medium — grace period helps, but failure modes are subtle |
| Shell startup | 5 (incl. revert) | 05-12 | 🟡 Medium — real-shell testing needed |
| Rollout log discovery | 7 | 05-13 | 🟡 Medium — /proc-dependent, platform-specific |
| File viewer / editor | 6 | 05-12 | 🟡 Medium — race guards in place but untested end-to-end |
| Harness | 3 | 03-01 | 🟢 Low — well-tested, stable since March |
| Message dedup | 5 | 03-26 | 🟡 Medium — double-fix pattern suggests fragility |
| JSONL reader | 2 | 04-07 | 🟢 Low — mechanically fixed, well-tested |

---

## Recommended Docker Sandbox Pressure Tests (Priority Order)

### P0: Session lifecycle stress
```
1. Create /new session → immediately switch away → switch back → verify clean transcript
2. Create /new session → send a message → while agent running, create another /new
3. Resume a session → immediately create /new → verify correct transcript binding
4. Kill broker process → verify session shows idle, not stuck-busy
```

### P1: Queue injection timing
```
1. Queue a message → wait for agent to finish → verify injection after 10s grace
2. Queue a message while agent is mid-turn → verify no premature injection
3. Queue 3 messages → kill broker after first is sent → verify 2 remain queued
```

### P2: Scrollback under load
```
1. Session with 200+ messages → scroll to top → send new message → verify no cursor drift
2. Load older messages → agent emits response simultaneously → verify no duplicates
3. Switch sessions rapidly (5x in 2 seconds) → verify no stale content
```

### P3: Shell startup robustness
```
1. Install zsh + oh-my-zsh in Docker → create web session → verify launch completes
2. Create web session with non-existent cwd → verify error is shown (not hang)
3. Create 3 web sessions rapidly → delete all 3 → verify no orphan processes
```

### P4: File viewer races
```
1. Click 5 files rapidly → verify final content matches last click
2. Open 1MB+ file → immediately open a 10-line file → verify correct content
3. Edit a file → switch session → switch back → verify edit preserved
```
