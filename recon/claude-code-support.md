# Code Context — Claude Code (`cc`) Backend Support

## Files Retrieved

1. `codoxear/agent_backend.py` (full, ~70 lines) — Backend abstraction: `AgentBackend` dataclass, `_BACKENDS` registry, `normalize_agent_backend()`, `get_agent_backend()`, `infer_agent_backend_from_log_path()`.
2. `codoxear/broker.py` (full, 1751 lines) — PTY wrapper, log discovery, busy/idle state machine, socket server, session metadata sidecar. Backend-conditional logic at: lines 42 (AGENT_BACKEND), 130–180 (Pi session args), 370–530 (Pi log discovery), 590–640 (busy state from Pi messages), all of `_apply_rollout_obj_to_state`.
3. `codoxear/rollout_log.py` (full, ~900 lines) — Chat event extraction, delivery messages, idle heuristics (`_compute_idle_from_log`), token extraction, all operate on three record-type families: `event_msg`, `response_item`, `message` (Pi).
4. `codoxear/pi_log.py` (full, ~380 lines) — Pi-specific extractors: `pi_user_text`, `pi_assistant_text`, `pi_assistant_is_final_turn_end`, `pi_token_update`, `read_pi_session_header`, `read_pi_run_settings`.
5. `codoxear/util.py` (full, ~500 lines) — Session meta reading, session log iteration, proc-based open-file discovery, subagent detection.
6. `codoxear/server.py` lines 155–175 (config paths), 2258–2417 (launch defaults), 2630–2710 (Session dataclass), 4686–4800 (spawn_web_session), 6680–6800 (POST /api/sessions handler).
7. `codoxear/sessiond.py` (full, ~350 lines) — Headless session helper. Same pattern: backend from env, pty.fork, log discovery, socket server.
8. `tests/test_launch_defaults.py` — Tests for `_read_codex_launch_defaults`, `_read_pi_launch_defaults`, `_read_new_session_defaults`.
9. `tests/test_broker_busy_state.py` — Tests for `_apply_rollout_obj_to_state`, `_should_clear_busy_state`, including Pi message types.
10. `tests/test_idle_heuristics.py` — Tests for `_compute_idle_from_log` across codex and Pi record types.
11. `tests/test_session_log.py` — Session log classification, discovery, `read_session_meta_payload`.
12. `tests/test_session_resume.py` — Resume candidate listing, `spawn_web_session` for codex and pi backends.

## Key Code

### Backend Registry (`agent_backend.py`)

```python
@dataclass(frozen=True)
class AgentBackend:
    name: str              # "codex" | "pi"
    bin_env_var: str        # "CODEX_BIN" | "PI_BIN"
    home_env_var: str       # "CODEX_HOME" | "PI_HOME"
    default_bin: str        # "codex" | "pi"
    default_home_dirname: str  # ".codex" | ".pi"
    sessions_relpath: tuple[str, ...]  # ("sessions",) | ("agent", "sessions")

_BACKENDS: dict[str, AgentBackend] = {
    "codex": CODEX_BACKEND,
    "pi": PI_BACKEND,
}
```

### Log Path Inference (`agent_backend.py`)

```python
def infer_agent_backend_from_log_path(path: Path) -> str | None:
    if name.startswith("rollout-") and name.endswith(".jsonl"):
        return "codex"
    if "/.pi/agent/sessions/" in path_text and name.endswith(".jsonl"):
        return "pi"
    return None
```

### Session Meta Reading (`util.py`)

- **Codex**: scans for `{"type": "session_meta", "payload": {...}}` record
- **Pi**: first JSONL record `{"type": "session", "id": "...", "cwd": "...", ...}`

### Busy/Idle State Machine (broker.py `_apply_rollout_obj_to_state`)

Three record families:
- `event_msg` (codex native): `user_message`, `agent_message`, `agent_reasoning`, `task_complete`, `turn_complete`, `error`
- `response_item` (codex native): `function_call`, `function_call_output`, `reasoning`, assistant `message` with `content[].output_text`
- `message` (Pi native): role-based: `user`→start turn, `assistant` with `text`/`toolCall`/`thinking`, `toolResult`

### Log Discovery

- **Codex**: `/proc/PID/fd` scan for open files matching `rollout-*.jsonl` under sessions dir
- **Pi**: extension bridge writes a marker file (`pi-active-sessions/broker-PID.json`) pointing to the active session log

### Session Log Iteration (`util.py`)

- **Codex**: `rglob("rollout-*.jsonl")` under sessions dir
- **Pi**: `rglob("*.jsonl")` under sessions dir, validated against sessions_dir containment

### Chat Event Extraction (`rollout_log.py`)

`_single_chat_event` and `_extract_chat_events` handle all three record families, producing normalized `{role, text, message_class, message_id, ts}` events.

### Launch Defaults (server.py)

`_read_new_session_defaults()` returns:
```python
{
    "default_backend": "codex",
    "backends": {
        "codex": { ... codex config ... },
        "pi": { ... pi config ... },
    }
}
```

### Spawn Web Session (server.py)

Backend-conditional argument construction:
- Codex: `-c`, `--model`, `--disable goals`, `--dangerously-bypass-approvals-and-sandbox`, `resume <id>`
- Pi: `--provider`, `--model`, `--thinking`, `--session <log_path>`

Environment: `CODEX_WEB_AGENT_BACKEND`, `CODEX_WEB_MODEL*`, backend-specific `*_HOME`.

## Architecture

```
                    ┌──────────────────┐
                    │  agent_backend.py │  Backend registry & path config
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
         broker.py      sessiond.py    server.py
         PTY wrapper    Headless       HTTP server
         + log watcher  session        + session mgr
              │              │              │
              ▼              ▼              ▼
         ┌─────────┐  ┌──────────┐  ┌──────────┐
         │util.py  │  │util.py   │  │rollout_  │
         │proc scan│  │proc scan │  │log.py    │
         │meta read│  │meta read │  │chat parse│
         └─────────┘  └──────────┘  │pi_log.py │
                                    └──────────┘
```

**Data flow**: Backend CLI writes JSONL → broker discovers log via /proc or marker file → broker reads JSONL for busy state → server reads JSONL for chat events/idle → UI polls server API.

**Backend dispatch pattern**: Almost every backend-dependent behavior uses `if backend == "codex": ... else: ...` (implicit pi). No abstraction layer — the `AgentBackend` dataclass handles only paths/bins.

## Claude Code Log Format (from `~/.claude/`)

### Path structure
```
~/.claude/
├── settings.json          # model, effortLevel, permissions
├── settings.local.json    # per-machine permissions
├── history.jsonl           # session index (sessionId, project, display, timestamp)
└── projects/
    └── -home-yiwen-codex-web/     # project dir name = cwd path with / → -
        ├── <uuid>.jsonl            # main session log
        └── <uuid>/
            └── subagents/
                └── agent-<hash>.jsonl  # subagent logs
```

### JSONL Record Types

| type | role/subtype | key fields | maps to codoxear event |
|------|-------------|------------|----------------------|
| `user` | `message.role=user` | `uuid`, `parentUuid`, `sessionId`, `timestamp`, `cwd`, `promptId`, `message.content` (string or `[{type:text,text:...}]` or `[{type:tool_result,...}]`) | `{role:"user", text:...}` |
| `assistant` | `message.role=assistant` | `uuid`, `parentUuid`, `sessionId`, `timestamp`, `message.content` (list of `{type:text,text:...}`, `{type:thinking,...}`, `{type:tool_use,id:...,name:...,input:...}`), `message.stop_reason` (`end_turn`, `tool_use`, `stop_sequence`), `message.usage`, `message.model` | `{role:"assistant", text:...}` |
| `system` | `turn_duration` | `durationMs`, `timestamp` | turn-end signal |
| `system` | `stop_hook_summary` | `hookCount`, `preventedContinuation` | informational |
| `progress` | - | `data.type` (e.g. `hook_progress`) | activity signal |
| `file-history-snapshot` | - | snapshot data | ignored for chat |
| `queue-operation` | - | `operation` (`enqueue`/`dequeue`), `content` | informational |
| `last-prompt` | - | `lastPrompt`, `leafUuid` | metadata |
| `permission-mode` / `mode` | - | `permissionMode` / `mode` | metadata |
| `attachment` | - | file attachment data | informational |

### Key structural differences from codex/pi

1. **No `session_meta` record.** Session ID is in every record's `sessionId` field. The first user record acts as implicit session header.
2. **No `event_msg` wrapper.** Records use top-level `type: "user" | "assistant" | "system"` (not `event_msg` with nested `payload.type`).
3. **No `response_item` wrapper.** The `assistant` type directly contains the API response `message` object.
4. **Tool use is inline.** `assistant` content includes `{type: "tool_use", name, id, input}`. Tool results appear as `user` records with `message.content: [{type: "tool_result", tool_use_id, content}]`.
5. **`stop_reason` is on `message`**, not the outer record. Values: `end_turn`, `tool_use`, `stop_sequence`.
6. **`system/turn_duration`** serves as the explicit turn-end signal.
7. **Session paths use project-dir naming** (`-home-yiwen-codex-web`), not date-based dirs like codex.
8. **Subagent logs** are in `<session-uuid>/subagents/agent-<hash>.jsonl`, not separate rollout files.
9. **`isSidechain`** field indicates branched conversation paths (unique to CC).
10. **`usage`** includes `input_tokens`, `output_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`, plus `service_tier`, `speed`.

### CC CLI Flags (from `claude --help`)

| Flag | Description |
|------|-------------|
| `--model <model>` | Model alias (`opus`, `sonnet`, `fable`) or full name |
| `--effort <level>` | `low`, `medium`, `high`, `xhigh`, `max` |
| `-c, --continue` | Resume most recent session in cwd |
| `--resume <id>` | Resume specific session by ID |
| `--fork-session` | Create new session ID when resuming |
| `--dangerously-skip-permissions` | Bypass permission checks |
| `-p, --print` | Non-interactive output mode |
| `--output-format <format>` | `text`, `json`, `stream-json` |
| `--name <name>` | Display name for session |
| `--add-dir` | Additional directories for tool access |
| `--no-session-persistence` | Don't save session to disk |
| `--settings <json>` | Override settings |
| `--bare` | Minimal mode, skip hooks/LSP/etc |

## Implementation Plan

### 1. Backend Registration (`agent_backend.py`)

Add `CC_BACKEND`:
```python
CC_BACKEND = AgentBackend(
    name="cc",
    bin_env_var="CC_BIN",           # or "CLAUDE_CODE_BIN"
    home_env_var="CLAUDE_CODE_HOME", # or "CC_HOME"
    default_bin="claude",
    default_home_dirname=".claude",
    sessions_relpath=("projects",),  # NOTE: CC uses project-scoped session dirs
)
```

Update `_BACKENDS`, `normalize_agent_backend()`.

**Open question**: `sessions_relpath=("projects",)` is correct for session *iteration* but CC organizes sessions by project-path-derived directory names. The existing `iter_session_logs` with `rglob("*.jsonl")` would work but must filter out `subagents/` directories and non-session files (`history.jsonl` etc).

### 2. Log Path Inference (`agent_backend.py`)

Add to `infer_agent_backend_from_log_path`:
```python
if "/.claude/projects/" in path_text and name.endswith(".jsonl"):
    return "cc"
```

Must also exclude `subagents/` path components.

### 3. Session Log Iteration (`util.py`)

Add CC session validation:
```python
def _is_cc_session_log_path(path: Path, *, sessions_dir: Path | None = None) -> bool:
    if path.suffix != ".jsonl":
        return False
    # Exclude subagent logs
    if "/subagents/" in str(path):
        return False
    # Exclude non-project files (history.jsonl etc)
    ...
```

Update `iter_session_logs`, `proc_open_rollout_logs_for_backend`.

### 4. Session Meta Reading (`util.py`)

CC has no `session_meta` record. Need a new function:
```python
def read_cc_session_header(path: Path) -> dict[str, Any] | None:
    """Read first user/assistant record to extract sessionId, cwd, model."""
    # First record with sessionId field
```

Update `read_session_meta_payload` to handle `cc` backend.

The CC session "meta" contains:
- `sessionId` (UUID) — present on every record
- `cwd` — present on every record
- `version` — CC version
- `gitBranch` — current branch
- No `source.subagent` equivalent, but subagent logs live in `subagents/` subdirs

**Subagent detection**: Check if log path contains `/subagents/` rather than reading a `source.subagent` payload field.

### 5. CC Log Normalization (`cc_log.py` — new file)

New module analogous to `pi_log.py`:

```python
def cc_user_text(obj: dict) -> str | None:
    """Extract user message text from CC log record."""
    if obj.get("type") != "user": return None
    msg = obj.get("message", {})
    if msg.get("role") != "user": return None
    content = msg.get("content")
    # Handle string content or list of {type:"text",text:...} or [{type:"tool_result",...}]
    # Filter out tool_result parts, isMeta records, command records
    ...

def cc_assistant_text(obj: dict) -> str | None:
    """Extract assistant text from CC log record."""
    if obj.get("type") != "assistant": return None
    msg = obj.get("message", {})
    # Collect all {type:"text"} parts from content
    ...

def cc_assistant_is_final_turn_end(obj: dict) -> bool:
    """True if this assistant message ends the turn (stop_reason=end_turn, no tool_use)."""
    ...

def cc_assistant_tool_use_count(obj: dict) -> int:
    """Count tool_use parts in assistant content."""
    ...

def cc_assistant_thinking_count(obj: dict) -> int:
    """Count thinking parts in assistant content."""
    ...

def cc_message_role(obj: dict) -> str | None:
    """Return normalized role: 'user', 'assistant', 'toolResult', 'system'."""
    ...

def cc_token_update(obj: dict) -> dict | None:
    """Extract context usage from CC usage data."""
    # CC doesn't have model_context_window in usage — need external lookup
    ...

def cc_is_turn_end(obj: dict) -> bool:
    """True if this record signals a turn has ended."""
    # system/turn_duration record
    ...

def read_cc_session_id(path: Path) -> str | None:
    """Read sessionId from first record."""
    ...

def read_cc_run_settings(path: Path) -> tuple[str|None, str|None, str|None]:
    """Extract (provider, model, effort) from CC session log."""
    # CC doesn't have provider concept in logs — model is on assistant records
    # Effort level is in settings.json, not in session log
    ...
```

### 6. Chat Event Extraction (`rollout_log.py`)

Add CC record handling to `_single_chat_event` and `_extract_chat_events`:

```python
if typ == "user":
    text = cc_user_text(obj)
    if text: return {role: "user", text, ts}

if typ == "assistant":
    text = cc_assistant_text(obj)
    if text:
        message_class = "final_response" if cc_assistant_is_final_turn_end(obj) else "narration"
        return {role: "assistant", text, message_class, message_id, ts}
    # Also handle tool_use count for metadata

if typ == "system" and obj.get("subtype") == "turn_duration":
    # Turn-end signal (similar to task_complete/turn_complete)
```

### 7. Busy/Idle Detection

#### Broker (`_apply_rollout_obj_to_state`)

Add CC record handling:

```python
if typ == "user":
    user_text = cc_user_text(obj)
    if user_text:
        # Start turn (same as codex user_message)
        st.pending_calls.clear(); st.busy = True; st.turn_open = True; ...
        return
    # Tool result records keep turn busy
    if _cc_is_tool_result(obj):
        st.busy = True; st.last_turn_activity_ts = now_ts; return

if typ == "assistant":
    if cc_assistant_tool_use_count(obj) > 0 or cc_assistant_thinking_count(obj) > 0:
        _reopen_turn_on_activity(st)
        st.busy = True; st.last_turn_activity_ts = now_ts; return
    if cc_assistant_text(obj) and cc_assistant_is_final_turn_end(obj):
        _close_turn_state(st); return
    if cc_assistant_text(obj):
        st.turn_has_completion_candidate = True
        st.busy = True; st.last_turn_activity_ts = now_ts; return

if typ == "system" and obj.get("subtype") == "turn_duration":
    _close_turn_state(st); return
```

#### Server idle heuristic (`_compute_idle_from_log`)

Add parallel CC handling in the scan loop.

### 8. Log Discovery in Broker

CC doesn't have Pi's extension bridge mechanism. Options:

**Option A (proc scan, like Codex):** Scan `/proc/PID/fd` for open `.jsonl` files under `~/.claude/projects/`. This is the most consistent approach.

**Option B (CC-specific session marker):** CC may write a file indicating the active session. Unknown — needs investigation.

**Recommendation**: Start with proc scan (Option A). The existing `proc_open_rollout_logs_for_backend` and `proc_find_open_rollout_log` already support per-backend path validation. Just need `_is_cc_session_log_path` to filter correctly.

**Challenge**: CC writes *many* files — the proc scan must distinguish the main session JSONL from subagent logs, snapshot files, etc. The `subagents/` path filter helps, but verify there aren't other non-session `.jsonl` files under `projects/`.

### 9. Launch Defaults (`server.py`)

Add `_read_cc_launch_defaults()`:
```python
def _read_cc_launch_defaults() -> dict[str, Any]:
    # Read ~/.claude/settings.json for model, effortLevel
    # CC doesn't have a provider concept — always Anthropic
    # Effort levels: low, medium, high, xhigh, max
    ...
```

Update `_read_new_session_defaults()` to include `cc` backend.

### 10. Session Spawn (`server.py`)

Add CC backend branch in `spawn_web_session`:
```python
elif backend_name == "cc":
    codex_args = ["--dangerously-skip-permissions"]
    if model: codex_args.extend(["--model", model])
    if reasoning_effort: codex_args.extend(["--effort", reasoning_effort])
    # No provider/service_tier for CC
    # Resume: codex_args.extend(["--resume", resume_id])
```

Update environment setup:
```python
elif backend_name == "cc":
    env.setdefault("CLAUDE_CODE_HOME", str(CC_HOME))
    env.pop("CODEX_HOME", None)
    env.pop("PI_HOME", None)
```

### 11. Server Session Discovery

Update `_discover_existing` and `refresh_session_meta` for cc backend awareness.

### 12. UI Updates (`static/index.html`, `app.js`, `app.css`)

- Add CC backend tab in new-session modal
- CC logo/icon for sidebar
- Effort level choices: `low`, `medium`, `high`, `xhigh`, `max`
- No provider selector (always Anthropic)
- No service_tier selector

### 13. PTY/Broker Considerations

CC's TUI is a Node.js application with a rich terminal UI. Key considerations:

- **Interrupt hint**: CC likely uses different PTY text for "working" indicators. Need to discover the pattern (e.g., `esc to interrupt` may or may not be present). May need `_DETACH_TRIGGER_PHRASES` and `_update_busy_from_pty_text` updates.
- **Session switch**: CC's `--continue`/`--resume` don't trigger the same "To continue this session, run..." output. May not need detach trigger for CC.
- **Bracketed paste**: CC's Node.js terminal likely supports bracketed paste for injection.
- **Shell startup**: Login shell wrapping should work the same.

### 14. Token/Context Usage

CC's `usage` object includes `input_tokens`, `output_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`, but **no `model_context_window`**. To compute context percentage:
- Look up model context windows from CC's model info (may need to maintain a static table or query CC)
- Or skip context percentage for CC initially

### 15. Resume Support

CC uses:
- `--resume <session-id>` to resume a specific session
- `-c` / `--continue` to resume the most recent session

For resume candidate listing, scan `~/.claude/projects/<project-dir>/*.jsonl` and extract `sessionId` from the first record.

## Tests Required

### Unit Tests (new)

1. **`test_cc_log.py`** — CC log parsing:
   - `cc_user_text` extracts text from string content and list content
   - `cc_user_text` returns None for tool_result-only records
   - `cc_user_text` returns None for isMeta user records
   - `cc_assistant_text` concatenates text parts
   - `cc_assistant_is_final_turn_end` returns True when `stop_reason=end_turn` and no tool_use
   - `cc_assistant_is_final_turn_end` returns False when `stop_reason=tool_use`
   - `cc_assistant_tool_use_count` counts tool_use parts
   - `cc_assistant_thinking_count` counts thinking parts

2. **`test_cc_backend_registration.py`**:
   - `normalize_agent_backend("cc")` succeeds
   - `get_agent_backend("cc").default_bin == "claude"`
   - `infer_agent_backend_from_log_path` returns "cc" for CC paths
   - `infer_agent_backend_from_log_path` returns None for subagent paths

3. **`test_cc_busy_state.py`** — Busy/idle for CC records:
   - User message starts turn
   - Tool_use in assistant keeps turn busy
   - Final assistant message with `stop_reason=end_turn` (no tool_use) closes turn
   - `system/turn_duration` closes turn
   - Tool result keeps turn busy
   - Thinking content keeps turn busy

4. **`test_cc_idle_heuristics.py`** — Idle detection for CC logs:
   - Fresh CC session (no messages) is idle
   - After user message, before assistant reply → busy
   - After `stop_reason=end_turn` assistant → idle
   - After `system/turn_duration` → idle
   - After tool_use → busy

5. **`test_cc_chat_events.py`** — Chat event extraction:
   - User text extracted correctly
   - Assistant text with final_response class
   - Tool_use increments tool count
   - Thinking increments thinking count
   - isMeta user records are skipped

6. **`test_cc_launch_defaults.py`**:
   - Read model/effort from `~/.claude/settings.json`
   - Default to `opus` model and `medium` effort when settings missing
   - Effort levels include `max`

7. **`test_cc_session_resume.py`**:
   - `spawn_web_session` constructs correct CC CLI args
   - Resume uses `--resume <id>`
   - Environment sets `CLAUDE_CODE_HOME`

### Integration/Smoke Tests

8. **`test_cc_session_meta.py`**:
   - `read_session_meta_payload` for CC backend reads `sessionId` from first user record
   - Subagent detection via path pattern (`/subagents/`)

### Existing Test Updates

9. **`test_launch_defaults.py`**: Add `test_read_new_session_defaults_includes_cc_backend`
10. **`test_idle_heuristics.py`**: Add CC log variants alongside existing codex/pi tests
11. **`test_broker_busy_state.py`**: Add CC record type tests
12. **`test_session_log.py`**: Add CC log classification and iteration tests
13. **`test_session_resume.py`**: Add `test_spawn_web_session_passes_cc_backend_to_broker`

## Risks & Open Questions

### High Risk

1. **CC log format stability.** Claude Code is a Node.js app under active development. The JSONL format (`user`/`assistant`/`system` types) may change. The format is not formally documented. **Mitigation**: Pin to known-good record shapes and fail gracefully on unknown types.

2. **No session_meta equivalent.** CC embeds session metadata (sessionId, cwd, model) on every record rather than a dedicated header. This means `read_session_meta_payload` must scan for the first meaningful record. **Risk**: Empty or corrupt session files.

3. **User message filtering.** CC user records include tool results, meta records (`isMeta: true`), command records (`<command-name>...</command-name>`). The chat extraction must distinguish genuine user prompts from internal tool-result routing. **Mitigation**: Filter on `isMeta`, `toolUseResult` presence, and XML command patterns.

4. **Multiple assistant records per turn.** CC emits one `assistant` record per API streaming chunk (thinking, text, tool_use are split into separate records with the same `message.id` but different UUIDs). The chat extractor must merge or handle these correctly to avoid duplicate display.

### Medium Risk

5. **Context window lookup.** CC doesn't include `model_context_window` in usage data. Need external table or query. **Mitigation**: Start without context percentage display for CC, add later.

6. **Proc scan false positives.** CC may have multiple `.jsonl` files open simultaneously (session, subagent, history). The discovery logic must carefully filter. **Mitigation**: Only match files under `projects/` that are not in `subagents/` directories and are not `history.jsonl`.

7. **Sidechain records.** CC has `isSidechain: true` for branched conversations. These should probably be excluded from chat event extraction. **Open question**: Does the main conversation always have `isSidechain: false`?

8. **CC's `--dangerously-skip-permissions`** may behave differently from Codex's `--dangerously-bypass-approvals-and-sandbox`. Need to verify the exact flag.

### Low Risk

9. **No service_tier or preferred_auth_method for CC.** CC always uses Anthropic API directly. These fields should be `None` for CC backend.

10. **`CLAUDE_CODE_HOME` vs `CC_HOME`.** CC uses `CLAUDE_CONFIG_DIR` environment variable to override the default `~/.claude`. Need to verify the exact env var name. **Default**: `~/.claude`.

11. **PTY text patterns.** CC's terminal output may use different escape sequences for the "working" indicator. The `_update_busy_from_pty_text` function may need CC-specific patterns.

12. **Session naming.** CC has `-n, --name` for session display names. Should the UI expose this?

### Decision Needed

- **Backend name**: `"cc"` (short) vs `"claude-code"` (descriptive)?
- **Environment variable prefix**: `CC_BIN`/`CC_HOME` vs `CLAUDE_CODE_BIN`/`CLAUDE_CODE_HOME`?
- **Session discovery**: proc scan only, or also support a CC-specific marker mechanism?
- **Minimum CC version**: What's the oldest supported Claude Code version?
- **Permission mode**: Default to `--dangerously-skip-permissions` for web-owned sessions?

## Start Here

Open **`codoxear/agent_backend.py`** first. This is the entry point for adding any new backend. Add `CC_BACKEND` to `_BACKENDS`, then follow the compile/import errors to find every file that needs updating. The `if backend == "codex": ... else: ...` pattern throughout the codebase will naturally guide you to every decision point.

After that, create **`codoxear/cc_log.py`** as the CC-specific log parser (analogous to `pi_log.py`), writing it test-first with sample CC JSONL records from `~/.claude/projects/`.
