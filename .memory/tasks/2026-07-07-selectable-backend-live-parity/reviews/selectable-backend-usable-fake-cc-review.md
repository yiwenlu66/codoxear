# Adversarial Review: Selectable-backend usable fake-Claude proof

**Commit:** `b59cf4c` on `recovery/product-gaps`
**Artifact directory:** `.memory/tasks/2026-07-07-selectable-backend-live-parity/browser-artifacts/backend-parity-fake-cc-19397/`
**Reviewer verdict:** **ACCEPT**
**Blockers:** 0
**Nonblocking findings:** 6

---

## Assertion falsification results

### (a) Browser path used Claude tab / model sonnet / reasoning max / direct launch

**Result: CONFIRMED**

`eval-before-start.json` captures DOM state immediately before "Start session" click:

| Field | Value | Source |
|---|---|---|
| Active backend tab | `Claude` (active=true) | `eval-before-start.json → result.backend[2]` |
| Model | `sonnet` | `eval-before-start.json → result.model` |
| Reasoning | `max` | `eval-before-start.json → result.reasoning` |
| tmux | `false` | `eval-before-start.json → result.tmux` |

Corroborating snapshots: `snapshot-claude-tab.txt` shows Claude/Codex/Pi tabs with Model combobox (not Provider/model, confirming cc-specific layout). `snapshot-before-start-after-fill.txt` shows model combobox expanded with `sonnet` selected. `snapshot-reasoning-menu.txt` shows reasoning dropdown with low/medium/high/xhigh/max options.

Browser driver notes (`browser-driver-notes.md`) record the click sequence: Claude tab → fill cwd → fill name → fill model sonnet → open reasoning → select max → uncheck tmux → Start.

The temporal sequence is further corroborated by `eval-claude-controls.json` (captured after Claude tab click but before filling), which shows Claude tab active (`class: "agentBackendTab active"`) with model still at default and reasoning at "medium"—consistent with the tab having been clicked before fields were filled.

### (b) Real bound cc session row, not failed-launch / synthetic sidecar

**Result: CONFIRMED**

`api-sessions-after-bind.json` session row:

| Field | Value | Significance |
|---|---|---|
| `session_id` | `broker-190` | Broker PID-based ID |
| `thread_id` | `69121efb-477b-47da-a1e2-bd10cb85aafd` | UUID from fake-claude |
| `agent_backend` | `cc` | Claude Code backend |
| `pid` | `192` | Live fake-claude PID |
| `broker_pid` | `190` | Live broker PID |
| `log_path` | `/home/tester/.claude/projects/-workspace/69121efb-...jsonl` | Real JSONL path |
| `busy` | `false` | Idle, not stuck |
| `launch_state` | absent | Not a failed-launch row |

Cross-reference with container evidence:
- `container/processes.txt`: PID 190 = broker, PID 192 = fake-claude child. PPID chain 192→190→1 confirms broker-spawned.
- `container/sidecars.jsonl.txt`: sidecar JSON has matching `session_id`, `log_path`, `sock_path`, `agent_backend=cc`, `control_capabilities.sync_send=true`.
- `container/session_launches.jsonl`: two ledger rows for launch `launch-1783411913492-ee057d12`: `starting` then `broker_spawned`, with `transport=direct`, `model=sonnet`, `reasoning_effort=max`.
- `container/tmux-list-sessions.txt`: "error connecting" (no tmux socket), confirming direct launch—not tmux.
- `container/socks-list.txt`: only `broker-190.json` present.

No `launch_state` field in the session row, no second failed session. `api-sessions-before-start.json` shows `sessions: []` before the test, confirming clean start.

### (c) Prompt reached fake backend through broker PTY

**Result: CONFIRMED**

Evidence chain:

1. `sentinel-prompt.txt`: `USER_SENTINEL_FAKE_CC_PROMPT_19397 please answer through fake claude`
2. `container/fake-claude-runtime-logs.txt` command log entry: `"prompt":"USER_SENTINEL_FAKE_CC_PROMPT_19397 please answer through fake claude"` with matching session_id `69121efb-...` and `fake_notice` label.
3. `container/fake-claude-jsonl-head.txt` third row: `type=user`, content matches sentinel exactly.
4. `container/processes.txt` confirms broker (PID 190) is parent of fake-claude (PID 192, PPID 190). Broker command line includes `--model sonnet --effort max`. Fake-claude received same args.
5. `fake-claude-runtime-logs.txt` invocation record confirms `env_model=sonnet`, `env_effort=max` from broker environment passthrough.

The prompt text matches character-for-character between `sentinel-prompt.txt`, `browser-fill-sentinel.txt` action, the command log, and the JSONL user row.

### (d) Final assistant sentinel rendered in browser and /messages/tail

**Result: CONFIRMED**

`api-messages-tail-after-send.json` events (3 entries):
1. Bootstrap assistant: `FAKE_CLAUDE_BOOTSTRAP_ASSISTANT_READY_not_real_provider`
2. User: `USER_SENTINEL_FAKE_CC_PROMPT_19397 please answer through fake claude`
3. Final assistant: `FAKE_CLAUDE_ASSISTANT_SENTINEL_usable_branch_19397 received: USER_SENTINEL_FAKE_CC_PROMPT_19397 please answer through fake claude` — with `message_class=final_response`, `transcript_state=bound`.

`eval-after-send-idle-browser-state.json` transcript text contains all three messages in the same order. `poll-browser-after-send.jsonl` line 2 (i=1, ok=true) shows the sentinel appearing after a brief poll.

`snapshot-after-send-sentinel.txt` shows the attach button disabled with "Wait for the current response to finish" (mid-send state). `snapshot-after-send-idle.txt` shows all controls re-enabled (idle state). The two snapshots prove the busy→idle transition was observed in the browser.

### (e) /api/sessions, /messages/tail, and #ctxChip token values agree

**Result: CONFIRMED**

| Source | `tokens_in_context` | `context_window` | `percent_remaining` | `max_input_tokens` | `reserved_tokens` |
|---|---|---|---|---|---|
| `api-sessions-after-send.json` | 4500 | 200000 | 98 | 183616 | 16384 |
| `api-messages-tail-after-send.json` | 4500 | 200000 | 98 | 183616 | 16384 |
| `eval-after-send-idle-browser-state.json` #ctxChip | text "Ctx 98%" | title "...4500/183616 tokens (16384 reserved; window 200000)." | — | — | — |

All three agree exactly. Token math verified: `max_input_tokens = context_window - reserved_tokens = 200000 - 16384 = 183616` ✓. `tokens_remaining = max_input_tokens - tokens_in_context = 183616 - 4500 = 179116` ✓. `percent_remaining = round(100 * 179116 / 183616) = 98` ✓.

Token source verified: `tokens_in_context = input_tokens + cache_read_input_tokens + cache_creation_input_tokens` from fake-claude JSONL usage. After bind: `1200 + 34 + 56 = 1290` (matches `api-sessions-after-bind.json`). After send: `4321 + 100 + 79 = 4500` (matches post-send values). The system correctly sums all input token components from the Claude Code usage format.

### (f) Idle real-session controls are enabled

**Result: CONFIRMED**

From `eval-after-send-idle-browser-state.json`:

| Control | `disabled` | Label |
|---|---|---|
| `#sendBtn` | `false` | Send |
| `#fileBtn` | `false` | View file |
| `#attachBtn` | `false` | Attach file (max 16.0 MB) |
| `#captureBtn` | `false` | Add photo (max 16.0 MB) |
| `#queueBtn` | `false` | Queued messages |
| `#unattendedBtn` | `false` | Unattended mode |
| `#ctxChip` | `false` | Ctx 98% |

All idle-state controls are enabled. Contrast with `eval-after-send-browser-state.json` (mid-send snapshot) where attach was disabled with "Wait for the current response to finish"—proving the system transitions from busy to idle correctly.

### (g) Proof boundary correctly excludes real Claude credentials/auth/schema/tool/provider behavior

**Result: CONFIRMED**

- `grep` for cookie values, `codoxear_auth`, `set-cookie`, `ANTHROPIC_API_KEY`, `CLAUDE_API_KEY`, bearer tokens, session token strings: **zero hits** across all 69 artifact files.
- `secret-marker-check.txt`: "no sensitive auth marker hits in artifacts"
- `browser-driver-notes.md`: "Password was the Docker sandbox default and is not recorded here."
- `api-login.json`: only `{"ok":true}`, no cookie value stored.
- `browser-fill-login.txt`: action log only ("Done"), no password value.
- `fake-claude` script: every JSONL row includes `"fake_notice":"FAKE_CLAUDE_CODE_FOR_CODOXEAR_DOCKER_ONLY"`. Script header docstring says "intentionally fake" and "must never be installed on the host."
- `VERIFICATION-REPORT.md` Limits section explicitly states: "does not prove real Claude credentials, real provider authentication, real Claude Code terminal UI behavior, or real API/tool error behavior."
- No real Claude Code binary, schema, tool invocation, or API response shapes beyond the minimal user/assistant JSONL structure.

---

## Artifact hygiene

| Check | Result |
|---|---|
| Cookies/auth headers | None found |
| API keys/secrets | None found |
| Private host paths | None (all paths are container-internal `/home/tester/...` or `/workspace`) |
| Credential values | None (login password not recorded) |
| Bulky logs | No. Total artifact directory: 288KB across 69 files. Largest structured file is `eval-claude-controls.json` (~12KB, full-page control inventory). |
| Empty files committed | 6 files are 0 bytes: `browser-click-reasoning-max.txt`, `container-after-stop.txt`, `stop.txt`, `git-diff-check.txt`, `git-diff-check-final.txt`, `git-staged-files-final.txt`. These are lifecycle markers where no output was expected or where an action produced no stdout. |
| Preflight session leak | `fake-install-preflight.txt` records a dry-run with session_id `692d0157-...`. Cleanup confirmed in `fake-preflight-cleanup.txt`. The real test session uses `69121efb-...`. No cross-contamination. |

---

## Nonblocking findings

1. **`provider_choice: "openai-api"` in cc session row.** `api-sessions-after-bind.json` shows `provider_choice: "openai-api"` for the cc-backend session. This is inherited from Codex defaults, not a Claude provider value. Not a correctness issue for the mechanics proof but could mislead a reader into thinking the Claude session used an OpenAI provider. The cc backend's `new_session_defaults` correctly shows `provider_choice: null`.

2. **CC model list includes "fable."** `api-sessions-before-start.json` → `new_session_defaults.backends.cc.models` contains `["sonnet", "opus", "fable"]`. "fable" is not a real Claude model. Product configuration concern; does not affect the proof, which uses "sonnet."

3. **`poll-after-start.jsonl` is 60 lines of 401 errors.** This is a curl-based server readiness poll that ran before browser authentication. All entries are `{"i": N, "error": "<HTTPError 401: 'Unauthorized'>"}`. Confirms server was reachable but adds noise to the artifact set. Harmless.

4. **`browser-click-reasoning-max.txt` is empty (0 bytes).** The first attempt to click "max" produced no stdout, but `browser-click-max-ref.txt` shows success ("Done"), and `eval-before-start.json` confirms reasoning=max was set. The flow completed correctly.

5. **`eval-claude-controls.json` is oversized for its purpose.** 101-element full-page control inventory (~12KB). Only ~6 controls are relevant to the proof claims. Not harmful but adds bulk.

6. **`eval-after-send-browser-state.json` and `eval-after-send-idle-browser-state.json` capture different temporal states.** The "after-send" capture shows mid-send busy state (attach disabled). The "idle" capture shows post-completion idle state. Both are useful evidence but the naming could be clearer. The VERIFICATION-REPORT correctly uses only the idle capture for the idle-controls claim.

---

## Accepted claim boundary

The following claim is supported by the committed evidence:

> Codoxear's Claude Code browser-created path works mechanically through real browser New Session tab selection (Claude, model sonnet, reasoning max, direct launch), broker/log binding, composer send over PTY, transcript/tail rendering, token projection, and idle control state—using a deterministic fake Claude Code executable inside a Docker container that writes Claude Code-shaped JSONL.

The following is correctly **excluded** from the proof:

- Real Claude provider authentication or API key validation
- Real Claude Code binary behavior, terminal UI, permission handling
- Real Claude API schema, tool invocations, error responses, streaming
- Multi-turn, multi-tool, or long-running session behavior
- Claude Code-specific session resume or conversation continuity

---

## Evidence table

| Artifact | Purpose | Key finding |
|---|---|---|
| `eval-before-start.json` | Pre-launch DOM state | Claude active, sonnet, max, tmux=false |
| `api-sessions-after-bind.json` | Session row after bind | cc backend, broker-190, thread_id set, no launch_state |
| `api-sessions-after-send.json` | Session row after send | tokens updated 1290→4500, busy=false |
| `api-messages-tail-after-send.json` | Tail API response | 3 events, transcript_state=bound, token matches |
| `eval-after-send-idle-browser-state.json` | Browser idle state | All controls enabled, sentinel in transcript, Ctx 98% |
| `eval-after-send-browser-state.json` | Browser mid-send state | Attach disabled (busy), sentinel present |
| `container/processes.txt` | Process tree | broker→fake-claude chain, correct args |
| `container/fake-claude-runtime-logs.txt` | Fake CLI logs | Exact prompt received, env vars passed |
| `container/fake-claude-jsonl-head.txt` | JSONL content | 4 rows, correct structure, fake_notice on all |
| `container/sidecars.jsonl.txt` | Sidecar metadata | cc backend, sonnet, max, sync_send=true |
| `container/session_launches.jsonl` | Launch ledger | direct transport, starting→broker_spawned |
| `fake-claude` | Fake executable source | Labeled, deterministic, PTY-aware |
| `secret-marker-check.txt` | Credential scan | No hits |
| `independent-review-summary.json` | Self-review | 9/9 checks passed |
| `poll-browser-after-send.jsonl` | Browser poll for sentinel | 2 polls: miss then hit, token update confirmed |

---

## Verdict

**ACCEPT.** All seven assertions are supported by cross-referenced, internally consistent evidence across API responses, browser DOM evaluations, container process/log captures, and sidecar metadata. No credentials or secrets are committed. The proof boundary is correctly stated and does not overclaim real-provider behavior. Six nonblocking findings noted, none affecting the core claims.
