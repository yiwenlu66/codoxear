# Selectable-backend Claude usable-branch proof (fake Claude)

## Outcome
Deterministic usable fake-Claude branch passed on Docker port `19397`. This proves Codoxear's Claude Code broker/log/send/outcome mechanics through the actual browser New Session UI without real Claude credentials. It is not a real-provider/authentication proof.

The fake executable was installed only inside the container at `/usr/local/bin/claude` and is preserved in this artifact directory as `fake-claude`. It is labeled `FAKE_CLAUDE_CODE_FOR_CODOXEAR_DOCKER_ONLY` in script, invocation log, command log, and JSONL rows.

## User-visible path exercised
Browser flow:

- Opened `http://127.0.0.1:19397/`, logged into the Docker sandbox.
- New Session → Claude tab.
- Working directory `/workspace`.
- Model `sonnet`.
- Reasoning `max`.
- `Create in tmux` unchecked, so this was a direct launch.
- Started session, selected/bound projected session.
- Sent `USER_SENTINEL_FAKE_CC_PROMPT_19397 please answer through fake claude` through the composer.

## Observed values
### `/api/sessions` after bind/send
From `api-sessions-after-send.json`:

- `session_id`: `broker-190`
- `thread_id`: `69121efb-477b-47da-a1e2-bd10cb85aafd`
- `agent_backend`: `cc`
- `cwd`: `/workspace`
- `transport`: `null` in row; launch ledger records `direct`
- `log_path`: `/home/tester/.claude/projects/-workspace/69121efb-477b-47da-a1e2-bd10cb85aafd.jsonl`
- `launch_state`: absent/null, so not a failed-launch row
- `busy`: `false`
- `model`: `claude-sonnet-4-5` from assistant usage row
- `reasoning_effort`: `max`
- token: `tokens_in_context=4500`, `context_window=200000`, `max_input_tokens=183616`, `percent_remaining=98`

### Sidecar metadata
From `container/sidecars.jsonl.txt`:

- `agent_backend`: `cc`
- `session_id`: `69121efb-477b-47da-a1e2-bd10cb85aafd`
- `sock_path`: `/home/tester/.local/share/codoxear/socks/broker-190.sock`
- `log_path`: same Claude JSONL path as `/api/sessions`
- `model`: `sonnet`
- `reasoning_effort`: `max`
- `transport`: `null` for direct broker sidecar
- `control_capabilities.sync_send`: `true`

### Launch ledger
From `container/session_launches.jsonl`:

- `launch_id`: `launch-1783411913492-ee057d12`
- records: `starting`, then `broker_spawned`
- `agent_backend`: `cc`
- `model`: `sonnet`
- `reasoning_effort`: `max`
- `transport`: `direct`
- `cwd` / `requested_cwd`: `/workspace`

Current successful launches do not append a distinct `log_bound` ledger row; the live sidecar provides the log-bound path.

### Browser transcript and controls
From `eval-after-send-idle-browser-state.json` and `snapshot-after-send-idle.txt`:

- Transcript contains the user sentinel.
- Transcript contains final fake assistant sentinel: `FAKE_CLAUDE_ASSISTANT_SENTINEL_usable_branch_19397 received: ...`.
- Transcript does not show a failed-launch row.
- `#ctxChip`: text `Ctx 98%`, title `Context input: 4500/183616 tokens (16384 reserved; window 200000).`
- Idle real-session controls: send, file, attach, capture, queue, and unattended are enabled.
- Body overflow check: `scrollWidth=1280`, `clientWidth=1280`, `overflow=false`.

### `/messages/tail`
From `api-messages-tail-after-send.json`:

- `transcript_state`: `bound`
- `busy`: `false`
- token matches `/api/sessions`: `tokens_in_context=4500`, `context_window=200000`, `percent_remaining=98`
- events include bootstrap assistant, user sentinel, and final fake assistant sentinel.

### Broker/fake CLI command path
From `container/processes.txt`:

- broker process: `/usr/local/bin/python3 -m codoxear.broker --cwd /workspace -- --dangerously-skip-permissions --model sonnet --effort max`
- fake CLI child: `python3 /usr/local/bin/claude --dangerously-skip-permissions --model sonnet --effort max`

From `container/fake-claude-runtime-logs.txt`:

- invocation recorded argv `--model sonnet --effort max`, env `CODEX_WEB_MODEL=sonnet`, `CODEX_WEB_REASONING_EFFORT=max`
- command log recorded the exact browser prompt, proving the send reached fake Claude through the broker PTY

From `container/fake-claude-jsonl-head.txt`:

- JSONL contains Claude Code-shaped `user` and `assistant` rows with matching `sessionId`, `cwd`, assistant `model=claude-sonnet-4-5`, and usage.

## Defects fixed / blockers
No product-code defects were exposed. No product code was changed.

## Review
`independent-review-summary.json` re-parsed the saved API, browser, sidecar, ledger, and fake command evidence. All nine checks passed: real bound CC row, `sonnet`/`max`, tail/browser sentinels, token agreement, idle controls enabled, sidecar log bind, ledger direct launch, and fake prompt receipt.

## Commands/tests run
- `CODOXEAR_DOCKER_PORT=19397 scripts/codoxear-docker-sandbox preflight`
- `CODOXEAR_DOCKER_PORT=19397 scripts/codoxear-docker-sandbox start`
- container-only fake install via `docker cp` + `docker exec -u root install -m 0755 /tmp/fake-claude /usr/local/bin/claude`
- browser New Session and composer-send flow via `agent-browser`
- API captures with `curl` against `http://127.0.0.1:19397`
- focused tests: `python3 -m pytest -q tests/test_cc_log.py tests/test_cc_session_log.py tests/test_backend_launch_adapter.py` → `26 passed`
- `git diff --check` → passed
- `CODOXEAR_DOCKER_PORT=19397 scripts/codoxear-docker-sandbox stop`

## Limits
The proof replaces Claude Code with a deterministic fake inside Docker. It proves Codoxear mechanics after a Claude-shaped executable starts, writes a log, stays alive, receives PTY input, and writes final assistant usage. It does not prove real Claude credentials, real provider authentication, real Claude Code terminal UI behavior, or real API/tool error behavior.

## Artifact hygiene
The temporary auth jar was removed. `secret-marker-check.txt` records no sensitive auth marker hits in artifacts.
