# EPISTEMIC

## Phenomenon
Codoxear exposes selectable backend tabs as product promises. The tested question is whether the browser-created Claude Code path with reasoning `max` preserves truthful user-visible semantics across both pre-log failure and usable log-bound sessions.

## Accepted mechanisms
### Pre-log failed launch
Docker port 19396 did not contain a usable `claude` executable. The browser New Session flow still carried the selected backend/model/effort (`cc`, `sonnet`, `max`) into the launch ledger and broker argv. The broker failed before log bind, and Codoxear projected that as a synthetic failed-launch row instead of a silent idle or misleading real session.

The failed-launch row is intentionally not a real session. `/api/sessions` exposed `launch_state=failed`, `launch_stage=broker_early_exit`, `model=sonnet`, `reasoning_effort=max`, `busy=false`, and `log_path=null`; `/messages/tail` exposed `transcript_state=failed` with an assistant error row. Browser controls for send/composer, queue, attach/capture, file, and unattended were disabled with failed-launch labels; API send/enqueue/file-list/attachments rejected the launch id as `404 unknown session`. Local recovery affordances remained available: Details, Copy details, New like this, and Dismiss launch. See OPS entries for the accepted failed-branch commits/artifacts.

### Deterministic usable fake-Claude launch
Docker port 19397 installed a container-only fake `/usr/local/bin/claude`, explicitly labeled `FAKE_CLAUDE_CODE_FOR_CODOXEAR_DOCKER_ONLY`, before browser launch. The fake wrote Claude Code-shaped JSONL under container `~/.claude/projects/...`, kept the file/process alive, read broker PTY stdin, and appended user/final assistant rows with `claude-sonnet-4-5` usage.

The actual browser New Session flow selected Claude, `sonnet`, `max`, direct launch, and cwd `/workspace`. Codoxear bound a real CC session row (`session_id=broker-190`, `thread_id=69121efb-477b-47da-a1e2-bd10cb85aafd`) to `/home/tester/.claude/projects/-workspace/69121efb-477b-47da-a1e2-bd10cb85aafd.jsonl`; it was not a failed row (`launch_state=null`, `busy=false`). A browser composer send of `USER_SENTINEL_FAKE_CC_PROMPT_19397...` reached the fake through the broker, proven by `/tmp/fake-claude-commands.jsonl`, and the transcript plus `/messages/tail` rendered `FAKE_CLAUDE_ASSISTANT_SENTINEL_usable_branch_19397...` as a final assistant response. The session stayed in real-session state with send/file/attach/capture/queue/unattended enabled when idle. CC usage projected consistently: `/api/sessions`, `/messages/tail`, and browser `#ctxChip` all reported the 4,500-token context input as `Ctx 98%` / `Context input: 4500/183616 tokens (16384 reserved; window 200000).` See OPS entry 2026-07-07T08:16:42Z.

The launch ledger for a successful direct web launch currently records `starting` and `broker_spawned` with backend/model/effort/transport but not a distinct `log_bound` success record. The sidecar is the live/log-bound evidence source for successful sessions.

## Evidence basis
- Task init: `68b5b51`.
- Failed-launch proof: `1cbd477` plus clarification `9195f5c`, artifacts under `.memory/tasks/2026-07-07-selectable-backend-live-parity/browser-artifacts/backend-parity-19396/`.
- Failed-launch review: `3abc5b6`, accepted with no blockers.
- Usable fake-Claude proof: `b59cf4c`, artifacts under `.memory/tasks/2026-07-07-selectable-backend-live-parity/browser-artifacts/backend-parity-fake-cc-19397/`.
- Usable fake-Claude review: `d76981f`, accepted with zero blockers and six nonblocking findings.
- OPS entries contain the command/test/browser/API evidence summary.

## Ruled out in this environment
- The unavailable-Claude branch did not disappear, idle silently, become a usable session, or enable real-session actions.
- The deterministic usable fake-Claude branch did not require host Claude credentials and did not use a prewritten sidecar shortcut; broker discovery bound the live JSONL produced by the fake executable inside the container.
- CC log binding, browser send, transcript rendering, token projection, and idle control-state projection all worked for the fake provider mechanics.

## Residual uncertainty
This is not a real-provider/auth proof. It proves Codoxear broker/log/send/outcome mechanics for the Claude Code path when a Claude-shaped executable behaves deterministically. A real Claude Code installation with credentials could still expose provider-specific behavior outside this fake: authentication prompts, API-key validation, CLI schema drift, permission handling, terminal UI quirks, streaming/tool rows, multi-turn/resume behavior, long-running tool turns, or real API errors.

## Nonblocking observations
- Raw `/api/sessions` still carries `provider_choice="openai-api"` on CC rows, while the user-visible details path previously showed Provider as unset for CC. This is cosmetic raw-JSON noise unless the UI starts exposing it as a CC provider.
- Browser `snapshot-after-send-sentinel.txt` caught attach/capture temporarily disabled while the just-sent turn was still settling; `snapshot-after-send-idle.txt` and `eval-after-send-idle-browser-state.json` are the idle real-session control-state evidence.
- The CC model list exposed by defaults includes `fable`; this proof only exercises `sonnet`, so any future claim about `fable` requires separate product/config validation.
- `poll-after-start.jsonl` contains pre-browser-auth 401 readiness noise; it is not used as session mechanics evidence.

## Current claim
Selectable-backend Claude/max browser parity is proven for both accepted branches available under Docker constraints: truthful failed-launch projection when `claude` is absent, and deterministic usable-session broker/log/send/outcome mechanics with a container-only fake Claude Code executable. No product-code defect was exposed by the usable fake-Claude proof.
