## Objective
Refactor Codoxear's architecture and frontend structure after the major feature branch, using the current `develop` state as the base. The goal is not to add broad new product features; it is to make the implementation easier to reason about, safer to change, and better aligned with the product invariants while preserving user-visible behavior unless a refactor exposes a clear user-experience defect that should be fixed.

Pause correction, 2026-06-12: this structural refactor task is parked. User review found that the older major feature task was overclaimed and still has real product gaps. Do not continue structural refactor as the active objective until the older feature task is repaired and accepted. Refactor work may resume only after provider/model selection, UI action placement/topbar cleanup, long-chat ergonomics, responsiveness evidence, file-viewer polish, and scoped backend/capability claims are addressed or explicitly parked by the user.

Ontology correction: refactor is not an acceptance object by itself. It is permitted only when it clarifies or preserves a product promise, workflow, source of truth, or invariant. Refactor progress must not be used to offset missing user-facing behavior.

Done means:
- A single reviewable refactor acceptance branch exists, based on current `develop`; prefer `refactor/structural-ux-review` unless a better branch name is explicitly recorded. `main` remains untouched and no merge to `main` happens without explicit user approval.
- The refactor removes meaningful structural coupling from the current monoliths, especially `codoxear/server.py` and `codoxear/static/app.js`, without changing public API semantics, runtime state formats, or UX behavior accidentally.
- Each refactor has an explicit invariant, acceptance checks, and evidence. Refactors must be mechanical where possible and semantic only when the old mechanism is demonstrably wrong.
- The project still preserves Codoxear's product model: shared broker for CLI/web sessions, minimal UI, GTD-style sidebar without nesting, sparse chat rendering, and Linux-first local-session control.
- Validation includes unit/source/runtime tests, isolated Docker server smoke, and a mandatory user-perspective browser review with screenshots using the `agent-browser` skill against a sandboxed Docker deployment.
- The final report states what was refactored, what behavior was proven preserved, what UX/UE issues were found by browser use, which were fixed, which were parked, and why.

## Workbench
Current state correction:
- This refactor branch exists as preserved work/evidence, but it is not the active acceptance path.
- The older major feature task must be repaired first. Its prior `develop` acceptance claim is invalidated by product gaps, especially the missing real provider/model selector and the cluttered top-bar/action-placement design.
- Live checkout `/home/yiwen/codex-web` must remain on `main` unless the user explicitly asks otherwise. All recovery/refactor work must use an isolated worktree.
- Previous refactor work may be mined later, but do not treat it as progress that compensates for missing feature-task behavior.

Known structural observations from the pre-prompt inspection:
- `codoxear/server.py` is about 7,839 lines. `SessionManager` is about 2,579 lines with 86 methods. `Handler.do_GET` is about 1,192 lines; `Handler.do_POST` is about 980 lines.
- `codoxear/static/app.js` is about 9,894 lines with about 410 functions, most stateful UI logic inside one `renderApp()` closure.
- `codoxear/broker.py` is about 1,828 lines and mixes PTY control, backend launch, log discovery, socket protocol, and busy-state tracking.
- `codoxear/rollout_log.py` has duplicate single-record and batch chat extraction paths.
- `server.py`, `broker.py`, `sessiond.py`, `util.py`, and `pty_util.py` contain overlapping low-level helpers.
- Existing recon files are useful but may contain superseded details from before the Unattended rename; prefer `recon/final-acceptance-summary.md` and this prompt for current scope.

Refactor is parked. Required next actions before resuming this prompt:
1. Return to `.memory/tasks/2026-06-11-major-refactor-new-features/PROMPT.md` and execute the product recovery tasks there.
2. Fix the real new-session provider/model selector gap rather than preserving the old separate/fake selector.
3. Redesign top-bar/action placement according to Codoxear's minimal UI model rather than adding a generic overflow menu.
4. Browser-validate the repaired product in isolated Docker on desktop and mobile-ish viewports.
5. Record which original feature gaps are fixed, which are honestly scoped/parked, and why.
6. Only then resume structural refactor phases from this prompt.

Observed failure modes to guard against:
- Treating "tests pass" as sufficient even if the UI becomes awkward, visually broken, confusing, or slow.
- Moving code across modules while accidentally changing route status codes, JSON shapes, localStorage keys, DOM ids/classes, runtime state filenames, or API paths.
- Flattening backend differences into silent fallbacks; unsupported combinations should still fail loudly.
- Changing queue/unattended/busy timing while pretending the refactor was mechanical.
- Introducing a bundler/framework/toolchain change as part of refactoring without explicit user approval.

## Context
Required project context:
- `AGENTS.md` for architecture notes, product philosophy, validation norms, and live-session safety rules.
- Prior acceptance summary: `recon/final-acceptance-summary.md`.
- Prior architecture/UI/bug recon: `recon/architecture-review.md`, `recon/ui-ergonomics.md`, `recon/git-history-bugs.md`, with awareness that some older Harness compatibility notes were superseded.
- Core code: `codoxear/server.py`, `codoxear/broker.py`, `codoxear/sessiond.py`, `codoxear/rollout_log.py`, `codoxear/pi_log.py`, `codoxear/cc_log.py`, `codoxear/util.py`, `codoxear/pty_util.py`, `codoxear/voice_push.py`, `codoxear/static/app.js`, `codoxear/static/app.css`, `codoxear/static/index.html`.
- Validation tooling: `scripts/codoxear-docker-sandbox`, `docker/sandbox.Dockerfile`, existing `tests/`.
- Browser operation: use the Pi `agent-browser` skill for real Chrome interaction and screenshots during UX/UE review.

Relevant existing tests to preserve and extend:
- Route/auth/static: `tests/test_auth_cookie.py`, `tests/test_url_prefix.py`, `tests/test_static_assets.py`, `tests/test_docker_sandbox_source.py`.
- Session/log/discovery: `tests/test_session_log.py`, `tests/test_session_resume.py`, `tests/test_stale_sidecars.py`, `tests/test_sessions_pending_log_idle.py`, `tests/test_message_cursor.py`, `tests/test_message_index.py`.
- Queue/unattended: `tests/test_server_queue_persistence.py`, `tests/test_queue_sweep_idle_guard.py`, `tests/test_unattended_sweep.py`, `tests/test_unattended_mode_source.py`.
- Backend defaults/capabilities: `tests/test_launch_defaults.py`, `tests/test_reasoning_effort_source.py`, `tests/test_cc_*`, `tests/test_claude_backend_source.py`.
- UI/source/runtime: `tests/test_chat_scrollback_source.py`, `tests/test_chat_transcript_runtime.py`, `tests/test_chat_navigation_source.py`, `tests/test_file_viewer_source.py`, `tests/test_file_picker_session_state.py`, `tests/test_new_session_model_options_source.py`, `tests/test_voice_*`.

## Task specifications

### 1. HTTP routing decomposition
Refactor `codoxear.server.Handler` so `do_GET` and `do_POST` become dispatchers rather than thousand-line route ladders.

Target route groups:
- static/auth/settings
- sessions and session diagnostics
- messages/tail/live/history/export
- files and git viewer
- queue
- unattended
- voice/notifications/audio
- launch/new-session/resume candidates

Invariants:
- Public route paths, auth behavior, status codes, JSON fields, cache headers, and error messages remain unchanged unless a test or browser observation proves the old behavior is wrong.
- URL prefix behavior remains unchanged.
- Client disconnect handling remains quiet.

Acceptance checks:
- Existing route/source tests pass.
- Add route-dispatch source or unit tests if the dispatch table/route grouping becomes nontrivial.
- Docker server smoke passes after the routing phase.

### 2. Persistent state-store extraction
Extract persistent JSON state ownership out of `SessionManager` while preserving all filenames and formats.

Initial stores:
- Unattended config (`unattended.json`)
- queue state (`session_queues.json`)
- file history (`session_files.json`)
- aliases (`session_aliases.json`)
- sidebar metadata (`session_sidebar.json`)
- hidden sessions (`hidden_sessions.json`)
- recent cwd list (`recent_cwds.json`)

Invariants:
- Existing JSON files remain readable and writable without migration.
- Atomic write pattern remains write-temp-then-`os.replace`.
- Delete-session cleanup uses one explicit lifecycle path that covers every per-session store.
- Invalid JSON should still fail loudly where it currently fails loudly; do not add silent compatibility fallbacks.

Acceptance checks:
- Existing queue, unattended, file-history, sidebar, hidden-session, and recent-cwd tests pass.
- Add or keep a test proving session deletion clears all relevant per-session state without touching unrelated sessions.

### 3. Pure helper module extraction
Move unrelated pure/helper clusters out of `server.py` into bounded modules. Preserve import compatibility temporarily if needed.

Candidate modules:
- `codoxear/auth.py`: password, HMAC secret, cookie sign/verify, auth helpers.
- `codoxear/message_cursor.py`: cursor signing, verification, encode/decode, history cursor attachment.
- `codoxear/file_viewer.py`: file kind detection, text/image/pdf/video inspection, PNG repair, byte ranges, video preview helpers.
- `codoxear/git_viewer.py`: git path resolution, diff/numstat, changed-file listing, worktree helpers.
- `codoxear/launch_defaults.py`: Codex/Pi/Claude launch defaults, model/provider/reasoning normalization.
- `codoxear/launch_records.py`: launch attempt records and launch transcript projections.

Invariants:
- Module extraction is behavior-preserving.
- Public CLI entry points continue to import and run.
- Tests importing old `server.py` helper names either continue via re-export during transition or are migrated with no behavior change.

Acceptance checks:
- Relevant focused tests pass after each module extraction.
- Full Docker suite passes before moving to the next high-risk phase.

### 4. `rollout_log.py` chat extraction unification
Unify duplicate chat-event extraction paths.

Current issue:
- `_single_chat_event()` and `_extract_chat_events()` duplicate backend/type dispatch for Codex, Pi, and Claude Code records.

Target:
- Introduce one record classifier that can produce a chat event plus metadata deltas/signals.
- Page/history/live/export paths should use the same classification semantics.
- Metadata accumulation (thinking/tools/system, turn flags, token snapshots) should be separate from chat-event text extraction.

Invariants:
- Same events, message classes, timestamps, message ids, cursor behavior, token extraction, and idle interpretation.
- Adjacent assistant dedupe semantics remain intact.

Acceptance checks:
- Message, transcript export, idle heuristics, server chat flags, Pi, and Claude log tests pass.
- Add a regression that proves a single fixture produces identical events through tail/page/live/export paths where applicable.

### 5. Shared runtime utility deduplication
Deduplicate low-level helpers across `server.py`, `broker.py`, `sessiond.py`, `util.py`, and `pty_util.py`.

Targets:
- JSONL offset reading
- PID/process-group liveness
- process termination helpers where safe
- path equivalence
- PTY write/inject/window-size helpers
- session-id/log-path helper logic

Invariants:
- Broker, sessiond, and server behavior remain unchanged.
- No dependency cycle is introduced.
- Shared utility modules remain small and mechanism-focused.

Acceptance checks:
- Grep/source check shows one canonical definition per helper class where feasible.
- Broker/sessiond fail-closed tests and JSONL offset tests pass.

### 6. Backend adapter boundary
Create a backend adapter boundary so Codex/Pi/Claude behavior is not scattered across server, broker, util, and rollout code.

Adapter responsibilities may include:
- CLI args and resume args
- environment isolation
- supported launch fields
- model/provider/reasoning validation surface
- sessions directory and log discovery predicates
- run settings/session metadata reading

Invariants:
- Shared broker architecture remains intact.
- Backend-specific differences remain explicit; do not hide unsupported settings through silent fallbacks.
- Existing `agent_backend.py` registry remains the seed of the adapter model or is extended compatibly.

Acceptance checks:
- Backend registration and launch-default tests pass.
- Codex/Pi/Claude source tests pass.
- Add a source or unit test showing that adding a backend primarily means adding adapter registration rather than editing route logic.

### 7. Buildless frontend modularization
Refactor `codoxear/static/app.js` into explicit internal modules/factories while keeping one shipped static `app.js` and no framework/bundler unless the user explicitly approves a toolchain change.

Initial module boundaries:
- Core/API/perf/formatting helpers
- backend config and launch defaults helpers
- markdown renderer and local file reference parsing
- transcript controller: identity, tail cache, pending user echo, polling, history loading, DOM windowing
- session sidebar renderer/actions
- Unattended menu state and save debounce
- new-session dialog state/rendering
- file viewer/search/editor/PDF/video handling
- queue/diagnostics/help viewers
- voice/notifications/audio runtime
- composer/send/enqueue/attachments/iOS viewport guard

Invariants:
- Shipped UI remains static and minimal.
- `index.html` still loads the expected static assets.
- Existing DOM ids/classes, API calls, localStorage keys, keyboard shortcuts, and visible copy remain unchanged unless a browser UX finding justifies a change.
- No framework rewrite, no workspace rewrite, no bundler, no `import`/`export` in the first refactor pass.

Acceptance checks:
- Existing JS source and Node VM tests pass.
- Add source tests for module/factory boundaries and for no accidental bundler/import/export introduction.
- Add or extend Node VM tests for extracted pure frontend modules.

### 8. Busy/idle authority research before semantic change
Do not immediately rewrite busy/idle semantics. First make the current semantics explicit and observable.

Current risk:
- Broker reports busy from PTY/log state; server also computes idle from logs and sometimes overrides broker state. Prior history shows stale broker busy was a real bug, so simply trusting broker is unsafe.

First refactor target:
- Introduce an explicit runtime-state projection with fields such as `broker_busy`, `log_idle`, `effective_busy`, `effective_source`, and `queue_len`.
- Preserve current effective behavior initially.
- Document which paths use which signal: sidebar, diagnostics, queue sweep, Unattended sweep, message polling.

Only after tests and evidence should any authority change be proposed.

Acceptance checks:
- Existing idle, queue, unattended, and broker busy tests pass.
- Add disagreement-case tests that pin current behavior before changing it.
- Any semantic change requires predictions and evidence for queue/unattended timing.

### 9. Mandatory post-refactor browser UX/UE review
This is a hard requirement. Do not stop at "tests pass" or "API works."

After structural refactoring is substantially complete and before final acceptance:
- Use the `agent-browser` skill to operate a real Chrome browser against a standalone Docker sandbox deployment only.
- Use isolated app/session state. Do not touch live server, live sessions, live brokers, or live runtime app dir.
- Seed or create synthetic sessions/logs/files as needed so the UI has realistic data: multiple sessions, long chat, pending/failed launch row if feasible, queue, Unattended settings, file viewer candidates, backend tabs, and diagnostics.
- Take screenshots of core surfaces and keep artifact paths in task evidence:
  - login
  - session list/sidebar on desktop and mobile-ish widths
  - selected session chat with long transcript
  - chat search and user-turn navigation
  - new-session dialog including Codex/Pi/Claude backend states
  - file viewer/search/editor or read-only states
  - queue viewer
  - Unattended menu
  - settings/voice/notifications if available
  - diagnostics/help where relevant
- Interact as a user, not as an API tester: click, type, scroll, resize/mobile viewport, use keyboard navigation, open/close menus, inspect focus behavior, and observe whether the UI remains understandable and responsive.
- Reason intensively about interaction logic: discoverability, visual hierarchy, stale state, accidental complexity, mobile touch ergonomics, keyboard escape/enter behavior, empty/error/loading states, and whether refactoring introduced subtle UI regressions.
- Record observations separately from interpretations. Screenshots are evidence; claims must point to observed browser behavior.
- Fix refactor-introduced UX regressions before final acceptance. For deeper pre-existing UX issues, either fix if small and compatible with the refactor scope, or document them as parked product decisions with screenshots and rationale.

Acceptance checks:
- Browser UX review artifact exists in `recon/` or task memory, with screenshot paths and issue/fix decisions.
- At least one desktop and one mobile-ish viewport pass are performed.
- Final summary explicitly states why the product remains minimal and responsive from a user's perspective, not only why tests pass.

## Constraints
Hard rules:
- Do not touch live sessions.
- Do not touch, stop, restart, or kill the live server.
- Do not kill `codoxear-broker` or underlying backend CLI processes.
- Do not use live runtime state under `~/.local/share/codoxear` for validation. Use Docker sandbox roots and isolated HOME/app dirs.
- Do not merge to `main` without explicit user approval.
- Do not commit secrets, provider credentials, live logs, runtime sockets, app state, bulky screenshots outside intended artifact paths, or scratch artifacts.
- Do not use `git add -A`, `git add .`, or broad staging when unrelated files may exist.
- Do not add silent fallbacks that hide broken contracts.
- Do not introduce a frontend framework, bundler, workspace rewrite, or static asset pipeline unless the user explicitly approves that product/tooling change.
- Do not change public API routes, persisted state filenames/formats, DOM ids/classes, localStorage keys, or user-visible semantics as an accidental side effect of extraction.
- Do not claim a semantic refactor is behavior-preserving unless tests and/or browser evidence constrain the relevant behavior.
- Keep commits atomic: one extraction boundary, invariant, or behavior-preserving migration per commit where possible.
- Run targeted validation after each phase and full isolated Docker validation at major checkpoints.
- Use `agent-browser` for the mandatory UX/UE review; raw HTTP checks, source tests, and screenshots from tools other than a real browser are not a substitute.
