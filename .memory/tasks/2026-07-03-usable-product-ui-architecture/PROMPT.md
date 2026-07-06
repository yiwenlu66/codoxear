## Objective
Advance Codoxear on `/home/yiwen/codex-web-product-recovery` (branch `recovery/product-gaps`) from tests-green recovery to a **fully usable product with a decent UI and clean code architecture**, evidenced from the user's perspective.

"Fully usable" is a user-perspective claim, not a test-suite claim: core flows must work end-to-end in an isolated Docker deployment driven through a real browser (desktop and mobile viewports), including real backend sessions where credentials allow. "Decent UI" means coherent visual hierarchy, correct loading/empty/error states, touch ergonomics, and readable information density within Codoxear's minimal-UI product model. "Clean architecture" means each user-visible state has one semantic authority; app-shell concentrations shrink only when that improves a concrete product or verification mechanism.

Do not merge, promote, or modify `/home/yiwen/codex-web` or `main` without explicit user approval.

## Current roadmap (2026-07-04, HEAD `e718cd9`)
Codoxear is a product under continued iteration, not a promotion-proposal exercise and not an open-ended refactor project. Major named browser controllers are extracted and accepted through Voice/settings/notifications. Further extraction is allowed when it is the right intervention for a concrete product defect or verification mechanism; extraction is not the organizing principle.

The product promise is narrower and stricter than "the backend will answer well": every selectable backend path either works or fails visibly, and every sent prompt produces a browser-visible result: answer, explicit backend error, explicit no-answer completion, or explicit interruption.

Roadmap:
1. **Finish current truthfulness blockers.** Startup discovery must tolerate bad runtime metadata; no-answer turns must be visible without false positives; backend/provider errors must render as error-styled transcript messages; completed backend activity must leave the session idle; Docker verification must be isolated from the live runtime.
2. **Certify transcript message projection.** For Pi, Codex, and Claude Code where credentials allow, verify New Session -> send -> a visible transcript message or recovery message. Answer, no-answer, backend error, and interruption are messages shown where they occur, not session-state categories and not log-parser internals.
3. **Audit binary busy/idle authority.** Busy/idle is a boolean, not a taxonomy. The existing sendability, queueability, interrupt affordance, blue/gray indicator, and spinner/animation must all derive from the same binary truth. When the backend is no longer working, the session is idle regardless of why it stopped; the reason appears as a message/recovery item, not as a busy-state variant.
4. **Certify the user-facing surfaces below.** File browsing/editing, repository status/diff, and mobile companion use are first-class product surfaces, not optional polish. Browser evidence must cover them explicitly.
5. **Fix evidence-backed failures only.** Each fix must name the false UI/local-state contract. Use direct patch or extraction according to mechanism; delegate code/test work to executors. Re-verify affected flows and any state authority they share.
6. **Handle bounded non-product debt after product-critical evidence.** The known remaining route-test seam is `tests/test_file_upload.py` using server-global monkeypatching; residual source tests are converted when they block safe changes, not as release blockers by default.
7. **Acceptance and next iteration.** Full validation plus clean-room review decides whether remaining work is blocker, impairing, or polish. Do not package a promotion proposal as the deliverable.

Negative evidence: the attempted sidebar/swipe extraction after `c3693df` was discarded because it followed “next extractable cluster” instead of a proven Codoxear product mechanism. If direction feels vague again, ask the `theorist`/Fable agent before dispatching implementation.

## Surface certification map
This map is an internal planning and acceptance tool, not product terminology. User-facing communication must name the actual surface: sessions, transcript, files, repository status, attachments, mobile controls, diagnostics, settings, or notifications. Each item below must be certified from the browser in an isolated Docker deployment unless explicitly marked as code-only.

1. **Launch, auth, and session inventory**
   - Login/logout and cookie persistence.
   - Sidebar/session list state, aliases, hidden/stale sessions, backend icons, recent cwd ordering.
   - New Session dialog for Codex/Pi/Claude tabs: provider/model/reasoning defaults, cwd/worktree choices, launch provenance, failed-launch display, dismissal/retry.
   - Bad sidecar metadata must hide or skip the bad session without crashing startup or hiding valid neighbors.

2. **Transcript message projection**
   - Successful answer renders as an assistant message.
   - Backend/provider error renders as a transcript message with error styling, not narration.
   - Backend completes without answer renders an explicit no-response transcript message, without false positives when alternate assistant row forms exist.
   - User interruption renders as a visible transcript/recovery message where appropriate.
   - Search, older-history pagination, and live polling preserve the same transcript messages.
   - Server restart or second server discovery rehydrates transcript messages from backend logs, not volatile UI state.

3. **Binary busy/idle, sendability, and indicators**
   - There must be one binary busy/idle authority. Broker busy, log idle, `/api/sessions` ready/sendable flags, composer disabled state, queue button state, interrupt affordance, sidebar blue/gray indicator, chat spinner/animation, mobile status text, and unattended sweep eligibility must be mechanical projections of that boolean.
   - No new busy states, labels, colors, legends, or mapping tables belong here. The existing visual language may show only busy versus idle. Messages explain what happened; they do not redefine session state.
   - Required corner cases: pre-log first send; backend error message; no-answer message; interrupted turn; queued prompt while busy; queue drain; backend process exit before log bind; log tail pagination; stale sidecar; rediscovery after server restart.
   - A mismatch among backend activity, send controls, queue controls, the blue/gray indicator, or animation is a blocking defect because it tells the user contradictory facts about the same binary state.

4. **Transcript and navigation**
   - Sparse chat rendering remains readable on desktop and mobile.
   - Search, match navigation, older-history loading, scroll retention, copy/export, markdown/code/table rendering, and message identity survive appended logs and polling.
   - Empty/loading/error states explain what the user can do next.

5. **Send, queue, interrupt, and unattended execution**
   - Plain send, queued send while busy, queue persistence, queue cancellation/drain, interrupt, and recovery from failed sends.
   - Unattended mode respects the same busy/idle authority and never fires against a turn that is still active or invisibly failed.

6. **Files and editing**
   - File picker/search/session file history, open-by-path, path boundary enforcement, directory/binary/non-UTF/large-file handling.
   - Viewer readability, markdown/media/video preview behavior where supported.
   - Editor open/edit/save/cancel, stale-content conflict recovery, lock behavior, diff/preview, save error display.
   - Upload/download/attachment paths, including failure visibility and no leakage outside the session cwd policy.
   - Mobile file viewer/editor usability: touch targets, keyboard/viewport behavior, readable diffs and errors.

7. **Repository status and diffs**
   - Git status/diff/log/file diff from the selected session cwd.
   - Dirty/untracked/renamed/deleted states; binary/non-UTF filenames; nested repo or non-repo boundaries.
   - Git viewer must not imply a write happened; read-only views must stay read-only.
   - Mobile git readability and navigation.

8. **Diagnostics and recovery**
   - Details/diagnostics panel, launch ledger, backend/run settings, log paths with safe redaction, copyable actionable errors.
   - Failed launch, backend crash, stale socket, malformed metadata, missing log, auth failure, and provider failure states.
   - Recovery flows: dismiss, retry, delete/hide, and rediscover without touching live runtime.

9. **Mobile companion usability**
   - 390x844-class viewport and real touch ergonomics for: login, sidebar, new session, chat read/send/queue/interrupt, status indicators, search/history, file viewer/editor, git viewer, diagnostics, settings/notifications, unattended.
   - No hover-only controls; keyboard opening must not strand the composer or hide critical status.
   - Information density must remain readable without turning the phone UI into a debug console.

10. **Voice, settings, and notifications**
   - Voice/settings/notification controllers remain composed on the live page.
   - Permission-denied, unsupported, and unavailable states are visible and do not corrupt sendability or unattended state.

11. **UI polish tranche**
   - Fix the ledger's blocking and impairing defects first; then polish items worth their diff.
   - Preserve product invariants: GTD flat sidebar, sparse chat rendering, minimal top bar, fail-loud errors.

12. **Architecture and test debt tranche**
   - Replace duplicated authorities when evidence shows two code paths define the same user-visible fact differently.
   - Do not chase line count or split modules without a concrete mechanism.
   - Replace remaining internal monkeypatch seams in route tests when they are known liabilities, especially `tests/test_file_upload.py`.
   - Reduce source-named tests when the checked behavior is executable and the conversion supports product-critical changes.

13. **Acceptance**
   - Full local pytest + Docker `test` + Docker `smoke`.
   - Browser re-verification of every fixed ledger item on desktop and mobile.
   - Independent clean-room review scoped to product truthfulness, surface completeness, and architecture authority, not just test mechanics.

## Context
- Task memory: `.memory/tasks/2026-07-03-usable-product-ui-architecture/`.
- Project memory: `.memory/project/ARCHITECTURE.md` (ownership map, invariants, failure modes) and `.memory/project/VALIDATION.md` (commands). Read these first after any compaction.
- Prior task `.memory/tasks/2026-06-11-major-refactor-new-features/` is complete (8 certification items + challenged-review fixes). Prior task `.memory/tasks/2026-06-12-structural-refactor-ux-review/` is parked; its mandatory browser-review requirement folds into this task's surface certification map.
- Key learned risk: pytest green missed a live `/api/sessions` 500 and failed-launch UX defects. Server smoke + browser evidence are mandatory for usability claims.

## Delegation (user-corrected 2026-07-03: main agent must not own code-level details)
- ALL code-level work is subagent work: file edits, fix implementation, test writing, debugging to patch, refactors. The main agent does not write product or test code itself.
- Main agent owns: browser/UX evidence collection (decisive artifacts), mechanism-level diagnosis framing, contract authoring, worker diff review, git commits, ledger ranking, acceptance judgment, memory.
- Execution agents currently available: `executor` for code/test/debug/refactor work, `critic` for independent review, `theorist` for product-model/roadmap correction. Use `theorist`/Fable before acting if the roadmap becomes generic or process-only.
- Contracts must state: goal, context files to read first, files in scope, hard constraints (no commit/no staging/no out-of-scope edits), validation commands, output shape, stop rules.
- Model status 2026-07-03: glm-5.2 recovered and verified async+foreground. deepseek-v4-flash validated backup. gpt-5.5 for harder work; opus-4-8 for design/review. Local `pi -p` CLI is the fallback if the runner breaks (see .memory/local/pi-subagents-runner-repair.md).
- Parallel-edit discipline: concurrent contracts must have disjoint file scopes; never run the full local suite for acceptance while a contract is in flight (in-flight edits collide with the run — observed with contract 3).

## Constraints
- Do not edit, restart, merge, or promote `/home/yiwen/codex-web`; do not kill live sessions/brokers.
- Docker sandbox only for broker/server/session/tmux verification; never port 8743; never host live app dir. Host-side throwaway `HOME` is not an isolation boundary and must not be used as a substitute for Docker.
- Do not use pattern-based host process cleanup (`pkill -f`, `killall`, broad `pgrep | xargs kill`) in agent-run verification. Cleanup must be exact PID-scoped for a process just started by the agent, or container-scoped via Docker teardown.
- No `git add -A` / broad staging; functional commits separate from memory/docs commits; small coherent checkpoints.
- Fail loud; no silent fallbacks; preserve public API/state-format compatibility unless a defect requires change.
- Do not claim completion from tests alone; usability claims require browser evidence; acceptance requires Docker validation and clean-room review.
