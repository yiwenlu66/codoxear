## Objective
Continue the Codoxear product recovery and architecture refactor in `/home/yiwen/codex-web-product-recovery` on branch `recovery/product-gaps`. The work is not to make isolated repairs look safe; it is to move real ownership of state, behavior, and failure semantics out of tangled centers until the system has coherent subsystems.

Do not merge, promote, or modify `/home/yiwen/codex-web` or `main` without explicit user approval.

## Workbench
1. Reconcile the current dirty `codoxear/static/app.js` save-conflict edit: either complete it as part of a real file-viewer/editor behavior extraction or revert it before further runtime work.
2. Move file-viewer/editor state and behavior out of inline `app.js` DOM construction into a stateful frontend controller module.
3. Move transcript/chat rendering, loaded-history state, search navigation, and scroll scheduling out of `app.js` into coherent transcript modules.
4. Replace scattered backend-specific launch/log/runtime branches with explicit backend adapter objects for Codex, Pi, and Claude Code.
5. Make busy/idle/readiness a single runtime model consumed by listing, messages, queue, unattended mode, diagnostics, and broker/sessiond integration.
6. Consolidate per-session persistent state lifecycle in the session store layer so deletion, load, save, and migration are not hand-maintained across many JSON maps.
7. Resolve `sessiond.py` as either a broker-compatible headless runner sharing the same state/control/log machinery or a documented/deprecated reduced-fidelity path.
8. Replace source-text and monkeypatch scaffolding with executable behavior tests and real interfaces wherever the dependency is not OS/process/socket/network I/O.

## Context
- Active checkout: `/home/yiwen/codex-web-product-recovery`.
- Protected checkout: `/home/yiwen/codex-web` on `main`; do not edit, restart, merge, or promote it.
- Task memory: `.memory/tasks/2026-06-11-major-refactor-new-features/`.
- Current reference checkpoint: `recon/refactor-entry-checkpoint.md`; use it for preserved product semantics and already-moved ownership, not as a stopping document.
- Current git state at this rewrite included an uncommitted runtime edit in `codoxear/static/app.js` around file-save conflict behavior. Treat it as unstable until completed or reverted.
- Server route/controller extraction and much of `SessionManager` decomposition are already materially advanced. The most important remaining concentrations are stateful frontend behavior in `app.js`, backend-specific branching across launch/log/runtime code, busy/idle authority, per-session persistent lifecycle, and test scaffolding that preserves old tangles.

## Task specifications

### 1. Current dirty frontend edit
The dirty `app.js` save-conflict edit must not remain as an isolated conditional or event-handler rearrangement. Inspect the diff, the file-viewer/editor state around it, and the tests that would execute it. Then choose one of two outcomes:

- Complete the movement by extracting save-conflict behavior into the same file-viewer/editor controller work described below, with tests that execute behavior rather than only searching source text.
- Or revert the dirty runtime edit before starting a larger architecture change elsewhere.

Why: carrying an uncommitted local repair encourages more local repairs. The working tree must not become a pile of unrelated half-moves.

### 2. File-viewer/editor frontend architecture
Create a stateful frontend module for the file viewer/editor, for example `codoxear/static/app_file_viewer.js` or a small set of file-viewer/editor modules if one module becomes incoherent.

Move these responsibilities out of `app.js`:
- active file identity and line/git/api token state;
- file-open request lifecycle, abort/currentness, mode resolution, and success/failure application;
- draft file load behavior;
- file-save request lifecycle, body construction, response application, conflict behavior, reload/keep-editing behavior, and error rendering;
- dirty/unavailable copy-only behavior;
- paste dialog and touch/keyboard editor actions that depend on file-editor state;
- file-viewer toolbar/editability capability decisions.

Keep DOM nodes and external app dependencies explicit. Prefer a controller factory that receives dependencies such as API call, toast, modal isolation, file-picker callbacks, and editor construction. Do not preserve global variables just because old tests reference them; preserve only user-facing behavior and truly public `window.Codoxear*` contracts.

Why: recent work improved individual file-viewer paths, but `app.js` still owns the state machine. The desired design is a file-viewer/editor subsystem with explicit state transitions, not scattered globals plus handler closures.

### 3. Transcript/chat architecture
Move transcript and chat behavior out of `app.js` into modules that own:
- loaded transcript window state;
- message row creation and update policy;
- older-message loading;
- search state, all-transcript hint state, and navigation;
- day separators/time chips/decorations;
- scroll-to-bottom, jump-to-latest, bottom-lock, and post-render scroll scheduling.

Use existing `app_transcript.js` and `app_message_rows.js` where they are suitable; create a controller module if the remaining stateful behavior needs one. The scroll scheduler must become a named owner rather than several independent call sites competing after render, typing, polling, and history loads.

Why: previous smooth-scroll attempts failed because several bottom-scroll schedulers can neutralize one another. The solution is not another call-site tweak; it is one owner for transcript rendering and scroll intent.

### 4. Backend adapter model
Create explicit backend adapter objects or classes for Codex, Pi, and Claude Code. Move backend-specific behavior into those adapters, including:
- launch argument and environment construction;
- session/log path recognition;
- session id extraction from logs/metadata;
- resume/run-settings semantics;
- log row parsing hooks needed by rollout/chat extraction;
- busy/idle signal interpretation where it depends on backend row format;
- provider/model/default/reasoning metadata projection.

Then replace scattered `if backend == ...`, `AGENT_BACKEND == ...`, and equivalent frontend/backend checks with adapter calls when those checks express backend semantics rather than UI labels.

Why: Claude Code support was added through many local branches. Adding or changing a backend should require adapter changes, not edits across launch config, broker, session discovery, runtime, rollout parsing, and frontend display code.

### 5. Busy/idle/readiness authority
Define one runtime state model for whether a session is busy, idle, interrupted-idle, ready for direct send, ready for queue promotion, eligible for unattended injection, and safe for diagnostics/listing display.

Move the model into `codoxear/session_runtime.py` or a closely related module that receives broker state, log evidence, backend adapter interpretation, queue state, pending attachments, and commit-unknown state. Listing, message routes, queue, unattended mode, diagnostics, and send/readiness code must consume the same model rather than recomputing partial answers.

Broker and sessiond should emit structured observations; server code should not override broker state with a parallel heuristic unless that rule is explicit in the runtime model.

Why: hybrid busy/idle authority causes inconsistent UI, queue, unattended, and send behavior. A single runtime model makes disagreements visible and testable.

### 6. Per-session persistent lifecycle
Strengthen `codoxear/session_store.py` into the owner of per-session persistent lifecycle. It should coordinate, at minimum:
- aliases;
- sidebar metadata;
- hidden sessions;
- file history;
- queues;
- unattended config;
- pending attachments;
- direct commit-unknown sends;
- recent cwd records;
- launch recovery links where applicable.

Deletion of a session, migration of old state, load/save ordering, and cleanup of stale ids should be expressed through store methods, not hand-updated in manager/list/control code. Keep existing JSON files if that is the fastest path, but hide them behind a lifecycle API so future storage changes are not spread through the app.

Why: the current JSON files are better organized than before, but per-session data still lacks one owner. Every new per-session feature should not require another manual edit to deletion and persistence code.

### 7. sessiond decision
Determine whether `sessiond.py` is a supported headless runner or legacy compatibility code.

If supported:
- make it share broker turn-state, terminal query, log-watcher, control-command, launch-record, and backend-adapter machinery wherever behavior should match;
- add tests proving parity for busy/idle, state/tail/control responses, and backend log binding.

If not supported:
- document the reduced-fidelity behavior;
- remove misleading shared-support claims;
- keep only the compatibility necessary for packaging/tests.

Why: sessiond is currently a parallel implementation with lower fidelity. Parallel implementations create hidden divergence.

### 8. Test architecture cleanup
Replace source-text tests with behavior tests when behavior can be executed. Keep source assertions only for packaging/static registration, public import compatibility, security-sensitive literal preservation, or non-executable build wiring.

Replace monkeypatch seams with explicit dependency injection where the dependency is internal app logic. Keep monkeypatching for OS/process/socket/network boundaries such as `subprocess.Popen`, `socket.socket`, filesystem probes, external binaries, and time when deterministic tests need it.

When a compatibility wrapper exists only because an earlier extraction avoided redesign, remove it after callers are moved. Do not add new wrappers unless a real public import, browser global, or external compatibility contract requires them.

Why: source-text and monkeypatch scaffolding are now part of the architecture problem. They should shrink as real interfaces appear.

## Operating mode
- Continue according to this workbench by default; do not yield merely because a bounded tranche completed.
- Keep the internal state organized as Deliverables, Completed, Next actions, and Parked user decisions. Surface those sections only when yielding is necessary.
- Before each action, reason through the mechanism, failure modes, and verification path; prefer reading, tracing, inspection, and causal reasoning over trial-and-error.
- Resolve crashes, bugs, and design mistakes without asking the user unless the next step is irreversible/high-risk or requires information only the user can provide.
- Use the strongest available verification, including Docker sandbox validation for acceptance-quality claims.
- Do not repeat the same command, edit, or analysis unless a concrete new reason changes what it can prove.
- Yield only when all deliverables are complete, the only remaining gap is a parked user decision, or the next step is irreversible/high-risk.
- Before any necessary yield, run a clean-room adversarial review with a dedicated subagent using the user intent, deliverables, completed evidence, remaining next actions, parked decisions, constraints, and changed artifacts; apply findings before yielding or surface the exact unresolved decision/risk.

## Constraints
- Do not edit, restart, merge, or promote `/home/yiwen/codex-web`.
- Do not kill live sessions, the live server, brokers, or backend CLI processes.
- Do not commit secrets, credentials, private logs, sockets, runtime state, bulky scratch output, or ignored artifacts.
- Do not use `git add -A`, `git add .`, or broad path staging; stage explicit files or hunks.
- Do not turn a local symptom repair into the whole change. A conditional, event-handler tweak, renamed variable, or source assertion is not enough unless it is part of moving ownership into the intended subsystem.
- Do not switch to another subsystem because the current subsystem requires a larger redesign. Read more, model the data flow, make the larger move, and repair breakage.
- Do not optimize for looking safe. Tests, reviews, and small diffs are tools; they are not the work.
- Do not use line count as architecture evidence. Report ownership, data flow, dependency direction, and failure behavior instead.
- Do not preserve an internal wrapper, monkeypatch path, global, or source-text pattern unless there is a real external contract or OS/process/socket boundary.
- Do not add silent fallbacks. Missing dependencies, invalid state, and contract violations must fail loudly or return explicit errors.
- Do not claim completion from local validation, clean-room review, or a commit. The task advances only when responsibilities actually move and the remaining entanglement shrinks.
- Keep functional commits separate from memory/documentation commits.
- Use Docker validation for acceptance-quality claims. Host validation may guide development but is not acceptance evidence unless the user changes this rule.
- Before changing an existing system, identify the current owner, the desired owner, what data crosses between them, and what would be invalid after the move.
