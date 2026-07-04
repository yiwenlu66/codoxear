## Objective
Advance Codoxear on `/home/yiwen/codex-web-product-recovery` (branch `recovery/product-gaps`) from "Workbench complete, tests green" to a **fully usable product with a decent UI and clean code architecture**, evidenced from the user's perspective.

"Fully usable" is a user-perspective claim, not a test-suite claim: core flows must work end-to-end in an isolated Docker deployment driven through a real browser (desktop and mobile viewports), including at least one real backend session. "Decent UI" means coherent visual hierarchy, correct loading/empty/error states, touch ergonomics, and readable information density within Codoxear's minimal-UI product model. "Clean architecture" means the remaining app-shell concentrations shrink, no duplicate authorities appear, and test scaffolding keeps converging on executable contracts.

Do not merge, promote, or modify `/home/yiwen/codex-web` or `main` without explicit user approval.

## Current roadmap (2026-07-04, HEAD `c3693df`)
Codoxear is a feature-complete release candidate awaiting whole-product certification, not an open-ended refactor project. Major named browser controllers are now extracted and accepted through Voice/settings/notifications. Further extraction is allowed when it is the right intervention for a concrete product or verification mechanism, but extraction is not the organizing principle.

Roadmap:
1. **Certify current product composition at `c3693df`.** One isolated Docker deployment, desktop and mobile browser, at least one real Pi session. Verify the changed controllers compose on one live page: New Session -> real launch; send/queue/interrupt; transcript/search/navigation; diagnostics/recovery; unattended; voice/settings; file/git flows. Output is a ranked defect/uncertainty ledger pinned to this HEAD.
2. **Resolve backend-parity scope.** Pi has real evidence; Codex and Claude Code use different log/busy/session-id mechanisms. Run real Codex/Claude evidence if credentials/environment allow; otherwise record the release boundary explicitly.
3. **Fix evidence-backed failures only.** Each fix must name the false UI/local-state contract. Use direct patch or extraction according to mechanism; delegate code/test work to executors. Re-verify only affected flows unless the change crosses controller boundaries.
4. **Handle bounded non-product debt after product-critical evidence.** The known remaining route-test seam is `tests/test_file_upload.py` using server-global monkeypatching; residual source tests are converted when they block safe changes, not as a release blocker by default.
5. **Clean-room review and promotion proposal.** Package evidence, validation, known boundaries, and ask before touching protected `/home/yiwen/codex-web`.

Negative evidence: the attempted sidebar/swipe extraction after `c3693df` was discarded because it followed “next extractable cluster” instead of a proven Codoxear product mechanism. If direction feels vague again, ask the `theorist`/Fable agent before dispatching implementation.

## Workbench
1. **Usability evidence pass**: scripted browser walkthroughs in an isolated Docker deployment, desktop (~1440w) and mobile (~390x844) viewports, with screenshots archived under this task's `browser-artifacts/`. Cover: login; sidebar/session list; new-session dialog for Codex/Pi/Claude tabs; send/queue/interrupt affordances; transcript rendering, search, older-history loading; file viewer/editor (open/edit/save/conflict/diff/preview/picker); git viewer; Details/diagnostics; unattended mode; settings/notifications; recovery flows (failed launch, dismiss); delete. Output: a ranked defect/improvement ledger (blocking > impairing > polish) recorded in EPISTEMIC.md.
2. **Real-backend end-to-end session**: run one real Pi session inside the Docker sandbox (copy only Pi auth/config per `.codex/skills/codoxear-docker-test/SKILL.md`; never print secrets). Prove from the browser: create session → send prompt → response renders → queue while busy → interrupt → transcript search → file viewer against the session cwd.
3. **UI polish tranche**: fix the ledger's blocking and impairing defects; then polish items worth their diff. Preserve product invariants: GTD flat sidebar, sparse chat rendering, minimal top bar, fail-loud errors.
4. **Architecture debt tranche**: (a) the originally named app-shell concentrations are done through Voice/settings/notifications; do not chase line count or split modules without a concrete mechanism. (b) replace remaining internal monkeypatch seams in route tests when they are known liabilities, especially `tests/test_file_upload.py`. (c) reduce source-named tests when the checked behavior is executable and the conversion supports product-critical changes.
5. **Acceptance**: full local pytest + Docker `test` + Docker `smoke` + browser re-verification of every fixed ledger item + independent clean-room review. Only then report.

## Context
- Task memory: `.memory/tasks/2026-07-03-usable-product-ui-architecture/`.
- Project memory: `.memory/project/ARCHITECTURE.md` (ownership map, invariants, failure modes) and `.memory/project/VALIDATION.md` (commands). Read these first after any compaction.
- Prior task `.memory/tasks/2026-06-11-major-refactor-new-features/` is complete (8 Workbench items + challenged-review fixes). Prior task `.memory/tasks/2026-06-12-structural-refactor-ux-review/` is parked; its mandatory browser-review requirement folds into this task's item 1.
- Key learned risk: pytest green missed a live `/api/sessions` 500 and two failed-launch UX defects. Server smoke + browser evidence are mandatory for usability claims.

## Delegation (user-corrected 2026-07-03: main agent must not own code-level details)
- ALL code-level work is subagent work: file edits, fix implementation, test writing, debugging to patch, refactors. The main agent does not write product or test code itself.
- Main agent owns: browser/UX evidence collection (decisive artifacts), mechanism-level diagnosis framing, contract authoring, worker diff review, git commits, ledger ranking, acceptance judgment, memory.
- Execution agents currently available: `executor` for code/test/debug/refactor work, `critic` for independent review, `theorist` for product-model/roadmap correction. Use `theorist`/Fable before acting if the roadmap becomes generic or process-only.
- Contracts must state: goal, context files to read first, files in scope, hard constraints (no commit/no staging/no out-of-scope edits), validation commands, output shape, stop rules.
- Model status 2026-07-03: glm-5.2 recovered and verified async+foreground. deepseek-v4-flash validated backup. gpt-5.5 for harder work; opus-4-8 for design/review. Local `pi -p` CLI is the fallback if the runner breaks (see .memory/local/pi-subagents-runner-repair.md).
- Parallel-edit discipline: concurrent contracts must have disjoint file scopes; never run the full local suite for acceptance while a contract is in flight (in-flight edits collide with the run — observed with contract 3).

## Constraints
- Do not edit, restart, merge, or promote `/home/yiwen/codex-web`; do not kill live sessions/brokers.
- Docker sandbox only for server/session testing; never port 8743; never host live app dir.
- No `git add -A` / broad staging; functional commits separate from memory/docs commits; small coherent checkpoints.
- Fail loud; no silent fallbacks; preserve public API/state-format compatibility unless a defect requires change.
- Do not claim completion from tests alone; usability claims require browser evidence; acceptance requires Docker validation and clean-room review.
