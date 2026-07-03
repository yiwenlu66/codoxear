## Objective
Advance Codoxear on `/home/yiwen/codex-web-product-recovery` (branch `recovery/product-gaps`) from "Workbench complete, tests green" to a **fully usable product with a decent UI and clean code architecture**, evidenced from the user's perspective.

"Fully usable" is a user-perspective claim, not a test-suite claim: core flows must work end-to-end in an isolated Docker deployment driven through a real browser (desktop and mobile viewports), including at least one real backend session. "Decent UI" means coherent visual hierarchy, correct loading/empty/error states, touch ergonomics, and readable information density within Codoxear's minimal-UI product model. "Clean architecture" means the remaining app-shell concentrations shrink, no duplicate authorities appear, and test scaffolding keeps converging on executable contracts.

Do not merge, promote, or modify `/home/yiwen/codex-web` or `main` without explicit user approval.

## Workbench
1. **Usability evidence pass**: scripted browser walkthroughs in an isolated Docker deployment, desktop (~1440w) and mobile (~390x844) viewports, with screenshots archived under this task's `browser-artifacts/`. Cover: login; sidebar/session list; new-session dialog for Codex/Pi/Claude tabs; send/queue/interrupt affordances; transcript rendering, search, older-history loading; file viewer/editor (open/edit/save/conflict/diff/preview/picker); git viewer; Details/diagnostics; unattended mode; settings/notifications; recovery flows (failed launch, dismiss); delete. Output: a ranked defect/improvement ledger (blocking > impairing > polish) recorded in EPISTEMIC.md.
2. **Real-backend end-to-end session**: run one real Pi session inside the Docker sandbox (copy only Pi auth/config per `.codex/skills/codoxear-docker-test/SKILL.md`; never print secrets). Prove from the browser: create session → send prompt → response renders → queue while busy → interrupt → transcript search → file viewer against the session cwd.
3. **UI polish tranche**: fix the ledger's blocking and impairing defects; then polish items worth their diff. Preserve product invariants: GTD flat sidebar, sparse chat rendering, minimal top bar, fail-loud errors.
4. **Architecture debt tranche**: (a) extract remaining `app.js` orchestration concentrations — chat search/navigation orchestration, new-session dialog state, queue/recovery panel rendering — into owned modules with executable tests; (b) replace internal monkeypatch seams in route tests (`patch.object(server, "MANAGER")`, `_require_auth`, `_json_response` patterns) with injected route deps; (c) reduce the 72 source-named tests where the checked behavior is executable.
5. **Acceptance**: full local pytest + Docker `test` + Docker `smoke` + browser re-verification of every fixed ledger item + independent clean-room review. Only then report.

## Context
- Task memory: `.memory/tasks/2026-07-03-usable-product-ui-architecture/`.
- Project memory: `.memory/project/ARCHITECTURE.md` (ownership map, invariants, failure modes) and `.memory/project/VALIDATION.md` (commands). Read these first after any compaction.
- Prior task `.memory/tasks/2026-06-11-major-refactor-new-features/` is complete (8 Workbench items + challenged-review fixes). Prior task `.memory/tasks/2026-06-12-structural-refactor-ux-review/` is parked; its mandatory browser-review requirement folds into this task's item 1.
- Key learned risk: pytest green missed a live `/api/sessions` 500 and two failed-launch UX defects. Server smoke + browser evidence are mandatory for usability claims.

## Delegation
- Foreground/async subagents for bounded, self-contained work: implementation (worker), audits (reviewer). Contracts must state goal, files in scope, validation commands, and stop rules.
- Model status 2026-07-03: `occ-glm/glm-5.2` provider-down (503 model_not_found) — this also killed async runs at child startup. `deepseek/deepseek-v4-flash` validated working. Try `occ/gpt-5.5` for harder work; `occ-claude/claude-opus-4-8` for design/review. Local `pi -p` CLI (read-only tools for reviews) is the proven fallback when the runner misbehaves.
- Main agent keeps: browser/UX evidence, ledger ranking, acceptance judgment, all git commits.

## Constraints
- Do not edit, restart, merge, or promote `/home/yiwen/codex-web`; do not kill live sessions/brokers.
- Docker sandbox only for server/session testing; never port 8743; never host live app dir.
- No `git add -A` / broad staging; functional commits separate from memory/docs commits; small coherent checkpoints.
- Fail loud; no silent fallbacks; preserve public API/state-format compatibility unless a defect requires change.
- Do not claim completion from tests alone; usability claims require browser evidence; acceptance requires Docker validation and clean-room review.
