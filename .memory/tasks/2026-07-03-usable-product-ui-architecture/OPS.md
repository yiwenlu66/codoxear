# Operational ledger

## 2026-07-03T15:05:00Z Task initialized
- Created this task after user directive: re-read delegation/memory instructions, plan next step, update task PROMPT; goal is a fully usable product with decent UI and clean code architecture.
- Created missing `.memory/project/` (ARCHITECTURE.md, VALIDATION.md) from distilled prior-task knowledge (commit `a8ec3c0`).
- Delegation health checks: async `worker` run died pre-output (stale-run reconciliation); foreground `worker` on `occ-glm/glm-5.2` failed with provider 503 `model_not_found` (no channel for glm-5.2); foreground `worker` on `deepseek/deepseek-v4-flash` returned `RUNNER OK`. Interpretation: runner infrastructure works foreground; earlier async deaths are consistent with child startup failing on the dead default model.
- Prior task `2026-06-11-major-refactor-new-features` closed at commit `02e3e64` with all 8 Workbench items complete plus challenged-review fixes (`dce2ce6`, `6ff957a`, `f4aaf46`, `541d4ab`, `43f4598`); validation: local `1349 passed, 136 subtests`, Docker test `1348 passed, 1 skipped`, Docker smoke 200 post-login.

## 2026-07-03T21:45:00Z Subagent API verified; async runner repaired
- User asked to verify subagent functionality and glm-5.2 availability. Foreground worker on `occ-glm/glm-5.2` returned `RUNNER OK` (run `16af76a8`) — glm-5.2 recovered and reachable via subagent API.
- Async lane kept failing (~1s silent death) independent of model. Diagnosed via /proc argv sampler + manual re-run of the captured command with stderr visible: the `~/.pi/agent/npm` pi-subagents install could not resolve `@earendil-works/pi-coding-agent` (module only present in `~/.npm-global` tree); spawn used `stdio: "ignore"` so the crash was invisible.
- Repaired with a symlink (`~/.pi/agent/npm/node_modules/@earendil-works/pi-coding-agent -> ~/.npm-global/...`). Verified: async worker on glm-5.2 returned `ASYNC GLM RESTORED` (run `61d484c0`). Full details in `.memory/local/pi-subagents-runner-repair.md` (machine-local, now git-ignored).
- Also learned: bare `pi` CLI in tmux lacks the occ-glm API key; tmux fallback must use ambient-credential providers.
- Delegation matrix now: async+foreground both work; glm-5.2 default for implementation contracts.
