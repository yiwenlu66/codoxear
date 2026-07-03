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

## 2026-07-03T22:35:00Z Workbench 1+2 round: real-session evidence, three live-only defects fixed
- Delegation restored end-to-end: async worker on glm-5.2 verified (`ASYNC GLM RESTORED`); three test-conversion contracts completed by workers (commits `889a8e7` message routes, `0a42e01` transcript export, `0697a3e` pending-log-idle; ~46 internal monkeypatch seams removed, all validated and reviewed before commit).
- UX walkthrough round 1 (desktop+mobile, sandbox 19083) captured in browser-artifacts d01-d12, m01-m03; ledger in EPISTEMIC. Fixed: ANSI garbage in failed-launch transcript (`fc620e6`), unattended-on-failed-launch error (`cefd7d6`).
- Workbench 2 real-backend setup: pi CLI installed in sandbox container (required Node 22 upgrade — distro Node 20 lacks `markAsUncloneable` for pi's undici); Pi config copied per docker-test skill; provider apiKeys are env-refs resolved into the throwaway sandbox copy only (chmod 600, never printed); root-owned ~/.npm cache chowned; settings.json stripped to defaults to avoid startup package installs.
- Live-session evidence (real Pi + deepseek-v4-flash, browser-driven): create -> auto-select -> send -> response render; second send via live-cursor path; queue-while-busy (badge/toast) -> automatic drain -> queued response; interrupt button ends turn cleanly; loaded+all transcript search (`1/2 loaded · 2 all`); file viewer opens repo README with explicit Monaco-timeout plain-text fallback; live token chip renders. Artifacts d13-d25.
- THREE live-only functional defects found by this flow and fixed:
  1. `af1fea4` live message polling broke on every bound-cursor poll (`_read_jsonl_records_from_offset` lost its 2 MiB default during route extraction `be7eeb3`); regression test drives handle_messages_live over a real log with a valid signed cursor.
  2. `acc232c` rollout_idle.py dropped eight pi/cc helper imports during decomposition — live NameError on Pi thinking rows (hit), latent NameErrors for Pi tool rows and all CC row paths; regression tests feed those row shapes through _analyze_log_chunk/_compute_idle_from_log.
  3. (Earlier round) `dce2ce6` recent_cwds default limit.
- Full local suite after everything: `1358 passed, 132 subtests passed`. Docker test (19084): `1357 passed, 1 skipped, 132 subtests`. Docker smoke (19085): 401/200 correct.
- Pattern now three-for-three: every functional break lived in a seam that unit tests faked away and only a live server+real session exposed. Live-session browser validation is non-negotiable acceptance evidence for this codebase.
