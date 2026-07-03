# Epistemic model

## Phenomenon
Codoxear passes its test suites and the 8-item refactor Workbench, but "fully usable product with decent UI" is an unmeasured claim: no systematic desktop+mobile browser walkthrough exists for the current HEAD, and no real backend session has been driven end-to-end from the browser in the sandbox.

## Live claims
- Test suites under-measure product usability: proven by the challenged review (live 500 + two failed-launch UX defects behind 1344 green tests). Browser + smoke evidence is the discriminating instrument.
- Architecture residuals are known, not hypothesized: `app.js` (~9.1k lines) still owns chat search/navigation orchestration, new-session dialog state, queue/recovery panels; 72 source-named tests; internal `server.MANAGER`-style monkeypatch seams in route tests.
- Delegation lane: foreground subagents work with `deepseek/deepseek-v4-flash`; `occ-glm/glm-5.2` is provider-down and is the probable cause of async runner deaths (child exits at startup before writing results).

## Open questions (highest value first)
1. What does the ranked UX defect ledger look like on current HEAD? (Workbench 1 — nothing can be prioritized honestly before this.)
2. Does a real Pi session work end-to-end from the browser in the sandbox? (Workbench 2 — the core product promise.)
3. Which architecture extraction gives the best defect-surface reduction per diff? (Decide after ledger; candidates: new-session dialog, chat search orchestration, queue/recovery panels.)

## Ruled out
- "NO BLOCKERS reviews prove zero issues" — falsified twice this task cycle.
