# OPS

## 2026-09-10T14:03:05+08:00

- Initialized correction task memory from the project template.
- Observed complaint screenshot: the transcript busy bubble displays only animated dots plus `tools: 12 · thinking: 5.5k · subagents: 2`; no child lines are visible.
- Initial working tree contains unrelated modified logo task memory and many unrelated untracked paths. These are excluded from this task.
- Current HEAD `2dde496f` contains hidden expandable typing-bubble details; deployed snapshot is reportedly `79d71532` and remains outside verification scope.

## 2026-09-10T14:16:00+08:00

- Inspected installed `pi-subagents` lifecycle-v3 producer source and retained status artifacts. Parallel runner sets each step to `running` independently and records per-step `model`, `toolCount`, and cumulative `tokens.total`; completed siblings transition independently. Chain status retains future steps as `pending`, so counting pending steps would falsely count work that has not started.
- Current Codoxear scanner emits one record per active top-level status file by selecting only `currentStep`; this collapses parallel children. Chosen correction: emit each modern `steps[i]` with `status == running`; retain one legacy fallback only when step records contain no lifecycle statuses.
- Traced freshness path: `/api/sessions` ETag covers full detail payload; fresh 200 responses replace catalog/index; selected-session refresh invokes `updateTypingStatsFromSession`. Current details bypass `sessionState` and directly mutate the typing runtime. Chosen correction adds atomic `subagentDetails` selected state and makes transcript store projection the sole renderer trigger for count/details.

## 2026-09-10T14:31:00+08:00

- Implemented Pi active-step projection, selected-session `subagentDetails` state, store-owned transcript projection, shared always-visible busy/idle detail rendering, compact `tokens used` labels, session-switch cleanup, CSS, contract docs, and behavioral coverage.
- First Docker targeted run exposed six harness/model defects: fake DOM nodes did not implement the browser `textContent = ""` child-clearing contract; null telemetry was coerced to numeric zero; and one transcript-controller stub lacked the new store projection method. Corrected the formatter and executable DOM harnesses. No production change was built atop the failed result.
- Docker targeted rerun: `52 passed in 2.84s` for Pi scanner, session store, transcript runtime, and cross-backend reconciliation.

## 2026-09-10T14:48:00+08:00

- Docker-only browser fixture served a real `/api/sessions` row backed by a fake isolated Pi control socket and lifecycle-v3 status file containing two parallel running steps. API returned two distinct records with role/model/tools/cumulative tokens.
- At 390×844, before any child-detail interaction, the busy transcript bubble rendered `tools: 12 · thinking: 5.5k · subagents: 2` plus `reviewer · model-a · tools: 3 · tokens used: 4.2k` and `executor · model-b · tools: 7 · tokens used: 8.1k`. `typingStats` had no role or `aria-expanded`; horizontal overflow was zero.
- After the parent turn settled while both children remained running, the idle transcript bubble automatically rendered the same two child lines under `▸2 subagents working`. Post-login browser errors and console messages were empty after clearing expected pre-login diagnostics.
- Retained artifacts under `/tmp/codoxear-inline-subagent-results-2026-09-10/`: `phone-busy-inline-children.png`, `phone-idle-inline-children.png`, API payloads, and `browser-evidence.json` with SHA-256s printed at capture time.
- Rebuilt tracked `codoxear/static/dist/app.bundle.js` from the ESM entrypoint.

## 2026-09-10T14:54:00+08:00

- An initial full-suite Docker invocation failed at setup for all tests because running the sandbox wrapper via sudo created the mounted HOME root-owned. This was an isolation-fixture permission failure (`PermissionError: /home/tester/.local/share/codoxear`), not a product result. Chowned only the throwaway sandbox root and reran.
- Full Docker suite after correcting fixture ownership: `1877 passed, 112 subtests passed`; three unrelated failures remained: two PDF.js Node-18 runtime failures and the pre-existing deploy-script clean-snapshot test under a dirty shared checkout. No assigned subagent-path test failed.
- Focused Docker gate including scanner, selected store, transcript flow, cross-backend projection, session listing, static assets, and wiring guard: `92 passed in 9.64s`.
