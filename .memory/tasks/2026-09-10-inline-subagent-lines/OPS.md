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

## 2026-09-10T15:22:37+08:00

- Deployment was explicitly authorized for reviewed commit `fda169685ffa7479091fe48ff89d67a576c931e4`.
- Prior deployed snapshot: `79d71532e03b946d78bf912a269fc1e04e8e3c04`; new deployed snapshot: `fda169685ffa7479091fe48ff89d67a576c931e4`.
- `scripts/deploy.sh fda169685ffa7479091fe48ff89d67a576c931e4` completed successfully, including its standard wiring/build checks and deployment smoke/health boundary; the script reported `smoke test passed` and `deployed fda169685ffa7479091fe48ff89d67a576c931e4`.
- Read-only post-deploy metadata confirmed the detached deploy worktree is clean at the exact reviewed commit and `codoxear-server.service` is active/running with `WorkingDirectory=/home/yiwen/.local/share/codoxear/deploy`.
- No live login, authenticated API session request, broker restart, or backend CLI restart was performed.

## 2026-09-11T12:14:58+08:00

- Accepted the follow-up UX correction: content-sized busy/idle activity bubbles with bounded wrapping, provider-prefix removal from model labels, and `tokens` in place of `tokens used`. Updated `PROMPT.md` before production edits.
- Docker real-UI baseline at 390×844 localized the sizing mechanism. With one short child (`qa · gpt-5.6-sol · tools: 1 · tokens used: 84`), the busy bubble still measured 325.59px inside a 370px row—exactly its phone `max-width: 88%`—because `.subagentDetails` computed to `flex: 0 0 100%`, `width: 299.59px`. With a qualified model, a child line computed `white-space: nowrap`, `overflow: hidden`, `clientWidth: 300`, `scrollWidth: 385`, proving clipping rather than wrapping.
- Predicted correction: make the two activity bubbles intrinsic inline grids, span details across the summary columns without percentage width/flex basis, and let lines wrap anywhere. A browser-only style probe supported the mechanism: the short busy bubble shrank from 325.59px to 273.42px, while the long case remained bounded at 325.59px and its line `scrollWidth` fell to its 300px `clientWidth` through wrapping.

## 2026-09-11T12:40:58+08:00

- Implemented only the requested follow-up: intrinsic grid sizing/wrapping for the existing busy and idle activity bubbles, qualified-model suffix display, and `tokens` label. Updated AGENTS and project architecture contracts; no sidebar, panel, toggle, state, producer, or counter logic changed.
- Added parsed-cascade CSS behavior coverage and updated executable transcript formatter/runtime expectations. Docker focused gate after rebuilding the tracked bundle: `76 passed in 9.89s`.
- Full Docker suite: `1879 passed, 112 subtests passed`; the same three unrelated failures remain—two PDF.js tests require a newer Node runtime than sandbox Node 18, and the deploy-script clean-snapshot test rejects the intentionally dirty shared checkout. Both new CSS tests and all scoped transcript tests passed.
- Docker real-server/browser fixture used a container-only Pi socket, Pi JSONL, and lifecycle-v3 status file; `/api/sessions` supplied qualified models and telemetry through the real catalog/store/transcript path. No fake DOM/store or reveal interaction was used.
- Busy and idle were each verified at 390×844 and 1280×800 with short `dexgem-responses/gpt-5.6-sol` and long `anthropic/claude-sonnet-4-5-20250929` API model values. The UI rendered `gpt-5.6-sol` / `claude-sonnet-4-5-20250929` and `tokens`; the provider prefixes and `tokens used` were absent.
- Short phone widths were 245.92px busy and 259.92px idle inside a 370px row, rather than the 325.59px maximum seen before correction. Long phone widths remained bounded at 325.59px; both lines wrapped to 28px (two 14px lines), and line/bubble `scrollWidth == clientWidth`. Desktop short/long bubbles remained intrinsic at 245.92/410.88px busy and 259.92/424.88px idle inside an 876px row. Body/document horizontal overflow was zero in all cases.
- Retained screenshots and raw geometry/API/browser diagnostics at `/tmp/codoxear-subagent-ux-results-2026-09-11/`; `verification-summary.json` is the concise sizing ledger and `SHA256SUMS` covers every screenshot and raw JSON artifact. Browser errors and console messages were empty after clearing expected pre-login diagnostics.
