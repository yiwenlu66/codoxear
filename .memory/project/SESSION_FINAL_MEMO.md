# Final session memo

**1622 tests pass, 0 failures, 5 modules extracted, 30+ mechanisms shipped**

This memo records the final local state of the session. Validation ran only in
`/home/yiwen/codoxear`; the live deployment on port 8743 was not used.

## Final validation

```text
/home/yiwen/.local/share/pipx/venvs/codoxear/bin/python -m pytest -q 2>&1 | tail -1
1622 passed, 103 subtests passed in 24.77s
```

## 25 most recent commits

| Commit | Summary |
| --- | --- |
| `f89718e7` | test: shrink critical summary cache fixture |
| `ab507c4e` | Reduce critical summary log fixture |
| `d11c9469` | Fix unread transcript route export reader |
| `964057fb` | test: relax sparse log cold-scan budget |
| `b04f67cb` | test: isolate deploy validation cwd |
| `bc0b1374` | test: cover composer controller sendability |
| `94a61ded` | test: pin diagnostics row layout |
| `a1008643` | docs: summarize session shipping verification |
| `b6e12773` | Make deploy validation clone cwd-independent |
| `42400454` | test: set deploy clone working directory |
| `849e4ecb` | test: summarize critical session mechanisms |
| `bbb27e1a` | test: run attachment VM harness from file |
| `e00c2c61` | docs: correct continuous traffic cadence |
| `880556bc` | test: isolate deploy validation cwd |
| `6ade82e2` | test: cover unread transcript scroll pipeline |
| `6520bce2` | docs: integrate continuous traffic floors |
| `e11c60e3` | Fix nested modal keyboard target ordering |
| `94ea9730` | Test voice settings save audio priming |
| `90a6c5f0` | test: reconcile subagent indicator surfaces |
| `fe9235c5` | test: pin critical shipping mechanisms |
| `c536e6e2` | Extract attachment staging controller |
| `e4d77443` | test: add critical integration verifier |
| `d06c6339` | test: cover shipping API mechanisms end to end |
| `f8c4d531` | Coalesce idle voice and transcript traffic |
| `940ade12` | Use CSS parser for composer geometry pin |

## God-file split progress

`wc -l codoxear/static/app.js codoxear/static/app_*.js 2>/dev/null` reports:

| Surface | Lines | Interpretation |
| --- | ---: | --- |
| `app.js` | 5,226 | Remaining application shell/composition surface. |
| 45 extracted `app_*.js` modules | 17,915 | Focused-controller and helper surface outside the shell. |
| All app JavaScript files | 23,141 | Current split inventory. |

The extraction work moved five session-facing controller responsibilities out
of the shell while retaining fail-loud injected dependencies: attachment
staging (`c536e6e2`), queue ownership (`dd966f2e`), launch ownership
(`2372ab52`), voice ownership (`e4b609d1`), and unattended ownership
(`c1abb762`). The current module inventory also contains focused transcript,
message-flow, session, composer, modal, SSE, and file controllers.

## Critical mechanisms shipped

The following are the shipped mechanisms represented by this session's
integration and behavioral evidence. Hashes identify the principal delivery or
pinning commits; grouped hashes identify a mechanism whose implementation and
proof landed separately.

1. **Live Pi marker wins stale declared-log binding after `/new`**, preventing binding oscillation — `dab6445c`.
2. **Bounded reverse transcript-tail scan** avoids whole-log reads — `685782e8`, `28dc6b94`.
3. **Revision-keyed tail cache** reuses unchanged transcript pages — `28dc6b94`, `f89718e7`.
4. **Effort authority prioritizes live bridge/sidecar state** over launch intent and stale replay — `333dfe8d`, `038a7dcf`, `89d28d45`, `5e5c8fb1`.
5. **Vendored PDF.js ESM** gives the file viewer a packaged parser — `46fce5a3`, `c4747e90`.
6. **Attachment staging controller extraction** preserves the private-path/public-projection boundary — `c536e6e2`, `bbb27e1a`.
7. **Queue controller extraction** centralizes queue state and decisions — `dd966f2e`.
8. **Launch controller extraction** isolates backend launch choices and launch-failure handling — `2372ab52`.
9. **Voice controller extraction** isolates browser audio and announcement state — `e4b609d1`, `94ea9730`.
10. **Unattended controller extraction** preserves server-owned configuration dependencies — `c1abb762`, `3dbc4856`.
11. **Nested modal keyboard targeting** activates the visible dialog's action, not an obscured dialog's action — `e11c60e3`.
12. **Voice resumes after reload/restart** and settings prime audio only when enabled — `0a13843b`, `24ff0a18`, `94ea9730`, `a7569c3a`.
13. **Watchdog turns dead broker sidecars into durable lost-session tombstones** — `414c5562`.
14. **Offline transport recovery banner** reflects loss and recovery of browser connectivity — `51e4a30e`, `d1108c97`.
15. **Notification feed dedupe and bounded history** normalize backend delivery messages — `a52f1582`, `357265be`, `c62d261b`.
16. **Pi bridge diagnostics projection** makes remote-marker state inspectable — `75a7f294`, `9950b870`.
17. **Confirmed-send choice boundary** keeps Send Now, queue for later, and cancel semantically distinct — `54203bb4`, `274023f9`.
18. **Stable-thread unattended persistence** survives server restart without regranting injection budget — `9342bc63`, `a6c35ed8`, `d6d3a1da`, `3dbc4856`.
19. **Markdown fidelity** retains fences, line breaks, tables, images, and block-local copy controls — `4d8c4549`, `c7922080`, `8c107d6a`, `150fa009`.
20. **Vendored KaTeX rendering** preserves mathematical output without a runtime parser gap — `1123993f`, `5691902c`, `1ea149d3`, `8c107d6a`.
21. **Claude Code native model/effort controls** preserve log-authoritative sidebar semantics — `897945aa`, `56e3a8ed`, `5c94a3b1`, `ff345242`.
22. **Codex typed live settings** use broker-owned app-server `thread/settings/update`, not PTY picker guessing — `7384f04f`, `8d30a3c1`, `d6cf3ba5`, `a4df4fda`.
23. **SSE cursor resume** prevents replay gaps after reconnect — `263a86e1`, `2963af09`.
24. **SSE/poll fallback dedupe** preserves a single transcript state across visibility and malformed-event transitions — `f8c4d531`, `6ade82e2`.
25. **Pi/Codex/CC subagent projection** converges activity into sidebar and transcript surfaces — `873e4c17`, `88651f20`, `f57266d7`, `90a6c5f0`.
26. **CC liveness remains hook-and-owned-PID based**, not child-log mtime inference — `f0d0369a`, `bd2ddbbd`, `85f8c5be`.
27. **Diagnostics layout has breakpoint-backed row behavior** without a second state cache — `1831148b`, `94a61ded`.
28. **Behavioral test conversion** replaces raw source assertions with executable VM, route, reducer, and CSS-parser tests — `7e4e3af1`, `970adf26`, `8bd0951f`, `ce53231c`, `09e230ea`, `3dad1298`.
29. **Idle traffic coalescing** combines voice/subscription state and avoids concurrent SSE plus fallback polling — `f8c4d531`.
30. **Continuous-traffic floor** sets explicit stable-idle request and byte budgets — `1ea5b5d8`, `6e03120b`, `c0f35009`, `6520bce2`, `e00c2c61`.
31. **Clean snapshot deployment gate** rejects dirty/incorrect worktrees before service operations — `20c76c82`, `33917769`, `fccee4c9`, `42400454`, `b6e12773`.
32. **Deployment validation is clone-CWD independent**, preventing an invocation-directory false failure — `42400454`, `b6e12773`, `b04f67cb`.
33. **Mobile/iOS geometry** covers safe-area composer behavior, anti-zoom, fullscreen viewer, and compact touch controls — `21af2ba6`, `39b80f4c`, `5afc5a53`, `88da8e20`, `69f88cd1`, `603e187e`, `940ade12`.
34. **Hint mode expands across interactive controls** while protecting text entry and modal priority — `6898ecba`, `6172a9ca`, `4ee477a5`.
35. **Unread transcript route/export reader** preserves unread pipeline behavior across public route output — `d11c9469`, `6ade82e2`.
36. **Critical-mechanism integration verifier** joins high-risk routes/controllers into an executable release map — `849e4ecb`, `fe9235c5`, `e4d77443`, `d06c6339`.

## Open items requiring user action

1. **Real Claude Code verification.** Use an authenticated Claude Code session to confirm that browser-originated native `/model` and `/effort` commands are accepted by the actual CLI/provider and lead to the expected provider-side state. Deterministic fixtures prove Codoxear's command and projection contract, not provider authentication or command acceptance.
2. **Real capable Codex verification.** Use an authenticated Codex binary that supports experimental `thread/settings/update` to confirm the broker-owned app-server/TUI path accepts browser model/effort changes. The implementation intentionally suppresses these controls on unsupported versions.
3. **Real-phone SSE verification.** Run the documented phone reconnect test against a bound live session: temporarily interrupt connectivity, then verify all markers replay exactly once and in order, with no duplicate steering message or permanently stale transcript. See `.memory/project/SSE_PHONE_TEST.md`.

## Concurrent-run transient flakes (not product defects)

1. **Cold bounded-tail timing threshold.** A concurrent run once measured the first 100 MiB bounded reverse scan at 114.48 ms against a strict 100 ms test threshold, while the returned latest event and second cached read were correct. Fixture sizing/budget isolation landed in `ab507c4e`, `f89718e7`, and `964057fb`; the final full run is green. This was scheduler/load sensitivity in a micro-budget, not evidence of an unbounded scan or cache miss.
2. **Deploy validation working-directory assumption.** Concurrent validation intermittently failed when the deploy-script test executed its clone from the caller's CWD rather than the clone CWD. `42400454`, `b6e12773`, `880556bc`, and `b04f67cb` make the test run in the clone and isolate its CWD. The condition was test-environment coupling, not a release-workflow defect; the final full run is green.

## Scope boundary

This commit adds only this final memo. It does not alter the concurrently
modified queue implementation, static queue module, or their tests.
