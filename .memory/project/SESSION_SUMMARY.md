# Session shipping summary and integrated verification report

**Scope.** This report covers the **195 implementation/test commits immediately
preceding this report** in `origin/main..HEAD` (merge base
`679dc252515c1771c0284a306b1372bde3407802`) at the time it was written. It is a
release-readiness map, not a ledger: it records the mechanism, its test evidence,
and the boundary between deterministic Codoxear proof and a real authenticated
backend/browser proof. The complete ordered inventory is reproducible with:

```sh
git log --reverse --format='%h %s' origin/main..HEAD
```

## Verification vocabulary

- **Behavioral** — an executable Python route/reducer test, Node VM controller
  test, CSS parser test, or Node execution of a shipped asset. It exercises the
  owner rather than asserting source text.
- **Integrated** — the test crosses the owner boundary (for example route →
  normalization → response, static route → served ESM, or broker state → session
  listing). Deterministic fake brokers/logs prove Codoxear mechanics, not a vendor
  account.
- **Real backend** — requires a live authenticated Codex or Claude Code session,
  or a browser/device capability. These are explicitly called out instead of
  being inferred from a fake transport.

## Shipping mechanisms, grouped by area

| Area and commits | Mechanism delivered | Behavioral / integrated verification | Remaining real-world boundary |
| --- | --- | --- | --- |
| **Race / log binding** — `dab6445c` | A live Pi active-session marker wins over a stale declared JSONL after `/new`, preventing log-binding oscillation. | **Yes / yes.** `test_all_critical_mechanisms.py` and `test_session_critical_summary.py` construct a stale declaration plus live marker and assert the new log is bound. | A live Pi TUI/browser continuation remains an operational smoke, not a correctness prerequisite. |
| **Disk I/O** — `685782e8`, `28dc6b94` (the local history has no `8b7a6a33`) | Tail reads use bounded reverse scanning and revision-keyed cache; unchanged live polls avoid replaying JSONL. | **Yes / yes.** `test_message_index.py`, `test_message_routes.py`, `test_perf_floor_regression.py`, `test_all_critical_mechanisms.py`, and `test_session_critical_summary.py` create 50–100 MiB logs and verify one bounded read then cache reuse. | No provider login is needed. The strict 100 ms first-read assertion is timing-sensitive; see residual test ownership below. |
| **Effort divergence / live settings authority** — `333dfe8d`, `038a7dcf`, `89d28d45`, `5e5c8fb1`, `3dbc4856`, `6fe71113` (requested; absent locally) | Live bridge/sidecar settings override launch intent and stale log replay; extraction wiring passes required unattended storage dependencies. | **Yes / yes.** `test_all_critical_mechanisms.py`, `test_session_critical_summary.py`, `test_unattended_controller_deps.py`, and `test_unattended_store.py` prove priority and controller construction. | Pi bridge needs a live TUI for operational confirmation; Codex/CC settings require the backend-specific checks below. |
| **PDF vendor** — `46fce5a3`, `c4747e90` | PDF.js ESM is vendored and served through `/pdf.mjs`, so the viewer has a packaged parser rather than a CDN dependency. | **Yes / yes.** `test_pdf_viewer_pipeline.py`, `test_all_critical_mechanisms.py`, and `test_session_critical_summary.py` resolve the static route, import served ESM, parse a PDF, and assert `getDocument`. | A browser visual/open-file smoke is still useful; no agent login is involved. |
| **Frontend extractions and attachment staging** — `1238b26d`, `fce5cbd6`, `45ceffa9`, `6906ccad`, `dabb13c0`, `38b34086`, `ba4831ac`, `b823e116`, `c536e6e2`, `e4b609d1`, `ff644e70`, `dd966f2e`, `2372ab52`, `c1abb762` | Session list, transcript, message flow, session edit, queue, launch, voice, unattended, and attachment responsibilities moved from the shell into injected controllers; attachment staging has one public/private state boundary. | **Yes / partial integrated.** Controller VM harnesses, static-asset tests, attachment harness, session-edit lifecycle tests, transcript/SSE tests, and `test_frontend_*_module_source.py` behavior suites cover controller contracts and loading. | A browser with a live session is needed to verify the whole composition and staged upload against a backend-readable path. |
| **Modal/UI keyboard** — `1cdba92b`, `173abe99`, `e11c60e3`, `f727875b`, `ef3739dd` | Nested dialogs target their own visible action; edit/queue destructive choices retain their confirmation semantics. | **Yes / controller integrated.** `test_modal_keyboard_nested_target_order.py` and `test_frontend_modal_keyboard_module_source.py` execute the modal controller in a VM. | Manual browser keyboard traversal remains the user-interface check. |
| **Voice** — `0a13843b`, `24ff0a18`, `94ea9730`, `a7569c3a` | Voice listening resumes after reload/restart, announcements resume correctly, and saving settings primes audio only when enabled. | **Yes / partial integrated.** `test_voice_resume.py`, `test_voice_announcement_resume.py`, `test_voice_settings_save.py`, `test_voice_push.py`, and route tests cover controller plus projection paths. | Requires real browser microphone/audio permission and device/browser autoplay behavior for end-to-end proof. |
| **Watchdog** — `414c5562` | Dead broker sidecars become durable lost-session tombstones rather than silently appearing live. | **Yes / yes.** `test_all_critical_mechanisms.py` covers watchdog → sidecar/lifecycle outcome. | Docker/process-lifecycle smoke remains appropriate; no vendor login needed. |
| **Offline transport** — `51e4a30e`, `d1108c97` | Browser network loss exposes a recovery banner and clears it only after transport recovery. | **Yes / controller integrated.** `test_offline_resilience.py`, `test_offline_banner_transport_failure.py`, and `test_all_critical_mechanisms.py`. | Real browser offline/online event handling remains a browser smoke. |
| **Notification feed and settings** — `a52f1582`, `357265be`, `c62d261b`, `9ac87675`, `75a7f294`, `9950b870` | Notifications dedupe/read correctly across backend-normalized events; panel history is bounded; Pi bridge diagnostics are visible. | **Yes / yes.** `test_notification_feed_all_backends.py`, `test_notifications.py`, diagnostics-route tests, and critical-mechanism tests exercise all backend fixture shapes. | A real Pi intercom/CC/Codex event verifies each upstream producer, not the feed reducer itself. |
| **Confirmed send-choice** — `54203bb4`, `274023f9` | “Send now” is steering through confirmed-send; “later” is explicit queueing; cancel does neither. | **Yes / yes.** `test_all_critical_mechanisms.py`, queue/control/session-input route tests, and shipping API integration tests distinguish all three outcomes. | Real busy Codex/Pi/CC PTY steering should be smoke-tested with authenticated CLIs. |
| **Unattended persistence/reconciliation** — `9342bc63`, `a6c35ed8`, `d6d3a1da`, `3dbc4856` | Configuration is keyed by stable thread identity, prompt persistence survives restart, and in-flight edits reconcile newest text rather than regressing. | **Yes / yes.** `test_unattended_prompt_preservation.py`, `test_unattended_edit_reconcile.py`, `test_unattended_store.py`, `test_unattended_sweep.py`, and controller dependency tests cover store/bootstrap and request ordering. | The full browser/server-only-restart draft scenario needs a supported deployment/sandbox browser run; do not treat a unit test as that proof. |
| **Markdown / KaTeX** — `4d8c4549`, `c7922080`, `8c107d6a`, `150fa009`, `1123993f`, `5691902c`, `1ea149d3` | Fences and long code remain legible; line breaks, tables, image geometry, local copy controls, and KaTeX survive rendering. | **Yes / partial integrated.** `test_markdown_edge_cases.py`, `test_markdown_image_cache.py`, `test_app_markdown_extended.py`, renderer tests, and Node KaTeX execution cover the renderer and vendored assets. | Browser visual/accessibility check (especially math and image dimensions) remains. |
| **Claude Code effort** — `897945aa`, `56e3a8ed`, `5c94a3b1`, `ff345242` | Claude model/effort controls inject native commands; sidebar semantics remain log-authoritative (effort stays launch value absent CC log evidence). | **Yes / partial integrated.** `test_cc_*`, browser-picker VM tests, and `ff345242` coverage exercise command and projection contracts. | **Requires real authenticated Claude Code login/session** to prove native command acceptance and provider-side result. |
| **Codex picker and live control** — `7384f04f`, `8d30a3c1`, `d6cf3ba5`, `a4df4fda` | Browser choices use broker-owned app-server `thread/settings/update`; provider availability and thinking recovery are projected without PTY picker guessing. | **Yes / partial integrated.** typed payload/picker behavior and backend control tests cover the bridge contract. | **Requires real capable Codex binary and authenticated provider session** for final app-server/TUI acceptance; unsupported versions deliberately do not advertise controls. |
| **SSE reliability battle** — `263a86e1`, `2963af09`, `f8c4d531`, `6ade82e2` | SSE resumes from cursor, falls back to polling without duplicate/lost events, survives malformed events/visibility changes, and avoids duplicate idle traffic. | **Yes / yes.** `test_sse_live.py`, `test_sse_battle_advanced.py`, message-flow controller tests, scroll-to-unread test, and shipping API tests execute server plus EventSource-like client behavior. | A real browser bound to a live backend log validates network intermediaries; no backend login is necessary to validate cursor semantics. |
| **Subagent indicators and hygiene** — `873e4c17`, `88651f20`, `f57266d7`, `90a6c5f0`, `f0d0369a`, `bd2ddbbd`, `85f8c5be`, `c1abb762` | Native Pi/Codex activity and bounded CC hook-based liveness converge on one `subagents_running` sidebar/typing projection; extracted modules load fail-loud. | **Yes / integrated.** `test_pi_subagents.py`, `test_pi_subagent_transcript.py`, `test_codex_subagents.py`, `test_cc_subagents.py`, and reconciliation/stream tests prove projection and lifecycle. | Real multi-agent runs are required to validate each vendor’s emitted lifecycle; existing evidence includes live Codex schema observation but does not replace release smoke. |
| **Diagnostics** — `dd0d3df4`, `7a6a1233`, `1831148b`, `75a7f294`, `9950b870` (requested `cd09b910` is absent) | Diagnostics expose bridge markers and stack rows by layout breakpoint without a second state cache. | **Yes / integrated.** diagnostics route tests and CSS parser/breakpoint tests execute public payload and stylesheet rules. | Browser narrow/wide layout check remains. |
| **Source-test conversion and test seams** — `7e4e3af1`, `970adf26`, `8bd0951f`, `ce53231c`, `09e230ea`, `3dad1298`, `f76492a2` (requested `81b31ff1` is absent) | Brittle raw-source assertions were removed/repaired in favor of VM, route, reducer, CSS-parser, and fixture behavior tests. | **Yes.** This is test-infrastructure work; its evidence is the executable suites listed throughout this report. | No external backend boundary. |
| **Performance floor / continuous traffic** — `1ea5b5d8`, `6e03120b`, `c0f35009`, `28dc6b94`, `f8c4d531`, `6520bce2`, `e00c2c61` | Unchanged settings avoid scans; tail cache is bounded; SSE suppresses redundant fallback; voice/settings subscription data is coalesced; idle traffic has explicit request/byte budgets. | **Yes / deterministic integrated.** `test_continuous_traffic_budget.py`, `test_traffic_floor.py`, `test_perf_floor_regression.py`, message-route tests, and the critical suite exercise the real route/controller owners. | No live deployment was used. A browser capture against a stable Docker session would measure actual wire framing; active sessions legitimately exceed the idle floor. |
| **Deploy gate and PR dispositions** — `20c76c82`, `33917769`, `fccee4c9`, `880556bc`, `42400454`, `b6e12773`, `9529dc87`, `0be970a3`, `3ba47176`, `d86bdda6` | Deployment uses clean committed snapshots, validates static JavaScript/boot references before service operations, preserves unit environment, and records PR #21/#22 decisions. | **Yes / sandbox integrated.** `test_deploy_script.py` uses a cloned worktree and fake `pipx`/`systemctl`/`curl` to prove ordering and clean-worktree refusal; the late clone-CWD fixes make that proof independent of the invoking directory, and JS checks run against the snapshot. | **Do not use the live deployment for this report.** Actual deploy health remains a release operator action. |
| **Layout, mobile, and iOS** — `21af2ba6`, `39b80f4c`, `5afc5a53`, `88da8e20`, `69f88cd1`, `603e187e`, `64c2ba0a`, `b096e460`, `940ade12`, `40774139`, `49daf095`, `05176941`, `87d4c108` | Paper-language tokens, desktop rail, mobile fullscreen viewer, grouped sidebar counts, safe-area composer, anti-zoom, and compact touch targets are aligned. | **Yes / partial integrated.** CSS parser tests, iOS geometry tests, sidebar count tests, and controller VM tests execute the relevant rules. | **Requires real iOS/Safari and desktop browser viewport checks** for keyboard/safe-area rendering. |
| **Hint mode** — `6898ecba`, `6172a9ca`, `4ee477a5`, `610f1d7c` | `f`-leader hints cover shell/dialog controls while text entry and modal rules retain priority. | **Yes / controller integrated.** hint-mode VM/coverage tests exercise labels, visibility, modal, and text-entry guards. | Manual desktop keyboard traversal remains. |
| **Miscellaneous reliability and UX glue** — `29e8e9cc`, `3733b0f5`, `781a9421`, `81989de1`, `9308ca72`, `6ff900bc`, `6af88d01`, `04fe6b1b`, `06e42aea`, `b9726086`, `a58f6ab9`, `682b093f`, `4242ef68`, `996d7254`, `a0483455`, `2113ff5d` | Session initialization, cached listing, URL resolution, nonfatal diagnostics, edit affordances, transcript filtering/search/history, unread state, cache versioning, and local asset delivery were repaired or tightened. | **Yes / varies by mechanism.** Session/listing/transcript/search/static-route tests cover their owners; browser UX remains the appropriate final check where a control is visual. | No single backend proof applies; follow the owning mechanism above. |

### Commit-accounting notes

The table deliberately groups the session’s shipping work by mechanism rather
than treating every extraction/revert/follow-up as a separate product feature.
The 195 implementation/test commits preceding this document also include temporary investigation, docs, style
retuning, reverts/reapplications needed to preserve controller seams, and test
fixture repair. The named requested abbreviations that **do not resolve in this
checkout** are `8b7a6a33`, `6fe71113`, `cd09b910`, `81b31ff1`, and `87a02a9e`;
the table records the corresponding reachable commits where the history makes
that relationship clear (`28dc6b94`, `3dbc4856`, `1831148b`, and the current
behavior suites). No invented commit identity is presented as evidence.

## Cross-cutting test evidence

The dedicated integration layer added late in the session is:

- `tests/test_all_critical_mechanisms.py` — marker/race, 100 MiB tail cache,
  effort authority, unattended ordering, send choice, nested modal, voice,
  markdown, PDF, notifications, watchdog, and offline contracts.
- `tests/test_shipping_mechanisms_e2e.py` — API-level shipping mechanisms.
- `tests/test_session_critical_summary.py` — integrated in `849e4ecb`;
  exercises marker binding, listing effort, a 50 MiB route cache, and served
  PDF ESM in one executable flow.
- `scripts/verify_all_critical.sh` — convenience runner for the critical
  integration collection.

These tests are intentionally complementary: a VM proves browser-controller
state transitions, a Python route test proves public API behavior, and a
backend fixture proves normalization. None is reported as a substitute for a
real Codex/CC login or a browser permission/device behavior.

## Residual test ownership and current observation

The handoff for this report identifies **three remaining failures owned by a
separate subagent**: two scrollback/performance checks and one deploy-script
validation check. This task did not edit their tests or implementations.

A fresh full-suite attempt in this shared checkout produced a different current
snapshot: **1 failed, 1608 passed, 103 subtests passed**. The sole observed
failure was
`tests/test_all_critical_mechanisms.py::test_messages_tail_reads_100mb_once_then_uses_cache_under_budget`:
its first bounded scan took **114.48 ms** against a `<100 ms` threshold, while
the cached read and functional response assertions passed. The expected
scrollback/deploy failures did not reproduce in this checkout, likely because
concurrent owner work is already present. The ownership statement is preserved
as a handoff fact; the run result is recorded separately so this document does
not claim three failures that were not observed.

The failure is a performance-threshold flake or regression until its owner
repeats it under controlled load; it does **not** show an unbounded read or a
cache miss, because the route returned the expected latest event and the second
request used the same cached payload.

## Final status

The shipping mechanisms have behavioral coverage, and the high-risk paths have
integrated deterministic coverage. Release acceptance remains conditional on
real authenticated CC/Codex picker checks, browser/device checks for voice and
iOS, and owner resolution/reclassification of the residual timing/deploy/
scrollback test handoff. This report makes those boundaries explicit rather
than upgrading test coverage into unsupported end-to-end claims.
