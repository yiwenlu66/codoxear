# Refactor-entry checkpoint for `recovery/product-gaps`

Date: 2026-06-26
Branch: `recovery/product-gaps`
Latest functional code checkpoint: `be7eeb3 extract session runtime and route controllers`
Protected checkout: `/home/yiwen/codex-web` on `main` was not modified or merged.

This checkpoint records the product-gap recovery state before any broad structural/frontend refactor. It is not merge approval.

## Closed product gaps in this recovery branch

Recent committed recovery checkpoints include:

- Transcript/search reliability:
  - bounded transcript search hint payloads and bounded count hints;
  - older-match search paging from unloaded transcript windows;
  - exact-by-default search-count API semantics, with bounded lower-bound UI hints;
  - search navigation now preserves all-transcript count evidence during Next/Previous navigation so an enabled `0 loaded · N all` path can load the offscreen match;
  - oversized JSONL records skipped by bounded transcript search now mark `match_count_truncated` instead of overstating exactness.
- Transcript/log robustness:
  - malformed sidecars skipped fail-closed;
  - live JSONL partial reads bounded;
  - batch chat extraction shares single-event construction;
  - Claude Code id-less tool-use placeholders generated once.
- Launch/session metadata:
  - fresh tmux launch metadata requires a live broker pid;
  - launch sidecar metadata validation hardened;
  - Pi provider/model launch path now passes explicit custom providers through instead of treating UI defaults as an API whitelist;
  - sidecar metadata schema/capability parsing now lives in `codoxear/sidecar_metadata.py`, while server discovery/refresh call sites keep fail-closed aliases;
  - failed web-owned launches now appear as recoverable non-session rows with a redacted in-chat recovery card, review-only New like this action, Dismiss/Copy details actions, and disabled send/enqueue/attach paths.
- Pi live-backed launch path:
  - `model_provider=anthropic`, `model=claude-haiku-4-5`, `reasoning_effort=low` launched through the web path, accepted a send, bound a log, produced an assistant final response, reached idle, and cleaned up in isolated Codoxear app state.
- Pi busy-after-interrupt recovery:
  - explicit web ESC is tagged only after a successful broker write and broker `interrupted_idle` is published only after busy actually clears;
  - Pi tool-call accounting tracks arbitrary string IDs exactly, including empty/whitespace/sentinel-looking IDs, preserves duplicate-ID multiplicity, and keeps absent/non-string IDs busy-closed until final/abort/error;
  - Pi registration, bind/rebind, and live tailing seed from complete JSONL rows without advancing over partial rows, replace stale pending calls on log switch, and discard stale tail batches;
  - confirmed-send barriers now require parseable JSON object row evidence and block send/queue/attachment plus list/messages/diagnostics busy display until resolved.
- Codex and Claude Code live web-send/log-bind paths:
  - isolated direct web-owned Codex broker under temp Codoxear app state reproduced a binding failure when Codex logged cwd as `/.tmp-on-ssd/...` while Codoxear filtered on `/tmp/...`;
  - rollout/new-log discovery now matches cwd by exact string or existing absolute filesystem identity (`samefile`) while failing closed for relative, tilde, unknown-user, and nonexistent payload cwd aliases;
  - after the Codex fix, the isolated browser composer sent a prompt, `/messages/tail` showed the expected user/assistant sequence ending in `CODEX_WEB_LIVE_OK_20260615`, and the session returned idle;
  - isolated Claude Code evidence showed CC writes logs under `.claude/projects` without a writable log fd visible to the current `/proc` discovery path, so CC now has a bounded closed-log fallback constrained by launch time, preexisting paths, and cwd identity;
  - CC header extraction now merges early metadata rows because live CC mode/permission rows carry `sessionId` before the first row with `cwd`;
  - after the CC fallback/header fix, a browser-sent prompt rebound placeholder `broker-1630561` to real thread `410ef3d0-6967-49cd-9488-45b30c40f5d6`; Claude's upstream gateway returned a terminal 503, which Codoxear rendered as an assistant API-error row and then reported idle.
- File/inline reference UX:
  - file candidates remain visible without git state;
  - equivalent inline refs merge only after inspected identity;
  - failed inline file inspections are not cached as durable facts;
  - file viewer modal focus restored;
  - sessions rooted at `/` can create valid relative descendant files through `/file/write` without the prior root-prefix false rejection;
  - git subprocess/path/pathspec/numstat/worktree helper logic now lives in `codoxear/git_ops.py`, with server wrappers preserving private names and `_run_git` patch seams.
- Queue/send/unattended UX:
  - unattended prompts gate on final assistant turns;
  - mobile composer stop control added;
  - read endpoints remain observation-only and do not promote queued prompts;
  - orphan, queued-recovery, and unknown-send sessions now render an in-chat recovery panel with safe review actions instead of opening to an empty disabled pane.
- Bounded frontend refactor tranches:
  - app URL/base-path resolution moved from `app.js` into `codoxear/static/app_url.js`;
  - local-storage access moved from `app.js` into `codoxear/static/app_storage.js`;
  - performance-sampling diagnostics moved from `app.js` into `codoxear/static/app_perf.js`;
  - API request/ETag/perf-sampling wrapper moved from `app.js` into `codoxear/static/app_api.js`;
  - file/viewer helper logic moved from `app.js` into `codoxear/static/app_file_helpers.js`;
  - session/sidebar render-state helper logic moved from `app.js` into `codoxear/static/app_session_helpers.js`;
  - viewport/media-query helper logic moved from `app.js` into `codoxear/static/app_viewport.js`;
  - polling-delay policy moved from `app.js` into `codoxear/static/app_polling.js`;
  - conversation-copy formatting moved from `app.js` into `codoxear/static/app_conversation_copy.js`;
  - video-preview error text formatting moved from `app.js` into `codoxear/static/app_file_helpers.js`;
  - markdown rendering, markdown cache, markdown preview image routing, and local file-reference parsing moved from `app.js` into `codoxear/static/app_markdown.js`;
  - launch/backend/default/provider/model-memory helpers moved from `app.js` into `codoxear/static/app_launch.js`;
  - display/formatting/icon helpers moved from `app.js` into `codoxear/static/app_display.js`;
  - `index.html` loads `app_url.js`, `app_storage.js`, `app_perf.js`, `app_api.js`, `app_markdown.js`, `app_launch.js`, `app_display.js`, `app_file_helpers.js`, `app_session_helpers.js`, `app_viewport.js`, `app_polling.js`, `app_conversation_copy.js`, then `app.js`; `app.js` fails loudly if any helper is missing;
  - all helper scripts participate in static asset versioning, top-level static routing, and wheel packaging;
  - frontend static asset registration now has a single server-side manifest for version-hashed frontend files and exact top-level static routes;
  - URL-prefix behavior remains the same algorithm (`/static/index.html`, `/static/`, otherwise current directory) with root-like app paths resolved under the computed app base;
  - storage-denial behavior remains the same guarded contract: unavailable/throwing storage yields `null` for reads and `false` for writes/removes;
  - performance diagnostics preserve the 200-sample window, nonnegative-value filter, percentile/rounding policy, and public `window.codoxearPerf` entry point;
  - API extraction preserves URL-prefix resolution, sessions ETag/304 reuse, private not-modified marker identity, JSON parse/error behavior, API timing sample names, and cleanup-time cache clearing through `window.CodoxearApi`;
  - file-helper extraction preserves literal file-list paths, location-suffix stripping behavior, text/diff kind decisions, blocked-file messages, and priority offset formatting through `window.CodoxearFileHelpers`;
  - session-helper extraction preserves failed/pending launch selectability, review/waiting/later grouping, sidebar entry/signature semantics, and fast-session detection through `window.CodoxearSessionHelpers`;
  - viewport-helper extraction preserves mobile width, reduced-motion, desktop-action, and touch-control media query semantics through `window.CodoxearViewport`, including the pre-existing `isMobile()` undefined/falsy return when `matchMedia` is absent;
  - polling-helper extraction preserves session/secondary visibility delays, message poll fast/running/idle/hidden/offline/error-backoff branch order, and kick-delay normalization through `window.CodoxearPolling`, while timers and mutable polling counters remain in `app.js`;
  - conversation-copy extraction preserves role filtering, falsy-text coercion, trailing-whitespace trim, locale timestamp labels, section headers, and separator formatting through `window.CodoxearConversationCopy`, while API/clipboard/button/toast side effects remain in `app.js`;
  - video-preview error formatter extraction preserves message/string trimming and fallback text through `window.CodoxearFileHelpers`, while preview fetch/auth/video-state/DOM side effects remain in `app.js`;
  - markdown extraction preserves chat/file-preview wrappers, session-scoped image blob routing, local file-reference parsing, and non-literal `openFileReference()` parser behavior through the exported `window.CodoxearMarkdown` boundary;
  - launch-helper extraction preserves Pi/Codex/Claude backend normalization, default launch settings, provider/model memory, model-specific reasoning choices, providerless Pi model memory, Claude provider ignoring, URL-prefixed logo paths, and app-owned failed-launch redaction;
  - display-helper extraction preserves tooltip fallback priority, byte/time/relative-age formatting, session display labels, short session IDs, and SVG icon markup through `window.CodoxearDisplay`;
  - recovery prompt preview and recent-cwd fuzzy scoring now live in the display-helper boundary while recovery security/session state and recent-cwd UI state remain app-owned;
  - file-picker matching/scoring extraction preserves exact/basename/token/subsequence scoring, draft path normalization, Unicode folded match ranges, normalized candidate scoring, and comparator tie-breaks through `window.CodoxearFileHelpers`, while file-search state, candidate maps, API calls, DOM highlighting, picker rendering, file-open actions, validation caches, and file-viewer state remain app-owned;
  - chat-search display extraction preserves snippet compaction and all-transcript hint role labeling through `window.CodoxearDisplay`, while row text extraction, rendered-row matching, search timers, transcript-search API calls, loaded/all count state, DOM status updates, focus/navigation, and load-older actions remain app-owned;
  - file-picker source/label extraction preserves candidate-source normalization and source-section label text through `window.CodoxearFileHelpers`, while file identity keys, clone/merge semantics, candidate maps/cache, changed-file API calls, DOM section insertion/rendering/highlighting, active file/open behavior, file-viewer state, focus/timers, and recovery/security behavior remain app-owned;
  - file-editor cursor-helper extraction preserves inserted-text cursor arithmetic through `window.CodoxearFileHelpers`, while Monaco/editor access, paste execution, file dirty state, touch-selection reset, selection application, focus, DOM, file-viewer availability, save/edit behavior, timers, APIs, and recovery/security behavior remain app-owned;
  - file-editor delete-key helper extraction preserves key-to-Monaco-delete-command mapping through `window.CodoxearFileHelpers`, while event filtering/lowercasing, native delete suppression, Monaco/editor access, `editor.trigger`, touch-selection reset, focus, toast/error behavior, DOM, file-viewer availability, save/edit behavior, timers, APIs, and recovery/security behavior remain app-owned;
  - model-option match helper extraction preserves deterministic model-search text matching through `window.CodoxearLaunch`, while model-option construction, exact/prefix/contains ordering, result slicing, provider/model selection, rendering, local/session state, memory persistence, focus/menu behavior, APIs, DOM, timers, recovery/security behavior, and launch-dialog state remain app-owned;
  - diagnostics helper extraction preserves provider-display formatting and copy-text row formatting through `window.CodoxearSessionHelpers`, while backend normalization remains `app_launch.js`/app-owned and diagnostics API fetch, row construction, mutable copy state, clipboard, DOM/buttons/backdrop, focus, auth-loss handling, error recovery, timers, and recovery/security behavior remain app-owned;
  - queue normalizer extraction preserves modern/legacy queue API-payload normalization through `window.CodoxearSessionHelpers`, while queue refresh, API fetch, auth/error handling, draft preservation, viewer item assignment, empty text, rendering, mutation locks, move barriers, send/enqueue/delete/update/move behavior, DOM, focus, timers, and recovery/security behavior remain app-owned.
- Backend/server architecture tranche:
  - `codoxear/session_runtime.py` is the shared runtime authority for broker busy state, queue length, interrupted-idle override, confirmed-send boundary gating, remote readiness, and token fallback selection; message polling, diagnostics, queue promotion, direct send readiness, attachment readiness, and sidebar/list-session busy display now call through that mechanism instead of each recomputing partial busy/ready policy;
  - `codoxear/session_store.py` owns persistent session maps for unattended config, aliases, sidebar metadata, hidden sessions, file history, queues, pending attachment IDs, commit-unknown sends, and recent cwd records, while `SessionManager` keeps compatibility properties and runtime/control behavior;
  - message/tail/live/history/search/export controller behavior now lives in `codoxear/message_routes.py`, with cursor HMAC encode/decode injected from the server secret boundary and cursor mismatches still surfacing as conflict responses;
  - session file read/write payload parsing, optimistic version checks, create/update behavior, and symlink-parent escape coverage now live in `codoxear/file_routes.py`, while lower-level path, file-kind, and response primitives remain in their existing modules;
  - failed-launch ledger/redaction/transcript fallback behavior now lives in `codoxear/launch_ledger.py`;
  - queue/enqueue HTTP validation, status mapping, and legacy `queue` response compatibility now live in `codoxear/queue_routes.py`, while queue mutation and recovery barriers remain manager-owned;
  - session control POST validation/status mapping for delete/edit/rename/send/unattended/interrupt/attachment injection/pending-clear/commit-unknown-clear now lives in `codoxear/control_routes.py`, while manager methods remain the control-plane mutation boundary;
  - diagnostics GET composition now lives in `codoxear/diagnostics_routes.py`, sharing `session_runtime.py` for busy/ready/token state and keeping provider/sidebar/git details as explicit dependencies;
  - Git changed-files, diff, and file-version API semantics now live in `codoxear/git_routes.py`, with `git_ops.py` remaining the subprocess/path/pathspec authority and server providing dependency seams.
- Browser/desktop UX:
  - desktop notifications focus the target session;
  - Pi custom provider/model browser behavior now has executable JS/VM coverage;
  - long-transcript per-message copy controls now use a roving active button so the accessibility/tab order has one enabled copy control instead of one repeated control per rendered message;
  - Details diagnostics can be copied from the dialog using only rendered label/value rows, not the raw diagnostics object;
  - Details can open a review-only New Session dialog with copied launch settings from an allowlisted diagnostics subset; Pi provider semantics use actual `model_provider`, not synthetic `provider_choice`, including providerless and sparse-metadata cases;
  - file picker search results highlight exact/fuzzy query matches using DOM text nodes and Unicode-safe folded-index mapping, without changing path identity;
  - markdown fenced code blocks use a light Codoxear-themed surface, and markdown tables wrap/contain normal wide content with internal horizontal scroll only for impossible many-column cases;
  - video files expose an explicit compatible-MP4 preview action that preflights the server transcode route, surfaces route/ffmpeg errors in status text, and avoids relying only on opaque media-element errors.

## Latest validation evidence

Latest code-validation evidence after the Codex and Claude Code live binding repairs:

- Focused CC/Codex discovery validation after hardening: `python3 -m py_compile codoxear/util.py codoxear/broker.py codoxear/cc_log.py tests/test_broker_proc_rollout.py tests/test_claude_backend_source.py tests/test_cc_log.py` plus `tests/test_broker_proc_rollout.py`, `tests/test_claude_backend_source.py`, `tests/test_cc_log.py`, `tests/test_cc_chat_and_idle.py`, `tests/test_cc_busy_state.py`, `tests/test_session_resume.py`, and `tests/test_stale_sidecars.py` -> `104 passed`.
- Full local suite: `python3 -m pytest -q` -> `952 passed, 104 subtests passed`.
- Docker sandbox suite: `scripts/codoxear-docker-sandbox test` -> `951 passed, 1 skipped, 104 subtests passed`.
- Isolated live Codex proof: temp HOME/app state on port 19044, real `CODEX_HOME`, direct web-owned broker, temp cwd trust accepted, bootstrap log bound, browser composer sent the final prompt, `/messages/tail` and browser DOM showed assistant `CODEX_WEB_LIVE_OK_20260615`, and session state returned `busy=false`, `queue_len=0`.
- Isolated live Claude Code proof: temp HOME/app state on port 19048, real Claude config symlinked without printing secrets, direct web-owned broker, temp cwd trust accepted, browser composer sent the prompt, fallback rebound `broker-1630561` to thread `410ef3d0-6967-49cd-9488-45b30c40f5d6` and log path `/home/yiwen/.claude/projects/--tmp-on-ssd-codoxear-live-cc4-EeoSB2-work/410ef3d0-6967-49cd-9488-45b30c40f5d6.jsonl`, `/messages/tail` and browser DOM showed the user prompt plus Claude's synthetic assistant API-error row, and session state returned `busy=false`, `queue_len=0`.
- Clean-room critic subagent `5df64f7b-12c0-4e8c-a65b-f36985c79e35` returned `NO BLOCKERS` for the earlier Codex cwd-alias fix.
- Clean-room critic subagent `05290a8a-033a-46c1-ab02-c0d8f52d3254` found two CC fallback blockers: post-fork known-log snapshotting could skip a fast-created CC log forever, and relative `--cwd` could fail absolute CC cwd matching. Both were fixed by prelaunch snapshotting, absolute broker cwd expansion, and focused regressions.
- Clean-room critic subagent `62c6924a-cbdf-4535-b3d8-d6886680fd2a` confirmed those fixes, then found a large-first-row blocker in the bounded CC header scan. The cap now bounds row start offsets, so a valid first CC row larger than 512 KiB remains discoverable while rows starting after the window remain ignored.
- Final narrow critic subagent `6f5dbf25-e41e-4467-8760-66e781c6809e` returned `NO BLOCKERS` for the committed CC fallback/header repair (`c1280cb fix Claude Code closed-log binding`).

Latest Docker-only evidence after backend/server architecture tranche:

- Runtime/controller architecture focused validation: `python3 -m pytest -q tests/test_file_inspect.py tests/test_git_ops.py tests/test_file_picker_search_source.py tests/test_session_resume.py` -> `130 passed, 52 subtests passed`; earlier focused route/runtime/store/controller validation after diagnostics extraction -> `54 passed, 4 subtests passed`.
- Full Docker sandbox suite after the stable server architecture diff: `CODOXEAR_DOCKER_PORT=18920 scripts/codoxear-docker-sandbox test` -> `1028 passed, 1 skipped, 107 subtests passed`.
- Fresh clean-room architecture review `04337c97-39f1-4aff-9221-8bb90c030a3e`, saved to `/tmp/codoxear-architecture-runtime-routes-review.md`, returned `NO BLOCKERS` for runtime busy/ready/token semantics, broker-queue log-parse short-circuiting, `SessionManager.__new__` compatibility, persistence path rebinding, queue/control/diagnostics/git route error mappings, file write security, message cursor/HMAC behavior, and launch redaction/recovery.
- Functional architecture commit created: `be7eeb3 extract session runtime and route controllers`. This evidence does not claim real mobile/device behavior, live backend lifecycle expansion, or completion of the remaining inline file/tail/unattended route seams.

Latest Docker-only evidence after frontend helper extractions:

- URL helper focused Docker validation: `scripts/codoxear-docker-sandbox test tests/test_frontend_url_module_source.py tests/test_static_assets.py tests/test_url_prefix.py tests/test_session_polling_source.py` -> `29 passed, 3 subtests passed`.
- URL helper Docker runtime route check under `CODEX_WEB_URL_PREFIX=/codoxear`: in-container requests returned `/codoxear/api/me -> 401`, `/codoxear/app_url.js?v=test -> 200` with helper content, and `/codoxear/app.js?v=test -> 200`.
- URL helper full Docker sandbox suite: `scripts/codoxear-docker-sandbox test` -> `955 passed, 1 skipped, 107 subtests passed`.
- URL helper read-only critic subagent `82cb6205-46b9-428c-97e2-ded96036dd5a` returned `NO BLOCKERS` for static serving, asset versioning, packaging, script ordering, fail-loud behavior, URL-prefix behavior, CSP, service-worker path behavior, and broad UI semantic scope. It did not run tests.
- Host-side prefix server/browser evidence for the URL helper tranche was discarded after the user clarified validation must be Docker-only.
- Storage helper focused Docker validation: `scripts/codoxear-docker-sandbox test tests/test_storage_robustness_source.py tests/test_static_assets.py tests/test_frontend_url_module_source.py tests/test_file_picker_session_state.py tests/test_new_session_model_options_source.py` -> `39 passed, 3 subtests passed`.
- Storage helper Docker runtime route check under `CODEX_WEB_URL_PREFIX=/codoxear`: in-container requests returned `/codoxear/api/me -> 401`, `/codoxear/app_url.js?v=test -> 200`, `/codoxear/app_storage.js?v=test -> 200` with helper content, and `/codoxear/app.js?v=test -> 200`.
- Storage helper full Docker sandbox suite: `scripts/codoxear-docker-sandbox test` -> `955 passed, 1 skipped, 107 subtests passed`.
- Storage helper read-only critic subagent `5679462c-49f1-4e14-aecb-ada1b99a3f80` returned `NO BLOCKERS` for storage-denial behavior, script ordering, fail-loud dependency, static routing/versioning/package inclusion, CSP/path behavior, and helper-name compatibility. It did not run tests.
- Perf helper focused Docker validation: `scripts/codoxear-docker-sandbox test tests/test_frontend_perf_module_source.py tests/test_static_assets.py tests/test_frontend_url_module_source.py tests/test_storage_robustness_source.py tests/test_session_polling_source.py` -> `31 passed, 3 subtests passed`.
- Perf helper Docker runtime route check under `CODEX_WEB_URL_PREFIX=/codoxear`: in-container requests returned `/codoxear/api/me -> 401`, `/codoxear/app_url.js?v=test -> 200`, `/codoxear/app_storage.js?v=test -> 200`, `/codoxear/app_perf.js?v=test -> 200` with `window.CodoxearPerf`, and `/codoxear/app.js?v=test -> 200`.
- Perf helper full Docker sandbox suite: `scripts/codoxear-docker-sandbox test` -> `958 passed, 1 skipped, 107 subtests passed`.
- Perf helper read-only critic subagent `690d8a0d-02e9-4a46-974c-f0d925df8523` returned `NO BLOCKERS` for sample-window/filter/percentile/rounding semantics, script order, `window.codoxearPerf` compatibility, static routing/versioning/package inclusion, CSP/path behavior, and Docker-only evidence. It did not run tests. Its non-blocking stale-HTML/new-JS note remains within the existing static-shell freshness limitation and is handled by no-store/versioning rather than silent fallback.
- Static registry focused Docker validation: `scripts/codoxear-docker-sandbox test tests/test_static_assets.py tests/test_url_prefix.py tests/test_frontend_url_module_source.py tests/test_storage_robustness_source.py tests/test_frontend_perf_module_source.py` -> `24 passed, 3 subtests passed`.
- Static registry Docker runtime route check under `CODEX_WEB_URL_PREFIX=/codoxear`: `/codoxear/api/me -> 401`; `/codoxear/app_url.js`, `/codoxear/app_storage.js`, `/codoxear/app_perf.js`, `/codoxear/app.js`, `/codoxear/app.css`, `/codoxear/favicon.png`, `/codoxear/manifest.webmanifest`, `/codoxear/service-worker.js`, and `/codoxear/static/app.js` all returned 200.
- Static registry full Docker sandbox suite: `scripts/codoxear-docker-sandbox test` -> `959 passed, 1 skipped, 107 subtests passed`.
- Static registry read-only critic subagent `35973f5f-5be7-4c9f-80f4-15ac71ced0e9` returned `NO BLOCKERS` for route mappings, URL-prefix behavior, `/static/*` preservation, version hash order/coverage, unchanged `_send_static`/CSP/cache/content-type/package behavior, and Python compatibility. Its non-blocking notes were pre-existing or non-behavioral for exact-route matching.
- Markdown helper focused local validation after extraction: `node --check codoxear/static/app.js`, `node --check codoxear/static/app_markdown.js`, and `python3 -m pytest -q tests/test_file_viewer_source.py tests/test_file_picker_search_source.py tests/test_markdown_renderer_source.py tests/test_markdown_tables.py tests/test_static_assets.py` -> `80 passed`.
- Markdown helper focused Docker validation: `scripts/codoxear-docker-sandbox test tests/test_file_viewer_source.py tests/test_file_picker_search_source.py tests/test_markdown_renderer_source.py tests/test_markdown_tables.py tests/test_static_assets.py` -> `80 passed`.
- Markdown helper Docker runtime route check under `CODEX_WEB_URL_PREFIX=/codoxear`: `/codoxear/api/me -> 401`; `/codoxear/app_markdown.js?v=test -> 200`.
- Markdown helper full Docker sandbox suite: `scripts/codoxear-docker-sandbox test` -> `962 passed, 1 skipped, 107 subtests passed`.
- Clean-room critic subagent `85d8069d-4e2f-40ef-bf96-eee6545b280d` found a real missing-global blocker after extraction: the non-literal `openFileReference()` path still called `parseLocalFileRef()` after it became private to `app_markdown.js`. The fix exported `parseLocalFileRef`, added the app wrapper/fail-loud check, and added an executable VM regression for the non-literal branch.
- Clean-room critic subagent `4773fc5a-bbbf-4d0e-b29d-e65061972fcf` returned `NO BLOCKERS` after the parser export fix; focused reviewer `de87e7a2-ae7e-4120-9b54-0f6ad5232a9d` also returned `NO BLOCKERS` for the five extraction acceptance blockers. Residual risks: non-literal direct calls with an inline suffix such as `src/app.py:7` still do not consume the parsed line when `ref.line` is absent, which appears pre-existing and outside the extraction regression; duplicate `stripPathLocationSuffix()` copies can drift if suffix semantics change.
- Launch helper focused local validation after extraction and storage-contract repair: `node --check codoxear/static/app.js`, `node --check codoxear/static/app_launch.js`, real-order VM load of `app_url.js` + `app_storage.js` + `app_launch.js`, and `python3 -m pytest -q tests/test_launch_ui_source.py tests/test_new_session_model_options_source.py tests/test_static_assets.py tests/test_claude_backend_source.py tests/test_reasoning_effort_source.py` -> `44 passed`.
- Launch helper focused Docker validation: `scripts/codoxear-docker-sandbox test tests/test_launch_ui_source.py tests/test_new_session_model_options_source.py tests/test_static_assets.py tests/test_claude_backend_source.py tests/test_reasoning_effort_source.py tests/test_frontend_url_module_source.py tests/test_storage_robustness_source.py` -> `50 passed, 3 subtests passed`.
- Launch helper Docker runtime route check under `CODEX_WEB_URL_PREFIX=/codoxear`: `/codoxear/api/me -> 401`; `/codoxear/app_launch.js?v=test -> 200`; `/codoxear/app.js?v=test -> 200`.
- Launch helper full Docker sandbox suite: `scripts/codoxear-docker-sandbox test` -> `964 passed, 1 skipped, 107 subtests passed`.
- Clean-room critic subagent `cf0565f1-c700-4f51-8dc3-d09808ca58b1` found a real storage-contract blocker: `app_launch.js` initially required non-existent `CodoxearStorage.storageGetItem/storageSetItem/storageRemoveItem` names while `app_storage.js` exports `getItem/setItem/removeItem`. The fix switched to the real storage API and added a real-order module-load VM regression.
- Focused reviewer `f3542620-2f37-4c18-9f1b-7de723caabf9` returned `NO BLOCKERS` after the storage-contract repair for wrapper coverage, module isolation, provider/default/reasoning behavior, app-owned failed-launch redaction, and static loading/versioning/routing/packaging. Residual risk: focused review scope only; broader browser UX was not re-reviewed in this tranche.
- Display helper focused Docker validation: `CODOXEAR_DOCKER_PORT=18836 scripts/codoxear-docker-sandbox test tests/test_frontend_display_module_source.py tests/test_button_tooltips_source.py tests/test_static_assets.py tests/test_send_button_source.py tests/test_file_viewer_source.py tests/test_launch_ui_source.py -q` -> `51 passed`.
- Display helper Docker runtime route check under `CODEX_WEB_URL_PREFIX=/codoxear`: `/codoxear/api/me -> 401`; `/codoxear/app_display.js?v=test -> 200`; `/codoxear/app.js?v=test -> 200`; `/codoxear/ -> 200` and the rendered index referenced `app_display.js?v=...`.
- Display helper full Docker sandbox suite: `CODOXEAR_DOCKER_PORT=18837 scripts/codoxear-docker-sandbox test` -> `969 passed, 1 skipped, 107 subtests passed`.
- Clean-room critic `a1a8273c-19a5-45d5-a395-d32c0a301ac9` returned `NO BLOCKERS` for contract drift, load order, static asset/versioning gaps, source-test blind spots, and runtime dependency/fallback issues. Its non-blocking guard suggestions were applied with formatter/icon coverage and versioned-index asset existence/route tests before commit.
- API helper focused Docker validation: `CODOXEAR_DOCKER_PORT=18844 scripts/codoxear-docker-sandbox test tests/test_frontend_api_module_source.py tests/test_session_polling_source.py tests/test_auth_cleanup_source.py tests/test_static_assets.py tests/test_frontend_url_module_source.py tests/test_frontend_perf_module_source.py` -> `40 passed, 3 subtests passed`.
- API helper Docker runtime route check under `CODEX_WEB_URL_PREFIX=/codoxear`: `/codoxear/api/me -> 401`; `/codoxear/app_api.js?v=test -> 200`; `/codoxear/app.js?v=test -> 200`; `/codoxear/ -> 200`, and `app_api.js` contained `window.CodoxearApi`.
- API helper full Docker sandbox suite: `CODOXEAR_DOCKER_PORT=18845 scripts/codoxear-docker-sandbox test` -> `973 passed, 1 skipped, 107 subtests passed`.
- Clean-room critic `8080791f-65d9-482f-8a84-4abb2388d6b5` returned `NO BLOCKERS` for contract drift, load order, static/version/package wiring, source-test blind spots, runtime dependency/fallback issues, cleanup cache clearing, 304 marker identity, URL-prefix resolution, and perf sample names. Its suggestions for `api_messages_init_ms` and error-contract VM coverage were applied before commit.
- File helper focused Docker validation: `CODOXEAR_DOCKER_PORT=18849 scripts/codoxear-docker-sandbox test tests/test_frontend_file_helpers_source.py tests/test_file_viewer_source.py tests/test_file_picker_search_source.py tests/test_file_picker_session_state.py tests/test_static_assets.py -q` -> `65 passed`.
- File helper Docker runtime route check under `CODEX_WEB_URL_PREFIX=/codoxear`: `/codoxear/api/me -> 401`; `/codoxear/app_file_helpers.js?v=test -> 200`; `/codoxear/app.js?v=test -> 200`; `/codoxear/ -> 200`, and `app_file_helpers.js` contained `window.CodoxearFileHelpers`.
- File helper full Docker sandbox suite: `CODOXEAR_DOCKER_PORT=18850 scripts/codoxear-docker-sandbox test` -> `976 passed, 1 skipped, 107 subtests passed`.
- Clean-room critic `d3fbcd89-e7ce-4f46-87c4-39164d6a27f3` returned `NO BLOCKERS` for contract drift, load order, static asset/versioning/package coverage, source-test blind spots, runtime dependency/fallback issues, literal path preservation, suffix stripping behavior, blocked-file byte formatting, wrapper coverage, and file-picker VM harness updates. Its guard suggestions for newline no-suffix preservation and zero viewer-limit behavior were applied before commit.
- Session helper focused Docker validation: `CODOXEAR_DOCKER_PORT=18856 scripts/codoxear-docker-sandbox test tests/test_frontend_session_helpers_source.py tests/test_sidebar_gtd_source.py tests/test_session_polling_source.py tests/test_voice_push_source.py tests/test_chat_scrollback_source.py tests/test_static_assets.py tests/test_launch_ui_source.py -q` -> `70 passed`.
- Session helper Docker runtime route check under `CODEX_WEB_URL_PREFIX=/codoxear`: `/codoxear/api/me -> 401`; `/codoxear/app_session_helpers.js?v=test -> 200`; `/codoxear/ -> 200`, and the served helper contained immutable `SESSION_SIDEBAR_GROUPS`.
- Session helper full Docker sandbox suite: `CODOXEAR_DOCKER_PORT=18857 scripts/codoxear-docker-sandbox test` -> `979 passed, 1 skipped, 107 subtests passed`.
- Clean-room critic `a02bb3b9-2860-486d-a79d-0448cc5be833` returned `NO BLOCKERS` for contract drift, failed/pending launch selectability, review/waiting/later grouping, sidebar render-signature semantics, app-owned redaction/label boundary, load order, static/version/package coverage, and runtime dependency/fallback issues. Its non-blocking suggestions for redaction-boundary assertions and immutable exported group metadata were applied before commit.
- Viewport helper focused Docker validation: `CODOXEAR_DOCKER_PORT=18865 scripts/codoxear-docker-sandbox test tests/test_frontend_viewport_module_source.py tests/test_file_viewer_source.py tests/test_sidebar_touch_mode.py tests/test_chat_navigation_source.py tests/test_overlay_accessibility_source.py tests/test_static_assets.py -q` -> `63 passed`.
- Viewport helper Docker runtime route check under `CODEX_WEB_URL_PREFIX=/codoxear`: `/codoxear/api/me -> 401`; `/codoxear/app_viewport.js?v=test -> 200`; `/codoxear/static/app_viewport.js?v=test -> 200`; `/codoxear/ -> 200`, and the served helper contained `window.CodoxearViewport` plus the preserved `isMobile()` expression.
- Viewport helper full Docker sandbox suite: `CODOXEAR_DOCKER_PORT=18867 scripts/codoxear-docker-sandbox test` -> `985 passed, 1 skipped, 107 subtests passed`.
- Clean-room delegate review `da503187-4b55-415a-88fe-e41c33d4b3e6` returned `NO BLOCKERS` for exact media-query strings, absent-`matchMedia` behavior, touch OR semantics, desktop combined query, reduced-motion call-site behavior, fail-loud guard, wrapper/call-site names, static asset/version/package wiring, and real-module test coverage.
- Polling helper focused Docker validation: `CODOXEAR_DOCKER_PORT=18873 scripts/codoxear-docker-sandbox test tests/test_frontend_polling_module_source.py tests/test_frontend_viewport_module_source.py tests/test_session_polling_source.py tests/test_static_assets.py -q` -> `34 passed`.
- Polling helper Docker runtime route check under `CODEX_WEB_URL_PREFIX=/codoxear`: `/codoxear/api/me -> 401`; `/codoxear/ -> 200`; `/codoxear/app_polling.js -> 200`; `/codoxear/static/app_polling.js -> 200`.
- Polling helper full Docker sandbox suite: `CODOXEAR_DOCKER_PORT=18874 scripts/codoxear-docker-sandbox test` -> `989 passed, 1 skipped, 107 subtests passed`.
- Clean-room delegate review `927f869e-57ea-4c7d-9b8e-b44397727336` returned `NO BLOCKERS` for delay equivalence, app-owned mutable state, fail-loud guard behavior, static asset/version/package wiring, and module encapsulation. Its positive requested kick-delay test suggestion was applied before commit; remaining notes are maintenance risks, not behavior regressions.
- Conversation-copy helper focused Docker validation: `CODOXEAR_DOCKER_PORT=18880 scripts/codoxear-docker-sandbox test tests/test_frontend_conversation_copy_source.py tests/test_frontend_polling_module_source.py tests/test_transcript_export.py tests/test_static_assets.py -q` -> passed.
- Conversation-copy helper Docker runtime route check under `CODEX_WEB_URL_PREFIX=/codoxear`: `/codoxear/api/me -> 401`; `/codoxear/ -> 200`; `/codoxear/app_conversation_copy.js?v=test -> 200`; `/codoxear/static/app_conversation_copy.js?v=test -> 200`.
- Conversation-copy helper full Docker sandbox suite: `CODOXEAR_DOCKER_PORT=18881 scripts/codoxear-docker-sandbox test` -> `994 passed, 1 skipped, 107 subtests passed`.
- Clean-room delegate review `4a1b533a-5dee-4b1a-8a82-c7556f7b39b9` returned `NO BLOCKERS` for side-effect ownership, fail-loud guard behavior, formatter semantic equivalence, load order, static asset/version/package wiring, and test coverage. Earlier review attempts failed before findings due subagent model-configuration errors.
- Video-preview error formatter focused Docker validation: `CODOXEAR_DOCKER_PORT=18882 scripts/codoxear-docker-sandbox test tests/test_frontend_file_helpers_source.py tests/test_file_viewer_source.py tests/test_static_assets.py -q` -> `40 passed`.
- Video-preview error formatter full Docker sandbox suite: `CODOXEAR_DOCKER_PORT=18883 scripts/codoxear-docker-sandbox test` -> `994 passed, 1 skipped, 107 subtests passed`.
- Clean-room delegate review `729062d7-003f-4282-8d8e-96dbe8ba2eac` returned `NO BLOCKERS` for semantic equivalence, fail-loud guard behavior, wrapper/call-site preservation, side-effect ownership in `app.js`, real-helper VM coverage, and no static asset wiring requirement.
- Recovery prompt preview helper focused Docker validation: `CODOXEAR_DOCKER_PORT=18884 scripts/codoxear-docker-sandbox test tests/test_frontend_display_module_source.py tests/test_chat_scrollback_source.py tests/test_static_assets.py -q` -> `42 passed`.
- Recovery prompt preview helper full Docker sandbox suite: `CODOXEAR_DOCKER_PORT=18885 scripts/codoxear-docker-sandbox test` -> `994 passed, 1 skipped, 107 subtests passed`.
- Deterministic equivalence check against the pre-extraction inline formatter passed for 10 representative edge cases, and a helper-body side-effect probe found no redaction/session/DOM/API references.
- Clean-room delegate review `ba7cbeaf-14d0-49d6-a5c1-3f412201d830` returned `NO BLOCKERS` for semantic equivalence, fail-loud guard behavior, wrapper/call-site preservation, app-owned recovery/security side effects, real-helper VM coverage, and no static asset wiring requirement.
- Recent-cwd score helper focused Docker validation: `CODOXEAR_DOCKER_PORT=18886 scripts/codoxear-docker-sandbox test tests/test_frontend_display_module_source.py tests/test_static_assets.py -q` -> `16 passed`.
- Recent-cwd score helper full Docker sandbox suite: `CODOXEAR_DOCKER_PORT=18887 scripts/codoxear-docker-sandbox test` -> `994 passed, 1 skipped, 107 subtests passed`.
- Deterministic equivalence check against the pre-extraction inline scorer passed for 11 representative cases, and a helper-body side-effect probe found no recent-cwd state/DOM/API references.
- Clean-room delegate review `79326319-1d34-4be7-b352-12aa500d5d11` returned `NO BLOCKERS` for semantic equivalence, fail-loud guard behavior, wrapper/call-site preservation, app-owned recent-cwd UI/state boundary, real-helper VM coverage, and no static asset wiring requirement.
- File-picker helper focused Docker validation: `CODOXEAR_DOCKER_PORT=18890 scripts/codoxear-docker-sandbox test tests/test_frontend_file_helpers_source.py tests/test_file_picker_search_source.py tests/test_file_viewer_source.py tests/test_static_assets.py -q` -> `63 passed`.
- File-picker helper full Docker sandbox suite: `CODOXEAR_DOCKER_PORT=18891 scripts/codoxear-docker-sandbox test` -> `994 passed, 1 skipped, 107 subtests passed`.
- Deterministic equivalence check against the pre-extraction inline bodies passed for file-search scoring, draft path normalization, Unicode folded/range mapping, normalized candidate scoring, and entry comparator tie-breaks; helper side-effect probe found no file state/DOM/API references.
- Clean-room delegate review `d52041b0-0f7b-4cb7-986b-99ffccd2c32d` saved to `/tmp/codoxear-file-picker-helper-review.md` returned `NO BLOCKERS` for scoring/path/range/comparator equivalence, fail-loud guard coverage, wrapper preservation, `CodoxearDisplay.baseName` dependency, app-owned file state/API/DOM boundaries, real-helper test loading, and no static wiring requirement.
- Chat-search display helper focused Docker validation: `CODOXEAR_DOCKER_PORT=18894 scripts/codoxear-docker-sandbox test tests/test_frontend_display_module_source.py tests/test_chat_navigation_source.py tests/test_static_assets.py -q` -> `27 passed`.
- Chat-search display helper full Docker sandbox suite: `CODOXEAR_DOCKER_PORT=18895 scripts/codoxear-docker-sandbox test` -> `994 passed, 1 skipped, 107 subtests passed`.
- Deterministic equivalence check against the pre-extraction inline bodies passed for representative chat-search snippet/hint cases; helper side-effect probe found no chat DOM/state/API references.
- Replacement clean-room delegate review `48a1112c-dedd-475b-be3c-bfb882fcb56f` saved to `/tmp/codoxear-chat-search-display-review.md` returned `NO BLOCKERS` for snippet/hint semantics, fail-loud guard coverage, wrapper preservation, app-owned chat-search state/API/DOM/load-older boundaries, real-helper test execution, and no static wiring requirement. The earlier delegate `edc2af3c-a1c9-4e0b-b4be-f0fb8ef9f29b` failed before findings due model output-config and was not counted as a substantive review.
- File-picker source/label focused Docker validation: `CODOXEAR_DOCKER_PORT=18896 scripts/codoxear-docker-sandbox test tests/test_frontend_file_helpers_source.py tests/test_file_picker_search_source.py tests/test_file_viewer_source.py tests/test_static_assets.py -q` -> `63 passed`.
- File-picker source/label full Docker sandbox suite: `CODOXEAR_DOCKER_PORT=18897 scripts/codoxear-docker-sandbox test` -> `994 passed, 1 skipped, 107 subtests passed`.
- Deterministic equivalence check against the pre-extraction inline bodies passed for 13 source-normalization cases and 10 section-label cases; helper side-effect probe found no DOM/API/file-state/timer/focus/storage references.
- Clean-room delegate review `26c0dc6c-8ce0-4714-9c6c-e17422448ada` saved to `/tmp/codoxear-file-source-label-review.md` returned `NO BLOCKERS` for pure helper ownership, fail-loud guard coverage, wrapper/call-site preservation, app-owned file identity/cache/API/DOM/viewer/focus/timer/recovery/security boundaries, static wiring sufficiency, and test coverage.
- File-editor cursor-helper focused Docker validation: `CODOXEAR_DOCKER_PORT=18898 scripts/codoxear-docker-sandbox test tests/test_frontend_file_helpers_source.py tests/test_file_viewer_source.py tests/test_file_picker_search_source.py tests/test_static_assets.py -q` -> `63 passed`.
- File-editor cursor-helper full Docker sandbox suite: `CODOXEAR_DOCKER_PORT=18899 scripts/codoxear-docker-sandbox test` -> `994 passed, 1 skipped, 107 subtests passed`.
- Deterministic equivalence check against the pre-extraction inline body passed for 33 cursor cases across empty/falsy text, single-line text, LF/CRLF/CR normalization, leading/trailing newlines, and multiple start positions; helper side-effect probe found no DOM/API/editor/file-state/timer/focus/storage references.
- Clean-room delegate review `4d727bae-5f06-462e-ba1e-986659d18308` saved to `/tmp/codoxear-cursor-helper-review.md` returned `NO BLOCKERS` for semantic equivalence, fail-loud guard coverage, wrapper scope/call-site preservation, app-owned editor/paste/dirty/focus behavior, static wiring sufficiency, and branch-covering tests.
- File-editor delete-key helper focused Docker validation: `CODOXEAR_DOCKER_PORT=18900 scripts/codoxear-docker-sandbox test tests/test_frontend_file_helpers_source.py tests/test_file_viewer_source.py tests/test_file_picker_search_source.py tests/test_static_assets.py -q` -> `63 passed`.
- File-editor delete-key helper full Docker sandbox suite: `CODOXEAR_DOCKER_PORT=18901 scripts/codoxear-docker-sandbox test` -> `994 passed, 1 skipped, 107 subtests passed`.
- Deterministic equivalence check against the pre-extraction inline body passed for 10 key inputs including `undefined`, `null`, blank, lowercase `backspace`/`delete`, uppercase variants, unknown, `0`, and `false`; helper side-effect probe found no DOM/API/editor/file-state/timer/focus/storage references.
- Clean-room delegate review `cc8e023e-2ba7-492f-a3b8-e3b1714948f7` saved to `/tmp/codoxear-delete-key-helper-review.md` returned `NO BLOCKERS` for pure helper ownership, fail-loud guard coverage, wrapper/call-site preservation, app-owned event filtering/lowercasing/native suppression/Monaco trigger/touch reset/focus/toast behavior, static wiring sufficiency, and test coverage.
- Model-option match helper focused Docker validation: `CODOXEAR_DOCKER_PORT=18902 scripts/codoxear-docker-sandbox test tests/test_launch_ui_source.py tests/test_new_session_model_options_source.py tests/test_static_assets.py -q` -> `39 passed`.
- Model-option match helper full Docker sandbox suite: `CODOXEAR_DOCKER_PORT=18903 scripts/codoxear-docker-sandbox test` -> `994 passed, 1 skipped, 107 subtests passed`.
- Deterministic equivalence check against the pre-extraction inline body passed for 10 cases covering empty query, exact/prefix/contains, case-insensitive search text, fallback model field, blank search text with model fallback, null option, and no-match; helper side-effect probe found no DOM/API/session/dialog-state/timer/focus/storage references.
- Clean-room delegate review `5992087b-7600-48de-a9ab-6c0a060861a7` saved to `/tmp/codoxear-model-option-match-review.md` returned `NO BLOCKERS` for pure helper ownership, semantic equivalence, fail-loud guard coverage, wrapper/call-site preservation, app-owned filtering order/sessionModelOptions/rendering/provider-selection/memory/focus/menu behavior, static wiring sufficiency, and test coverage.
- Diagnostics helper focused Docker validation: `CODOXEAR_DOCKER_PORT=18904 scripts/codoxear-docker-sandbox test tests/test_diagnostics_source.py tests/test_frontend_session_helpers_source.py tests/test_static_assets.py -q` -> `19 passed`.
- Diagnostics helper full Docker sandbox suite: `CODOXEAR_DOCKER_PORT=18905 scripts/codoxear-docker-sandbox test` -> `994 passed, 1 skipped, 107 subtests passed`.
- Deterministic equivalence check against pre-extraction `app.js` inline bodies passed for 11 provider cases and 5 copy-text cases, including `claude`/`claude-code` alias normalization through the app wrapper; refined helper side-effect probe found no DOM/API/session/modal/timer/focus/storage references.
- Clean-room delegate review `87b32795-9336-49a9-b2d0-170748b2a68e` saved to `/tmp/codoxear-diagnostics-helper-review.md` returned `NO BLOCKERS` for semantic equivalence, alias normalization preservation, fail-loud guard coverage, app-owned diagnostics state/API/DOM/clipboard/auth-error boundaries, load-order sufficiency, export wiring, and test coverage.
- Queue normalizer focused Docker validation: `CODOXEAR_DOCKER_PORT=18906 scripts/codoxear-docker-sandbox test tests/test_frontend_session_helpers_source.py tests/test_queue_button_source.py tests/test_static_assets.py -q` -> `17 passed`.
- Queue normalizer full Docker sandbox suite: `CODOXEAR_DOCKER_PORT=18907 scripts/codoxear-docker-sandbox test` -> `994 passed, 1 skipped, 107 subtests passed`.
- Deterministic equivalence check against pre-extraction `app.js` inline body passed for 9 payload cases covering null/undefined/empty input, invalid `items`, modern item filtering, flag booleanization, legacy string filtering, post-filter legacy IDs, `items`-over-`queue` priority, and empty items; helper side-effect probe found no DOM/API/queue state/timer/focus/storage references.
- Clean-room delegate review `890a40d4-1092-4540-8337-f74167c972de` saved to `/tmp/codoxear-queue-normalizer-review.md` returned `NO BLOCKERS` for helper placement/export, fail-loud guard coverage, wrapper/call-site preservation, modern/legacy semantic equivalence, `items` priority, app-owned queue state/API/DOM/focus/timer/recovery/security boundaries, load-order sufficiency, syntax validity, test coverage, and minimality.
- Advisory future-target scout `83934278-8784-4af1-9179-580e7df91d74` saved to `/tmp/codoxear-next-pure-helper-scout-after-queue.md` found no remaining safe pure-helper extraction candidates. Parked non-candidates are `redactedLaunchErrorText`/`sessionLaunchLabel` (pinned/security-sensitive), `launchPresetProviderChoice` (pinned by launch-dialog source-slicing tests), and remaining wrappers/state readers/dead code/DOM-browser-side-effect/orchestration logic.

Prior Pi busy-after-interrupt evidence remains valid:

- Focused Pi/server JSONL validation: `python3 -m py_compile codoxear/server.py codoxear/broker.py codoxear/pi_log.py codoxear/util.py` plus `tests/test_broker_busy_state.py`, `tests/test_read_jsonl_from_offset.py`, `tests/test_sessions_pending_log_idle.py`, and `tests/test_server_queue_persistence.py` -> `187 passed, 26 subtests passed`.
- Adjacent readiness/interrupt/source validation: `tests/test_broker_busy_state.py`, `tests/test_interrupt_semantics_source.py`, `tests/test_file_upload_module_source.py`, `tests/test_idle_heuristics.py`, `tests/test_sessions_pending_log_idle.py`, `tests/test_server_queue_persistence.py`, `tests/test_queue_sweep_idle_guard.py`, `tests/test_diagnostics_source.py`, `tests/test_launch_provenance.py`, `tests/test_session_sidebar_priority.py`, `tests/test_server_chat_flags.py`, `tests/test_sessiond_fail_closed.py`, `tests/test_send_button_source.py`, `tests/test_read_jsonl_from_offset.py`, and `tests/test_broker_fail_closed.py` -> `310 passed, 38 subtests passed`.
- Clean-room critic subagent `809c69e7-147b-4201-aed0-4f1565b0cb94` returned `NO BLOCKERS`; residual risks are repeated reads of huge unterminated partial Pi JSONL rows until newline/EOF and unobserved normal empty Pi final-close assistant rows.

Prior video preview evidence remains valid:

- Focused video/file-viewer validation: node syntax check plus `tests/test_file_viewer_source.py`, ffmpeg transcode fixtures in `tests/test_file_inspect.py`, and `tests/test_video_preview_cache.py` -> `33 passed`.
- API fixture under isolated Docker: generated odd-dimension MPEG4/PCM MKV; `/api/files/read` returned `kind=video` and `video_preview_url`; `/api/files/video_preview` returned `video/mp4`; ffprobe showed H.264/yuv420p and even encoded dimensions.
- Browser fixture under isolated Docker: preview preflight `Range: bytes=0-0` returned `206` with `Content-Range`; Chromium loaded metadata from the preview URL.
- VM regression: 500 JSON preview route error surfaced into fileStatus and did not set the video source.

Recent clean-room reviews returned no blockers after fixes:

- `/tmp/codoxear-pi-provider-ui-behavior-review2.md`
- `/tmp/codoxear-search-navigation-count-review.md`
- `/tmp/codoxear-roving-copy-buttons-review3.md`
- `/tmp/codoxear-root-cwd-resolve-review.md`
- `/tmp/codoxear-oversized-search-review.md`
- `/tmp/codoxear-recovery-panel-review6.md`
- Clean-room critic subagent review of sidecar extraction diff and call sites -> no blocker findings; non-blocking source-test brittleness was reduced before commit.
- Clean-room critic subagent review of Details-copy diff -> no blocker findings for stale-session binding, secret-copy risk, accessibility/focus, or sparse-UI risk.
- Clean-room critic review of file-picker highlight diff found a Unicode slicing bug; folded-index mapping plus `İfoo.py`/emoji regressions fixed it. Re-review found no blockers.
- Clean-room architecture review of git helper extraction found detached-HEAD semantic drift; `git_ops.current_git_branch()` was corrected to preserve `HEAD`. Targeted re-review and critic review found no blockers.
- Clean-room critic review of Details → New like this found Pi provider corruption risks in direct presets, diagnostics provider display, duplicate/recent options, remembered providerless choices, and sparse metadata. Each counterexample was fixed with regressions; final re-review found no blockers for Pi provider corruption, auto-start, focus, or sparse UI behavior.
- Clean-room critic review of markdown rendering first found a hidden-overflow/fixed-layout clipping counterexample for many-column tables. The final implementation uses auto overflow plus auto table layout; re-review found no blockers for clipping, page/bubble overflow, copy semantics, or chat/file-preview markdown paths.
- Iterated clean-room critic review of failed-launch recovery found and drove fixes for immediate POST response leakage, quoted/unclosed env syntax, nested launch-attempt diagnostics, colon/JSON secret syntax, redaction idempotence, failed-launch attach POST affordance, raw server/broker launch persistence and stderr, and Authorization/Auth Bearer/Basic values. Final review found no remaining failed-launch secret leakage/persistence path or mutation/autostart regression in inspected scope.
- Clean-room critic review of video preview/transcoding found no blockers for transcode correctness, route error surfacing, stale request guards, sparse/contextual UI, or file/session identity. Its non-blocking failure-path coverage note was addressed with a VM regression before commit.

Recent isolated browser evidence:

- Synthetic 180-turn Codex transcript under Docker app state reproduced and then fixed the all-transcript-search Next no-op. After the fix, clicking Next from `0/0 loaded · 1 all` emitted `/messages/search?...order=latest&before=...` and `/messages/history?cursor=...`, then showed `1/1 loaded · 1 all` and `Loaded transcript match` with no captured JS errors.
- The same long transcript had 60 message-copy button nodes but exactly one enabled/tabbable/accessibility-visible copy button after the roving-copy fix. Inactive samples were disabled, `tabIndex=-1`, `aria-hidden=true`, `opacity:0`, `visibility:hidden`, and `pointer-events:none`. Hidden-focus counterexamples with `Alt+↑` and `Alt+Shift+↑` remained false.
- Synthetic recovery fixtures under isolated Docker app state verified the in-chat recovery panel: orphan recovery did not fetch `/messages/tail`, Review queue opened preserved prompts, clearing an unknown marker and deleting queue items kept panel/buttons/focus synchronized, transcript-backed live appends kept the panel as the latest recovery surface, and focused panel actions survived rapid panel rebuilds.
- Pi live backend evidence exists for one current configured provider/model path as described above. Codoxear app/session state was isolated; the backend provider configuration came from the user's existing real Pi environment and was handled without printing secret values.
- Failed-launch fixture under isolated Docker app state verified redacted card/transcript/sidebar rendering for env, JSON, Authorization/Bearer, Auth/Basic, and tail secrets; send, queue, and attach were disabled; sidebar duplicate/rename were absent; New like this remained review-only.
- Video preview fixture under isolated Docker app state verified that a generated non-browser-safe MKV transcodes through the server preview route to browser-loadable MP4 metadata after a range preflight.

## Invariants broad refactoring must preserve

Any broad frontend/server refactor must keep these product semantics explicit and mechanically preserved:

1. **Send commit boundary:** HTTP `/send` success means the broker/sessiond path accepted the prompt or returned explicit unknown-commit recovery state; reads must not promote queued prompts.
2. **Unknown commit state blocks unsafe actions:** unresolved direct/queued uncertainty blocks send, enqueue, attach, sweep, reorder, and silent destructive cleanup bypasses; recovery UI may explain/review/clear explicit markers but must not silently resume mutation paths.
3. **Git/file identity:** changed-file paths are repo-root-relative literals; candidate identity is `(gitPath, path)`; path text must not be normalized destructively. Visual highlighting may wrap displayed substrings but must preserve original path strings for titles, copy/open actions, and identity keys. Git helper extraction must preserve literal pathspec handling and existing server wrapper/patch seams.
4. **Inline refs:** ambiguous inline refs route through the identity-aware picker; failed/truncated project search is ambiguity, not uniqueness proof.
5. **Broker state:** `busy` is bool and `queue_len` is nonnegative non-bool int; malformed state is fail-closed, not coerced. Explicit web ESC may set `interrupted_idle` only after a successful broker write, and server-side consumers may accept it only when broker busy is false and broker queue is empty.
6. **Stale busy/confirmed-send override:** stale broker busy can be overridden only with idle log evidence or validated broker `interrupted_idle`, empty queue, and a cleared confirmed-send barrier. Confirmed-send barriers require parseable JSON object row evidence; raw bytes, blank/malformed rows, arrays/scalars, and trailing partial rows are not commit evidence.
7. **Sidecar discovery:** malformed sidecar metadata is skipped/logged; fresh launch metadata requires a live broker pid; stale discovery still tolerates pid placeholders where explicitly allowed. Schema/type/capability parsing belongs in `codoxear.sidecar_metadata`; consumers may prune/skip only through explicit validation failure, not coercion.
8. **Transcript scale:** live JSONL readers stay bounded. Broker Pi tailing must not advance over incomplete rows and must process completed oversized rows, but pathological unterminated partial rows remain a known expense until newline/EOF.
9. **Search semantics:** count is exact by default only when all records in scope were parseable under the bounded line cap; skipped oversized records make `match_count_truncated` true. Bounded counts are lower bounds; `count_max` is incompatible with `order=latest`; UI hints stay sparse and server-clipped.
10. **Search navigation:** navigation refresh may recompute loaded DOM matches without discarding already-known all-transcript count evidence.
11. **Modal/accessibility focus:** active dialogs must receive focus immediately; focus must not remain in inert/`aria-hidden` content; message-copy controls must not flood tab/accessibility traversal. Dialog copy actions should copy rendered/allowlisted rows rather than hidden raw response objects.
12. **Pi launch providers:** Pi CLI/config is authority for provider names. UI defaults are hints, not an API whitelist; explicit provider/model pairs must not inherit stale bare-model reasoning constraints. Synthetic diagnostics `provider_choice` must not be treated as actual Pi provider state; providerless Pi sessions must remain providerless through copied launch presets, recent model selection, memory, parsing, and start request construction.
13. **Minimal UI philosophy:** keep the topbar sparse; utility controls belong in contextual rails/surfaces, not a generic dumping-ground menu.
14. **No silent fallbacks:** absence, malformed contracts, or unsupported combinations should fail loudly with recoverable UI when possible.
15. **Markdown containment:** code blocks should remain readable in the light UI; tables should wrap normal wide content and use internal scroll only when the column count cannot physically fit without clipping.
16. **Failed launches are recoverable non-sessions:** failed web-owned launches may be reviewed, dismissed, copied, or used to prefill a reviewed New Session dialog, but must not accept send/enqueue/attach or duplicate/rename autostart mutations. Failed-launch diagnostics shown through UI/API, persisted in `session_launches.jsonl`, or written to launch-failure stderr must be redacted through the shared launch-failure sanitizer.
17. **Video preview is explicit and diagnosable:** compatible MP4 preview generation may be requested automatically for known unsafe containers or manually through the contextual file-viewer video action. The client preflights the preview route and surfaces JSON/text route errors instead of hiding ffmpeg failures behind media-element fallback.

## Parked limits and decisions

The branch is stronger than the historical `develop` summary, but these limits remain explicit:

- Merge/promote to `main` still requires explicit user approval.
- Broad structural/frontend refactor is not complete; this checkpoint only defines its entry state.
- Real-browser/manual backend exercise of the Details → New like this button remains incomplete; source/VM tests, full pytest, Docker, and critic review cover the implemented semantics.
- Markdown rendering evidence covers CSS/source tests and headless Chromium fixtures, not real mobile-device or assistive-technology review.
- Codex live response evidence now covers the direct web-owned broker/browser-send/final-response path in isolated app state. Tmux web-owned Codex isolation remains caveated because a tmux launch attempt inherited the long-lived tmux server HOME and was not accepted as isolated proof.
- Claude Code live log binding and terminal API-error rendering are now validated under isolated HOME. A successful live Claude model answer remains incomplete because the inference gateway returned terminal 503 connection failures during validation.
- Real mobile-device, assistive-tech, slow-network, huge-transcript, and full live backend lifecycle evidence remain incomplete.
- Pi busy-after-interrupt evidence is deterministic fixture/source/server/broker validation plus full local/Docker suites, not a live Pi TUI/browser interruption replay.
- Smooth scrolling for Jump to latest remains parked until scheduler/runtime harness evidence exists.
- Non-UTF-8 Git filenames are replacement-decoded rather than byte-literal end-to-end.
- Symlink containment checks are pre-open/read/write, not atomic against concurrent local filesystem mutation.

## Recommended next step

The bounded pure-helper extraction wave has reached a natural stop under the current invariants: the post-queue scout found no remaining safe deterministic argument-only target. Further work should not force wrappers, state readers, unused/dead code, security-sensitive redaction/label logic, launch-dialog source-sliced logic, or DOM/API/mutable-state/browser-side-effect orchestration into helper modules. Resume only with explicit broader ownership/design approval or a newly identified candidate that clears the same deterministic/no-state/no-DOM/no-dead-code bar, with focused Docker, full Docker, and exactly one clean-room review before any functional commit.
