# Codoxear `recovery/product-gaps` Acceptance Summary

Date: 2026-06-15
Branch: `recovery/product-gaps`
Protected live checkout: `/home/yiwen/codex-web` on `main` was not modified, merged into, restarted, or promoted.
Latest functional code checkpoint before this summary: `70fc3a1 centralize frontend static asset registry`; later commits through `935b402 docs: refresh recovery action prompt` are documentation/task-state updates.

## Candidate status

`recovery/product-gaps` is a validated, reviewable recovery candidate for explicit user approval. It is not merge approval. Promotion to `main` remains parked until the user explicitly authorizes it.

The branch resolves the product gaps recorded in `recon/refactor-entry-checkpoint.md` under the evidence below, while keeping unsupported areas explicitly parked. Recent frontend/static work used Docker-only validation, per user instruction; host-side prefix evidence from the URL-helper tranche was discarded and is not used as acceptance evidence.

A final pre-summary clean-room review (`90a8597a-fa0a-4a8a-b240-8e5989960b39`) returned `NO BLOCKERS` for current `HEAD` before this summary rewrite. It verified that closed claims are supported by checkpoint evidence, recent helper/static refactors preserve fail-loud/static-routing/versioning behavior, parked limitations are not claimed solved, and no tracked live runtime artifacts/secrets were found.

## Integrated recovery work

### Backend/live-session reliability

- **Pi busy-after-interrupt repair:** explicit web ESC is recorded only after a successful broker write; `interrupted_idle` is accepted only when broker busy is false and queue is empty. Pi tool-call accounting now preserves arbitrary string IDs and duplicate-ID multiplicity, stays busy-closed for absent/non-string IDs until final/abort/error, and seeds/tails only complete JSONL rows.
- **Codex web-send/log binding:** direct web-owned Codex launch in isolated app state reproduced cwd-alias log binding failure; rollout discovery now accepts exact cwd strings or existing absolute filesystem identity via `samefile()` and fails closed for relative, tilde, unknown-user, or nonexistent aliases.
- **Claude Code closed-log binding:** Claude Code no-writable-fd logs are discovered through a bounded fallback constrained by prelaunch snapshots, launch time, cwd identity, and merged early header rows. Races from post-fork snapshots, relative broker cwd, and large first JSONL rows were fixed with regressions.
- **Confirmed-send/busy barriers:** confirmed-send barriers require parseable JSON object row evidence and block unsafe mutation/read-busy paths until resolved.

### Recovery, launch, and session UX

- Failed web-owned launches render as recoverable non-session rows with redacted in-chat recovery cards, review-only New-like-this action, Dismiss/Copy details actions, and disabled send/enqueue/attach paths.
- Orphan, queued-recovery, and unknown-send sessions render an in-chat recovery panel with safe review actions instead of a blank disabled pane.
- Pi provider/model launch paths preserve Pi CLI/config authority: explicit custom providers pass through, synthetic diagnostics `provider_choice` is not treated as actual Pi provider state, and providerless Pi sessions remain providerless through copied presets/recent selections/start requests.

### Transcript/search and long-session orientation

- Transcript search uses bounded hint payloads and count hints; exact counts remain exact only when bounded parsing saw every scoped record, while oversized skipped JSONL records mark `match_count_truncated` instead of overstating certainty.
- Search navigation preserves all-transcript count evidence during Next/Previous navigation and can load offscreen matches from unloaded transcript windows.
- Live JSONL partial reads remain bounded; Pi tailing does not advance over incomplete rows.
- Long-transcript per-message copy controls now use a single roving enabled/tabbable control instead of flooding keyboard/accessibility traversal.

### File, Git, markdown, and media UX

- File picker candidates remain visible without git state, preserve `(gitPath, path)` identity, and highlight exact/fuzzy matches with Unicode-safe folded-index mapping without rewriting path identity.
- Inline file references route ambiguity through the identity-aware picker; failed inline inspections are not cached as uniqueness facts.
- Git subprocess/path/pathspec/numstat/worktree helper logic lives in `codoxear/git_ops.py`, preserving literal pathspec handling and server wrapper patch seams.
- Markdown fenced code blocks use a light Codoxear-themed surface; markdown tables contain/wrap normal wide content and use internal horizontal scroll only for impossible many-column cases.
- Video files expose an explicit compatible-MP4 preview action; the client preflights the preview route and surfaces route/ffmpeg errors instead of hiding conversion failures behind media-element fallback.

### Bounded frontend/static refactors

- URL/base-path resolution moved to `codoxear/static/app_url.js`; `app.js` fails loudly if `window.CodoxearUrls` is absent.
- Local-storage access moved to `codoxear/static/app_storage.js`; denied/throwing storage still yields `null` for reads and `false` for writes/removes.
- Performance sampling moved to `codoxear/static/app_perf.js`; the 200-sample window, nonnegative filter, percentile/rounding policy, and public `window.codoxearPerf` entry point are preserved.
- Frontend static registration now uses `FRONTEND_ASSET_FILES` for version-hashed assets and `TOP_LEVEL_STATIC_ASSETS` for exact top-level static routes. Public static URLs, `/static/*`, CSP/cache/content-type behavior, package-data globs, and URL-prefix behavior remain unchanged.

## Validation evidence

### Current final-review evidence

- Clean-room final review `90a8597a-fa0a-4a8a-b240-8e5989960b39` returned `NO BLOCKERS` for branch `recovery/product-gaps` at pre-summary `HEAD 935b4028ea798ee885250fa12e184c2ffd302ee6` with a clean worktree.
- The review checked checkpoint support, recent helper/static source and tests, backend/live-binding closure claims, parked limitations, refreshed prompt state, and absence of tracked live runtime artifacts/secrets.

### Recent Docker-only frontend/static validation

- URL helper focused Docker: `scripts/codoxear-docker-sandbox test tests/test_frontend_url_module_source.py tests/test_static_assets.py tests/test_url_prefix.py tests/test_session_polling_source.py` -> `29 passed, 3 subtests passed`.
- URL helper full Docker: `scripts/codoxear-docker-sandbox test` -> `955 passed, 1 skipped, 107 subtests passed`.
- Storage helper focused Docker: `scripts/codoxear-docker-sandbox test tests/test_storage_robustness_source.py tests/test_static_assets.py tests/test_frontend_url_module_source.py tests/test_file_picker_session_state.py tests/test_new_session_model_options_source.py` -> `39 passed, 3 subtests passed`.
- Storage helper full Docker: `scripts/codoxear-docker-sandbox test` -> `955 passed, 1 skipped, 107 subtests passed`.
- Performance helper focused Docker: `scripts/codoxear-docker-sandbox test tests/test_frontend_perf_module_source.py tests/test_static_assets.py tests/test_frontend_url_module_source.py tests/test_storage_robustness_source.py tests/test_session_polling_source.py` -> `31 passed, 3 subtests passed`.
- Performance helper full Docker: `scripts/codoxear-docker-sandbox test` -> `958 passed, 1 skipped, 107 subtests passed`.
- Static registry focused Docker: `scripts/codoxear-docker-sandbox test tests/test_static_assets.py tests/test_url_prefix.py tests/test_frontend_url_module_source.py tests/test_storage_robustness_source.py tests/test_frontend_perf_module_source.py` -> `24 passed, 3 subtests passed`.
- Static registry route probe under `CODEX_WEB_URL_PREFIX=/codoxear`: `/api/me -> 401`; `app_url.js`, `app_storage.js`, `app_perf.js`, `app.js`, `app.css`, `favicon.png`, `manifest.webmanifest`, `service-worker.js`, and `static/app.js` all returned `200` in Docker.
- Static registry full Docker: `scripts/codoxear-docker-sandbox test` -> `959 passed, 1 skipped, 107 subtests passed`.

### Backend/live-binding and product-gap validation

- Pi busy-after-interrupt focused validation: Pi/server JSONL tests -> `187 passed, 26 subtests passed`; adjacent readiness/interrupt/source validation -> `310 passed, 38 subtests passed`; clean-room critic `809c69e7-147b-4201-aed0-4f1565b0cb94` returned `NO BLOCKERS`.
- Codex cwd-alias repair validation: focused regressions through `tests/test_broker_proc_rollout.py`, `tests/test_session_resume.py`, and `tests/test_stale_sidecars.py`; final full local -> `943 passed, 104 subtests passed`; Docker -> `942 passed, 1 skipped, 104 subtests passed`; final critic `5df64f7b-12c0-4e8c-a65b-f36985c79e35` returned `NO BLOCKERS`.
- Isolated Codex live proof: temp HOME/app state, real `CODEX_HOME`, direct web-owned broker, accepted temp cwd trust, browser composer sent `Reply with exactly CODEX_WEB_LIVE_OK_20260615 and nothing else.`, transcript tail/browser DOM showed `CODEX_WEB_LIVE_OK_20260615`, and session state returned `busy=false`, `queue_len=0`.
- Claude Code fallback/header validation after critic-driven fixes: focused CC/Codex validation -> `104 passed`; full local -> `952 passed, 104 subtests passed`; Docker -> `951 passed, 1 skipped, 104 subtests passed`; final critic `6f5dbf25-e41e-4467-8760-66e781c6809e` returned `NO BLOCKERS`.
- Isolated Claude Code live proof: temp HOME/app state, direct web-owned broker, accepted temp cwd trust, browser prompt reached the real CC log, fallback rebound placeholder to thread `410ef3d0-6967-49cd-9488-45b30c40f5d6`, upstream returned terminal 503, Codoxear rendered an assistant API-error row, and session state returned `busy=false`, `queue_len=0`.
- Video preview validation: focused video/file-viewer tests -> `33 passed`; Docker API fixture produced browser-compatible MP4 from odd-dimension MPEG4/PCM MKV; Chromium metadata load succeeded after range preflight; VM regression confirmed route errors surface without setting a stale video source.
- Long-transcript/recovery/browser fixture evidence in isolated Docker app state covered search loading from unloaded transcript matches, roving copy-button accessibility, recovery panel actions, failed-launch redaction/mutation blocking, and video preview metadata load.

### Review evidence

Recent clean-room reviews returned no blockers after fixes for the Pi busy repair, Codex cwd alias binding, Claude Code fallback/header repair, URL helper extraction, storage helper extraction, performance helper extraction, static registry centralization, markdown containment, video preview, failed-launch recovery, search navigation/count semantics, recovery panel UX, file-picker Unicode highlighting, git helper extraction, and Details/New-like-this provider semantics.

## Scoped limitations and parked decisions

- Merge/promote to `main` still requires explicit user approval.
- Successful live Claude model-text response remains unproven because validation hit terminal upstream 503 connection failures; current evidence covers live send, log binding, API-error rendering, and idle recovery.
- Broad structural/frontend refactoring is not complete; only the bounded helper/static-registry tranches listed above are claimed.
- Real mobile-device, assistive-technology, slow-network, huge-transcript, service-worker lifecycle, and CDN/cache deployment evidence remain uncollected.
- Full live backend lifecycle evidence remains incomplete beyond the scoped live paths described here.
- Pi busy-after-interrupt evidence is deterministic fixture/source/server/broker validation plus full local/Docker suites, not a live Pi TUI/browser interruption replay.
- Codex live response evidence covers direct web-owned broker/browser-send/final-response in isolated app state. Tmux web-owned Codex isolation remains caveated because a tmux launch attempt inherited a long-lived tmux server HOME and was not accepted as isolated proof.
- Real-browser/manual backend exercise of Details -> New like this remains incomplete; source/VM tests, full suites, Docker, and critic review cover implemented semantics.
- Smooth Jump to latest remains parked until scheduler/runtime harness evidence exists.
- Non-UTF-8 Git filenames are replacement-decoded rather than byte-literal end-to-end.
- Symlink containment checks are pre-open/read/write, not atomic against concurrent local filesystem mutation.
- Service worker, manifest, and favicon remain outside the frontend asset-version hash; this is pre-existing behavior, not a new claim from the static registry refactor.
- A stale cached old shell plus new helper-dependent `app.js` would fail loudly instead of silently recomputing missing helpers; default no-store/versioning mitigate this within existing freshness semantics.

## User decision required

The branch is ready for explicit user review/approval under the evidence above. The next action is a user decision: approve promotion/merge planning for `recovery/product-gaps`, request additional validation, or request more changes. No promotion to `main` is authorized by this summary.
