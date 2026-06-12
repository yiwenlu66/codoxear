# OPS

Append-only operational ledger for the structural refactor + UX review task. Each record must include `date '+%Y-%m-%d %H:%M'`.

## 2026-06-12 10:07
- Started actual structural refactor task from tracked `PROMPT.md`.
- Created branch `refactor/structural-ux-review` from `develop` before code edits. `main` remains untouched.
- Initial state was clean on `develop`; branch now carries only task-memory ledger edits.

## 2026-06-12 10:07
- Baseline validation before source refactoring: local `node --check codoxear/static/app.js` passed.
- Baseline isolated Docker suite: `scripts/codoxear-docker-sandbox test` passed (`429 passed, 2 skipped in 10.02s`).

## 2026-06-12 10:10
- Phase 1 source refactor: extracted message cursor signing/verification/encode/decode/history-cursor attachment into new `codoxear/message_cursor.py`.
- Kept compatibility wrappers in `codoxear/server.py` so existing route code and tests can continue importing `_encode_message_cursor`, `_decode_message_cursor`, `_attach_history_cursors`, and `MessageCursorError` from `codoxear.server` during transition.
- Updated `tests/test_message_route_source.py` so it asserts route usage in `server.py` and cursor attachment semantics in `message_cursor.py` instead of requiring implementation code to live in the server monolith.
- Targeted validation: `python3 -m py_compile codoxear/server.py codoxear/message_cursor.py` passed; `scripts/codoxear-docker-sandbox test tests/test_message_cursor.py tests/test_message_route_source.py tests/test_message_index.py tests/test_transcript_export.py -q` passed (`13 passed`).

## 2026-06-12 10:13
- Phase 2 source refactor: extracted cookie/HMAC auth helpers into new `codoxear/auth.py` with a `CookieAuthSettings` value object.
- Kept server-level compatibility wrappers for `_sign_cookie`, `_verify_cookie`, `_parse_cookies`, `_require_auth`, and `_set_auth_cookie` so existing tests/imports and route code remain stable during the transition.
- Targeted validation: `python3 -m py_compile codoxear/server.py codoxear/auth.py` passed; `scripts/codoxear-docker-sandbox test tests/test_auth_cookie.py tests/test_url_prefix.py tests/test_static_assets.py tests/test_client_disconnects.py -q` passed (`18 passed`).

## 2026-06-12 10:16
- Phase 3 state-store refactor: extracted Unattended prompt cleaners, prompt rendering, and `unattended.json` load/save ownership into `codoxear/unattended.py` with `UnattendedStore`.
- `SessionManager` now delegates `_load_unattended` and `_save_unattended` to the store while preserving the in-memory `_unattended` map and existing public methods/routes.
- Added `tests/test_unattended_store.py` for direct file-format, atomic-save output shape, loud legacy `text` rejection, cleaner defaults, and prompt rendering coverage. Updated source test to assert validation wording in the new owner module.
- Targeted validation: `python3 -m py_compile codoxear/server.py codoxear/unattended.py` passed; `scripts/codoxear-docker-sandbox test tests/test_unattended_store.py tests/test_unattended_sweep.py tests/test_unattended_mode_source.py tests/test_unattended_input_source.py -q` passed (`18 passed`).

## 2026-06-12 10:17
- Major checkpoint after message-cursor, auth, and Unattended-store extractions: local `node --check codoxear/static/app.js` passed.
- Full isolated Docker suite: `scripts/codoxear-docker-sandbox test` passed (`433 passed, 2 skipped in 9.54s`). Test count increased from baseline because new direct Unattended store tests were added.

## 2026-06-12 10:19
- Phase 4 route refactor: extracted shared URL-prefix parsing into `Handler._parse_prefixed_request_path()` and static GET dispatch into `Handler._handle_static_get()`.
- This removes duplicated GET/POST URL-prefix handling and starts route decomposition while preserving route order and static asset behavior.
- Added `tests/test_route_decomposition_source.py` to pin shared prefix parsing and static route helper usage.
- Targeted validation: `python3 -m py_compile codoxear/server.py` passed; `scripts/codoxear-docker-sandbox test tests/test_route_decomposition_source.py tests/test_url_prefix.py tests/test_static_assets.py tests/test_auth_cookie.py tests/test_docker_sandbox_source.py -q` passed (`17 passed`).

## 2026-06-12 10:23
- Phase 5 route refactor: extracted voice/notification/audio GET handling into `Handler._handle_voice_get()` and matching POST handling into `Handler._handle_voice_post()`.
- Added a shared `Handler._read_json_body()` helper for the extracted POST group; route semantics and response bodies are intended to remain identical.
- Extended `tests/test_route_decomposition_source.py` to pin voice route helpers.
- Targeted validation: `python3 -m py_compile codoxear/server.py` passed; `scripts/codoxear-docker-sandbox test tests/test_route_decomposition_source.py tests/test_voice_push.py tests/test_voice_push_source.py tests/test_voice_playback_source.py tests/test_auth_cookie.py tests/test_url_prefix.py -q` passed (`44 passed`).

## 2026-06-12 10:25
- Phase 6 utility dedupe: changed broker `_read_jsonl_from_offset` wrapper to delegate to canonical `codoxear.util.read_jsonl_from_offset` after a quiet missing-file existence check.
- Added broker regressions for ignoring partial appended JSONL lines and preserving the broker's quiet missing-file return contract.
- Targeted validation: `python3 -m py_compile codoxear/broker.py` passed; `scripts/codoxear-docker-sandbox test tests/test_broker_fail_closed.py tests/test_broker_busy_state.py tests/test_read_jsonl_from_offset.py -q` passed (`57 passed, 1 skipped`).

## 2026-06-12 10:25
- Major checkpoint after route decomposition and broker JSONL reader dedupe: local `node --check codoxear/static/app.js` passed.
- Full isolated Docker suite: `scripts/codoxear-docker-sandbox test` passed (`436 passed, 2 skipped in 9.54s`). Test count increased from 433 because route-decomposition and broker JSONL wrapper tests were added.

## 2026-06-12 10:28
- Phase 7 frontend refactor: introduced buildless `BackendConfig` factory inside `codoxear/static/app.js` for backend normalization, display/logo helpers, launch defaults, provider choices, reasoning choices, and fast support.
- Preserved existing function names through aliases so current UI call-sites and source/VM tests remain stable. No bundler, framework, module script, import, or export was introduced.
- Added `tests/test_frontend_module_boundaries_source.py` to pin the buildless factory boundary and single static `app.js` delivery contract.
- Targeted validation: local `node --check codoxear/static/app.js` passed; `scripts/codoxear-docker-sandbox test tests/test_frontend_module_boundaries_source.py tests/test_claude_backend_source.py tests/test_reasoning_effort_source.py tests/test_new_session_model_options_source.py tests/test_launch_ui_source.py -q` passed (`13 passed`).

## 2026-06-12 10:28
- Major checkpoint after frontend `BackendConfig` factory extraction: local `node --check codoxear/static/app.js` passed.
- Full isolated Docker suite: `scripts/codoxear-docker-sandbox test` passed (`438 passed, 2 skipped in 9.71s`). Test count increased from 436 because frontend module-boundary tests were added.

## 2026-06-12 10:36
- Browser sandbox setup found a real auth regression before UX review: isolated server `/api/login` returned 500 with `NameError: name 'hmac' is not defined` because the auth extraction left `_is_same_password()` in `server.py` calling `hmac.compare_digest` without importing `hmac`.
- Fixed by restoring the `hmac` import in `codoxear/server.py` and adding `tests/test_auth_cookie.py::test_password_compare_wrapper_uses_constant_time_compare`.
- Targeted validation: local `python3 -m pytest tests/test_auth_cookie.py -q` passed (`4 passed`); Docker `scripts/codoxear-docker-sandbox test tests/test_auth_cookie.py -q` passed (`4 passed`).

## 2026-06-12 11:03
- Mandatory browser UX review ran via `agent-browser` against isolated Docker sandbox `codoxear-sandbox-ux-18802` on `http://127.0.0.1:18802/`, with container-local mock broker sockets and synthetic Codex/Pi/Claude logs under `/home/tester`.
- Desktop evidence before fixes: login succeeded; long Codex chat loaded 60 rows/30 user turns; loaded search for `UNIQUE-UX-NEEDLE` reported `1/2 loaded` and scrolled to the marker; Load older expanded to 180 rows/90 user turns; Codex/Pi/Claude new-session tabs worked; file search/read opened `docs/review-notes.md`; queue viewer showed two editable queued items; Unattended menu showed seeded values; Settings/Help/Details opened.
- Browser-discovered issues: custom modals leaked background controls into the accessibility snapshot; opening Settings while Unattended was open left the Unattended popover stacked behind it; Pi/Claude new-session modes hid the provider control but left an orphan `Provider` label/blank column; Help copy still said supported backends were only Codex and Pi.
- Applied frontend fixes in `codoxear/static/app.js` and `codoxear/static/app.css`: added shared modal isolation (`inert` + `aria-hidden` on `.app`), modal-open transient overlay closure, whole-provider-field hiding plus collapsed layout, and Help copy listing Claude.
- Targeted validation before browser recheck: `node --check codoxear/static/app.js` passed; local `python3 -m pytest tests/test_overlay_accessibility_source.py tests/test_claude_backend_source.py tests/test_frontend_module_boundaries_source.py -q` passed (`8 passed`); Docker equivalent passed (`8 passed`).
- Fixed browser evidence: New Session Claude modal accessibility snapshot contained only modal controls; `.app` had `inert` and `aria-hidden="true"`; provider field display was `none`; row had `providerHidden`; Settings opened with `#unattendedMenu` display `none`; mobile New Session closed sidebar, hid Provider, and exposed only modal controls; Help contained `Codex, Pi, and Claude`; browser errors/console were empty after fixed interactions.
- Screenshots saved under `.memory/tasks/2026-06-12-structural-refactor-ux-review/browser-artifacts/`, including desktop/mobile before and after fix images.

## 2026-06-12 11:06
- Full validation after modal overlay fix initially failed `tests/test_file_viewer_source.py::TestFileViewerSource::test_file_open_race_guard_is_wired_through_fetch_and_render` because the source test pinned adjacency between `cancelPendingFileOpen()` and `fileBackdrop.style.display = "block";`.
- Updated the assertion to require the intentional new sequence `cancelPendingFileOpen(); prepareModalOpen(); fileBackdrop.style.display = "block";`, preserving the race-guard invariant while acknowledging modal isolation prep.
- Targeted validation: local and Docker single test passed.
- Final full validation: local `node --check codoxear/static/app.js` passed; full isolated Docker `scripts/codoxear-docker-sandbox test` passed (`443 passed, 2 skipped in 9.66s`).

## 2026-06-12 11:08
- User reported that the refactor frontend changes affected their live server because work had been done in `/home/yiwen/codex-web`, the checkout the live server can serve static files from.
- Corrective action: preserved the refactor branch by creating isolated worktree `/home/yiwen/codex-web-refactor-structural-ux-review` at commit `d4a3f25`, switched the original checkout `/home/yiwen/codex-web` back to `develop` (`60de01d`), then checked out `refactor/structural-ux-review` in the new worktree.
- Current isolation state: original checkout is `develop`; refactor branch is in `/home/yiwen/codex-web-refactor-structural-ux-review`. Further refactor work must use the isolated worktree path.

## 2026-06-12 11:32
- User corrected task ordering: the older major feature request was overclaimed and must be repaired before structural refactor continues.
- Updated task prompts so product-gap recovery is the active priority and the structural refactor prompt is explicitly parked until those gaps are fixed, browser-validated, and honestly scoped.
- Concrete recovery gates now include real provider/model selection, top-bar/action placement redesign, long-chat ergonomics, responsiveness evidence, file-viewer polish, incomplete git-history pressure coverage, and scoped backend/reasoning claims.

## 2026-06-12 11:40
- Created clean isolated recovery branch/worktree `/home/yiwen/codex-web-product-recovery` from `develop` so feature recovery can happen before parked structural refactor history.
- Ported the prompt correction from the parked refactor worktree and strengthened task prompts around product promises/workflows/invariants/evidence as the acceptance ontology.
- Any imported structural-refactor ledger history is parked evidence from the separate refactor worktree; it must not be treated as active recovery progress or acceptance proof for this branch.
