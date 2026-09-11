# OPS — immutable evidence trail

## 2026-09-05 — Implementation of commits 1–6 (this session)

- Start state: one uncommitted edit in `app_file_editor_ops.js` (Escape modal-dismissal branch removed, Tab trap kept) from a prior dead attempt. Verified it matched plan commit 1 and built on it.
- Escape audit (grep `Escape` across `codoxear/static/*.js`), classification:
  - Removed as dismissal: `app_file_editor_ops.js` capture branch (confirm/paste/unsaved/viewer/send-choice/queue/help/diag/voice/session-edit/new-session), `app_chat_search.js` input Esc-close, `app_new_session.js` document Esc-close.
  - Kept as transient/focus: hint-mode exit, file-picker dropdown, composer model picker + blur, unattended menu, touch-selection exit, new-session cwd/model dropdown menus (input-keyed).
- Monaco command IDs verified against vendored `monaco/vs/editor.api-CalNCsUg.js` before use: `cursorLeft/Right/Up/Down`, `cursorHome`, `cursorEnd`, `cursorWordStartRight/Left`, `cursorWordEndRight`, `cursorMove` with `by: "halfPage"`, `undo`, `redo`, `deleteRight` all registered (`id:"..."` in the built editor API).
- Hint controller `enter()` was already public — commit 5 needed no exposure change (plan anticipated it might).
- Commits (in order): `921323d1` Esc policy, `e8b654f5` vim skeleton, `f3bfc096` Monaco motions, `6ab7ce6e` scroll fallback, `4344d342` f hint entry, `2ca164f0` help/AGENTS docs, `711c2bdb` O-fix (review fix on top).
- Verification commands run per stage and at end: `.venv/bin/python -m pytest -q` (1768 passed, 112 subtests, final), `python3 scripts/check_wiring.py` (passed, no new allowlist entries, pinned counts untouched), `npx esbuild ... --bundle` (builds), `python3 scripts/check_js_refs.py codoxear/static` (passed).
- Tests added: `tests/test_escape_modal_policy_source.py` (reverse-pinned Esc policy: editor-ops listeners leave dialogs open, Tab trap intact, new-session dialog stays open), `tests/test_frontend_file_vim_module_source.py` (13 behavioral VM tests: sub-mode machine, verbs, motions, scroll fallback, guards, hint entry). Reverse-pinned `test_frontend_chat_search_module_source.py` Esc test; updated `test_frontend_wiring.py` pinned `createFileOpsOptions` keys.
- Bugs found and fixed during self-review/test-driven work:
  - `deleteLine` initially bypassed `withWritableEditor` (no readOnly lift, no dirty sync) — caught by test sequencing accident, fixed.
  - dd range used shorthand `endLine` key instead of Monaco's `endLineNumber` — caught by exact-equality VM test.
  - `O` computed `insertLine = lineNumber - 1` (split the previous line; invalid range at line 1) — caught in post-commit self-review, fixed in `711c2bdb`.
- Deviations from plan (deliberate, both recorded in commit messages):
  - Ctrl-d/u on Monaco surfaces use `cursorMove {by: "halfPage"}` instead of `setScrollTop(±viewport/2)`: keeps cursor and viewport coherent (raw scrollTop drifts the view away from the cursor; j then snaps it back). Scroll surfaces still use scrollTop per plan.
  - Vim normal mode integrates with `fileEditorCapabilities` via a `vimNormalActive` gate (new optional dep) instead of fighting `syncFileEditorReadOnly` from a second writer; consequence: Ctrl-S save is insert-mode-only while in normal mode.
- Not done (per task scope): Docker behavioral verification (plan commit 7) and deploy. Working tree clean except pre-existing untracked `.memory/`, `.pi-subagents/`, `node_modules/`, `docker/iso-broker-16447.json`.

## 2026-09-05 fix-up round (review findings, commits beb463ef..cd203ca6)

- Independent review found 7 issues; all addressed.
- beb463ef "Tighten file viewer vim key handling": x modifier guard; pending g/d reject modifier keys (d Ctrl-d deletes nothing, g Ctrl-g/Meta-g jumps nowhere — key consumed, prefix cleared); Space excluded from printable swallow (isPlainPrintable → isSwallowedPrintable) in view+normal so native Close-button activation works; dd on last line deletes preceding newline ((n-1,maxcol)-(n,maxcol)), single-line → empty; pendingPrefix cleared on every Escape (incl. unconsumed view-mode Esc and dirty-normal toast) and on viewer close (finishHide hook in app_file_ops.js calls vim syncEditMode; lifecycle hide() funnels every close through finishHide — pinned by test in test_file_viewer_source.py). 7 new vim-module tests.
- 64037495 "Save with Ctrl-S from vim normal mode": operations receive activeFileEditorInsertIdleTextWritable (capabilities with ignoreVimNormal:true → idleTextWritable; keeps editMode/editable/file-view/!unavailable/!plain-fallback/!savePending/textKind gates), replacing activeFileEditorIdleTextWritable in the createFileViewerOperationsOptions select contract. Vim layer never consumes Ctrl-S (modifier fall-through, pinned). Controller-level test drives real save POST with vimNormalActive:true.
- f9924093 help overlay line: chat search closes via Close button / Tab+Enter (no Esc).
- cd203ca6 bundle rebuild via `npx esbuild codoxear/static/app.js --bundle --minify --outfile=codoxear/static/dist/app.bundle.js --format=esm`; verified fileVimMode ×5 and key!==" " (Space exclusion) in bundle; node --check OK.
- Verification: .venv/bin/python -m pytest -q → 1777 passed, 112 subtests; python3 scripts/check_wiring.py → pass (no new allowlist entries); scripts/docker_verify.sh HEAD (cd203ca6) → PASS (bundle boots, login, sidebar render, no page/console errors). No deploy (not requested).
- Prediction made before tests: ctrl-x would produce 0 triggers / not consumed — confirmed via standalone node run after two test-authoring count-capture mistakes (counts must be captured immediately after each press, not at JSON time).
