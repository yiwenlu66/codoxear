# Clean-room review — `fa74c6a Remove upload attachment index contract`

**Repo:** `/home/yiwen/codex-web-product-recovery` (branch `recovery/product-gaps`, HEAD = commit under review)
**Scope:** Review only. No edits, staging, or commits. Working tree confirmed clean.

## Verdict: ACCEPTED (no blockers, no required fixes)

The commit removes the vestigial `attachment_index` request-contract cleanly. The frontend stops sending it, the route stops reading/validating it (so legacy values are ignored, not rejected), the dead route-layer `attachment_inject_text` dependency is dropped, and send-boundary numbering is untouched. All validation branches, the `/inject_image` alias, staging-only semantics, and legacy-client compatibility are preserved and covered by tests. Full suite reproduces `1795 passed, 132 subtests passed`.

---

## Findings by review question

### Q1 — Frontend no longer sends `attachment_index` in `/inject_file`? ✅ YES
- `codoxear/static/app.js:6899` — the single upload call now sends `body: { filename: uploadName, data_b64: b64 }`; the previous `attachment_index: stagedAttachments.length + 1` is gone (diff).
- `grep attachment_index codoxear/static/` → **no matches** anywhere in static assets.
- `grep inject_file|inject_image codoxear/static/` → exactly one hit (`app.js:6899`), and it is inside `stageFiles()` (defined at `app.js:6853`). No `/inject_image` client call exists.
- Guarded by source test `tests/test_attach_button_source.py:54` `assertNotIn('attachment_index:', source)` and `:61` `assertEqual(source.count('/inject_file'), 1)`.

### Q2 — Handler ignores `attachment_index`, preserves all other validation/order? ✅ YES
Mechanism: `_handle_inject_attachment` (`codoxear/control_routes.py:293`) no longer reads `obj.get("attachment_index")` at all, so **any** value/type passes through silently (true ignore, not a narrowed accept). Every other check is preserved in the same order:
- `filename` — `control_routes.py:302` `not isinstance(filename, str) or not filename.strip()` → 400 `filename required`.
- `data_b64` — `:305` `not isinstance(data_b64, str) or not data_b64` → 400 `data_b64 required`.
- **Readiness before decode/stage** — `:309` `manager.attachment_staging_ready(session_id)` runs before `b64decode` (`:322`) and before `stage_uploaded_file` (`:329`). Confirmed by `test_inject_attachment_ready_check_precedes_base64_decode` and `test_inject_attachment_checks_readiness_before_decoding_or_staging`.
- **base64** — `:322-325` `b64decode(..., validate=True)` except → 400 `invalid base64`.
- **size/path** — `:329-332` stage `ValueError` → 413 when message starts `file too large`, else 400.
- **staging** — `:334` `manager.add_staged_attachment(...)`.
- **session errors** — readiness `KeyError`→404, `session_not_ready_error`→409, generic→409, `not ready`→409; staging `KeyError`→404 (with `out_path.unlink()` cleanup), `ValueError`→400 (with cleanup).
- Coverage: full 17-case matrix in `tests/test_control_routes.py:350-585` (missing filename/data_b64, invalid base64, oversize 413, generic 400, readiness 404/409/409/409, add_staged 404/400, unauthorized, JSON-serializable happy path).

### Q3 — Only route-layer dep removed; `file_upload.attachment_inject_text` remains and still used at send boundary? ✅ YES
- Definition retained: `codoxear/file_upload.py:155` `def attachment_inject_text(attachment_index, path)` (still raises `ValueError` for idx ≤ 0).
- Still consumed at the confirmed send boundary: `codoxear/session_send.py:7` import and `:78-80` `"".join(attachment_inject_text(idx, Path(...)) for idx, entry in enumerate(staged_entries, start=1))` → `Attachment N:` prefix built at commit time. `session_send.py` and `file_upload.py` are **not** in the commit's file list, so send-boundary numbering is untouched.
- Route-layer dep removed in three places: `control_routes.py:25` dataclass field, `server.py:51` import, and `server_route_deps.py` (both `server_route_caps` cap and the `ServerRouteDepsFactory` wiring). Verified dead in the parent: `git show 4afd953:codoxear/control_routes.py` had `attachment_inject_text` only as a dataclass field with **zero call sites** in the handler — so this was pure dead-dependency removal.
- No dangling references: `grep _attachment_inject_text codoxear/` → none (only negative `assertNotIn` guards in `tests/test_file_upload_module_source.py:35,58`). Guard `test_file_upload_module_source.py:47` still asserts `def attachment_inject_text(` exists in the module.

### Q4 — Avoids reintroducing backend key/PTY writes or a new commit boundary? ✅ YES
- Handler path is stage-only: no `inject_keys`, `inject_attachment_keys`, `paste`, or PTY writes anywhere in `_handle_inject_attachment` (grep of the function body is empty). Only `add_staged_attachment` is invoked.
- Guarded by `tests/test_file_upload_module_source.py:60` `assertNotIn("manager.inject_attachment_keys(session_id, seq)", block)` and behavioral `test_inject_attachment_stages_without_backend_paste` / `test_inject_attachment_ignores_legacy_attachment_index` (`not any(call[0] == "inject_attachment" ...)`).
- No new commit boundary: `session_send.py` is unchanged; the only `Attachment N:` generation remains at the existing confirmed-send commit point.

### Q5 — Tests aligned incl. legacy-index compatibility and no `/inject_file` outside `stageFiles()`? ✅ YES
- New `test_inject_attachment_ignores_legacy_attachment_index` (`tests/test_control_routes.py:488`) iterates legacy values `(0, 1, True, "1")` and asserts each → last response `200` and no `inject_attachment` call. This covers the two values the old handler rejected (`True`, `"1"` → 400) plus the previously-accepted ints, proving the relaxation.
- Obsolete tests removed: `test_inject_attachment_non_integer_index_is_400` and `test_inject_attachment_accepts_legacy_index_but_does_not_generate_text` (subsumed).
- `_deps`/`_inject_deps` no longer inject `attachment_inject_text`; `_good_body` no longer includes `attachment_index`; missing-filename / missing-data_b64 cases updated to drop the field.
- "No `/inject_file` outside `stageFiles()`" is enforced structurally by `test_attach_button_source.py:61` `assertEqual(source.count('/inject_file'), 1)` plus per-producer `assertNotIn('api(.../inject_file'...)` on the picker block; grep independently confirms the single call site.
- `tests/test_file_upload.py:266-302` still exercises `attachment_inject_text` numbering, unicode paths, non-positive-index rejection, and module identity — confirming the send-boundary helper contract is intact.

### Q6 — Hidden compatibility risk with `/inject_image` alias or old clients? ✅ NONE material
- `/inject_file` and `/inject_image` both route to the same `_handle_inject_attachment` (`control_routes.py:86-90`), so the relaxation applies identically to the alias. The alias is exercised by `test_inject_attachment_stages_without_backend_paste` (posts to `/api/sessions/s1/inject_image`, asserts stage-only + 200).
- Old clients that still send any `attachment_index` are strictly more likely to succeed (value ignored), never newly broken. Response body never contained `attachment_index`, so the response contract is unchanged.
- Only behavioral delta: a malformed `attachment_index` combined with a missing `data_b64` now returns `data_b64 required` instead of the old `attachment_index must be an integer`. This is the intended removal of the field's error surface; no client should depend on that message. Nonblocker.

---

## Required fixes
None.

## Nonblocker observations
1. **Stale historical scout note** — `.memory/tasks/2026-07-03-usable-product-ui-architecture/upload-attachment-scout.md:33,38,87` still documents the old `{ filename, data_b64, attachment_index }` contract and its validation matrix. This is point-in-time task memory from a *different, earlier* task, not a live API/contract doc, and is out of scope for this commit. Leaving it is acceptable; optionally add a one-line note that `attachment_index` was retired in `fa74c6a` if that scout file is still consulted. (The active task's `EPISTEMIC.md`/`OPS.md` already record the removal accurately.)

## Evidence checked
- `git show fa74c6a` (full diff) and `git show 4afd953:codoxear/control_routes.py` (parent) — confirmed `attachment_index` was validated-but-unused and `attachment_inject_text` dep had no call sites pre-commit.
- `git show --name-only fa74c6a` — 7 files; `session_send.py`/`file_upload.py` **not** among them.
- `grep` sweeps: `attachment_index` absent from `codoxear/static/` and all code (only `.memory/` notes + `file_upload.py` param name); single `/inject_file` call site; no `_attachment_inject_text` runtime refs.
- `node --check codoxear/static/app.js` → OK.
- `git diff --check` → clean.
- Focused: `pytest tests/test_control_routes.py tests/test_attach_button_source.py tests/test_file_upload_module_source.py` → 38 passed; broadened upload/send set (`+ test_file_upload.py test_send_ack.py test_composer_sendability_source.py test_send_button_source.py`) → 84 passed.
- Full suite: `pytest -q` → **1795 passed, 132 subtests passed** (matches claimed evidence).
- `git status --porcelain` and `git diff --cached` → empty (no edits, no staged files).
