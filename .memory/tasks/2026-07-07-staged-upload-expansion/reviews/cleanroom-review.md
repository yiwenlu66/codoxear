# Clean-room adversarial review — Codoxear staged-upload expansion (first slice)

**Repo reviewed:** `/home/yiwen/codex-web-product-recovery` @ `b1e6bc2` (branch `recovery/product-gaps`, working tree clean).
**Protected checkout `/home/yiwen/codex-web` not touched.** No edits/staging/commits made by this review.
**Commit range:** baseline `38e2120` → impl `e1c8315` → proof `b1e6bc2`.

## Verdict: **ACCEPTED** — no blockers.

Every product invariant in the six deliverables holds under adversarial inspection. The core "stage now, commit at send" boundary is enforced in code, covered by behavioral tests I independently reproduced (240 passed, 22 subtests), and demonstrated end-to-end by an honest Docker proof whose fake broker logs the actual socket commands. Six nonblocking observations are listed; none violate a stated invariant, and none require a fix to accept this slice.

---

## Invariant-level verification

### D1 — Server-owned staged list, stable per-session entries, multi-file picker ✅
- Truth lives in `SessionStore.staged_attachments: dict[sid -> list[entry]]` (`session_store.py`), persisted to `staged_attachments.json` (`server_config.py:243`, wired `server.py:914/931-932`). Entries carry stable `id` (`uuid4().hex`, `session_pending_state.py:70`), `display_name`, `filename`, `path`, `size`, `created_ts`. `_clean_staged_attachment_entry` validates+dedupes on load/save; `load_all` merges staged keys into `pending_attachment_ids` so projection survives restart.
- Multi-file: `imgInput` gains `multiple` (`app.js`), upload handler iterates `Array.from(imgInput.files)` staging each. `stage_uploaded_file` now disambiguates same-millisecond collisions (`file_upload.py`, test `test_same_millisecond_same_name_gets_unique_path`).
- Projected per-session into `/api/sessions` (`session_listing.py:52/216/348`), so counts are server-authoritative for every session, not just the selected one.

### D2 — Stage-only: attach/remove/clear never write to backend PTY/control socket ✅ (strongest evidence)
- Upload route `_handle_inject_attachment` (`control_routes.py:288`) now calls `manager.add_staged_attachment(...)` and no longer calls `inject_attachment_keys`; the bracketed-paste sequence is gone. On `KeyError`/`ValueError` it `unlink()`s the staged file (no orphan).
- New routes are store-only: GET `attachments`→`list_staged_attachments`, POST `attachments/delete`→`remove_staged_attachment`, POST `attachments/clear`→`clear_staged_attachments`.
- `grep` for `_sock_call|inject_keys|send_keys|sendall|socket` across `session_pending_state.py`, `session_store.py`, `file_upload.py` returns **nothing** — the staging path is pure filesystem + in-memory store.
- `inject_file`/`inject_image` map only to the stage-only handler (`control_routes.py:87-92`); no HTTP route reaches `inject_attachment_keys`.
- Route test `test_inject_attachment_stages_without_backend_paste` asserts `not any(call[0] == "inject_attachment" ...)`.
- Docker proof (fake broker recording every command): after multi-file upload → `{state:319, send:0, keys:0}`; after remove+clear → `{state:536, send:0, keys:0}` (`docker-calls-after-upload-summary.json`, `docker-calls-after-clear-summary.json`).

### D3 — Browser shows server-derived identity/count, per-file removal, clear-all ✅
- `stagedAttachments` is a mirror of server state: `setSelectedSessionStagedAttachments` writes both `sessionIndex[sid].staged_attachments` and the composer from the route response; `syncStagedAttachmentsFromSelectedSession()` runs on every `refreshSessions` poll and on select, reconciling local state **down** to server truth (kills stale-high badge). Badge = `Math.max(local, serverListCount, serverPending?1:0)` — can only transiently over-count, self-corrects next poll.
- Chip identity is not just filename: `display_name · fmtBytes(size) · id.slice(0,8)`, tooltip = backend path. Per-chip `×` → `attachments/delete`; `Clear` → `attachments/clear`; both re-sync from the response and surface errors via toast (401 → `handleAppAuthLoss`).
- Partial multi-file failure is surfaced (`attached N; M failed: …`) and successful entries are kept.
- Browser proof: `badge:"2"` + two chips → remove → `badge:"1"` → clear → `badge:"", chipCount:0, trayDisplay:"none"` (`browser-after-multifile-upload/-remove-one/-clear-all.json`).

### D4 — Send is the only commit boundary ✅
- `SessionSendCoordinator.send` (`session_send.py`): under lock snapshots `staged_entries` (only when `allow_pending_attachment`), builds `attachment_prefix` of `Attachment N: <path>\n` lines, prepends → `committed_text`. `committed_text` is what is sent, recorded as the prelog user message, and stored in the commit_unknown record.
- Success → `clear_staged_attachments` (deletes files + clears list + syncs projection). Commit-unknown / not-ready / injection error raise **before** the clear, so entries persist; the commit_unknown record carries the attachment-bearing text for retry.
- Direct send without the flag while pending → `require_send_preconditions` raises (`session_input.py:27`) — no silent drop. Queue drain and unattended sweep call `send` with `allow_pending_attachment=False`; `enqueue` refuses while `pending_attachment` (`session_queue.py:102`), and staging is refused while a queue is active (readiness) — so queue and staged attachments are mutually exclusive and a queued send cannot drop/duplicate attachments.
- Behavioral tests exercise the real coordinator: `test_staged_attachments_compose_at_confirmed_send_boundary_and_clear_on_success` (asserts exact `send` payload + cleared state) and `test_staged_attachments_survive_commit_unknown_send` (asserts preserved entry + commit_unknown text). Docker proof: exactly one `send` with two `Attachment N:` lines then user text, `keys:0`; commit_unknown path returns 504, `after_count:1`, `after_pending:true`, `last_has_attachment:true`.

### D5 — Filesystem safety & compatibility projections ✅
- 0600: `stage_uploaded_file` `os.chmod(..., 0o600)`; Docker `docker-upload-files-after-upload.txt` shows mode `600`.
- Symlink safety: `remove_staged_attachment_file` refuses symlink session dir, requires `target.parent.resolve() == subdir.resolve()`, rejects `.`/`..` and real dirs, and `unlink()`s without following; `remove_session_uploads` deliberately does not `resolve()` the target and unlinks a symlink entry itself. Both validate `session_id` against traversal and are scoped to `<root>/<sid>` (sibling preservation). Tests `test_remove_staged_attachment_file_unlinks_only_direct_child`, `test_broken_symlink_entry_is_unlinked`, plus the sibling-untouched assertions.
- `delete_session` pops staged state and calls `remove_session_uploads` unconditionally (removes the whole session upload dir, cleaning orphans not in the list); `DeletedSessionStateChanges.staged_attachments` drives `save_staged_attachments`.
- Backend-readable paths: absolute `Attachment N: /…/uploads/<sid>/<file>` (verified in Docker send payload).
- Compatibility projection: `_sync_pending_projection_locked` keeps `session.pending_attachment` and `pending_attachment_ids` equal to staged-list non-emptiness on every add/remove/clear; `clear_pending_attachment` delegates to `clear_staged_attachments`. Docker final state: `staged_attachments.json` and `pending_attachments.json` both list only the commit_unknown session; the sent session's dir is empty. Two sessions coexisted safely (sibling preservation demonstrated live).

### D6 — Evidence separation ✅
- `git diff --stat` per range: `e1c8315` touches only `codoxear/*` + `tests/*`; `b1e6bc2` touches only `.memory/tasks/2026-07-07-staged-upload-expansion/browser-artifacts/**` (35 files, 587 insertions). Functional and proof commits are disjoint.

---

## Findings (all NONBLOCKING)

- **N1 — `attachments/clear` route does not catch the symlink-guard `ValueError`.**
  `control_routes.py:179 _handle_attachments_clear` catches only `KeyError`; `session_store.py:441 clear_staged_attachments` loops `remove_staged_attachment_file` with no try/except. A malformed/tampered staged path (e.g. session subdir replaced by a symlink) would raise mid-loop → HTTP 500, in-memory list left intact while some files already deleted (list/disk divergence).
  *Mechanism/impact:* staged paths are always server-generated direct children, so this is unreachable without write access to the app dir (already-compromised host). Fails loud (500), not silent. `attachments/delete` handles the same error as 400. Hardening: catch `ValueError` in the clear route (and/or per-entry in the store loop) and return 400, popping regardless.

- **N2 — Absolute server path exposed to the browser.**
  `session_listing.py:216` ships each staged entry's absolute `path` to the authenticated client on every poll for every session; the UI uses it only for the chip tooltip. Minor information exposure (reveals app-dir layout/username); pre-existing pattern (`inject_file` already returned `path`). Basename would suffice.

- **N3 — `inject_attachment_keys` / `SessionAttachmentCoordinator` is now dead for the HTTP surface.**
  Still defined (`session_attachment.py`) and bound (`session_manager_method_bindings.py:136`) but no route reaches it. A latent PTY-write method retained for tests. Consider removing in a later cleanup to eliminate the possibility of a future route re-wiring a pre-send write.

- **N4 — `attachment_index` request field is vestigial.**
  `control_routes.py:298` still validates it (must be int) but staging ignores it; send numbers references via `enumerate`. Cosmetic dead parameter; the browser still computes/sends it.

- **N5 — Narrow concurrent stage-during-active-send race.**
  Staging takes only the manager lock, not the input lock; a stage landing between `require_send_preconditions` and the post-success `set_pending_attachment(False)` branch could momentarily desync `pending_attachment` vs the staged list. Guarded in practice by `attachment_staging_ready` refusing while `queue_sending_item_id`/runtime is non-idle, and self-corrected on the next poll/reload (`load_all` re-merges staged keys). Theoretical, single-user-per-session context.

- **N6 — Staging is gated on broker send-readiness.**
  `attachment_staging_ready` (`session_readiness.py:114/122/146`) still requires `sync_send_supported` and runtime `direct_send`, so a file cannot be pre-staged while the agent is busy. Deliberate (fails closed; a session that cannot commit should not accumulate attachments), and `require_key_write_errors=False` correctly decouples staging from the old key-injection capability. UX limitation, not a correctness defect.

## Required fixes
None to accept this slice. N1 is the only item worth a follow-up (defensive hardening); the rest are cleanup/UX notes.

## Evidence checked
- `git status` / `git log` / `git show 38e2120 --stat`; `git diff --stat` for `38e2120..b1e6bc2`, `38e2120..e1c8315`, `e1c8315..b1e6bc2`; `git check-ignore` (PROMPT.md gitignored).
- Read impl diffs + current source: `file_upload.py`, `control_routes.py`, `session_store.py`, `session_pending_state.py`, `session_send.py`, `session_readiness.py`, `session_listing.py`, `server.py`, `server_handler.py`, `session_attachment.py`, `session_queue.py`, `session_input.py`, `static/app.js`.
- `grep` sweeps: `inject_attachment_keys`, `attachment_staging_ready` vs `attachment_injection_ready`, `allow_pending_attachment` call sites, socket-write ops in staging modules, route table mapping.
- Independent validation: `node --check codoxear/static/app.js` (OK); `git diff --check 38e2120..e1c8315` (clean); focused `pytest` over the touched + attachment/send/store/queue suites → **240 passed, 22 subtests passed** (consistent with the reported 233).
- Proof artifacts read: `VERIFICATION-REPORT.md`, `docker-calls-after-{upload,clear,send}-summary.json`, `docker-unknown-calls-summary.json`, `api-unknown-summary.json`, `browser-after-{multifile-upload,remove-one,clear-all,send}.json`, `docker-final-state.txt`, `docker-upload-files-after-upload.txt`, `fake_upload_session.py`, `docker-server.txt`.
- Secret/bulk scan of proof dir: 152K total, largest 7.5K; `api-login.json` = `{"ok":true}`; only `hmac_secret` hit is a filename in a `ls` listing (mode 0600, no content); no cookies/tokens/passwords committed.
- Live Docker re-verification intentionally skipped: the committed proof already logs the discriminating socket commands (send/keys counts, exact send payload, 504 preservation) against the real recovery server, and the unit/route/store suite was reproduced. A fresh run would be redundant with no remaining live uncertainty to resolve.
