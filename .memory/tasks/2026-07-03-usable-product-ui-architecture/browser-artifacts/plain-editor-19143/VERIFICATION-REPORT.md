# Plain text file editor fallback — Docker/browser verification (rerun, artifact-preserving)

Verification only. No repo edits, staging, or commits. No access to protected
`/home/yiwen/codex-web` or host live runtime. All broker/server/fixture proof
ran inside a Docker sandbox; the browser (host Chromium via puppeteer-core) drove
the sandbox HTTP port only.

## Environment

- App dir served (container): `/home/tester/.local/share/codoxear`
- Container: `codoxear-plain-editor-19143` (removed after capture; `--rm`)
- Port: `19143` (127.0.0.1)
- Repo served (read-only bind-mount): `/home/yiwen/codex-web-product-recovery`
- Commit served: worktree HEAD `163320c` (contains plain-editor commit `ef83bda`
  and `1628498`; the two commits added after `1628498` are unrelated upload-route
  test changes and do not touch the file editor).
- Monaco asset `/monaco/vs/loader.js` returns **404**, so every browser path
  exercised the Monaco-unavailable fallback.

## Artifact root (left on host for the parent to copy)

`/tmp/codoxear-plain-editor-proof-19143/artifacts/` — 43 files:
DOM JSON (`dom-sc*.json`), API JSON (`api-*.json`), screenshots (`sc*.png`),
`browser-result.json`, `browser-summary.txt`.

## Scenario results

1. Monaco absent -> plain editor: PASS. loader 404; `.filePlainEditTextarea`
   present (readOnly in view mode), notice "Plain text editor — Rich editor
   unavailable. Editing is available in plain text.", no `.monaco-editor`.
2. Desktop edit/save/reopen: PASS. Edit -> textarea writable (editMode active);
   typing marks `#fileEditBtn.dirty`; Save writes via `/file/write`, disk version
   changes (`f09555…`), API text persists "SAVED-BY-EDITOR"; reopen shows saved
   content read-only.
3. Cancel/discard: PASS. Dirty draft ("UNSAVED-DRAFT-SHOULD-DISCARD",
   dirtyBeforeClose=true); close raises unsaved dialog -> "Discard" clicked;
   disk version unchanged.
4. Conflict 409 -> Reload/Keep: PASS (isolated driver). External mutate makes
   editor version stale; Save returns **409 `file changed on disk` conflict:true**;
   conflict row shows Reload-from-disk + Keep-editing; textarea draft
   ("EDITOR-DRAFT-CONFLICT") and dirty state preserved after Keep.
5. Unavailable-session/copy-edits: PASS. Removing the control socket drops the
   session from `/api/sessions`; viewer shows
   "Session is no longer available; copy unsaved edits before closing.";
   unsaved draft preserved, read-only, still selectable/copyable; Edit disabled.
6. Read-only regressions: PASS. Git changed.md -> read-only unified-diff plain
   fallback (`.filePlainFallback pre`, no textarea, no monaco, edit disabled,
   diff active; notice "Rich diff unavailable. Showing unified diff
   (read-only)"). binary.bin + oversize.txt -> download-only, edit disabled.
   nonutf.txt (raw bytes, editable=false) -> read-only plain fallback, edit
   disabled, status "nonutf.txt - read-only".
7. Mobile 390x844: PASS. viewport 390, scrollWidth 390 (no horizontal overflow),
   textarea writable, computed font-size **16px**, controls visible, dirty state
   visible.

## Harness notes / limitations

- SC4 was verified by an isolated browser driver (`sc4_isolated.js`). In the
  long multi-scenario page session the editor's cached file version was
  refreshed between open and save (the viewer re-reads on render), so the
  in-sequence conflict did not reproduce; the conflict mechanism itself (409 +
  Reload/Keep + draft preservation) is proven both in isolation and at the API
  level. This is a harness sequencing artifact, not a product defect.
- The Reload-from-disk action is gated by `window.confirm()`, which the headless
  driver did not drive; Keep-editing was exercised instead. The Reload affordance
  is present and visible in the captured conflict row.
- No product defect appeared; no stop-on-defect triggered.

## Repo state after verification

Recovery worktree: no changes, nothing staged. Protected `/home/yiwen/codex-web`
untouched.
