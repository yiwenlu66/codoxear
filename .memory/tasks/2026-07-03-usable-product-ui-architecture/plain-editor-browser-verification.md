All seven verification scenarios PASS with evidence preserved. Here is the result.

## Verification result: PASS (all 7 scenarios)

**Environment:** Docker sandbox `codoxear-plain-editor-19143` on port `19143`; container app dir `/home/tester/.local/share/codoxear`; repo served read-only from `/home/yiwen/codex-web-product-recovery` at HEAD `163320c` (contains plain-editor commit `ef83bda`; the two newer commits are unrelated upload-route tests). `/monaco/vs/loader.js` returns 404, so every path exercised the Monaco-unavailable fallback. Host Chromium (puppeteer-core) drove only the sandbox HTTP port.

**Artifacts preserved on host (do not delete — parent to copy):** `/tmp/codoxear-plain-editor-proof-19143/artifacts/` — 43 files: per-scenario DOM JSON, API JSON, screenshots, `browser-result.json`, and `VERIFICATION-REPORT.md`.

**Per-scenario outcome:**
1. Monaco absent → plain editor: `.filePlainEditTextarea` (readOnly view), notice present, no `.monaco-editor`. ✓
2. Edit/save/reopen: Edit→writable, typing→`#fileEditBtn.dirty`, Save→`/file/write` writes, version changes to `f09555…`, reopen shows saved content. ✓
3. Discard: dirty draft + unsaved dialog → "Discard" clicked → disk version unchanged. ✓
4. Conflict 409: editor Save returns `409 file changed on disk conflict:true`; conflict row with Reload/Keep; draft + dirty preserved after Keep. ✓
5. Unavailable session: "copy unsaved edits before closing" status; draft preserved, read-only, selectable; Edit disabled. ✓
6. Read-only regressions: changed.md diff → read-only unified-diff fallback (no textarea, edit disabled); binary/oversize → download-only; nonutf → read-only fallback. ✓
7. Mobile 390×844: font-size 16px, no horizontal overflow, writable textarea, dirty visible. ✓

**Two harness notes (not product defects):** SC4 was confirmed via an isolated driver because in the long multi-scenario page session the editor's cached version was refreshed between open and save (the viewer re-reads on render); the conflict mechanism itself is proven in isolation and at the API level. Reload-from-disk is gated by `window.confirm()` (not driven headlessly); Keep was exercised and Reload is visibly present in the screenshot.

**Cleanup/state:** Container removed (exact name); browser closed. Recovery worktree: no changes, nothing staged. Protected `/home/yiwen/codex-web` untouched. No `pkill/killall` or broad host cleanup used.