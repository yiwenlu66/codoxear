# Monaco-first editor/diff verification

Docker sandbox: `codoxear-monaco-19317` on `http://127.0.0.1:19317/` with container HOME `/home/tester`; app dir resolved to `/home/tester/.local/share/codoxear`.

## Claims exercised

1. **Vendored Monaco is served in a clean Docker browser deployment.**
   - `/monaco/vs/loader.js` returned HTTP 200 (`api-monaco-loader.status`, `api-monaco-loader.headers`).
   - The browser page exposed `window.require === "function"` and `window.monaco === "object"` after opening a file (`browser-open-editor-result.json`).

2. **Editable text files use Monaco, not a writable textarea baseline.**
   - Opening `notes.txt` produced `#fileDiff .monaco-editor`, `window.monaco.editor.getEditors().length === 1`, and `.filePlainEditTextarea === false`.
   - Editing through the Monaco model and clicking the UI Save button wrote the file through `/file/write`; a subsequent `/file/read` contained `SAVED-BY-MONACO` (`browser-save-result.json`).

3. **Repository diff uses Monaco diff editor, not a plain unified-diff substitute.**
   - Opening changed `changed.md` in diff mode produced `.monaco-diff-editor === true` and no `.filePlainEditTextarea` / no `pre.filePlainFallback` (`browser-diff-result.json`).
   - Monaco diff model values contain the old and modified file bodies, including `NEW monaco diff line` in the modified side (`browser-diff-models.json`).

4. **Monaco-unavailable state fails loud and read-only.**
   - In a fresh browser session, `window.require`/`window.monaco` were removed before opening a file. The UI rendered explicit “Code editor unavailable” copy, no Monaco editor, no `.filePlainEditTextarea`, and the edit affordance carried the unavailable explanation (`browser-monaco-unavailable.json`).
   - In that state the diff toggle was disabled and no plain unified diff substitute was rendered (`browser-diff-unavailable.json`).

5. **Save conflict controls preserve Monaco drafts and can reload disk content.**
   - External write changed `notes.txt`, then saving the stale Monaco draft rendered the conflict row.
   - “Keep editing” preserved the exact Monaco draft `UNSAVED-MONACO-DRAFT\n`.
   - “Reload from disk” with confirm accepted loaded `EXTERNAL-DISK-VERSION\n` into Monaco (`browser-conflict-result.json`).

## Raw artifacts

- `fake_monaco_session.py` — deterministic fake broker/session/repo fixture inside Docker only.
- `api-summary.json`, `api-sessions.json`, `api-file-notes.json`, `api-git-diff-changed.json`.
- `browser-open-editor-result.json`, `browser-save-result.json`, `browser-diff-result.json`, `browser-diff-models.json`, `browser-monaco-unavailable.json`, `browser-diff-unavailable.json`, `browser-conflict-result.json`.
- `browser-monaco-diff.png`, `browser-monaco-unavailable.png`.
- `docker-isolation.txt`, `docker-fixture-state.txt`, `docker-server.log`.
