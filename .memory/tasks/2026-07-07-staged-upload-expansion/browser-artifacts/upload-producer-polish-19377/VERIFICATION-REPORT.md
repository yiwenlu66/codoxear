# Upload producer polish verification

Docker sandbox: `codoxear-producer-polish-19377` on `http://127.0.0.1:19377/` with container HOME `/home/tester`. Fake broker sidecar advertised `sync_send:true` and `key_write_errors:false`.

## Claims exercised

1. **Mixed text+file paste preserves text and stages files.**
   - Browser dispatched a real `paste` event on `#msg` with two no-name image `File` objects and `text/plain` clipboard data.
   - The listener prevented default (`pasteDefaultPrevented` = `True`) and the textarea became `'prefix-MIXED-TEXTsuffix'` (`mixedPasteTextPreservedBeforeSend` = `True`).
   - The staged upload route was called twice through the shared producer path; captured filenames were `['pasted-1783393046685.jpg', 'pasted-1783393046685-2.webp']`.

2. **Pasted image fallback names use MIME extensions.**
   - No-name `image/jpeg` and `image/webp` clipboard files produced `.jpg` and `.webp` upload names (`pastedNamesHaveMimeExtensions` = `True`).
   - Public staged entries and chips used those names and contained no backend path key (`publicPastePayloadContainsPathKey` = `False`, `chipTitlesContainSlash` = `False`).

3. **Drag/drop highlight clears on window leave and off-composer drop remains a navigation fail-safe.**
   - File `dragenter` on `.composer` activated the highlight (`activeAfterDragEnter` = `True`), and a window-leave `dragleave` cleared it (`activeAfterWindowLeave` = `False`).
   - A later off-composer window `drop` with files was prevented (`offComposerDropDefaultPrevented` = `True`), cleared the highlight (`activeAfterOffComposerDrop` = `False`), and did not stage extra files (`offComposerDropDidNotStage` = `True`).

4. **Send remains the only backend commit boundary.**
   - Fake broker command summary had `1` `send` call and `0` `keys` calls.
   - The send payload contained absolute private upload paths for `.jpg` and `.webp` attachments (`send_contains_jpg_attachment` = `True`, `send_contains_webp_attachment` = `True`) plus the user prompt (`send_contains_prompt` = `True`).
   - Confirmed send cleared staged entries: `attachmentsAfterSend` = `{'attachments': [], 'ok': True, 'pending_attachment': False}`.

## Validation

- `node --check codoxear/static/app.js`
- Focused implementation tests: `11 passed`
- Full local suite: `1793 passed, 128 subtests passed`
- `git diff --check`
- Docker gate on port `19378`: `1792 passed, 1 skipped, 128 subtests passed`
- Docker smoke on port `19378`: pre-login `/api/me` 401, post-login `/api/sessions` 200, app dir `/home/tester/.local/share/codoxear`

## Raw artifacts retained

- `fake_upload_session.py`
- `upload-producer-polish-driver.js`
- `browser-producer-polish-result.json`
- `docker-calls-summary.json`
- `docker-test-19378.txt`
- `docker-smoke-19378.txt`
- `docker-final-state.txt`
