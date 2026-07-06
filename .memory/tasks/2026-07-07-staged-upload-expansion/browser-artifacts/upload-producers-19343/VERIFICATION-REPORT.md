# Paste/drop staged-upload producer verification

Docker sandbox: `codoxear-producers-19343` on `http://127.0.0.1:19343/` with container HOME `/home/tester`; fake broker sidecar advertised `sync_send:true` and `key_write_errors:false`.

## Claims exercised

1. **File-bearing paste is a pure staged-upload producer.**
   - Synthetic `paste` event on `#msg` carried `paste-alpha.txt` as a `File`.
   - Event was `defaultPrevented:true`, textarea stayed empty, browser rendered one staged chip and badge `1`, and the server attachment API returned one entry with `pending_attachment:true`.
   - Fake broker command summary after paste had only `state` calls; zero `send` and zero `keys`.

2. **Text-only paste remains browser-owned.**
   - Synthetic text-only paste was not prevented (`defaultPrevented:false`), inserted `TEXT-ONLY-PASTE` into the textarea, and did not create another staged entry.
   - Broker command summary again had zero `send` and zero `keys`.

3. **Composer drag/drop stages files and prevents page navigation.**
   - Synthetic file `dragover` on `.composer` was prevented and set `.composer.drop-active`; synthetic `drop` staged `drop-beta.txt` and `drop-gamma.txt`, cleared the highlight, rendered three total chips, and server attachment API returned three staged entries.
   - Broker command summary after drop had zero `send` and zero `keys`.

4. **Off-composer file drop is a navigation fail-safe, not an attach producer.**
   - Synthetic file `drop` on `window` was prevented, URL stayed `http://127.0.0.1:19343/#session=upload-proof`, chip count stayed `3`, and server attachment API still returned three staged entries.
   - Broker command summary had zero `send` and zero `keys`.

5. **Send remains the only backend commit boundary.**
   - Sending `please process paste and drop files` produced exactly one fake-broker `send` and zero `keys`.
   - The send payload began with three generated `Attachment N: <path>` lines for the pasted/dropped staged files, followed by the user text.
   - After confirmed send success, browser chips were gone and server attachment API returned zero entries with `pending_attachment:false`.

## Local validation before proof

- `node --check codoxear/static/app.js`
- `node --check codoxear/static/app_file_helpers.js`
- Focused frontend/upload suite: `183 passed, 22 subtests passed`
- Full local suite: `1787 passed, 132 subtests passed`
- `git diff --check`

## Raw artifacts

- `browser-after-paste.json`, `api-after-paste.json`, `summary-after-paste.json`, `docker-calls-after-paste-summary.json`
- `browser-after-text-paste.json`, `api-after-text-paste.json`, `summary-after-text-paste.json`, `docker-calls-after-text-paste-summary.json`
- `browser-after-drop.json`, `api-after-drop.json`, `summary-after-drop.json`, `docker-calls-after-drop-summary.json`
- `browser-after-offzone-drop.json`, `api-after-offzone-drop.json`, `summary-after-offzone-drop.json`, `docker-calls-after-offzone-drop-summary.json`
- `browser-after-send.json`, `api-after-send.json`, `summary-after-send.json`, `docker-calls-after-send-summary.json`
- `fake_upload_session.py`, `docker-final-state.txt`, `docker-server.txt`, `combined-evidence.json`
