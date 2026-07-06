# Capture staged-upload producer verification

Docker sandbox: `codoxear-capture-19357` on `http://127.0.0.1:19357/` with container HOME `/home/tester`; fake broker sidecar advertised `sync_send:true` and `key_write_errors:false`.

## Claims exercised

1. **Capture control is present and capture-capable.**
   - Browser DOM exposed `#captureBtn` enabled with title `Capture photo (max 16.0 MB)`.
   - Hidden `#captureInput` had `accept='image/*'` and `capture='environment'`.

2. **Capture is a pure staged-upload producer.**
   - A synthetic captured JPEG `File` with no filename was installed on `#captureInput.files` and the real `change` listener was dispatched.
   - Browser rendered one staged chip and badge `1`; textarea stayed `''`.
   - Server attachment API returned one entry with display name `captured-1783381957888.jpg`, backend-readable path `/home/tester/.local/share/codoxear/uploads/capture-proof/1783381957894_captured-1783381957888.jpg`, size `14`, and `pending_attachment:true`.
   - Fake broker command summary after capture had `send_count=0` and `keys_count=0`; only `state` calls occurred.

3. **Send remains the only backend commit boundary.**
   - Clicking the visible send button produced `send_count=1` and `keys_count=0`.
   - The single send payload was `'Attachment 1: /home/tester/.local/share/codoxear/uploads/capture-proof/1783381957894_captured-1783381957888.jpg\nplease process captured photo'`.
   - After confirmed send success, browser chips were `[]`, badge was `''`, and server attachment API returned `attachments: []` with `pending_attachment:false`.

## Local validation before proof

- `node --check codoxear/static/app.js`
- `node --check codoxear/static/app_display.js`
- `python3 -m pytest -q tests/test_attach_button_source.py tests/test_frontend_display_module_source.py` → `13 passed`
- focused upload/control/frontend suite → `184 passed, 22 subtests passed`
- full local suite → `1796 passed, 132 subtests passed`
- `git diff --check`

## Raw artifacts

- `browser-ready.json`
- `browser-after-capture.json`
- `docker-calls-after-capture-compact.json`
- `browser-after-send.json`
- `docker-calls-after-send-compact.json`
- `combined-evidence.json`
- `fake_capture_session.py`
- `docker-session-seed.txt`
- `docker-final-state.txt`
- `docker-server.txt`
