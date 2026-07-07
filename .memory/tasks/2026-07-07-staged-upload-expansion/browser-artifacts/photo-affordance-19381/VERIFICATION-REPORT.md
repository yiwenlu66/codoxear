# Photo attachment affordance verification

Docker sandbox: `codoxear-photo-affordance-19381` on `http://127.0.0.1:19381/` with container HOME `/home/tester`. Fake broker sidecar advertised `sync_send:true` and `key_write_errors:false`.

## Claims exercised

1. **The visible producer no longer promises guaranteed camera capture.**
   - `#captureBtn` title before staging was `Add photo (max 16.0 MB)`.
   - `#captureBtn` aria-label before staging was `Add photo (max 16.0 MB)`.
   - The hidden mobile-capable input still had `accept='image/*'` and `capture='environment'`.

2. **The producer still routes through the real staged-upload path.**
   - The driver installed a no-name JPEG `File` on `#captureInput.files` and dispatched the real `change` listener.
   - The delayed `/inject_file` wrapper observed the transient toast `staging photo...` with events `[{'event': 'inject-start', 'toast': 'staging photo...', 'ts': 1783399337325}, {'event': 'inject-response', 'toast': 'staging photo...', 'ts': 1783399337331}]`.
   - The staged display name was `photo-1783399337321.jpg` and the browser chip title was `photo-1783399337321.jpg · 14 B · attachment 742960a0`.
   - The textarea stayed `''` and the badge became `1`.

3. **Public/private staged-upload boundary remains intact.**
   - `/attachments` returned `1` staged entry with no public `path` key (`stagedEntriesHavePathKey` = `False`).
   - `/attachments` + `/api/sessions` public payload did not contain the upload root (`publicPayloadContainsPath` = `False`).
   - Broker calls after staging were `{'state': 27}` with `send_count=0` and `key_count=0`.

4. **Explicit send remains the only backend commit boundary.**
   - After clicking send, staged entries cleared: badge ``, chips `[]`, API `{'attachments': [], 'ok': True, 'pending_attachment': False}`.
   - Broker calls after send were `{'state': 32, 'send': 1}` with `send_count=1` and `key_count=0`.
   - The single send payload was `Attachment 1: /home/tester/.local/share/codoxear/uploads/photo-proof/1783399337329_photo-1783399337321.jpg
please process added photo`.

## Validation

- `node --check codoxear/static/app.js`
- Focused source tests: `11 passed`
- Full local suite: `1793 passed, 128 subtests passed`
- `git diff --check`
- Docker gate on port `19382`: `1792 passed, 1 skipped, 128 subtests passed`
- Docker smoke on port `19382`: pre-login `/api/me` 401, post-login `/api/sessions` 200, app dir `/home/tester/.local/share/codoxear`

## Raw artifacts retained

- `fake_photo_session.py`
- `photo-affordance-stage-driver.js`
- `photo-affordance-send-driver.js`
- `browser-ready.json`
- `browser-after-photo-stage.json`
- `browser-after-send.json`
- `docker-calls-after-stage-compact.json`
- `docker-calls-after-send-compact.json`
- `docker-test-19382.txt`
- `docker-smoke-19382.txt`
- `docker-final-state.txt`
