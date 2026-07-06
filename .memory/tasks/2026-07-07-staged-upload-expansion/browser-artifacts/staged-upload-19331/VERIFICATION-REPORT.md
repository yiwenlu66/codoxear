# Staged upload expansion verification

Docker sandbox: `codoxear-upload-19331` on `http://127.0.0.1:19331/` with container HOME `/home/tester`; app dir resolved under `/home/tester/.local/share/codoxear`.

## Claims exercised

1. **Upload is stage-only before send.**
   - Fake broker sidecar advertised `sync_send: true` and `key_write_errors: false`.
   - Browser multi-file upload staged two files and rendered two attachment chips with backend-readable staged paths (`browser-after-multifile-upload.json`).
   - Server attachment list returned two entries and `pending_attachment: true` (`api-attachments-after-browser-upload.json`).
   - Broker call log after upload contained state probes only; no `send` and no `keys` command (`docker-calls-after-upload-summary.json`).

2. **Users can remove one attachment and clear all before send.**
   - Removing the first chip left one staged chip (`browser-after-remove-one.json`).
   - Clear removed all chips, hid the tray, and server list returned zero entries with `pending_attachment: false` (`browser-after-clear-all.json`, `api-attachments-after-clear.json`).
   - Broker call log after remove/clear still contained no `send` or `keys` command (`docker-calls-after-clear-summary.json`).

3. **Send is the attachment commit boundary.**
   - Browser re-staged two files and sent `please use staged uploads`.
   - Fake broker received exactly one `send` command and zero `keys` commands; send text began with two generated `Attachment N: <path>` lines followed by the user text (`docker-calls-after-send-summary.json`, summarized in command output).
   - After confirmed send success, browser chips were gone and server attachment list was empty/pending false (`browser-after-send.json`, `api-attachments-after-send.json`).

4. **Commit-unknown preserves staged attachments.**
   - API staged one file on a second Docker-only fake session, then fake broker returned `commit_unknown` for a confirmed send.
   - Route returned HTTP 504 with `commit_unknown: true`.
   - The staged attachment remained listed with `pending_attachment: true`; fake broker saw a send payload with `Attachment 1:` and no `keys` command (`api-unknown-summary.json`, `docker-unknown-calls-summary.json`).

## Validation commands

- `node --check codoxear/static/app.js`
- Focused pytest for upload/control/send/store/frontend source suites: `233 passed, 22 subtests passed`.
- `git diff --check`
- Full local pytest: `1782 passed, 132 subtests passed`.

## Raw artifacts

- `fake_upload_session.py`, `fake_upload_unknown_session.py`
- API/browser JSON files under this directory
- `docker-calls-after-upload-summary.json`, `docker-calls-after-clear-summary.json`, `docker-calls-after-send-summary.json`, `docker-unknown-calls-summary.json`
- `docker-final-state.txt`, `docker-server.txt`
