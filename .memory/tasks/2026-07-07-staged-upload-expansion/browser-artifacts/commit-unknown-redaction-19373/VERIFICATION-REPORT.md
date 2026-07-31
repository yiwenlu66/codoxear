# Commit-unknown attachment preview redaction verification

Docker sandbox: `codoxear-unknown-redaction-19373` on `http://127.0.0.1:19373/` with container HOME `/home/tester`. Fake broker sidecar advertised `sync_send:true` and `key_write_errors:false`; its `send` response returned `commit_unknown:true` to deterministically exercise the unknown-send recovery state.

## Claims exercised

1. **Commit-unknown recovery preview is public-safe.**
   - Browser staged one file through the real hidden `#imgInput` change listener and sent via the visible `#sendBtn`.
   - `/api/sessions` projected `commit_unknown_send:true` with `commit_unknown_send_text` equal to `'commit unknown redaction prompt'`.
   - `rowCommitUnknownTextEqualsPrompt` was `True` and `rowCommitUnknownTextContainsAttachmentLine` was `False`.
   - `rowContainsUploadRoot`, `publicPayloadContainsUploadRoot`, and `bodyContainsUploadRoot` were `False`, `False`, and `False`.

2. **Staged-list public redaction still holds across commit-unknown.**
   - After the unknown send, `/attachments` preserved the staged entry with no `path` key (`stagedEntriesHavePathKey` = `False`).
   - The pre-send chip title was `commit-secret.txt · 12 B · attachment 26ee75c7` and `chipTitlesContainSlash` was `False`.

3. **Private committed text still preserves backend-readable attachment paths.**
   - Fake broker recorded `1` `send` and `0` `keys` calls.
   - The send payload contained the absolute internal upload path (`send_contains_attachment_absolute` = `True`) and the user prompt (`send_contains_user_text` = `True`).
   - Container `commit_unknown_sends.json` kept full private `text` with upload root (`private_record_text_contains_upload_root` = `True`) while storing public `display_text` = `'commit unknown redaction prompt'` with no upload root (`private_record_display_text_contains_upload_root` = `False`).

## Validation

- Focused regressions: `4 passed` for commit-unknown display text/fallback redaction and private display text persistence.
- Broader focused suite: `160 passed, 18 subtests passed`.
- Full local suite: `1793 passed, 128 subtests passed`.
- `node --check codoxear/static/app.js`, `node --check codoxear/static/app_display.js`, and `git diff --check` passed.
- Docker gate on port `19376`: `1792 passed, 1 skipped, 128 subtests passed`.
- Docker smoke on port `19376`: pre-login `/api/me` 401, post-login `/api/sessions` 200, app dir `/home/tester/.local/share/codoxear`.

## Raw artifacts retained

- `fake_commit_unknown_session.py`
- `commit-unknown-redaction-driver.js`
- `browser-commit-unknown-result.json`
- `docker-calls-summary.json`
- `docker-private-state.txt`
- `docker-test-19376.txt`
- `docker-smoke-19376.txt`
