# Staged attachment path redaction verification

Docker sandbox: `codoxear-redaction-19371` on `http://127.0.0.1:19371/` with container HOME `/home/tester`. Fake broker sidecar advertised `sync_send:true` and `key_write_errors:false`.

## Claims exercised

1. **Pre-send staged attachment public payloads omit backend paths.**
   - Browser staged two files through the real hidden `#imgInput` change listener.
   - Captured `/inject_file` responses returned `attachment`/`attachments` entries containing `id`, `display_name`, `filename`, `size`, and `created_ts`, with no `path` key.
   - `/api/sessions/redaction-proof/attachments` and the `redaction-proof` row from `/api/sessions` likewise returned staged entries with no `path` key.
   - `preSendContainsUploadRoot` was `False` and `preSendStagedEntriesHavePathKey` was `False`.

2. **Browser-visible chips do not expose absolute upload paths.**
   - Chip titles were `['alpha-secret.txt · 11 B · attachment a7d8da91', 'beta-secret.txt · 10 B · attachment 0cd6dce9']`.
   - `chipTitlesContainSlash` was `False` and `chipTextsContainSlash` was `False`.

3. **Send remains the backend path commit boundary.**
   - Fake broker command summary had `1` `send` call and `0` `keys` calls.
   - The send payload contained absolute internal upload paths for both attachments: attachment 1 absolute = `True`, attachment 2 absolute = `True`.
   - The send payload also contained the user text (`send_contains_user_text` = `True`).
   - After confirmed send, `/attachments` returned `{'attachments': [], 'ok': True, 'pending_attachment': False}` and browser chips were `[]`.

## Local and Docker validation

- `node --check codoxear/static/app.js`
- Focused redaction/upload suite: `200 passed, 18 subtests passed`
- Full local suite: `1791 passed, 128 subtests passed`
- `git diff --check`
- Docker sandbox unit gate on port `19372`: `1790 passed, 1 skipped, 128 subtests passed`
- Docker smoke on port `19372`: pre-login `/api/me` 401, post-login `/api/sessions` 200, container app dir `/home/tester/.local/share/codoxear`

## Raw artifacts retained

- `path-redaction-driver.js`
- `fake_redaction_session.py`
- `browser-redaction-result.json`
- `docker-calls-summary.json`
- `docker-test-19372.txt`
- `docker-smoke-19372.txt`
- `docker-final-state.txt`
- `docker-server-19371.log`
