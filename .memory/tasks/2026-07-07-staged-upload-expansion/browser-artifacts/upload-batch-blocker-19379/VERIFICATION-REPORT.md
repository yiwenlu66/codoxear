# Upload batch blocker recheck verification

Docker sandbox: `codoxear-batch-blocker-19379` on `http://127.0.0.1:19379/` with container HOME `/home/tester`. Fake broker sidecar advertised `sync_send:true` and `key_write_errors:false`.

## Discriminating setup

The proof stages a two-file picker batch. The browser driver delays delivery of the first `/inject_file` response after the server has staged `first.txt`. During that delay it deliberately creates a public unknown-send marker via `/send` with `allow_pending_attachment:true` and a fake-broker `commit_unknown:true` response. This is an external state change between batch items; the upload producer should re-read current session state and stop before uploading `second.txt`.

## Claims exercised

1. **Per-file blocker recheck stops the batch before the second upload.**
   - Browser captured `1` `/inject_file` request(s): `['first.txt']`.
   - The fetch timeline shows the first inject response, a commit-unknown marker (`status` `504`), then release of the first response; no second inject started.
   - The final toast was `'attached 1; stopped: Resolve the unknown send before attaching a file'` (`stoppedToast` = `True`).

2. **Partial success remains visible and public-safe.**
   - `/attachments` preserved exactly `1` staged entry after the stopped batch.
   - The staged entry had no public `path` key (`stagedEntriesHavePathKey` = `False`), and chip titles had no slash (`chipTitlesContainSlash` = `False`).

3. **Backend write boundary is not widened by upload.**
   - Fake broker summary had `0` `keys` calls.
   - The only `send` was the deliberate proof marker (`send_count` = `1`, `marker_send_count` = `1`), not an upload producer action.
   - Broker state observed the first staged file and reported busy (`busy_after_first_seen` = `True`); max staged-file count stayed `1` because the second upload did not occur.

## Validation

- `node --check codoxear/static/app.js`
- Focused implementation tests: `11 passed`
- Full local suite: `1793 passed, 128 subtests passed`
- `git diff --check`
- Docker gate on port `19380`: `1792 passed, 1 skipped, 128 subtests passed`
- Docker smoke on port `19380`: pre-login `/api/me` 401, post-login `/api/sessions` 200, app dir `/home/tester/.local/share/codoxear`

## Raw artifacts retained

- `fake_batch_blocker_session.py`
- `upload-batch-blocker-driver.js`
- `browser-batch-blocker-result.json`
- `docker-calls-summary.json`
- `docker-test-19380.txt`
- `docker-smoke-19380.txt`
- `docker-final-state.txt`
