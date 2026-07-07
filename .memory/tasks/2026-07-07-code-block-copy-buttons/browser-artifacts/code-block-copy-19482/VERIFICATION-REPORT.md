# Per-code-block copy browser proof

Scope: Docker-only Codoxear instance on port 19482 with a synthetic Codex transcript containing one assistant message with prose and two fenced code blocks.

## Validated behavior

- Assistant markdown rendered two `<pre>` blocks and two block-local `.code-copy-btn` controls.
- Clicking the first block button copied exactly `printf 'alpha <tag> & value'`.
- Clicking the second block button copied exactly `{"beta": 2, "note": "second block"}`.
- Neither block copy payload included surrounding prose.
- Message-level copy still copied the full raw assistant markdown, including prose and both fenced code blocks.
- Desktop buttons measured 30×30 px with default opacity 0.72; mobile buttons measured 44×44 px.
- Desktop and mobile evaluations reported no page-level horizontal overflow.

## Evidence files

- Docker focused tests: `docker-focused-19480.txt` (`37 passed`).
- Docker smoke: `docker-smoke-19481.txt` (`pre_login_api_me_status=401`, `post_login_sessions_status=200`).
- Synthetic session harness: `container/fake_code_copy_session.py`.
- API session/tail evidence: `api/sessions-before-proof.pretty.json`, `api/tail-before-proof.pretty.json`.
- Desktop browser proof: `browser/code-copy-proof-result.json`, `browser/snapshot-after-copy.txt`.
- Mobile browser proof: `browser/mobile-code-copy-proof-result.json`, `browser/mobile-snapshot-after-copy.txt`.
- Cleanup proof: `container/docker-stop.txt`, `container/docker-ps-after-stop.txt`.
