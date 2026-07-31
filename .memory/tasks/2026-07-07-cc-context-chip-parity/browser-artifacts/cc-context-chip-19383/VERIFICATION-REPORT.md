# Claude Code context chip verification

Docker sandbox: `codoxear-cc-context-19383` on `http://127.0.0.1:19383/` with container HOME `/home/tester`. Fake sidecar advertised `agent_backend: cc` and pointed to a synthetic Claude Code JSONL log under `/home/tester/.claude/projects/...`.

## Claims exercised

1. **Claude Code assistant usage now projects token state.**
   - `/api/sessions` row `cc-context-proof` had `agent_backend=cc` and model `claude-sonnet-4-5`.
   - Session token was `{'as_of': '2026-07-07T04:55:10.000Z', 'context_window': 200000, 'max_input_tokens': 183616, 'percent_remaining': 18, 'reserved_tokens': 16384, 'tokens_in_context': 150000, 'tokens_remaining': 33616}`.
   - The token uses `tokens_in_context=150000` from `input_tokens 100000 + cache_read_input_tokens 20000 + cache_creation_input_tokens 30000`; the synthetic `output_tokens 9999` is excluded.
   - Context window was `200000` for `claude-sonnet-4-5`, with `reserved_tokens=16384`, `max_input_tokens=183616`, and `percent_remaining=18`.

2. **The browser renders the existing backend-agnostic context chip.**
   - `#ctxChip` display was `flex`.
   - Chip text was `Ctx 18%`.
   - Chip title was `Context input: 150000/183616 tokens (16384 reserved; window 200000).`.

3. **Message polling carries the same token shape.**
   - `/messages/tail` token was `{'as_of': '2026-07-07T04:55:10.000Z', 'context_window': 200000, 'max_input_tokens': 183616, 'percent_remaining': 18, 'reserved_tokens': 16384, 'tokens_in_context': 150000, 'tokens_remaining': 33616}`.
   - Transcript events rendered the synthetic CC user prompt and assistant final answer: `['user', 'assistant']`.

4. **No backend write boundary changed.**
   - Fake broker command summary was `{'state': 438}` with `send_count=0` and `key_count=0`.
   - The proof only reads session state and transcript; no send/queue/attach path is exercised or modified by this slice.

## Validation

- `python3 -m py_compile codoxear/cc_log.py codoxear/rollout_tokens.py`
- Focused tests: `59 passed`
- Full local suite: `1798 passed, 128 subtests passed`
- `git diff --check`
- Docker gate on port `19384`: `1797 passed, 1 skipped, 128 subtests passed`
- Docker smoke on port `19384`: pre-login `/api/me` 401, post-login `/api/sessions` 200, app dir `/home/tester/.local/share/codoxear`

## Raw artifacts retained

- `fake_cc_context_session.py`
- `cc-context-chip-driver.js`
- `api-sessions-initial.json`
- `api-sessions-after-browser.json`
- `api-messages-tail.json`
- `browser-cc-context-chip.json`
- `docker-calls-compact.json`
- `docker-test-19384.txt`
- `docker-smoke-19384.txt`
- `docker-final-state.txt`
