# CC unknown-model token clearing proof

## Outcome
Passed on Docker/browser port `19411`. A mapped Claude Code usage row first projected a visible context chip. After appending a newer Claude Code assistant usage row with an unmapped model, the same session projected `token: null` through `/api/sessions` and `/messages/tail`, and the browser hid/disabled `#ctxChip` while preserving the transcript rows.

This proves Codoxear clears stale context pressure when the newest CC usage row cannot be mapped. It does not add or guess a context window for the unknown model.

## Fixture
A Docker-only fake CC broker/session script (`fake_cc_unknown_token_session.py`) created one terminal-owned CC sidecar/socket under the container app dir and a Claude-shaped JSONL log under container `/home/tester/.claude/projects/...`.

Initial log rows:
- user: `cc unknown token clearing fixture`
- assistant mapped model `claude-sonnet-4-5` with usage `4321 + 100 + 79 = 4500`, text `KNOWN_MODEL_TOKEN_SENTINEL visible before unknown`

Mutation:
- appended assistant unmapped model `claude-unmapped-future-9` with usage and text `UNKNOWN_MODEL_CLEAR_SENTINEL should hide context chip`

The fake broker state always returned `token: null`; therefore known-token projection came from Codoxear log parsing, not broker state.

## Evidence
Before appending the unknown row:
- `/api/sessions` token: `tokens_in_context=4500`, `context_window=200000`, `percent_remaining=98` (`api-sessions-known.json`).
- `/messages/tail` token matched (`api-tail-known.json`).
- Browser `#ctxChip`: visible/enabled, text `Ctx 98%`, title `Context input: 4500/183616 tokens (16384 reserved; window 200000).` (`browser-known-state.json`).

After appending the unknown row:
- `/api/sessions` row for `cc-unknown-token-proof` has `token: null`, `busy=false` (`api-sessions-after-unknown.json`).
- `/messages/tail` has `token: null` and includes the known assistant row plus `UNKNOWN_MODEL_CLEAR_SENTINEL should hide context chip` (`api-tail-after-unknown.json`).
- Browser transcript contains the unknown sentinel; `#ctxChip` is `display:none`, `disabled=true`, empty text/title; body overflow remains false (`browser-after-unknown-state.json`).
- `container/cc-log.jsonl.txt` preserves the exact mapped and unmapped usage rows.
- `container/fake-broker-calls-summary.json` records only broker state/tail calls and zero send/key calls.

## Validation commands
- Local full pytest: `python3 -m pytest -q` → `1812 passed, 134 subtests passed in 25.27s`.
- Focused local suite after adding live-route coverage: `210 passed, 4 subtests passed`.
- Docker focused tests on port `19412`: `68 passed, 4 subtests passed`.
- Docker smoke on port `19413`: pre-login `/api/me` 401, post-login `/api/sessions` 200, app dir `/home/tester/.local/share/codoxear`.
- `git diff --check` passed.

## Hygiene
No cookies, auth headers, credential values, or private-key contents are stored. API defaults contain the literal auth-method label `apikey`; that is a static configuration value, not a credential. Container secret/private key filenames were removed from app-dir listings, and raw repeated broker-call logs were reduced to a command-count summary.
