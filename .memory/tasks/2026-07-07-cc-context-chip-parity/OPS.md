# Operational ledger

## 2026-07-07T04:52:00Z — Task initialized

- Objective: make Claude Code sessions surface a truthful context-usage chip using CC assistant usage records and conservative model context-window mapping.
- Prior scout: `/tmp/codoxear-next-product-slice-after-upload.md` recommended pivoting off completed upload work to CC context-chip parity.
- Implementation contract: `/tmp/cc-context-chip-next-slice-contract.md`.
- Initial observations:
  - `codoxear/rollout_tokens.py::_extract_token_update()` has Pi and Codex token extraction but no CC branch.
  - `codoxear/cc_log.py` has CC message/run-setting helpers but no token/context function.
  - `codoxear/static/app.js::setContext()` already renders `#ctxChip` for any backend when token is non-null.
- External docs observation: Anthropic context-window docs state `input_tokens`, `cache_read_input_tokens`, and `cache_creation_input_tokens` all count toward the context window; most non-1M Claude models including Sonnet 4.5 and Haiku 4.5 have 200k context, while Sonnet 4.6+/5, Opus 4.6+/4.7+/4.8, Fable 5, and Mythos 5/Preview have 1M.

## 2026-07-07T05:00:00Z — CC context usage functional implementation

Prediction: if Claude Code assistant `message.usage` is parsed into the same token-update shape as Codex/Pi, then CC session rows and message polls can render the existing frontend context chip without frontend changes.

Intervention:
- Functional commit: `a8059be Project Claude Code context usage`.
- Mechanism: `codoxear/cc_log.py` now exposes `cc_token_update()` and `cc_model_context_window()`. The parser reads assistant `message.usage` and `message.model`, computes `tokens_in_context = input_tokens + cache_read_input_tokens + cache_creation_input_tokens`, excludes `output_tokens`, resolves documented model windows conservatively, and returns the shared token dict shape with default reserved tokens.
- Conservative mapping correction: an initial executor patch mapped undocumented future Sonnet 4.7/4.8/4.9 variants to 1M; main tightened this before commit so `claude-sonnet-4-7` returns no token until explicitly documented/mapped.
- `codoxear/rollout_tokens.py::_extract_token_update()` now checks CC token updates after Pi and before Codex token_count rows.

Validation:
- `python3 -m py_compile codoxear/cc_log.py codoxear/rollout_tokens.py`.
- Focused tests (`tests/test_cc_log.py tests/test_server_chat_flags.py tests/test_cc_chat_and_idle.py`) → `59 passed`.
- Full local `python3 -m pytest -q` → `1798 passed, 128 subtests passed`.
- `git diff --check` → clean.

## 2026-07-07T05:08:00Z — CC context chip Docker/browser proof

- Proof commit: `e57f3ca Record Claude Code context chip proof`.
- Artifact dir: `.memory/tasks/2026-07-07-cc-context-chip-parity/browser-artifacts/cc-context-chip-19383/`.
- Docker sandbox: `codoxear-cc-context-19383` on port `19383`; stopped by exact container name after proof.
- Fake CC sidecar advertised `agent_backend: cc` and pointed at a synthetic Claude Code log under `/home/tester/.claude/projects/...`.
- Synthetic assistant usage: model `claude-sonnet-4-5`, `input_tokens=100000`, `cache_read_input_tokens=20000`, `cache_creation_input_tokens=30000`, `output_tokens=9999`.
- Observed `/api/sessions` token: context window `200000`, tokens in context `150000`, max input `183616`, reserved `16384`, percent remaining `18`.
- Observed `/messages/tail` token with the same shape and transcript events for the synthetic CC user+assistant rows.
- Browser `#ctxChip` rendered visible with text `Ctx 18%` and title `Context input: 150000/183616 tokens (16384 reserved; window 200000).`.
- Fake broker command summary contained only `state` calls (`send_count=0`, `key_count=0`).
- Docker gate on port `19384` passed (`1797 passed, 1 skipped, 128 subtests passed`) and smoke passed (`/api/me` 401 before login, `/api/sessions` 200 after login).

Evidence status: functional/proof committed; clean-room review `175836c6-d9e7-4c4a-8916-2f5303513c64` launched and pending.

## 2026-07-07T05:13:00Z — Clean-room review accepted

- Review artifact: `.memory/tasks/2026-07-07-cc-context-chip-parity/reviews/cc-context-chip-review.md`.
- Review commit: `fa0778d Record Claude Code context chip review`.
- Recommendation: ACCEPT; no blockers.
- Reviewer evidence:
  - Product code changes limited to `codoxear/cc_log.py`, `codoxear/rollout_tokens.py`, and `tests/test_cc_log.py`; proof-only changes under the task artifact directory.
  - `cc_token_update()` parses assistant rows, sums prompt/cache input tokens, excludes output tokens, and emits the shared token dict shape.
  - `rollout_tokens._extract_token_update()` adds CC without altering Pi/Codex branch bodies.
  - Synthetic proof token came from the CC log parser because the fake broker returned `token:null`.
  - Browser/API proof showed visible `Ctx 18%`, matching `/api/sessions` and `/messages/tail` token payloads, and zero send/key broker calls.
  - Focused audit rerun: `tests/test_cc_log.py -> 15 passed`; direct unknown-model check returned no token.
- Nonblocking residuals recorded in EPISTEMIC.md: no separate browser artifact for unknown-model hiding, and existing last-known-token semantics would retain a previous chip if a session later changed from known to unmapped model.

Decision: CC context-chip parity is accepted for known conservatively mapped Claude Code models. Update task/project memory and close the slice.
