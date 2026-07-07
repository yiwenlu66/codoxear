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
