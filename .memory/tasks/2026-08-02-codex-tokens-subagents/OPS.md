# OPS

## 2026-08-02T19:35:00Z

- Initialized task memory. Working tree had pre-existing untracked `.memory/tasks/2026-07-31-issue-triage/OPS.md` and `.pi-subagents/`; they are excluded from this task.
- Read architecture, validation norms, prior Codex live-control evidence, existing token reducer, Pi subagent scanner, and listing projection.
- `codex` is absent from PATH and `~/.codex` has no configuration/session files. A previously unpacked released 0.146.0 executable is available under `/tmp/codex-release-probe/.../bin/codex`; prior live-control probes in that directory provide an isolated provider harness.

## 2026-08-02T19:32:00Z

- Started two isolated, owned `tmux` Codex `exec --json` probes with temporary `CODEX_HOME` and a DexGem Responses provider config. First model `gpt-5.6-sol` failed with capacity; retrying `gpt-5.5` succeeded without touching any broker or existing Codex process.
- The real completed rollout `/tmp/codoxear-real-codex-verify.YF3OS6/.../rollout-...019fc23f-58cf-7f01-802d-ce07bb7c6b7e.jsonl` recorded a token-count row with `total_token_usage.reasoning_output_tokens=23`, `last_token_usage.reasoning_output_tokens=23`, and `model_context_window=258400`. Native exec output independently reported `reasoning_output_tokens=23`.
- Pre-fix `_analyze_log_chunk` returned one thinking block but zero thinking tokens for that complete native log because it discarded the first cumulative snapshot. This falsified the assumed first-snapshot handling. Changed the reducer to add the first snapshot and retain it as baseline; direct runtime projection now reports 1 block, 23 tokens, total 23, and the actual frontend typing renderer reports `mode=tokens`, `tools: 0 · thinking: 23`.

## 2026-08-02T19:38:00Z

- A separate real Codex/DexGem `features.multi_agent_v2=true` run emitted child rollout headers. Parent `019fc244-b030-7de3-954a-6278909a128b` had a child `019fc244-bff1-70b2-8a32-dab03858a459` whose native header source was `subagent.thread_spawn.parent_thread_id=019fc244-b030-7de3-954a-6278909a128b`.
- While that run was active, `scan_active_codex_subagents()` returned the child and `build_active_session_rows_snapshot()` projected the real parent row as `subagents_running=1`. After all child rollouts emitted `task_complete`, the scanner returned zero even when passed a completed child as writable; terminal child rows take precedence over liveness. The scanner recognizes headers, requires a writable same-UID rollout FD or the short mtime grace for nonterminal children, and caches full scans for two seconds.
- Added focused behavioral coverage for first Codex cumulative snapshot, child header/liveness behavior, and Pi/Codex listing identity selection. These tests were added but not executed, per task constraint. `py_compile`, Node syntax, `git diff --check`, and workspace diagnostics passed.
- A direct cross-backend listing projection returned `{pi:1,codex:1,cc:0}` for injected Pi and Codex activity maps, confirming the new Codex key does not alter Pi's log-path key or project a count to Claude Code. An initial version of that one-off validation iterated the snapshot wrapper rather than `.rows`; it raised `TypeError` before making any assertion and was immediately corrected.
- Removed only the four task-owned tmux sessions after their Codex commands exited; no broker or pre-existing Codex process was stopped.
