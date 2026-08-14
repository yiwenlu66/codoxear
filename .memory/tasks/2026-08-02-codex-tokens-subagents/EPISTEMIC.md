# EPISTEMIC

## Phenomenon

Codoxear’s Codex typing counter assumes `event_msg.payload.info.total_token_usage.reasoning_output_tokens` is cumulative, while active child rollout logs are currently ignored by the parent-session subagent projection.

## Accepted mechanism

Codex 0.146.0 native `event_msg.payload.type == "token_count"` rows contain `info.total_token_usage.reasoning_output_tokens`. The field is cumulative inside the observed rollout. The first token snapshot after a fresh user-turn scan must be added (equivalent to a zero baseline); subsequent monotonic snapshots contribute only their difference. A smaller value is a fork/rollback rebaseline and contributes nothing.

Codex child lineage is present in the native child header at `session_meta.payload.source.subagent.thread_spawn.parent_thread_id`. The parent session is keyed by its `thread_id`, while Pi remains keyed by its log path. A child is live only if a same-UID process has its rollout open writable, or it was modified inside the eight-second writer-discovery grace period, and its latest decisive event does not close the work episode. The scan is cached for two seconds; retained completed child files age out and terminal rows suppress even fresh/open completed files.

## Verified live evidence

- Released `codex-cli 0.146.0` through DexGem model `gpt-5.5` completed an arithmetic reasoning turn. Its final token-count row contained `total_token_usage.reasoning_output_tokens: 23` and `last_token_usage.reasoning_output_tokens: 23`. The repaired reducer projected `thinking_blocks: 1`, `thinking_tokens: 23`, and cumulative total 23. The actual transcript typing renderer selected `mode: "tokens"` and rendered `tools: 0 · thinking: 23`.
- A released Codex/DexGem multi-agent turn created native child headers. Parent `019fc244-b030-7de3-954a-6278909a128b` had child `019fc244-bff1-70b2-8a32-dab03858a459` with `thread_spawn.parent_thread_id` equal to that parent; the live session-list projection returned `subagents_running: 1`. After the child rollout completed and exceeded the grace period, the scanner returned zero for the parent.

## Current justified claim

The token projection now matches the real emitted schema and correctly includes the first cumulative snapshot. Codex and Pi subagents use their native parent identities but converge on one public `subagents_running` field, preserving the existing sidebar marker and idle activity bubble behavior. Claude Code has no subagent scan path and remains zero.
