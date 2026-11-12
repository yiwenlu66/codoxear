# Compaction fixtures

`pi-real.jsonl` is a six-entry window copied from a real Pi v3 session (November
12, 2026). Entry identities, parent links, timestamps, compaction metadata and
usage are preserved. Chat prose is replaced, tool arguments/results are removed,
and compaction summary/details/systemMessage content is stripped. The summarizer
used 20,834 tokens; the *next assistant* reports 62,227 context tokens. Those are
intentionally different.

`cc.jsonl` is synthetic, based on the official Claude Code transcript example:
https://code.claude.com/docs/en/sub-agents (`compact_boundary`,
`compactMetadata.trigger`, `compactMetadata.preTokens`). `isCompactSummary` is the
best-effort persisted-summary exclusion; it has not been observed in a real
Claude run in this environment.

`codex.jsonl` is a synthetic checkpoint/completion pair, based on Codex's
`CompactedItem` persistence and context-compaction completion lifecycle. It is
not a capture of a real Codex run. Both legacy notification-only and checkpoint-
only forms are exercised, as is the response_item-wrapped checkpoint variant.
