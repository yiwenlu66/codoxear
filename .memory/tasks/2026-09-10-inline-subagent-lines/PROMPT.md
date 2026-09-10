## Objective
Implement automatically visible compact per-child activity lines inside both busy and idle transcript activity bubbles, with accurate fresh child telemetry from API through selected-session state and no click/toggle UI.

## Workbench
1. Trace Pi producer records and selected-session projection freshness.
2. Replace hidden expandable details with automatic inline child lines in both transcript activity states.
3. Add cross-layer behavioral coverage and Docker/browser phone-width proof.
4. Rebuild bundle, update contract docs, review, and commit only scoped files.

## Context
- Repository: `/home/yiwen/codoxear`
- Complaint screenshot: `/home/yiwen/.local/share/codoxear/uploads/broker-1219519/1789019919932_image.png`
- Existing hidden-detail implementation: commit `2dde496f`; deployment remains at `79d71532` and must not be probed.
- Preserve all unrelated logo task memory changes and existing untracked files.

## Task specifications
- Show one compact line per active child directly beneath the existing compact summary in the busy typing bubble.
- Show the same child lines in the idle transcript subagent activity bubble whenever active child activity exists.
- No sidebar-only disclosure, click expansion, panel, popover, modal, button, toggle state, or fake unknown rows.
- Selected-session catalog updates must refresh details when telemetry changes but child count does not.
- Pi rows correspond to actual active child steps (parallel/chain), including distinct role/model/tool/token telemetry; labels describe cumulative token usage, not context.
- Preserve Codex/Claude Code best-effort rows.
- Behavioral coverage: two parallel Pi children with different telemetry; one completes; same-count metric update; busy→idle continuity; session-switch cleanup.
- Docker-only tests and browser verification at screenshot-like phone width without clicking.
- Rebuild tracked bundle and update AGENTS actual UI contract.

## Constraints
- Do not deploy or probe live service/API/brokers.
- Do not modify unrelated logo memory or untracked files.
- Do not add imperative cross-module rendering relays.
- Do not use source-string tests.
- Commit only explicit scoped files after staged-diff review.
