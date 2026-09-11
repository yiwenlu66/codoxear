## Objective
Implement automatically visible compact per-child activity lines inside both busy and idle transcript activity bubbles, with accurate fresh child telemetry from API through selected-session state and no click/toggle UI. Correct their presentation so each activity bubble is content-sized up to the available transcript width, child telemetry is concise and wraps without horizontal overflow, and each header label begins immediately after its dots/marker at the intended gap regardless of child-line length.

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
- Activity bubbles must use only the width their wrapping content needs, bounded by the available transcript width; short content must not stretch across the row, while long model/tool lines wrap without horizontal overflow at 390px.
- In child detail lines, render a `provider/model` model identifier as `model`; leave bare model names unchanged.
- Label cumulative usage as `tokens`, replacing `tokens used`.
- Verify the real API-to-UI path for both busy and idle states with short and long actual model identifiers at phone and desktop widths. Capture screenshots and raw computed geometry (`bubble width`, `available width`, `scrollWidth`, `clientWidth`, text, and header marker-to-label gap) without relying on a fake DOM/store.
- Follow-up regression report `/home/yiwen/.local/share/codoxear/uploads/broker-1219519/1789103548085_IMG_3099.jpeg`: the idle dots remain at the left edge but `▸2 subagents working` is displaced toward the center because a spanning detail row contributes width to an `auto` marker track. Constrain the marker track to its intrinsic marker width in both idle and busy grids; the marker-to-label gap must remain unchanged between short and long two-child telemetry at 390×844 and desktop sizes.
- Rebuild tracked bundle and update AGENTS actual UI contract.

## Constraints
- Do not deploy or probe live service/API/brokers.
- Do not modify unrelated logo memory or untracked files.
- Do not add imperative cross-module rendering relays.
- Do not use source-string tests.
- Commit only explicit scoped files after staged-diff review.

## Follow-up request (2026-09-11): bubble alignment/spacing refinement

User: "alignment and spacing is a bit weird in the bubble. refine." — a small
visual refinement after shared-header commit `5d62d020`. No new features,
panels, toggles, sidebar, or counters.

- One consistent shared layout for busy+idle bubbles: matched internal padding
  and predictable left edges. Child lines align at bubble content-left (same
  left edge as the marker group) in BOTH states; drop the idle-only hanging
  `padding-left` indent whose 18px never matched the 22px header text indent.
- Header dots+summary keep their tight fixed gap, vertically centered against
  the first summary line.
- Tighten header→detail separation (8px → `--space-2`) and add modest
  inter-child separation (`--space-1`) so wrapped continuation visually belongs
  to its row.
- Existing spacing/type tokens only; no font shrinking, no label changes.
  Shrink-wrap, long-line wrapping, provider/model suffix, `tokens` label stay.
- Shared component CSS only — no per-theme structural patches.
- Concurrent voice/notification edits (including other `app.css` hunks) must
  remain untouched; stage only owned hunks; task-only git archive for browser
  verification. Docker-only testing; no deploy.
