# EPISTEMIC

## Phenomenon
The sidebar is the primary session navigation surface. Backend, launch ownership, and reasoning effort are visible there; model identity must also be visible because it distinguishes same-project sessions that use different backends/models.

## Accepted mechanism
The sidebar row builder now computes `modelTxt = sidebarModelText(s)` and builds `.metaText` from `[stateTxt, modelTxt, cwdBase, branchTxt].filter(Boolean).join(" | ")`.

`sidebarModelText(s)`:
- reads only `s.model` from the existing session-list row;
- trims whitespace;
- returns an empty string for absent, empty, whitespace-only, or case-insensitive `default`;
- returns the trimmed non-default model string otherwise.

Rendering still uses the `text:` property, so model strings are assigned as text content rather than HTML. Existing backend logo, owner icon, effort marker, fast-session icon, title row, swipe actions, Details/diagnostics, and API/backend semantics are unchanged.

## Accepted claim
The sidebar model-display gap is fixed in commits `373b39f` (implementation), `b419699` (proof), and `69f736f` (review). Meaningful model names now appear between age and cwd in actual sidebar rows; null/empty/default values are omitted; long provider/model strings truncate through existing `.metaText` ellipsis without body overflow.

## Evidence basis
- Source/runtime tests cover `sidebarModelText` for null/missing/empty/whitespace/default/gpt/claude/long provider-model cases, verify `stateTxt | model | cwdBase | branchTxt` ordering, and check effort/fast markers remain separate.
- Local focused pytest passed (`40 passed`), full local pytest passed (`1817 passed, 134 subtests`), Docker focused pytest passed (`40 passed`), and Docker smoke passed.
- Docker/browser proof in `browser-artifacts/sidebar-model-display-19431/` created five fake sidecar/socket sessions and verified desktop plus mobile sidebar metadata: `gpt-5.4` and `claude-sonnet-4-5` appear, `default` and empty models are omitted, and the long provider/model row is truncated on mobile with `bodyOverflow=false`.
- Clean-room review in `reviews/sidebar-model-display-cleanroom-review.md` accepted the result with no blockers.

## Boundary
This proves sidebar projection of already-available model metadata. It does not change model selection, backend launch behavior, provider choice semantics, Details/diagnostics, or API schema.
