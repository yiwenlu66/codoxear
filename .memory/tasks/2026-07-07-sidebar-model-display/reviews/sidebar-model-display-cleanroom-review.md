# Clean-room adversarial review: sidebar model-name display

**VERDICT: ACCEPT**

No blockers found. The implementation is minimal, correct, and well-evidenced.

## Evidence inspected

### Implementation diff (f40026e..373b39f)

8 lines changed in `codoxear/static/app.js`. No other production files touched. No server, API, backend, or schema changes.

**`sidebarModelText(s)` helper** — Takes session object, extracts `s.model`, trims whitespace, returns empty string for null/undefined/empty/whitespace-only/case-insensitive "default". Returns trimmed model otherwise.

**metaText construction** — Refactored from template-literal with inline ternaries to `[stateTxt, modelTxt, cwdBase, branchTxt].filter(Boolean).join(" | ")`. Semantic equivalence with old code verified: `stateTxt` is always truthy (either `"starting"` or a formatted age); `baseName()` returns `""` for falsy cwd; `branchTxt` is `""` when branch absent. `filter(Boolean)` correctly removes all falsy segments, producing identical output to the old ternary chain when model is absent.

**Ordering** — `modelTxt` is computed after `stateTxt` and before `cwdBase`/`branchTxt`, matching the required `stateTxt | model | cwdBase | branchTxt`.

**Escaping** — The `el()` helper uses `n.textContent = v` for the `text:` property (verified in `app_dom.js:7`). Model text with HTML special characters renders as literal text. No XSS vector.

### Tests (test_sidebar_model_display_source.py)

3 test methods, 40 assertions total:

1. **Function behavior** — Extracts `sidebarModelText` from app.js source, runs 10 cases via Node.js: null session, missing model, null model, empty, whitespace, "default", mixed-case "DeFaUlT" with whitespace, "gpt-5.4" (trimmed), "claude-sonnet-4-5", long provider/model string.
2. **Ordering** — Verifies the array `[stateTxt, modelTxt, cwdBase, branchTxt]` order via string-index comparison in the source block, and the exact `filter(Boolean).join(" | ")` expression.
3. **Marker preservation** — Verifies effort markers and fast-session markers remain structurally separate from the metadata text span.

### Proof artifacts (sidebar-model-display-19431/)

Docker+browser proof with 5 fixture sessions:

| Session | API model | Expected | Desktop metaText | Mobile metaText |
|---|---|---|---|---|
| model-codex-gpt | `gpt-5.4` | show | `3h ago \| gpt-5.4 \| model-codex-gpt` ✓ | same ✓ |
| model-cc-sonnet | `claude-sonnet-4-5` | show | `3h ago \| claude-sonnet-4-5 \| model-cc-sonnet` ✓ | same ✓ |
| model-pi-long | 80-char provider/model | truncate | visible, scrollWidth > clientWidth ✓ | truncated=true ✓ |
| model-default-omitted | `default` | omit | `3h ago \| model-default-omitted` ✓ | same ✓ |
| model-empty-omitted | `null` (from empty) | omit | `3h ago \| model-empty-omitted` ✓ | same ✓ |

- `bodyOverflow=false` on both viewports (1280×720 desktop, 390×844 mobile).
- Screenshots exist: desktop-sidebar.png (59KB), mobile-sidebar.png (44KB).
- Effort markers preserved: each row shows correct effort marker text and title.

### Validation commands

- Local focused pytest: 40 passed
- Local full pytest: 1817 passed, 134 subtests
- Docker focused tests: 40 passed
- Docker smoke: pre-login 401, post-login 200
- `git diff --check`: passed

### Hygiene

- No secrets in committed artifacts (only `"preferred_auth_method": "apikey"` label, tokens all null).
- No changes to server.py, broker.py, sessiond.py, rollout_log.py, pi_log.py, util.py, or any backend file.
- Clean working tree, no staged files.
- CSS `.metaText` has `overflow:hidden; text-overflow:ellipsis; white-space:nowrap` — confirmed pre-existing.

## Non-blocking observations

1. **No branch-present proof case** — All fixture sessions have `git_branch: null`, so the 4-segment `stateTxt | model | cwdBase | branchTxt` rendering was not browser-verified. The code and source tests cover this structurally (trivial array join mechanics). Not a risk.

2. **Source test fragility** — Tests extract JS function bodies by string landmarks (`function sidebarModelText(s)`, `function sessionIdFromHash`). Reformatting or moving those functions would break extraction. This is an established project pattern, not a new risk introduced by this slice.

3. **Desktop `truncated` key absent** — The mobile JSON includes `truncated: true/false` per row but the desktop JSON omits it. The overflow is still prevented by CSS; this is a proof instrumentation difference across viewport-specific scripts, not a rendering gap.

## Blockers

None.
