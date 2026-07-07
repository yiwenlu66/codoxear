# Clean-room adversarial review: reasoning-effort sidebar markers

**Reviewer:** subagent (clean-room, read-only)
**Branch:** `recovery/product-gaps` in `/home/yiwen/codex-web-product-recovery`
**Commits:** 674f5f8, f59086a, 937e09e
**Date:** 2026-07-07

## VERDICT: ACCEPT

No blockers found. The implementation is minimal, correct, and scoped exactly to
the display-truth gap. Backend semantics are untouched.

---

## Evidence inspected

### Implementation diff (f59086a)

| File | Change |
|---|---|
| `codoxear/static/app.js` | Replaced inline ternary chain with a frozen lookup object `REASONING_EFFORT_MARKERS` and a `reasoningEffortMarker()` function. Added entries: `max→M+`, `minimal→m`, `off→–`. Existing entries `xhigh→X`, `high→H`, `medium→M`, `low→L` preserved verbatim. |
| `codoxear/static/app.css` | Added three CSS rules: `.effort-max` (#a9536c), `.effort-minimal` (#6f8374), `.effort-off` (#777777). Existing four rules unchanged. |
| `tests/test_reasoning_effort_source.py` | Added `test_sidebar_reasoning_effort_markers_cover_supported_values` (regex-extracts the full mapping from source, asserts all 7 entries, verifies fallback expression, class template, title template) and `test_sidebar_reasoning_effort_css_covers_new_values` (asserts CSS rules exist for max/minimal/off). |

**Backend files changed:** None. `server.py`, `launch_config.py`, `agent_backend.py`, `app_launch.js`, `app_new_session.js` are all untouched.

### Mapping correctness

The lookup `REASONING_EFFORT_MARKERS[effortTxt] || ""` is safe because:
- All defined values are non-empty strings (no falsy collision).
- Unknown/undefined keys return `undefined`, which `||` coerces to `""`.
- The guard `if (effortMark)` prevents creating a DOM span for unknown values.
- `effortTxt` is normalized via `String(s.reasoning_effort || "").trim().toLowerCase()`, handling null/undefined/whitespace/casing.

### Browser proof (937e09e)

The DOM probe (`dom-probe.json`) shows 7 sidebar rows with exact expected text, classes, and titles:

| Session | Text | Class | Title |
|---|---|---|---|
| effort-codex-low | L | effortMark effort-low | reasoning effort low |
| effort-codex-medium | M | effortMark effort-medium | reasoning effort medium |
| effort-codex-high | H | effortMark effort-high | reasoning effort high |
| effort-codex-xhigh | X | effortMark effort-xhigh | reasoning effort xhigh |
| effort-pi-off | – | effortMark effort-off | reasoning effort off |
| effort-pi-minimal | m | effortMark effort-minimal | reasoning effort minimal |
| effort-cc-max | M+ | effortMark effort-max | reasoning effort max |

Desktop (1280×720) and mobile (390×844) marker summaries both show `bodyOverflow: false` and all `ok: true`. Screenshots visually confirm distinct colored markers in both viewports.

### Test results

| Test run | Result |
|---|---|
| Local focused pytest | 42 passed |
| Local full pytest | 1814 passed, 134 subtests |
| Docker focused pytest | 42 passed |
| Docker smoke | pre-login 401, post-login 200, app dir correct |

### Git state

- No staged files (`git diff --cached` empty).
- `git status --porcelain` clean.
- Three commits: task init, functional change, proof recording. Properly separated.

### Security

No credentials, cookies, bearer tokens, or API keys in committed artifacts. The literal `"apikey"` in API defaults is a static configuration label, not a secret value. `"token": null` entries are broker protocol field names.

---

## Non-blocking concerns

1. **M+ marker width asymmetry.** The `M+` marker renders at 15.5px width (vs 12px for single-char markers). The CSS `flex: 0 0 auto` allows this to grow naturally and no overflow was observed, but if future markers were wider this pattern could compress `metaText`. Non-blocking because it works today and the marker set is finite.

2. **`||` vs `hasOwnProperty` guard.** The fallback `|| ""` would incorrectly suppress a marker whose value was `""`, `0`, `false`, or `null`. All current values are non-empty strings, so this is safe. A `hasOwnProperty` check or nullish coalescing (`?? ""`) would be marginally more defensive. Non-blocking.

3. **CSS test scoping.** `test_sidebar_reasoning_effort_css_covers_new_values` only asserts the 3 new CSS rules, not the 4 existing ones. The existing rules were untouched and are covered implicitly by the visual proof. Non-blocking.

4. **Source-regex test fragility.** The test uses a regex to extract the mapping from app.js source text. If the whitespace or formatting changes, the regex could break. This is the standard approach for Python-testing JS source in this codebase and is accepted practice. Non-blocking.

---

## Acceptance criteria check

| Criterion | Status |
|---|---|
| Existing `xhigh→X`, `high→H`, `medium→M`, `low→L` unchanged | ✅ Verified in diff and DOM probe |
| `max→M+`, `minimal→m`, `off→–` render correctly | ✅ Verified in diff, DOM probe, screenshots |
| Unknown efforts render no marker | ✅ `REASONING_EFFORT_MARKERS[unknown]` returns `undefined` → `""` → `if (effortMark)` skips |
| Class remains `effortMark effort-${effortTxt}` | ✅ Verified in source and DOM probe |
| Title remains `reasoning effort <value>` | ✅ Verified in source and DOM probe |
| No backend launch/default/session metadata changes | ✅ Only app.js, app.css, test file changed |
| Browser proof exercises actual sidebar rows | ✅ 7 fake sessions, desktop + mobile screenshots |
