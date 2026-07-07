# Context Chip Accessibility — Clean-Room Adversarial Review

**Review date:** 2026-07-07
**Commits reviewed:** 97876db (functional), 650d165 (proof artifacts) on `recovery/product-gaps`
**Verdict:** ACCEPT with nonblocking findings

---

## 1. Evidence Inspected

| Source | What was checked |
|---|---|
| `git diff b68ef68..97876db` (app.js, app.css, test) | Full diff of the 3 changed files |
| `codoxear/static/app.js` (broader context) | `el()` helper via `app_dom.js`, `setContext()`, `onclick` handler, surrounding DOM construction, `topMeta`/`titleRow` structure |
| `codoxear/static/app.css` | `.status-chip`, `button.status-chip`, `button` global, `button:disabled`, `button.badge`, `.topMeta`, mobile breakpoint, global `outline` resets |
| `codoxear/static/app_dom.js` | Confirm `createElement` uses `setAttribute(k, v)` for `type`/`aria-label` (not `n.type = v`) |
| `codoxear/cc_log.py` | Verify `cc_token_update` math is untouched; `PI_DEFAULT_RESERVED_TOKENS = 16384` explains the 16384 reserved in proof |
| `codoxear/rollout_tokens.py` | Confirm token extraction chain unchanged |
| `tests/test_context_chip_accessibility_source.py` | All 4 tests pass; coverage of DOM construction, visibility boundaries, activation wiring, CSS density |
| `browser-proof-summary.txt` + 6 raw browser JSON files | TagName=BUTTON, type/typeProperty=button, aria-label, text, title, display, disabled, focusAfterProgrammatic, toast after click/Enter/Space, overflow |
| `browser_driver.js` | Self-contained DOM state evaluator; no side effects |
| `fake_cc_context_sessions.py` | Synthetic fixture; token math trace verified |
| `broker-call-summary.json` | `send_count=0, key_count=0` for both sessions |
| `docker-test.txt`, `docker-smoke.txt`, `focused-pytest.txt`, `full-pytest.txt`, `git-diff-check.txt` | All pass |

---

## 2. Frame Challenge

### 2.1 Does this slice solve the right product problem?

**Yes.** The `#ctxChip` was a `<span class="status-chip">` with `onclick`. Span-onclick controls are invisible to keyboard-only users (not in tab order, no keyboard activation) and screen readers report them as static text. Changing to `<button type="button" class="status-chip">` with `aria-label="Context usage details"` fixes both:

- Native Tab-focusability (button in sequential focus order)
- Native Enter/Space activation (button keybinding)
- Screen reader announcement: "Context usage details, button" conveys both identity and affordance

This is the correct intervention — a native control is always superior to ARIA role/tabindex/keydown emulation, and the CSS resets preserve the existing visual density.

### 2.2 Does the approach confuse status display with actionable control?

**No.** The chip has always been a hybrid: it displays context pressure (text "Ctx 18%", title with token count) AND clicking it shows a details toast. The change preserves this hybrid behavior exactly. The accessible name "Context usage details" names the action affordance; the visible text continues to name the status. Both coexist on the same element, which is legitimate for a control that both displays state and provides an interaction.

### 2.3 Is preserving the toast action (instead of a "Details" path) a problem?

**No.** The PROMPT.md says "Enter/Space activation opens the same Details path as pointer activation." The existing `onclick` handler dispatches to `setToast(...)` with token details. There is no separate "Details dialog" anywhere in the codebase — the toast *is* the Details display. All three activation methods (click, Enter, Space) trigger the identical toast. This satisfies the requirement.

### 2.4 Does `disabled=true` + `display:none` really prevent focus?

**Yes.** Both mechanisms independently prevent keyboard focus:

- `display:none` removes the element from the rendering tree and accessibility tree — Tab cannot reach it
- `disabled=true` on a button prevents programmatic and keyboard focus

Browser proof confirms: when `token=None`, `focusAfterProgrammatic=false`. When visible, `focusAfterProgrammatic=true`. The dual mechanism is robust (redundant but harmless).

### 2.5 Does the CSS button reset preserve layout and cause no regressions?

**Yes.** Analyzed the complete cascade:

| Layer | Selector | Specificity | Effect on chip |
|---|---|---|---|
| UA | `<button>` | varies | Default button chrome, padding, font — overridden below |
| Author | `button` (line 696) | 0,0,1 | border/background/color/border-radius/padding — all overridden by `.status-chip` (0,1,0 > 0,0,1) |
| Author | `button:disabled` (line 709) | 0,1,1 | `opacity: 0.6` — harmless because disabled always co-occurs with `display:none` |
| Author | `.status-chip` (line 776) | 0,1,0 | Sets display, align, padding, border-radius, border, background, color, font-size, line-height, white-space |
| Author | `button.status-chip` (line 788) | 0,1,1 | Resets appearance, margin, font-family, font-weight, cursor — overrides UA button chrome |
| Author (mobile) | `.status-chip` (line 2752) | 0,1,0 | Font-size 11px, padding 3px 8px — same specificity as base, applies last, wins |

No conflict: `.status-chip` author styles beat user-agent and global `button` styles. The `button.status-chip` reset fills the remaining gaps (appearance, margin, font). Browser proof confirms no horizontal overflow at 1280px (desktop) and 390px (mobile), chip dimensions stable (24px height desktop, 21px mobile).

**No `outline: none` affects this button.** Checked all 4 sites: `.chatSearchInput`, textarea, `.filePlainEditTextarea`, `.queueText:focus`. None target `button.status-chip`. Browser default focus ring (`:focus-visible`) applies naturally.

### 2.6 Do the source tests actually verify the contract?

**Yes, within their scope.** The 4 tests verify:

1. DOM construction: button element, attributes, aria-label, initial hidden+disabled state
2. Visibility boundaries: all three hidden paths (null tok, invalid tok) set display=none/disabled=true; visible path sets display=inline-flex/disabled=false; text/title preserved
3. Activation wiring: onclick still fires toast; no keydown/role/tabIndex emulation (native button behavior suffices)
4. CSS density: `.status-chip` properties intact; `button.status-chip` resets present; mobile breakpoint preserved

The tests are source-level (string matching in JS/CSS), not DOM-level. This is appropriate — browser-level behavior is verified by the Docker/browser proof. The combination provides full coverage.

### 2.7 Any backend or token semantics changed?

**No.** Diff scope: `app.js` (3 changed lines + 2 added lines), `app.css` (+8 lines), new test file. Zero changes to:
- `codoxear/pi_log.py`, `codoxear/cc_log.py`, `codoxear/rollout_tokens.py` — token extraction
- `codoxear/pi_context.py` — context math
- Any session, broker, adapter, or API module
- `setContext()` function body — only wrapper statements for `disabled` added, core token math untouched

### 2.8 Any secrets or inappropriate content in proof artifacts?

**No.** Checked all 24 committed proof artifacts:
- `fake_cc_context_sessions.py`: synthetic session IDs (`cc-context-chip-visible`, `cc-context-chip-no-token`), synthetic thread UUIDs, Docker container paths (`/home/tester/...`). No real credentials, tokens, API keys, or host paths.
- `broker-call-summary.json`: only call counts (send_count=0, key_count=0, state_count tallies). No content.
- Browser JSON, API snapshots: synthetic fixture data only.
- Docker outputs: standard test/smoke output, no secrets.

---

## 3. Blockers

**None.** All acceptance criteria are satisfied with direct evidence.

---

## 4. Nonblocking Findings

### NB-1: Accessible name could be more action-oriented
`aria-label="Context usage details"` is a noun phrase. On a button, screen readers announce "Context usage details, button." While adequate, an action-oriented label like `"Show context usage details"` or `"View context usage"` would better communicate that the button *does* something. The current label names the *topic* of the action rather than the action itself. **Severity: low.** No user is blocked — the button role already conveys interactivity.

### NB-2: `focus-visible` styling not explicitly verified
The CSS reset (`appearance: none`) does not suppress focus rings (no `outline: none` applies). Browser default `:focus-visible` rings should render. However, the browser proof JSON does not capture `outline` style or screenshot the focused state. The proof shows `focusBefore: true` before Enter/Space activation, confirming keyboard focus landed, but visual focus indicator was not explicitly checked. **Severity: very low.** Given no conflicting `outline` styles exist, the browser default should work.

### NB-3: Redundant `disabled` + `display:none`
Three code paths set both `ctxChip.style.display = "none"` and `ctxChip.disabled = true` simultaneously. `display:none` alone removes the element from tab order and accessibility tree; `disabled` alone prevents focus on a visible button. Either mechanism independently guarantees correct hidden behavior. The redundancy is harmless but unnecessary — if a future change ever set `disabled=true` without `display=none`, the `button:disabled { opacity: 0.6 }` rule would create visible dimming; and if `display=none` were removed but `disabled` kept, the chip would be invisible yet focusable (though browser behavior for disabled+hidden combos is brittle). **Severity: low.** Current code pairs them correctly in all paths.

### NB-4: Source tests are string-matching, not DOM-verifying
The 4 source tests grep for string patterns in app.js and app.css. They cannot catch issues like CSS specificity conflicts, browser quirks with `setAttribute("type", "button")`, or runtime DOM state errors. The Docker/browser proof covers these gaps, so this is an observation, not a defect. **Severity: informational.**

---

## 5. Acceptance Report

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Diff at 97876db changes only DOM construction (span→button), CSS (button.status-chip reset), and new source test. No backend, token, session, or upload changes. No scope widening."
    },
    {
      "id": "criterion-2",
      "status": "satisfied",
      "evidence": "Proof artifacts at .memory/tasks/2026-07-07-context-chip-accessibility/browser-artifacts/context-chip-accessible-19394/ include local pytest (43 focused + 1805 full passed), Docker test (1804 passed/1 skipped), Docker smoke (401→200), browser proof (tagName BUTTON, aria-label, text/title, click/Enter/Space toast, no-token hidden+disabled+unfocusable, desktop/mobile no-overflow, send_count=0/key_count=0)."
    }
  ],
  "changedFiles": [
    "codoxear/static/app.js",
    "codoxear/static/app.css",
    "tests/test_context_chip_accessibility_source.py"
  ],
  "testsAddedOrUpdated": [
    "tests/test_context_chip_accessibility_source.py"
  ],
  "commandsRun": [
    {
      "command": "python3 -m pytest -q tests/test_context_chip_accessibility_source.py -v",
      "result": "passed",
      "summary": "4 passed in 0.42s — all 4 source contract tests pass"
    },
    {
      "command": "git diff b68ef68..97876db (manual inspection)",
      "result": "passed",
      "summary": "No backend/token/session/upload changes; only app.js (ctxChip span→button + disabled bindings), app.css (button.status-chip reset), new test file"
    }
  ],
  "validationOutput": [
    "Source test: 4/4 passed",
    "Browser proof: tagName=BUTTON, type=button/button, disabled sync with display, aria-label stable",
    "Browser proof: toast after click/Enter/Space all = 'ctx 150000/200000 (18% left)'",
    "Browser proof: no-token state = display none, disabled true, focusAfterProgrammatic false",
    "Browser proof: send_count=0, key_count=0 in both sessions",
    "Browser proof: desktop overflow=false (1280px), mobile overflow=false (390px)",
    "CSS cascade analysis: no conflicts, .status-chip rules win over global button, button.status-chip resets UA chrome",
    "No secrets in any committed proof artifact"
  ],
  "residualRisks": [
    "Focus ring visibility not explicitly verified (browser default :focus-visible expected to work; no conflicting outline:none styles exist)",
    "Docker/browser proof exercises isolated server only — no live-host regression testing (by design per constraints)"
  ],
  "noStagedFiles": true,
  "diffSummary": "3 files changed, +115/-1 lines. app.js: ctxChip construction from span→button with type=button/aria-label, disabled bindings in setContext paths. app.css: button.status-chip reset (appearance, margin, font, cursor). test: 4 source contract tests for DOM/visibility/activation/CSS.",
  "reviewFindings": [
    "no blockers"
  ],
  "manualNotes": "Commit 650d165 (proof artifacts) contains 24 files with browser JSON, Docker outputs, and fake session fixtures — all synthetic, no secrets. The accessible name 'Context usage details' could be improved to 'Show context usage details' but this is nonblocking. The redundant disabled+display:none pairing is robust and harmless. Recommend accepting and proceeding to next slice."
}
```
