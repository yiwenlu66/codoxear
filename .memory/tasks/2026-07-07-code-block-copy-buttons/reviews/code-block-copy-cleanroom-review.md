# Clean-Room Adversarial Review: Code Block Copy Buttons

**Verdict: ACCEPT**

Branch: `recovery/product-gaps`
Commits: f7c748b → 1702f63 → 75585b9
Reviewer scope: read-only inspection, test execution, no edits

---

## Invariant Verification

### 1. Per-block copy buttons copy exactly that block's code text

**Satisfied.** `renderCodeBlock()` in `app_markdown.js` injects a `<button class="code-copy-btn">` inside each `<pre>` before the `<code>` element. The copy runtime (`app_code_copy.js`) resolves text via `closestElement(button, "pre")` → `pre.querySelector("code")` → `code.textContent`. This grabs only the `<code>` element's text within the enclosing `<pre>`.

Browser proof: `firstCopyExact: true` (copied `printf 'alpha <tag> & value'`), `secondCopyExact: true` (copied `{"beta": 2, "note": "second block"}`).

### 2. No surrounding prose or whole-message text in block copy

**Satisfied.** The `<code>` element's `textContent` is structurally isolated from prose in `<p>`, `<li>`, `<h*>` siblings.

Browser proof: `blockCopiesExcludeProse: true` — regex test confirmed neither block copy contained "prose between", "Here are two", or "End of answer".

### 3. Message-level raw markdown copy remains intact

**Satisfied.** The message-level copy button (`msg-copy-btn`) has its own direct `onclick` handler on the button element, completely separate from `chatInner`'s delegated click handler. The code copy handler in `chatInner` short-circuits on `.code-copy-btn` match only; non-matching clicks fall through to file reference handling. These are disjoint paths.

Browser proof: `messageCopyIncludesProseAndBothBlocks: true` — `copied[2]` contains full raw markdown with both fenced blocks and all prose.

### 4. Mobile target ≥44×44 without horizontal overflow

**Satisfied.** CSS `@media (max-width: 520px)` sets `.code-copy-btn` to `width: 44px; height: 44px; min-width: 44px; min-height: 44px; opacity: 1`. Pre gets `padding-right: 58px` (44 + 6 offset + 8 clearance). Button is `position: absolute` inside `position: relative` pre with `overflow: auto; box-sizing: border-box; max-width: 100%` — cannot cause page overflow.

Browser proof: mobile `rectsBefore` → `{width: 44, height: 44}`, `bodyOverflow: false`.

Desktop: 30×30 px buttons, `padding-right: 46px`, `bodyOverflow: false`.

### 5. No protected/live runtime dirs touched

**Satisfied.** All changes confined to `codoxear/static/`, `codoxear/static_routes.py`, `tests/`, and `.memory/tasks/`. No paths referencing `~/.local/share/codoxear` or `/home/yiwen/codex-web` in any changed files. Grep for `/home/yiwen` in artifacts returned zero matches. Working tree is clean, no staged files.

### 6. Proof artifacts are sanitized

**Satisfied.** No passwords, API keys, or real tokens in artifacts. Only `"token": None` appears — a fake broker state field, not a credential. All container paths use `/home/tester`. Docker container was stopped and removed (`docker-ps-after-stop.txt` is 0 bytes). `FAKE_NOTICE` marker present in session sidecar.

---

## Adversarial Analysis

| Check | Result |
|---|---|
| XSS via button injection | No risk. Button HTML is a hardcoded string literal with no user content. Code text is `escapeHtml()`-escaped. |
| Click delegation correctness | `closest(".code-copy-btn")` correctly resolves for `::before` pseudo-elements (they don't fire separate events). |
| Rapid click race condition | `WeakMap` timer cleanup via `clearTimeout` before setting new timer. No double-fire. |
| Unclosed fence edge case | Unclosed fences treated as text by `splitByFences`, producing no `<pre>` and no stray button. |
| Nested list code blocks | Both top-level and list-nested code blocks get buttons via shared `renderCodeBlock`. `closest("pre")` stops at own `<pre>`. |
| Script load order | `app_code_copy.js` loads after `app_clipboard.js`, before `app.js`. `defer` guarantees document-order execution. `app.js` fails-loud if module missing. |
| CSS z-index leakage | `z-index: 1` scoped inside `position: relative` pre. No interference with modal/overlay layers. |
| Scope widening | 6 source files + 3 test files changed. No unrelated modifications, no refactoring, no feature creep. |

---

## Test Results

```
tests/test_code_block_copy_source.py    5 passed
tests/test_markdown_renderer_source.py  8 passed
tests/test_static_assets.py            17 passed
────────────────────────────────────────────────
Total                                  30 passed
```

---

## Non-Blocking Observations

1. **Icon rendering**: The `⧉` (U+29C9) Unicode character used via `::before` may render inconsistently on older Android WebViews. The button remains functional and accessible via `aria-label` regardless. Cosmetic only.

2. **Desktop right padding increase**: Pre padding-right went from 12px to 46px (desktop) / 58px (mobile). Short code lines have more right whitespace. Standard trade-off for button placement.

3. **Button visibility on desktop**: Opacity 0.72 at rest, 1.0 on hover/focus. Discoverability is slightly reduced for non-hovering desktop users, but WCAG focus-visible state is handled.

---

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "6 source files + 3 test files changed, all within code-block copy scope. No unrelated modifications. renderCodeBlock adds button, app_code_copy.js handles click, CSS styles button, app.js delegates. No scope widening."
    },
    {
      "id": "criterion-2",
      "status": "satisfied",
      "evidence": "30 tests passed locally. Browser proof (desktop + mobile) in Docker confirms exact text copy, no prose leakage, message-level copy intact, 44x44 mobile targets, no overflow. Artifacts sanitized. No protected dirs touched."
    }
  ],
  "changedFiles": [
    "codoxear/static/app_code_copy.js",
    "codoxear/static/app_markdown.js",
    "codoxear/static/app.css",
    "codoxear/static/app.js",
    "codoxear/static/index.html",
    "codoxear/static_routes.py",
    "tests/test_code_block_copy_source.py",
    "tests/test_markdown_renderer_source.py",
    "tests/test_static_assets.py"
  ],
  "testsAddedOrUpdated": [
    "tests/test_code_block_copy_source.py (new, 5 tests)",
    "tests/test_markdown_renderer_source.py (4 assertions updated for copy button in pre)",
    "tests/test_static_assets.py (asset registration assertions added)"
  ],
  "commandsRun": [
    {
      "command": "python3 -m pytest tests/test_code_block_copy_source.py -v",
      "result": "passed",
      "summary": "5/5 passed in 0.53s"
    },
    {
      "command": "python3 -m pytest tests/test_markdown_renderer_source.py -v",
      "result": "passed",
      "summary": "8/8 passed in 0.68s"
    },
    {
      "command": "python3 -m pytest tests/test_static_assets.py -v",
      "result": "passed",
      "summary": "17/17 passed in 5.34s"
    },
    {
      "command": "git diff --name-only f7c748b^..75585b9 | grep -E '(.local/share/codoxear)'",
      "result": "passed",
      "summary": "No protected paths in diff"
    },
    {
      "command": "git diff --cached --name-only",
      "result": "passed",
      "summary": "No staged files"
    },
    {
      "command": "grep -rn '/home/yiwen' .memory/tasks/.../browser-artifacts/",
      "result": "passed",
      "summary": "No real home path leaks in artifacts"
    },
    {
      "command": "git status --short",
      "result": "passed",
      "summary": "Clean working tree"
    }
  ],
  "validationOutput": [
    "All 30 tests pass. Browser proof confirms: firstCopyExact=true, secondCopyExact=true, blockCopiesExcludeProse=true, messageCopyIncludesProseAndBothBlocks=true, bodyOverflow=false, mobile buttons 44x44, desktop buttons 30x30. Docker container cleaned up (docker-ps-after-stop.txt empty). No /home/yiwen paths in artifacts."
  ],
  "residualRisks": [
    "Unicode icon ⧉ (U+29C9) may render inconsistently on older Android WebViews — cosmetic only, button remains functional via aria-label"
  ],
  "noStagedFiles": true,
  "diffSummary": "New app_code_copy.js module with copy runtime; one-line renderCodeBlock change to inject button HTML; CSS for button styling with mobile 44x44 responsive override; app.js click delegation with short-circuit; index.html and static_routes.py asset registration; 5 new tests + existing test updates",
  "reviewFindings": [
    "no blockers found"
  ],
  "manualNotes": "Implementation is minimal and mechanically sound. The copy button is structurally isolated inside <pre>, the text extraction traverses only to the nearest <code> sibling, and the message-level copy path is completely orthogonal (direct onclick on msg-copy-btn vs delegated click on chatInner). All acceptance invariants verified with both unit tests and Docker/browser proof."
}
```
