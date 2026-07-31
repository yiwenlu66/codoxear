# Clean-room adversarial review: coarse-pointer code-copy touch targets

**Verdict: ACCEPT**

Branch `recovery/product-gaps`, functional commit `4fa373d`, proof commit `052d788`.

## 1. CSS scoping — coarse-pointer rule cannot enlarge desktop controls

The new rule lives inside `@media (hover: none) and (pointer: coarse)` (app.css line 3011). Both conditions must hold for the rule to fire. A fine-pointer desktop has `pointer: fine` and `hover: hover`; neither condition is met. The rule cannot activate on desktop.

Specificity is `.code-copy-btn` — same as the base rule at line 1245. The coarse-pointer block appears later in the file (line 3023 vs 1245), so cascade order lets it win when the media matches. The four properties (`width`, `height`, `min-width`, `min-height`) are set to `44px`, overriding the base `30px`.

The accompanying `.md pre { padding-right: 58px }` matches the phone rule's padding, preventing the 44px button from overlapping code. The base right padding is `46px`; a 44px button positioned `right: 6px` needs 50px clearance; 58px provides 8px of breathing room.

No other `.code-copy-btn` rules exist that could interfere — checked all seven occurrences in app.css. No JS or HTML changes in the functional commit.

**Pass.**

## 2. Test discrimination

`test_code_copy_css_coarse_pointer_tablet_touch_target` uses `css_media_block()` to extract the brace-delimited content of `@media (hover: none) and (pointer: coarse)`. It then:
- Asserts `.code-copy-btn` exists inside that extracted block
- Extracts the `.code-copy-btn { ... }` sub-rule and checks all four 44px properties
- Asserts `.md pre` exists inside the same block and checks `padding-right: 58px`

Before commit `4fa373d`, the coarse-pointer media block did not contain `.code-copy-btn`. The `assertIn(".code-copy-btn", coarse_block)` at line 196 would fail. This is a structural discriminator, not brittle full-file text matching — the helper parses nested braces and isolates the media block content.

**Pass.**

## 3. Browser proof fidelity — media emulation, not viewport-only

The CDP script (`run_cdp_coarse_code_copy_browser.py`) calls three CDP methods for each scenario:
- `Emulation.setDeviceMetricsOverride` — viewport size
- `Emulation.setTouchEmulationEnabled` — touch event generation
- `Emulation.setEmulatedMedia` — **explicit `pointer` and `hover` media feature overrides**, plus `any-pointer` and `any-hover`

The browser then reports `matchMedia('(pointer: coarse)').matches` and `matchMedia('(hover: none)').matches` in the page-level metrics. Results from `browser-results.json`:

| Scenario | pointer:coarse | hover:none | Button rect | Computed size | pre padding-right | Overflow | Clipboard |
|---|---|---|---|---|---|---|---|
| Touch tablet 768×1024 | `true` | `true` | 44×44 | all 44px | 58px | none | exact match |
| Touch phone 390×844 | `true` | `true` | 44×44 | all 44px | 58px | none | exact match |
| Desktop 1280×800 | `false` | — | 30×30 | all 30px | 46px | none | exact match |

The proof is media-query-driven, not viewport-width-driven. The tablet scenario at 768px is above the 520px phone breakpoint; it gets 44×44 solely from the coarse-pointer rule.

**Desktop `pointer:fine` limitation**: headless Chromium reports `matchMedia('(pointer: fine)').matches = false` even with `pointer: fine` emulation. However, `pointer: coarse` is `false`, which is the operative check — the coarse-pointer CSS rule requires `(pointer: coarse)` to fire. The 30×30 measured result confirms the rule did not fire.

Broker mutations: `send=0, keys=0, shutdown=0`. Only `state` polls observed (685 calls). Code-copy is client-side only.

**Pass.**

## 4. Artifact sanitization

- No `Set-Cookie`, `codoxear_auth`, or auth header values in any committed artifact
- Password references use `<sandbox-password>` (Docker sandbox default) or `<sandbox-password>` (script placeholder)
- The Chrome profile directory was removed after the run (`rm -rf raw/chrome-profile`)
- `smoke-isolation.txt` lists Docker-container paths (`/home/tester/`), not host paths
- `fake-sidecar.json` carries `fake_notice: "FAKE_COARSE_CODE_COPY_PROOF_DOCKER_ONLY"`
- `hmac_secret` and `webpush_vapid_private.pem` are listed by filename only (directory listing); contents not captured
- `broker-calls.jsonl` contains only `cmd:state` responses with no credentials

**Pass.**

## 5. Category errors

**Architecture contract scope**: ARCHITECTURE.md says "Mobile code-copy controls are part of the companion-device contract: at least 44x44 CSS px." The prior phone rule covered ≤520px. The new coarse-pointer rule covers all touch devices regardless of viewport width. Tablets are companion devices. Correct scope.

**Button visibility**: The base rule sets `opacity: 0.72`. The phone rule (≤520px) adds `opacity: 1`. The new coarse-pointer rule does not set opacity. On a coarse-pointer tablet, the button is 44×44 but at 0.72 opacity until `:hover` or `:focus-visible`. The button is never `display:none` or `visibility:hidden`. At 0.72 opacity the button is clearly visible. The stated contract is "at least 44x44 CSS px" — opacity is not part of the contract. This is a minor polish observation, not a violation.

**Padding impact on code layout**: 58px right padding (up from 46px base) reduces code area width by 12px on coarse-pointer devices. The `<pre>` uses `white-space: pre-wrap; overflow-wrap: anywhere; max-width: 100%` — code wraps rather than overflows. No layout breakage.

**msg-copy exclusion**: Task PROMPT.md explicitly excludes `.msg-copy-btn`. The architecture contract specifies "code-copy controls" specifically. Message-copy is a separate control with separate behavior. Correct scope boundary.

**Desktop proof sufficiency**: `pointer:coarse=false` proves the coarse-pointer rule does not fire. `pointer:fine=false` is a headless Chrome limitation, not a proof gap. The mechanism is the CSS conjunction `(hover: none) and (pointer: coarse)` — with `pointer:coarse=false`, the rule cannot match regardless of other conditions. 30×30 measured rect confirms.

**No JS behavioral change**: The functional commit modifies only `app.css` and the test file. No `app_markdown.js`, `app_code_copy.js`, `app.js`, or `index.html` changes. Copy behavior is purely CSS-sized.

**Pass.**

## Residual boundaries

1. **Opacity 0.72 on coarse-pointer tablets (>520px)**: The phone rule sets `opacity: 1` but the coarse-pointer rule does not. On touch tablets, the button is visible but slightly faded until hover/focus. A future polish commit could add `opacity: 1` to the coarse-pointer `.code-copy-btn` rule. This is not a contract violation.

2. **Desktop `pointer:fine` not positively proven**: Headless Chromium reports `fine=false` even with `pointer:fine` emulation. The proof relies on `coarse=false` which is sufficient for proving the coarse-pointer rule does not fire. A real desktop browser would show `fine=true`.

3. **Second code block copy isolation not browser-tested**: Only the first block's copy was verified in the browser. The session contains two blocks and `buttonCount:2` is confirmed. The per-block copy isolation is proven by the source test (`test_code_copy_runtime_copies_only_nearest_code_text`) which exercises `closest('pre')` scoping. Not a gap at the CSS level.

## Files reviewed

- `codoxear/static/app.css` — 9-line CSS addition in existing coarse-pointer media block
- `tests/test_code_block_copy_source.py` — 33-line addition: `css_media_block()` helper + `test_code_copy_css_coarse_pointer_tablet_touch_target`
- `.memory/tasks/2026-07-07-coarse-pointer-code-copy-targets/browser-artifacts/coarse-code-copy-19532/raw/browser-results.json` — 3-scenario proof
- `.memory/tasks/2026-07-07-coarse-pointer-code-copy-targets/browser-artifacts/coarse-code-copy-19532/VERIFICATION-REPORT.md`
- `.memory/tasks/2026-07-07-coarse-pointer-code-copy-targets/browser-artifacts/coarse-code-copy-19532/raw/run_cdp_coarse_code_copy_browser.py` — CDP proof script
- `.memory/tasks/2026-07-07-coarse-pointer-code-copy-targets/browser-artifacts/coarse-code-copy-19532/raw/fake_coarse_code_copy_session.py` — deterministic session
- `.memory/tasks/2026-07-07-coarse-pointer-code-copy-targets/browser-artifacts/coarse-code-copy-19532/raw/broker-call-summary.json` — zero mutations
- `.memory/tasks/2026-07-07-coarse-pointer-code-copy-targets/browser-artifacts/coarse-code-copy-19532/COMMANDS-RUN.md` — execution log
