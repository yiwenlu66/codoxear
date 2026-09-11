# OPS — iPhone settings bugs (append-only)

## 2026-09-06T03:20Z #1 Screenshot measurement (IMG_3089, 1179×2556, DPR 3)
- Row-scan: dialog spans x=58..1122 device px → 354.7 CSS px; textarea 94..1086 → 330.7 CSS px
  (dialog padding 12px each side ✓).
- Col-scan x=500: Clay swatch card top border y=429..432 (3 device px = 1 CSS px), bottom
  702..705 → card height 276 device px = 92 CSS px.
- Docker capture /tmp/codoxear-design-review.BY7gEJ/paper-light-settings-mobile.png (390 CSS px,
  DPR 1): Clay card y=83..175 = 92 px. Identical → no iOS zoom.

## 2026-09-06T03:22Z #2 Defect reproduction in Docker capture
- Crop of the same Docker capture (Paper selected): left outer ring merged with the inner border
  (no gap), right/top/bottom show the 1px gap. Same signature as the iPhone crops.

## 2026-09-06T03:25Z #3 CSS cause located
- app.css `.themeSwatch.active { outline: 2px solid var(--focus-ring); outline-offset: 1px }`
- app.css `.formBody { overflow-y: auto; overflow-x: hidden }` — no horizontal padding.
- `.choiceChips` already documents the same clip class and uses `padding: 2px` to absorb it.

## 2026-09-06T03:26Z #4 Text-entry audit (grep el("input"|"textarea"|"select"))
- Inputs: type=password (#pw, voiceApiKeyInput), text (cwd/name/model/worktree/filePicker/editName/voiceBaseUrl),
  search (chatSearchInput), number (unattended cooldown/remaining), date/time (edit snooze),
  checkbox/file/range (non-text). Textareas: #msg, queueText, settingsCustomCss, filePasteInput,
  unattendedRequest, unattendedPromptInput. No `<select>`. No untyped/email/url/tel inputs.
- Anti-zoom rule (app.css @media (max-width:880px),(pointer:coarse)) already covers every present
  text entry; broadening is hardening.

## 2026-09-06T03:55Z #5 Commits
- dcf3f3da bug 1 CSS (.themeSwatches padding 3px; anti-zoom selectors) + tests/test_settings_mobile_css.py
- 71bea229 bug 2 flatten settings (voice section inline; voice dialog retired) + tests + bundle
- cfbd9a75 bug 3 boot script theme-color + tests/test_theme_boot_script.py + AGENTS.md
- ab30e1fd docker_theme_verify.sh mobile probes
- Concurrent foreign edit observed (themes/clay.css) → committed by its owner as 897bc50d beneath mine.

## 2026-09-06T04:10Z #6 scripts/docker_theme_verify.sh HEAD (ab30e1fd) → PASS 25/25
- artifacts /tmp/codoxear-theme-verify.1Qz4rN
- 393x852: dialog 19..374 (w355) inside 393; body clip 31..362; active swatch 34..137, ring outset 3 →
  ring left edge 31 == body.left (inside); scrollWidth 331 == clientWidth 331; all four text entries 16px;
  voiceInline true; settingsVoiceBtn/voiceSettingsViewer absent. 390x844 analogous.
- Crop of 18-mobile-settings-393x852.png: ring gap visible on all four sides of the Paper card.
- Note: Chrome renders the dialog 355 CSS px wide at 393 — same as the iPhone JPEG measurement (354.7),
  a further confirmation that the phone was not zoomed.

## 2026-09-06T04:40Z #7 design_review.sh HEAD (ab30e1fd) → PASS 120 captures, 0 mismatches
- artifacts /tmp/codoxear-design-review.ylNNOa; settings captures show Appearance + inline
  Voice & notifications on desktop and 390px mobile for all six variants; ring gap visible on
  the selected card's outer edge (slate-dark mobile: Slate card right ring at 357 < dialog 374).

## 2026-09-06T04:55Z #8 docker_theme_verify.sh HEAD (6c898185, includes a concurrent foreign commit) → PASS 26/26
- artifacts /tmp/codoxear-theme-verify.Hq1sQd; voice_save_persists_and_closes_settings true:
  Save closed the dialog (status ""), reopen seeded https://voice.example/v1.
- Committed as $(git rev-parse --short HEAD).
