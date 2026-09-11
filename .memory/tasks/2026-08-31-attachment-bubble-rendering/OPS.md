# OPS log — attachment bubble rendering

## 2026-08-31T07:00Z — investigation
- Located inject point: `attachment_inject_text` (file_upload.py:159), used by
  session_send.py:85-89. Only other consumer of the grammar:
  session_listing.py `_redact_generated_attachment_prefix_paths` (sidebar
  privacy redaction, leading lines only).
- Real Pi log evidence: user event text contains both Attachment lines verbatim
  (~/.pi/agent/sessions/--home-yiwen-codoxear--/2026-08-31T06-43-50-*.jsonl).
- `_single_chat_event` projection returns full text including prefix (ran
  locally against the log).
- makeRow (app_message_rows.js): copy button onclick closed over ev.text;
  `void upgradeCandidateFileRefs(md)` at construction only.
- consumePendingUserIfMatches (app_transcript.js:46-67): swapped .md innerHTML
  + ts only. No copy rebind, no ref upgrade. rebuildDecorations = day-sep/group
  classes only.
- Confirmed pending echo text = raw composer text (app_message_flow.js:603-604).
- Vision subagent (terra): screenshots are NOT the bubble; they are crops of
  (1) composer queue+send cluster, (2) new-session backend tabs. No path text
  visible in either.

## 2026-08-31T07:20Z — implementation
- c0747b22: row.copyText authority + commit-time upgradeCandidateFileRefs;
  wiring select list updated; check_wiring.py passes.
- ff40bf64: attachmentDisplayMarkdown + chatMarkdownHtmlCached integration.
- tests/test_pending_user_commit.py (new, 2 tests),
  tests/test_app_markdown_extended.py (+3 tests: transform rules, chat render,
  passthrough). Full suite: 1751 passed, 112 subtests.
- Test venv: /tmp/codoxear-test-venv (pip install -e .[test]).

## 2026-08-31T07:30Z — verification
- Dispatched executor subagent 1444b304 for Docker behavioral verification
  (criteria a-d: settled bubble img loads via blob endpoint; non-image path is
  inlineFileLink; copy source contains both Attachment lines; same after
  reload). Port 19655, never 8743.

## 2026-08-31T08:05Z — first Docker verification FAILED (stale bundle)
- Subagent 1444b304 ran the full scenario against ff40bf64 image. All three
  fixes showed zero behavioral effect in the live-settled bubble; reload made
  paths clickable (old behavior). Root cause: image built from git archive
  serves the committed dist/app.bundle.js, which was stale — docker_verify.sh
  does not rebuild it (only deploy.sh does). Blob endpoints probed 200 OK, so
  server path resolution is confirmed fine.
- Rebuilt bundle with esbuild, committed 947146f4, resumed subagent for
  re-verification at new HEAD + full-size design screenshots (composer cluster,
  new-session backend tabs, desktop + narrow).
- Recorded the tracked-bundle freshness gate in .memory/project/VALIDATION.md.

## 2026-08-31T08:40Z — re-verification PASS at 947146f4
- (a) settled bubble contains blob-backed <img>, complete=true, naturalWidth>0.
- (b) non-image path upgraded to a.inlineFileLink after pending commit; no
  inert candidate spans remain.
- (c) real copy-button click captured "Attachment 1: ...png\nAttachment 2:
  ...txt\nwhat do you see?" — committed text with both prefixes.
- (d) reload preserves image + link projections.
- Evidence: /tmp/codoxear-attachment-reverify-results.IEduIV/
- Design screenshots captured (desktop 1280x900, narrow 480x900): composer
  empty/typed, new-session modal. Same dir, design-*.png.
- Environment note: UI-created Pi session cannot materialize a log without
  provider credentials in Docker; attachment test used harness-bootstrapped
  web-owned Pi broker with a real log. Known harness limitation, not app bug.

## 2026-08-31T09:10Z — design audit + pass
- Vision audit of 6 full-size screenshots (desktop 1280, narrow 480).
  Findings: composer attach/queue borderless (DELIBERATE: 87d4c108
  "Borderless utility buttons" — do not touch); queue glyph ambiguous vs
  topbar hamburger (medium); backend-tab active state weak per audit BUT
  tests/test_issue_archive_ui_contract.py pins "single 2px ink outline, not a
  doubled outline-plus-underline signal" — a prior user decision. Reverted my
  1px-border+selection-bar change and deleted its test; the pinned decision
  stands. Claude logo optically small (cc.svg mark fills ~42% of viewBox vs
  ~70% for codex/pi) — fixed by viewBox crop d1a60aed.
- Committed: b95b4c99 queue layers glyph; d1a60aed cc.svg crop; 3ddc8a1f
  bundle rebuild. Suite 1751 green, wiring guard green.
- Final visual verification running (run 13c45322) at 3ddc8a1f.

## 2026-09-03T04:45Z — final visual verification PASS at 3ddc8a1f
- Queue layers glyph reads distinctly from topbar hamburger; composer band
  still ~55px, send 32px square.
- Claude mark now fills tile comparably to Codex/Pi (old-vs-new 10x crop).
- No regressions: 2px active outline intact, no layout shift, caption
  unclipped, modal geometry unchanged.
- Skipped: sidebar Claude logo at 12px (no CC session in container). cc.svg is
  shared with the sidebar; risk accepted (strictly better fill at small size).
- Evidence: /tmp/codoxear-design-final-results.D9ZlEa/
- Task complete. Commits c0747b22..3ddc8a1f (6 commits). Not deployed.

## 2026-09-03T05:00Z — deployed
- scripts/deploy.sh 3ddc8a1f: snapshot worktree updated, bundle rebuilt from
  reviewed source, pipx reinstall, service restart, smoke test passed (initial
  load + reload rendered a session card). Health boundary 200/401 held.
