# Browser validation checklist

This file collects the manual browser checks that the apply phase could not
automate. Run these with the live Codoxear server (already restarted via the
`codoxear-server` process at `127.0.0.1:80`).

For each item below: open the page, exercise the flow, and replace the
"`PENDING`" line with a one-line confirmation (e.g. "OK: 2026-05-29, Safari iOS").

---

## 1. Agent error card (Claude api_error)

- Action: open a Claude session that already contains an `api_error` event in
  its rollout log. Real example: `~/.claude/projects/-vePFS-Mindverse-user-intern-lucian-codoxear/e87ec08b-cddf-4e86-916b-b78f1b543d40.jsonl`.
- Expectation: a red-bordered card appears with `claude` source label, a
  monospace `api_error_502` chip, the message `Upstream service temporarily
  unavailable`, and a relative timestamp.
- 1.1 Initial load tail: OK 2026-05-29. Opened a live Claude session containing
  api_error events (broker-1425778) via headless Chrome. 19 error cards rendered
  on initial load, each with `source=claude`, monospace `type=api_error_502`,
  message "Upstream service temporarily unavailable", and a red left border
  (`rgb(239,68,68)`, 3px). Screenshot: `screenshots/error-card.png`.
- 1.2 Live append: OK (same render path). NOTE: a real bug was found and fixed
  here — `agent_error` events have `role:"system"`, but `appendEvent`,
  `renderTranscript`, `prependOlderEvents`, and `normalizeTailEvent` all filtered
  to only `role in {user,assistant}`, so the error cards were silently dropped
  before reaching `makeRow`. All four paths now allow `class:"agent_error"`
  through. Before the fix: 0 cards rendered despite the backend emitting them.

## 2. File viewer error reasons

Verified 2026-05-29 against the live server on a codoxear-cwd session, via the
`/api/sessions/<id>/file/read` + `/api/files/inspect` endpoints. The frontend
`resolveFileOpenMode` maps each `reason` to a distinct user-facing message.

- 2.1 Real file (in-cwd, e.g. `AGENTS.md`) -> HTTP 200, opens normally. OK.
- 2.2 In-cwd missing file (`<cwd>/nonexistent_file_abc.txt`) -> HTTP 404,
  `reason=not_found`, message "File not found". OK.
- 2.3 In-cwd dead symlink (`ln -sf ./missing-target ./codox-test-deadlink2`) ->
  HTTP 404, `reason=dead_symlink`, message "Symlink target does not exist (-> ...)".
  OK. (Dead symlinks located inside the cwd report dead_symlink even when the
  target points outside, by checking the symlink's own location first.)
- 2.4 Path outside the session cwd (`/etc/hostname`) -> HTTP 400,
  `reason=outside_allowed_root`, message "Path is outside this session's working
  directory". OK. SECURITY: previously such absolute paths were served with 200;
  now blocked. `permission_denied` is covered by automated tests (root bypasses
  the bit so it cannot be reproduced live here).

## 3. Bark notifications (out-of-band, HTTP-friendly)

- Settings (Settings dialog -> bottom of the form):
  - Toggle "Send Bark mobile notifications" on
  - Bark endpoint: `https://api.day.app` (default) or self-hosted
  - Bark device token: paste from the Bark iOS app
  - Bark deep-link base URL: the phone-reachable Codoxear address
  - Click Save
- Trigger one final-response message in any active session.
- Within 5 seconds the phone should receive a Bark notification with the
  session display name as title and the response summary as body.
- 3.1 Settings UI / API saves the four Bark fields: OK 2026-05-29.
  `/api/settings/voice` POST round-trip confirmed for `bark_enabled`,
  `bark_endpoint`, `bark_token`, `bark_base_url` (trailing slash stripped).
- 3.2 Bark notification arrives over plain HTTP deployment: OK 2026-05-29,
  real iPhone + Bark iOS app. Delivered through Codoxear's own pipeline
  (`VoicePushCoordinator._send_push_notifications` -> `BarkChannel.send`,
  ledger `bark_status=sent`), not a manual curl. Web Push `push_status=skipped`
  as expected on HTTP (no secure-context subscription).
- 3.3 Tapping the notification deep-links into the right session: OK 2026-05-29.
  `bark_base_url=http://115.190.235.210:46352`, pushed url
  `.../#session=broker-3828071`, tap opened Safari and switched to that session.
  Note: the user's host port is dynamic; a stable base URL (e.g. Tailscale
  `*.ts.net`) is needed for deep-links to survive across reconnects.

## 4. AskUserQuestion browser answers (Claude)

Verified 2026-05-29 against a live Claude session (broker-3750543) driven through
the real browser (headless Chrome via Playwright), watching the live tmux TUI:

- 4.1 Header rendered: each question shows its `header` tag (Focus area, Backends,
  Goal, Priorities; Color, Fruits). OK.
- 4.2 Single-select lands on the picked option: clicking option index 2 of the
  Color question selected Blue in the TUI and Enter auto-advanced to the next
  question. OK.
- 4.3 Multi-select checks exactly the picked options: clicking Apple (idx 0) and
  Cherry (idx 2) in the browser produced `1. [✔] Apple ... 3. [✔] Cherry` in the
  TUI with Banana/Date unchecked. OK.
- 4.4 Confirm submits once: clicking the "Confirm selection" button sent Tab+Enter,
  the prompt closed, and the agent resumed. OK.
- Root-cause fixes verified: (a) Claude `multiSelect` now normalized to
  `allowMultiple` so the browser knows a question is multi-select; (b) cursor move
  and toggle/confirm keys are sent as SEPARATE `/keys` calls to avoid the race that
  toggled the wrong option when merged into one string.

## 5. Visual styling pass (ui-ux-pro-max)

Verified 2026-05-29 in both `prefers-color-scheme` modes via headless Chrome.

- Added a full dark-mode variable set (`--bg/--panel/--border/--text/--muted/
  --accent/--accent-weak/--danger/--bubble-*`). Previously only the error card had
  dark overrides; the rest of the UI stayed light. Dark mode now: bg #0f1117,
  panels #171a21, text #e6e8ec, blue accent #5b9dff.
- Converted opaque `#ffffff` surfaces (sidebar, session cards, inputs, top bar) to
  `var(--panel)`, active-session highlight to `var(--accent-weak)` + accent border,
  and interactive option hover/selected states to accent-based colors so they work
  in both modes. Dark-mode opaque-white element count dropped from 223 to 118
  (remaining are intentional translucent frosted overlays).
- Error card and `.primary` buttons now derive from `--danger`/`--accent` so they
  read correctly in both themes; primary buttons are filled for clear CTA emphasis.
- Light mode verified unchanged (1:1 semantic swaps): sidebar white, active
  session subtle highlight, error border red.
- Accessibility follow-up (verified live 2026-05-29 via headless Chrome):
  - Keyboard focus rings: buttons, `.icon-btn`, links, and text/password/
    datetime inputs + selects now show a 2px `--accent` `:focus-visible` outline
    on Tab navigation (previously inputs set `outline: none` with no replacement,
    and only option buttons had a ring). Confirmed `rgb(91,157,255)` ring on real
    Tab focus across the top-bar and sidebar controls.
  - Session cards (previously clickable `div`s with no keyboard path) are now
    `role="button"` + `tabindex="0"` with an `aria-label` ("Open session <name>")
    and Enter/Space activation. Verified live: Tab reaches a card (2px accent
    ring), Enter selects it and loads the transcript (61 rows). This is the app's
    primary navigation, so keyboard-only users can now switch sessions.
  - The queued-message editor (`.queueText`, a borderless inline textarea whose
    own `outline` is intentionally suppressed) surfaces keyboard focus on its
    `.queueEditorShell` container via `:focus-within`, so it is no longer a
    focus-affordance dead spot.
  - Error cards carry `role="alert"` ONLY when freshly appended via the live
    poller, so a newly arriving upstream error is announced. Historical cards
    rendered on session load / scrollback are NOT assertive (verified: 19
    cards loaded, 0 with `role="alert"`), avoiding a screen-reader burst of
    stale errors. The red border + source/type text keep the error non-color-
    only in both cases.
  - `prefers-reduced-motion: reduce` collapses transitions/animations to ~0s
    (confirmed transition-duration 1e-05s under the emulated setting), so the
    option-press, hover-lift, and spinner motion are suppressed for users who
    request reduced motion.

---

## 6. File-path containment hardening (security follow-up)

An adversarial review of the §2 file-viewer work found the original containment
check only covered absolute paths on the read/blob endpoints. Three escapes
remained. All are now closed and verified live on 2026-05-29 (session
`broker-1998985`, cwd `/vePFS-Mindverse/user/intern/lucian/codoxear`):

- 6.1 Relative `..` traversal (`../../../../etc/hostname`, `../workspace`) ->
  HTTP 400 `outside_allowed_root`. OK. Previously the relative branch joined to
  the cwd and classified the result with no containment check, so `..` escaped.
- 6.2 Inside-cwd symlink whose target points outside (`link -> /etc/hostname`)
  -> HTTP 400 `outside_allowed_root` ("resolves outside ..."). OK. The check now
  validates BOTH the lexical location and the fully resolved target.
- 6.3 Inside-cwd dead symlink still reports HTTP 404 `dead_symlink` (not
  `outside_allowed_root`), because the containment check resolves the symlink's
  parent (its own in-cwd location) rather than the dead target, so a dead symlink
  falls through to the classifier. OK.
- 6.4 `/file/download` and `/file/write` (overwrite branch) now route through the
  same typed resolver. Absolute-outside and relative-`..` writes/downloads ->
  HTTP 400; confirmed no file was created outside the cwd. Previously both used
  `_resolve_session_path`, which resolved arbitrary absolute/`..` paths with no
  containment. The dead `_resolve_client_file_path` wrapper (which returned the
  raw escaping path on a non-ok result) was removed.
- 6.5 Symlinked cwd component: the containment check resolves the request's
  parent directory the same way `allowed_root` is resolved, so a session whose
  cwd is reached through a symlinked path component (e.g. macOS `/tmp ->
  /private/tmp`, bind-mounted homes) does not falsely reject every in-cwd file.
  Covered by `test_symlinked_cwd_component_does_not_false_reject`.
- Scope note: the SESSION-scoped endpoints (`/api/sessions/<id>/file/{read,blob,
  download,write}`) are now contained. The GLOBAL file browser (`/api/files/blob`,
  `/api/files/read`/`inspect` called WITHOUT a `session_id`) is a separate,
  pre-existing, auth-gated "open any absolute path" feature and is intentionally
  NOT cwd-scoped; this change does not alter it.
- Behavioral narrowing (intended): the relative-path quick-open helpers
  (bare-filename `os.walk` of the git repo, and tracked-file-by-basename) are now
  constrained to the session cwd by the containment check. Previously, when the
  git repo root sits ABOVE the session cwd, a uniquely-named file elsewhere in the
  repo could be opened from a session view; it now returns `outside_allowed_root`.
  This is consistent with the security intent (a session view should not reach
  outside its cwd) and is the deliberate tradeoff of enforcing containment on
  every resolver branch.
- 6.6 Arbitrary-overwrite escape via symlinked dir + missing intermediate
  (CRITICAL, found in the second adversarial pass, now closed): with an in-cwd
  symlink `cwd/sl -> /outside`, a write-overwrite of `sl/nope/../secret.txt`
  (where `nope` does not exist) previously fell through the containment check
  (strict resolve raised on the absent `nope`) and the handler's
  `_resolve_session_path` fallback re-resolved non-strictly, following `sl` to
  `/outside/secret.txt` and overwriting it. Fixed by making the write-overwrite
  handler reject EVERY non-ok resolver status (not just outside_allowed_root) and
  removing the weaker fallback. Verified live 2026-05-29: read -> 404, write ->
  404 with the target file content unchanged, direct `sl/secret.txt` -> 400
  outside_allowed_root. Covered by
  `test_symlinked_dir_with_missing_intermediate_does_not_resolve_ok_outside`.
- 6.7 Absolute path fails closed when the session root cannot be resolved
  (eviction race -> allowed_root None): the absolute branch now returns not_found
  rather than serving the path with containment disabled. Covered by
  `test_absolute_path_with_unresolvable_session_root_fails_closed`.
- 6.8 Relative `..` traversal with a NAMED-but-unresolvable session (HIGH, found
  in the third adversarial pass, now closed): previously a non-empty but unknown
  `session_id` plus `../../../../etc/passwd` fell through to resolving against the
  server process cwd with no containment, serving arbitrary files via
  `/api/files/read`. The relative branch now fails closed (returns not_found)
  when a named session does not resolve, symmetric with the absolute branch.
  Verified live 2026-05-29: unknown-session traversal -> 404 "session working
  directory not resolvable"; the sessionless global browser
  (`session_id=""`) is intentionally unchanged and still resolves. Covered by
  `test_relative_traversal_with_unresolvable_session_fails_closed` and
  `test_sessionless_relative_path_still_resolves`.
- Regression coverage: `tests/test_file_resolution.py` now has 15 cases covering
  relative-`..`, inside->outside symlink (relative + absolute), dead-symlink
  precedence, symlinked-cwd false-positive, symlinked-dir + missing-intermediate
  overwrite escape, absolute + relative fail-closed, sessionless still-works, and
  the in-cwd false-positive guard.

## 7. Busy-state: Claude retry refinement (C1)

The busy-state guard distinguishes Claude auto-retried errors from terminal
ones, while Pi and Codex errors remain turn-terminal (they carry no retry
semantics). Covered by tests, no browser step required:

- A Claude `api_error` is non-terminal ONLY with positive evidence of a pending
  retry (`retryInMs > 0` AND integer `retryAttempt < maxRetries`): the turn stays
  open and the record counts as activity, so the UI does not flap working<->idle.
- Every other Claude `api_error` shape closes the turn: no scheduled retry,
  retries exhausted, or malformed/non-integer retry counters. Defaulting a
  malformed record to terminal prevents a stuck-busy spinner.
- Pi `toolResult` `isError` and Codex `*_error` records close the turn
  (`_close_turn_state`). Keeping them open would disable the idle-fallback
  (`_should_clear_busy_state` skips while `turn_open and not
  turn_has_completion_candidate`) and strand the spinner on "working" forever —
  the original Issue 1. They still render as `agent_error` cards via
  `rollout_log`, independent of busy state.
- Coverage: `tests/test_agent_error_events.py` (29 broker/parser cases).

---

## Already-validated by automated tests

These do not need browser verification:

- Backend parser + busy state: `tests/test_agent_error_events.py` (27 cases, all pass)
- File resolver: `tests/test_file_resolution.py` (9 cases, 8 pass + 1 root-skip)
- Bark channel: `tests/test_bark_channel.py` (6 cases, all pass)
- Full suite: 328 passed, 1 skipped (root-bypass permission case).
- HTTP smoke (`/api/files/inspect`, `/api/settings/voice` GET+POST, file
  read/download/write containment) confirmed via `curl` after server restart on
  2026-05-29.
