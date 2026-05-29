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

---

## Already-validated by automated tests

These do not need browser verification:

- Backend parser: `tests/test_agent_error_events.py` (13 cases, all pass)
- File resolver: `tests/test_file_resolution.py` (5 cases, 4 pass + 1 root-skip)
- Bark channel: `tests/test_bark_channel.py` (6 cases, all pass)
- HTTP smoke (`/api/files/inspect`, `/api/settings/voice` GET+POST) confirmed
  via `curl` after server restart on 2026-05-29.
