# OPS — sidebar snooze + drag (planning)

Append-only. All times local (2026-11-13).

## 2026-11-13 — investigation (no product code changed)

Read for the plan (paths relative to repo root):

- `codoxear/static/app_snooze.js` — presets only. No "indefinite"; `snoozePresetEpoch("none")`
  returns null, so "none" is expressed by the absence of a timestamp.
- `codoxear/static/app_session_edit.js` — snooze mode state machine
  (`none|evening|tomorrow|weekend|nextWeek|custom`), custom value written from
  `<input type=date>` + `<input type=time>`, save posts `/api/sessions/<id>/edit` with
  `name + priority_offset + snooze_until + dependency_session_id` (all fields always sent).
- `codoxear/static/app_shell.js:527-553` — chip construction; `:579-582` field layout;
  `editSnoozeCustomRow` holds the two native inputs.
- `codoxear/static/app.css:3212-3216` (date/time at `--ctl` = 44px), `:3325-3331` (chips at
  `--dialog-control-h` = 32px), `:4095-4106` (coarse-pointer `font-size: var(--font-xl)
  !important` on `input[type=date|time]`), `:3755` (`.customSnoozeRow` grid `1fr 140px`),
  `:3546` (`#editViewer .choiceChips` wraps), `:2626-2674` (`.filePickerMenu` /
  `.fileMenuItem`), `:258-264` (`#sessions` is the scroll container), `:363-373`
  (`.session { touch-action: pan-y }`).
- `codoxear/session_listing.py:153` (`clip01(time_priority + offset)`), `:425` (`snoozed`
  predicate), `:646-694` (`SIDEBAR_GROUP_RANKS`, `sidebar_group_key`, `sort_session_rows`),
  `:670` single order authority.
- `codoxear/server_config.py:349-350` — `SIDEBAR_PRIORITY_HALF_LIFE_SECONDS = 8h`,
  `SIDEBAR_PRIORITY_BUCKET_SECONDS = 30s`.
- `codoxear/session_cleaners.py:41-43` — offset range `[-1, 1]`.
- `codoxear/session_store.py:346-367` — `sidebar_state_for_session` prunes dead dependency and
  expired snooze (the pattern a rank prune would follow).
- `codoxear/session_ui_state.py:52-125` — `sidebar_meta_get` / `_validated_sidebar_entry` /
  `sidebar_meta_set` / `edit_session` (whole-row write, no partial semantics).
- `codoxear/control_routes.py:104-115` post route table; `:166-188` `_handle_edit` (requires
  `name`, always writes all four fields).
- `codoxear/session_routes.py:88-113` — non-session POST paths (`/api/quick-sessions`) are
  matched here, the precedent for a new `/api/sidebar-layout`.
- `codoxear/server_routing.py:23-34` — `match_session_route` requires exact arity, so a
  non-session `/api/...` path cannot be mistaken for a session route.
- `codoxear/diagnostics_routes.py:71-122` — recomputes `snoozed` and shows
  `priority_offset` / `final_priority` independently of the listing.
- `codoxear/session_snapshot_runtime.py:552-556` — `remap_session_identity` moves the whole
  `sidebar_meta` entry, so new meta keys follow a resume with no code change.
- `codoxear/static/app_sessions.js:82-141` (`bindSwipe`; axis gate at `:112-116`), `:165-172`
  (`refreshDeferred`), `:287-334` (touch/desktop card branches), `:171` + `:330` (render signature,
  empty-hint handling).
- `codoxear/static/app_session_helpers.js:43-64` — `sessionSidebarGroupKey` /
  `sidebarSessionEntries` partition only, never re-sort.
- `codoxear/static/app_dialog_menu.js:39-71` — `positionDialogMenu` (flip + max-height + width
  clamp) already suited to a 28-row / 48-row menu.
- `codoxear/static/app_transcript_render.js:303-305` — the direct-shortcut handler bails on
  `altKey`, so `Alt+↑/↓` is free.
- `tests/test_snooze_presets.py`, `tests/test_sidebar_render_order.py`,
  `tests/test_session_listing.py:417-442` — the behavioural harness style and the sort tests that
  will need new cases.
- `scripts/docker_theme_verify.sh` — the established docker + loopback-port + agent-browser
  harness to copy for a drag verifier.

## 2026-11-13 — negative results (kept so they are not re-proposed)

- Styling the native `<input type=date|time>` to match the chips was rejected: the coarse-pointer
  `font-size: var(--font-xl) !important` rule (`app.css:4105`) exists to stop iOS focus zoom
  and cannot be lowered for these two inputs without reintroducing that bug.
- Encoding "indefinite" as `math.inf` or a far-future stamp was rejected: `json.dumps(float("inf"))`
  emits `Infinity`, which `JSON.parse` rejects, so the listing row could hand the browser invalid
  JSON.
- Writing `priority_offset` from the drag was rejected on arithmetic, not taste: with
  `clip01(tp + offset)` an old row cannot out-rank a saturated row (score exactly 1.0 loses the
  `updated_ts` tie), and a constant offset on a decaying recency term inverts a young-row-below-
  older-row order in ~16 hours (numbers in `PLAN.md` §4.2).
- A month-grid calendar was rejected: 7 columns cannot give 44px targets inside a 366px phone
  dialog, and the app already ships a list-menu primitive that honours the touch rule.

## 2026-11-13 — artifact note

`PROMPT.md` in this directory is local-only: the repo `.gitignore` ignores `PROMPT.md` (25 older
task specs were force-added before the rule). The committed deliverable is `PLAN.md`, which is
self-contained; `OPS.md` carries the evidence trail.
