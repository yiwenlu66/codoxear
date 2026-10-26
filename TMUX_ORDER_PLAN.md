# Plan: keep tmux window order in sync with the sidebar order

Feature request: *automatically rearrange sessions in tmux; follow the same
order as in sidebar.*

Status: implemented (commits `f024d3c1`.. ) in two steps. Two decisions below were
superseded once the first live run showed the feature was not usable as planned:

- section 6.2 identifies windows by name (`tmux_window`); the shipped mirror
  **titles** each window with the sidebar name, so identity is the
  `@codoxear_nonce` window option stamped before the first rename;
- section 3 keeps non-codoxear windows in place; the shipped mirror moves them
  **after** the codoxear block so window `k` is sidebar row `k`.

See AGENTS.md ("Sidebar order is server-owned…") for the contract as shipped.

---

## 1. What the request means in this codebase

Each codoxear session that is launched with "Create in tmux" becomes **one
window** in a single tmux session (`CODEX_WEB_TMUX_SESSION`, default
`codoxear`). `session_launcher.py::_launch_tmux_broker` runs
`tmux new-window -d -P -t <sess>: -n <name> <cmd>`; the window name is
`safe_filename(f"{cwd.name}-{spawn_nonce[:6]}")` and is exported to the broker as
`CODEX_WEB_TMUX_WINDOW`, which the broker copies into its sidecar and the listing
row as `tmux_window` (`broker_metadata.py`, `session_runtime.py::session_transport_from_meta`).

So "rearrange sessions in tmux" can only mean: **permute window indices** inside
that tmux session. tmux has no sortable window list — the only ordering is
`#{window_index}` — so mirroring the sidebar is an explicit `swap-window`
sequence driven by codoxear.

tmux *sessions* (as opposed to windows) have no user-visible order at all, so
they cannot be the object of this feature. A window-per-session mirror is the
only reading that produces a visible effect.

## 2. What the sidebar order actually is today

Two implementations of the same idea exist, plus a grouping step:

1. `session_listing.py::sort_session_rows` (server, called at
   `session_list.py:147`) sorts rows by
   `(-final_priority, -updated_ts, -start_ts, session_id)`.
2. `app_session_refresh.js:74-83` re-sorts the same rows client-side with the
   same comparator, using `localeCompare` for the id tiebreak.
3. `app_session_helpers.js::sidebarSessionEntries` then **re-partitions** rows
   into three rendered groups — `Now` (`blocked=false, snoozed=false`),
   `Waiting` (`blocked`), `Later` (`snoozed`) — and renders group headers with
   counts in that order.

`final_priority` is time-decayed: `exp(-elapsed / 8h)` of
`now - updated_ts` (`updated_ts` = last conversation timestamp from the log),
bucketed at 30 s, plus the per-session `priority_offset`, forced to `0` when
snoozed or blocked. Launch-attempt rows carry `final_priority = 1.0`
(`launch_ledger.py:359`), so a starting session sits at the top.

Consequence: **the rendered order is not the comparator order.** The server
list interleaves `Waiting` and `Later` rows (both priority `0`, ordered by
`updated_ts`); the renderer splits them into two blocks. Any tmux mirror must
use the *rendered* order, i.e. group rank first, then comparator.

## 3. The invariant to own

> Let `L` be the most recent `/api/sessions` body a client received. Then,
> reading codoxear-owned windows in `CODEX_WEB_TMUX_SESSION` by ascending
> `window_index`, their sessions appear in the same relative order as in `L`'s
> **sidebar entry order** (group rank, then comparator), restricted to sessions
> that have an identifiable window.
>
> Additionally:
> - The **set** of indices occupied by codoxear windows is invariant across a
>   reconcile. Codoxear windows are permuted *among their own slots*; a
>   non-codoxear window (the user's own shells) never moves.
> - The reconciler never creates, destroys, renames, or selects a window, and
>   never changes which window a viewer is looking at.

The sidebar is the sole ordering authority. tmux follows it; tmux-side manual
moves are overwritten by the next reconcile (documented, and switchable off).

## 4. What already exists (reuse, do not rebuild)

- **Server-side ordering function**: `session_listing.py::sort_session_rows`,
  pinned by `tests/test_session_listing.py::test_sort_session_rows_orders_by_priority_recency_start_and_id`.
- **Row → window mapping**: every listing row already carries `tmux_session`
  and `tmux_window` (`build_active_session_row`), including launch-attempt rows
  (`launch_ledger.py:352-353`) and their `spawn_nonce`.
- **tmux discovery/capture**: `tmux_runtime.py` (`tmux_available`,
  `tmux_pane_snapshot`) with `run` injected for tests, plus
  `server._tmux_available` (30 s TTL cache) and `SessionRouteDeps.tmux_available`
  / `.tmux_session_name`.
- **One listing chokepoint**: `session_routes.py::_handle_sessions_list` is the
  only place a session list reaches a client, and it is ETag'd, so "the client
  received a new order" is already observable for free.
- **Threading pattern for background work**: `session_manager_bootstrap.py`
  worker-thread helpers (`stop_event`, daemon threads) and the `_sweep`
  coordinators on the manager.

## 5. Measured tmux behaviour the design depends on

Measured against **tmux 3.8** on this host, isolated servers (`-L <scratch>`),
never the live `default` session. These are the load-bearing facts:

| Property | Result |
| --- | --- |
| `-n <name>` on `new-window`/`new-session` pins the name | window-local `automatic-rename` becomes `0`, name survives the process running (tested with a global `set -g automatic-rename on` too) → `tmux_window` from metadata is a stable identity key |
| `swap-window -s @a -t @b` | swaps the two windows' indices in place; accepts window **ids**, so a plan is index-churn-proof; no renumbering, no gaps |
| `list-windows -F '#{window_index} #{window_id} #{window_name} #{window_active}'` | enough to plan a full permutation |
| User options readable per window | `list-windows -F '#{@codoxear_session}'` works (tmux ≥ 3.0), and window options travel with the window across a swap |
| `swap-window` **without** `-d`, one involved window is the current one | the viewer stays on the same *index* → **content changes** (client teleported to another agent's output) |
| `swap-window` **with** `-d`, one involved window is the current one | the viewer stays on the same *window* → content preserved, index changes |
| `swap-window` **with** `-d`, neither involved window is current | the viewer **is teleported anyway** (tmux 3.8 quirk: current window follows a swapped slot) |
| `swap-window` **without** `-d`, neither involved window is current | viewer untouched |
| Multiple attached clients | they share the session's current window in this configuration (a `select-window` moved both), so one rule can serve all viewers |

Derived rule, which the implementation must encode literally:

**pass `-d` if and only if one of the two windows involved in *this* swap is the
session's current window (`#{window_active}`), re-read immediately before each
swap.**

A pure swap-based permutation (`for each slot in ascending order: swap the
window that belongs there into it`) was validated against real tmux for
reverse, arbitrary, and mixed-with-foreign-windows targets: it converges in
≤ n−1 swaps, leaves foreign windows and the viewer's content intact, and emits
**zero** commands on a second pass (idempotence).

## 6. Design

### 6.1 Order authority: one implementation, on the server

- Extend `sort_session_rows` with a leading **group rank**
  (`Now` 0 / `Waiting` 1 / `Later` 2, precedence `blocked` → `snoozed` → else
  `now`, matching `sessionSidebarGroupKey`), so the server list order *is* the
  rendered sidebar order. Document it in the docstring as the single order
  authority.
- Delete the client comparator in `app_session_refresh.js`; the client renders
  the server's array order. `sidebarSessionEntries` keeps only the header
  emission (it becomes a pure partition-preserving walk).
- Why: one writer for one displayed value (see "State-authority principle" in
  AGENTS.md). Today the two comparators agree only by coincidence, and the
  rendered order already disagrees with both when `Waiting` and `Later` are
  both populated.
- Side benefits: `latestSessions` consumers (`app_new_session.js` resume menu,
  `app_quick_session.js`) inherit the sidebar order instead of a raw discovery
  order.

Risk check: nothing indexes `sessions[0]` expecting a particular row; ETag is
computed from the body, so an order change already forces a 200 + re-render.

### 6.2 Identity: session row ↔ tmux window

Primary key: `(row.tmux_session, row.tmux_window)`.

- Rows without both fields (`transport != "tmux"`, terminal-launched brokers
  without the env vars) are skipped — they have no window.
- A window name that resolves to **0** tmux windows → skip (window gone).
- A window name that resolves to **≥2** windows → ambiguous → skip, and report
  it in the pass summary. Never guess.
- A tmux window that no row names is **foreign**: it keeps its index forever
  (section 3).

Optional hardening (phase 5, explicitly deferrable): stamp
`set-option -w -t <win> @codoxear_session <session_id>` right after the window
is created and on rebind, then read it back via
`list-windows -F '#{@codoxear_session}'`. This only buys tolerance for a
user-issued `rename-window` / re-enabled `automatic-rename`; it does not change
the permutation logic. Keep it out of v1 so the launch path stays untouched.

### 6.3 Rearrangement

`reconcile_once(rows, tmux_session, tmux_bin, run)`:

1. `list-windows -t <sess> -F '#{window_index}\t#{window_id}\t#{window_name}\t#{window_active}'`
   → `{}`. Empty/error → return a "skipped" summary, no commands.
2. `target = [window_id of each row, in row order, where the row's
   (tmux_session, tmux_window) maps unambiguously]`; `slots = ascending indices
   of the windows those rows own`.
3. For `i, slot in enumerate(slots)`: re-read `window_active` (cheap, keeps the
   `-d` rule correct as windows move); if the window at `slot` is not the wanted
   one, `swap-window [-d] -s <id at slot> -t <wanted id>`.
4. Re-list and compare against the target; a mismatch is reported (never
   retried in a loop inside one pass — the next generation re-plans).

Properties: idempotent (empty command list when already ordered), bounded
(≤ n−1 swaps), permutation-only (index set preserved), viewer-preserving
(`-d` rule), and every target is a window **id** so no name parsing or index
arithmetic ever reaches tmux.

### 6.4 Trigger and cadence

Reconcile from the listing generation, not from a timer and not from the
browser:

- `_handle_sessions_list` computes `sessions = manager.list_sessions()` (already
  in sidebar order) and hands `sessions` to `tmux_order_sync.note_generation(rows)`
  **after** the response is written. Never blocks the request, never raises into
  it.
- The sync object holds the newest generation only (coalescing: generation N+1
  replaces N), a single-flight worker thread, a settle window (default 1.5 s
  quiet period) and a floor (`>= 10 s` between passes).
- Skip entirely when `tmux_available()` is false.
- Skip the tmux listing when the target order equals the last **applied** target
  AND the last pass succeeded — this is what makes a 304/no-op poll cost zero
  subprocesses, while still healing within 60 s (periodic forced re-list) if
  someone moved a window by hand.

Why not a timer: with no client polling, `list_sessions` is not called, so
`updated_ts`/priority never advance; a timer would reorder tmux against stale
data and diverge from what the sidebar would render. Tying the reconcile to the
listing generation makes "tmux mirrors the sidebar" literally true — it mirrors
the last list the product computed, and the ETag already makes "same list, no
re-render" free.

Why not the browser: it needs an open tab, races across tabs/phones, and
requires forcing a CORS-free POST path for what is a host-local side effect.

### 6.5 Configuration

| Env var | Default | Meaning |
| --- | --- | --- |
| `CODEX_WEB_TMUX_AUTO_ORDER` | `1` | master switch; `0` disables all tmux reordering |
| `CODEX_WEB_TMUX_ORDER_SETTLE_SECONDS` | `1.5` | quiet period after the newest generation |
| `CODEX_WEB_TMUX_ORDER_MIN_INTERVAL_SECONDS` | `10` | floor between passes |
| `CODEX_WEB_TMUX_ORDER_RELIST_SECONDS` | `60` | forced re-list even when the target order is unchanged |

Default on, because the request is for automatic behaviour; rollback is one env
var and needs no code change.

### 6.6 Observability

- One stderr line per *acting* pass:
  `tmux-order: session=codoxear swaps=2 order=<name,name,...>`
  and one per failure (`list-windows failed`, `ambiguous window <name>`,
  `swap @3<->@7 failed: ...`).
- `_record_metric("tmux_order_pass_ms", elapsed)` per pass, so `/api/metrics`
  shows the feature is alive and cheap; zero-swap passes are visible as a
  duration with no log line.
- Pass summaries are also returned by `reconcile_once`, which is what the tests
  assert on.

### 6.7 Failure modes

| Failure | Behaviour |
| --- | --- |
| tmux binary missing | `tmux_available()` false → no work |
| tmux server not running / session missing | `list-windows` rc≠0 → skipped summary, logged once per transition (not per poll) |
| window disappeared mid-pass | `swap-window` rc≠0 → abort the pass, next generation re-plans |
| duplicate window names | that name is skipped and reported; other windows still reconcile |
| user manually moved/renamed a window | next pass snaps the moved window back; a renamed window loses its identity and is skipped (stays put) |
| `renumber-windows` enabled in the user's config | irrelevant: indices are re-read every pass |
| server restart | tmux order persists; the first listing generation re-mirrors it |
| reconcile thread dies | single-flight flag reset in `finally`; next generation retries |
| many fast generations (agent streaming) | coalesced; at most one pass per `MIN_INTERVAL` |

## 7. Edge cases to handle explicitly

- **New session**: `new-window` appends at the end; the pass that follows the
  first listing generation moves it to its sidebar position (usually the top,
  `final_priority = 1.0`). Accepted visible jump, ~1 poll + settle.
- **Pending launch** (`launch_attempt` rows with `launch_state = "starting"`):
  they carry `tmux_window` and priority `1.0`, so they are ordered like any
  other row — but their window may not exist yet; the name simply fails to map
  and is skipped.
- **Pi `/new` / `/resume` in a window**: the sidecar's `tmux_window` comes from
  the env var and survives the rebind, so the new session row keeps the same
  window identity.
- **Snapshot restore**: restore launches with `create_in_tmux` reproduced from
  the recorded entry, so restored sessions get fresh windows and join the order.
- **`lost` / failed / direct-transport rows**: no window → skipped.
- **Several tmux sessions** (rows whose `tmux_session` differs from the server
  default, e.g. a config change): rows are grouped by `tmux_session` and each
  group reconciles against its own window list.
- **tmux session groups**: `list-windows` on a grouped session lists the shared
  windows; the slot invariant still holds because we only permute the windows we
  own.
- **A user's tmux on a different socket** (`-L`, custom `TMUX_TMPDIR`): windows
  created by codoxear always use the default socket (`_launch_tmux_broker`
  passes no `-L`), so the server reaches them with a plain `tmux` invocation.
  Terminally-launched brokers on another socket simply never map and are
  skipped. (Verified on this host: the unit sets no `TMUX_TMPDIR` and
  `/tmp/tmux-1000/default` exists.)

## 8. Alternatives rejected

- **`move-window` + `move-window -r` renumbering**: creates gaps/intermediate
  index collisions, renumbers foreign windows, and produces more visible churn
  than a straight swap permutation.
- **Sorting the tmux status bar from outside** (e.g. `status-format` rendering
  a sorted list from `@` options): display-only, duplicates tmux's own window
  list, and cannot be clicked through to the right window without more glue.
- **Client-side (browser) reconcile**: requires an open tab, races across
  clients, and moves host state from the browser.
- **tmux → sidebar direction**: the sidebar is the product's ordering surface
  (priority, snooze, dependency); deriving order from tmux indices would delete
  that semantics.
- **Stamp-only identity in v1**: the metadata name is already stable and
  per-row; stamping adds a launch-path mutation for no v1 benefit.

## 9. Phases

**Phase 1 — order authority (no tmux code yet).**
`session_listing.py`: add `sidebar_group_rank(row)`, prepend it to
`sort_session_rows`, document as the single order authority.
`app_session_refresh.js`: delete the comparator, render server order.
Tests: extend `tests/test_session_listing.py` ordering test with group cases
(blocked/snoozed/launch-attempt/ties across all four key fields); VM test via
`tests/frontend_module_loader.py` that `renderSessions` emits entries in input
order and groups render Now→Waiting→Later.

**Phase 2 — reconciler core (pure + fake runner).**
`tmux_runtime.py`: `list_window_rows(...)` / `swap_windows(...)` wrappers with
injected `run` (mirroring `tmux_pane_snapshot`).
`tmux_order.py`: `desired_window_order(rows, tmux_session)`,
`plan_swaps(slots, id_by_index, target_ids)`, `reconcile_once(...)`.
Tests: `tests/test_tmux_order_plan.py` (identity → no swaps; reverse; cycle;
foreign slots; missing subset), `tests/test_tmux_order_reconcile.py` (fake `run`
asserting the exact argv sequence and `-d` placement for the four measured
cases, and that a second pass emits nothing).

**Phase 3 — runtime wiring.**
`tmux_order_sync.py` (single-flight coalescing worker, seams for clock/submit so
tests never spawn threads), `SessionRouteDeps.note_listing_generation`,
`_handle_sessions_list` hook, `server.py` singleton + `server_config.py` knobs,
`server_route_deps.py` wiring. Tests: `tests/test_tmux_order_sync.py` with a
fake clock — coalescing drops superseded generations, the floor is respected,
errors never propagate to the request path.

**Phase 4 — behavioural verification in Docker.**
`scripts/docker_tmux_order_verify.sh`: real tmux inside the container, three
sessions launched with `create_in_tmux` plus one foreign window created
manually; assert (a) `list-windows` order equals the `/api/sessions` order,
(b) the foreign window keeps its index, (c) after a priority edit / snooze the
order follows, (d) a pty-attached viewer keeps seeing the *same window content*
across a reorder that moves the current window, (e) a second pass issues no
swaps. Same discipline as `scripts/docker_undelivered_send_verify.sh`: Docker
is the isolation boundary; no host repros, no pattern-based cleanup.

**Phase 5 — docs and optional hardening.**
AGENTS.md (new bullet under "Data flow" / "Ops notes"), `.memory/project/ARCHITECTURE.md`;
optional `@codoxear_session` stamping if rename tolerance turns out to matter.

## 10. Open decisions for you

1. **Default on or off?** I recommend on (`CODEX_WEB_TMUX_AUTO_ORDER=1`), off
   by env var when it annoys. The opposite default costs one manual enable.
2. **Churn tolerance.** With 8 h half-life decay and `updated_ts` = last
   conversation, the top slot changes on essentially every agent message. The
   floor (10 s) bounds it to one reorder per interval, but a fleet of active
   sessions will keep rearranging the status bar. Alternatives: only reorder
   when a session *finishes* a turn, or only reorder the `Now` group. I
   recommend mirroring exactly, with the floor — anything smarter is a second
   ordering rule to explain.
3. **Include `Waiting`/`Later` in the tmux order?** Recommended yes (it is what
   "same order as sidebar" means), lower slots for higher groups.
4. **Slot policy** for foreign windows: permute codoxear windows among their own
   indices (recommended, zero collateral damage), or pack codoxear windows into
   `0..n-1` and push the user's windows to the end.
5. **A manual "sync order now" affordance** (UI button / `POST
   /api/tmux-order/sync`)? Not needed if the mirror is automatic; useful only
   for debugging. Recommend no UI.

## 11. Non-goals

- Reordering tmux sessions (impossible/meaningless) or window *titles*.
- Syncing tmux → sidebar, or persisting a tmux-derived order.
- Creating, killing, or renaming windows; attaching/detaching clients;
  selecting windows.
- Reordering windows that no codoxear session owns.
- Guaranteeing an order for windows whose session cannot be identified (renamed
  by the user, or created by a non-codoxear launcher).
