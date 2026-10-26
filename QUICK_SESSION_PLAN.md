# Plan: quick sessions get a distinct name and their own working directory

Feature request: *quick session should be named something like "Quick Session xx"
and have a dedicated dir (e.g. `~/codoxear-xxx`) rather than `$HOME`.*

Status: plan only — nothing implemented.

---

## 1. What happens today

`app_quick_session.js::start()` calls `spawnSession("~", …)`, so the launch body
carries `cwd: "~"`, the server expands it (`launch_path_runtime.expand_user_path`),
and the broker sidecar records `/home/<user>`. Nothing distinguishes a quick
session from a session the user deliberately opened in `~`.

`session_display.sidebar_display_name` names a row by its trimmed alias, else
`basename(cwd)`. A quick session has no alias, so:

- every quick session is titled with the account name (`yiwen`) — identical to
  each other and to any session opened in `~`;
- the tmux window mirror and push notifications use that same value, so the
  window list and the notification title are indistinguishable too;
- they share one cwd, so file trees, `git status`, path completion, agent bash
  work and artifacts of one launch are visible to the next, and an agent that
  "tidies up its workspace" edits real home files.

`scripts/docker_quick_session_verify.sh` pins today's behavior: sidecar
`cwd == /home/tester`, launch body `cwd == "~"`, failure-dialog `cwd == "~"`.

So the naming problem is not cosmetic: the name must be derived from something
unique per workspace, and the workspace must be a thing that exists.

## 2. Naming scheme

**Name**: `Quick Session NN` — `NN` is the slot, zero-padded to two digits
(`Quick Session 01` … `Quick Session 99`, then `100`).
**Slot**: monotonic integer persisted in `<app_dir>/quick_sessions.json` as
`{"next_slot": N}`.

Invariants:

1. **Name ↔ directory.** Slot N is the same N in the label and in the path
   (`Quick Session 07` ⇄ `~/codoxear-quick/quick-07`). Moving in either
   direction needs no lookup.
2. **A slot is never reused.** Two live sessions can never share a name, and a
   name refers to exactly one workspace for the life of the machine's state.
3. **The label is an alias; identity is the slot.** Display keeps its existing
   single definition (`sidebar_display_name`: trimmed alias, else cwd basename).
   The quick flow sets the alias to `Quick Session NN`, so sidebar, tmux window
   title, notifications and copy-export agree. If the alias is ever missing the
   row reads `quick-07` — still the slot, still unique. Provisional
   launch-attempt rows (which never have an alias) derive the label from the
   slot in the cwd instead (§6.4).
4. **The name is a label, not a claim on files.** Renaming a quick session
   changes the label; the directory stays where it is.
5. **State loss cannot cause a collision.** On load,
   `next_slot = max(persisted_next_slot, max(existing quick-NN slot) + 1)`, so
   deleting `quick_sessions.json` cannot make a new allocation land on a
   directory that already holds work.

Why zero-padded: both `ls ~/codoxear-quick` and the sidebar sort lexically;
`quick-01 … quick-10` sorts correctly, `quick-1, quick-10, quick-2` does not.

### Overflow past 99

The padding is a **minimum width**, not a cap: `f"{slot:02d}"` renders slot 100
as `Quick Session 100` / `quick-100` — same mechanism, one digit more. Nothing
truncates, nothing collides, no name changes retroactively, and monotonicity is
unaffected. So the 99 boundary is not an identity problem, and there is no
rollover rule to design: rolling over (reusing 01…) would break invariant 2,
and widening the pad code-wide (`quick-001`) would rename nothing today but
cost a leading zero for a case that has not happened yet.

The one real consequence is **lexicographic order in a directory listing**:
`quick-100` sorts before `quick-99` (`'1' < '9'` at the first differing
character). Nothing in the app sorts by directory name — the sidebar sorts by
priority/time and the tmux mirror follows the sidebar — so the break is only
visible in `ls`, where `ls -v` (natural sort) restores `01, 02, … 99, 100`. Do
not "fix" it by parsing the number out of names into any ordering: the number
is identity, not a sort key.

Growth is one slot per quick launch, ever. 100 launches is reachable for a
heavy user, which is why the rule is stated rather than assumed; `Quick Session
137` is still readable and still unique. The parser side is one line:
`^quick-(\d+)$` → `int()`, so slot parsing is also width-independent
(`quick-100` → 100). A malformed counter (`next_slot` missing, non-int, ≤ 0)
is treated as 1, and the invariant-5 clamp still protects every existing
directory; an absurd but valid value only produces absurd numbers, never a
collision.

Why monotonic instead of "smallest free number". Reuse requires distinguishing
*abandoned* from *just allocated*, and at allocation time both look like an
empty directory. The only escapes are reservation state (a second source of
truth that goes stale) or mtime windows (still wrong after a suspend). A reuse
rule that gets it wrong puts two sessions in one directory under one name —
precisely the failure this feature exists to remove. Monotonic numbering has no
failure mode; its cost is that numbers grow with lifetime launch count.

Rejected label variants: `Quick NN` (shorter, better under phone-width
ellipsis — the sidebar truncates the tail, so the slot is the first thing lost;
keep it as the fallback if the measured width ever bites, decided by the CSS
measurement and not preemptively); `Quick Session 0824-3` (date-scoped; long,
and the row already shows a timestamp); `Quick Session 07 · pi` (the row
already shows a backend icon).

## 3. Path scheme

- Root: **`~/codoxear-quick`**, hardcoded (`Path.home() / "codoxear-quick"`).
  No env var, no setting: one fewer knob, and the default is the only value
  anyone would pick.
- Slot dir: `<root>/quick-NN`; full path `~/codoxear-quick/quick-07`.

Relocation is a symlink, not a knob: replacing `~/codoxear-quick` with a
symlink to another volume must work, which is why the containment test resolves
**both** sides (`Path(cwd).resolve()` against `quick_root().resolve()`) instead
of comparing a resolved cwd against a lexical root. A symlinked root therefore
still answers `is_quick_cwd` correctly for the recent-cwd filter and the
delete-leftover report.

Why one root rather than flat `~/codoxear-quick-07` directories: one entry in
`$HOME` instead of one per launch; one place to inspect, prune, or relocate
(symlink it to another volume); a root can be a mount point or symlink for a
small home partition.

Why the leaf repeats `quick-` rather than being bare `07`: the sidebar falls
back to the cwd basename when an alias is missing, and `07` alone reads like a
bug; `quick-07` is self-describing at any depth and costs nothing.

Why not spaces (`~/codoxear-quick/Quick Session 07`, which would produce the
exact label for free through the basename rule): a space in the cwd is a
permanent tax on every shell command the agent or the user types there, and
this repo already carries special handling for space-containing filenames.

Why not inside the app dir (`~/.local/share/codoxear/quick/07`): these are the
user's workspaces, not runtime state. They must be visible, browsable, and
survive an app-dir wipe.

Non-quick sessions are untouched: the dialog's cwd field, recent cwds, and `~`
launches behave exactly as today.

**Who creates the directory: the existing launch path.** `prepare_launch_plan`
already does `cwd_path.mkdir(parents=True, exist_ok=True)` for a missing cwd, so
the allocator only picks a number and returns a path. An allocation that never
reaches a launch (client dies between the two requests) costs a number and no
filesystem state.

**Fresh-directory trust modals.** A brand-new directory is exactly the shape
that raises "do you trust this folder" prompts, which the web UI cannot answer
and which eat injected sends (the Pi trust dialog already did this once — see
AGENTS.md "Pi launches always pass `--approve`"). All three backends are
already covered by construction: Codex receives
`projects.<exact cwd>.trust_level="trusted"` computed from `spawn_cwd`
(`codex_trust_override_for_path`), Pi gets `--approve`, and CC launches with
`--dangerously-skip-permissions`. So a fresh cwd adds no new modal — verify it
for CC anyway (§7).

## 4. API

New `POST /api/quick-sessions` — allocates one slot, nothing else. Authorized
like every other POST; no body.

```json
{"ok": true, "slot": 7, "name": "Quick Session 07",
 "cwd": "/home/yiwen/codoxear-quick/quick-07"}
```

It does not create the directory and does not spawn anything.

`POST /api/sessions` is unchanged. The client sends the returned `cwd` as an
ordinary absolute path, so the launch path, resume handling, snapshot
recording, tmux window creation and worktree handling all take their existing
branches.

Rejected: a `quick: true` flag on `POST /api/sessions`. One round trip is
nicer, but the name is only known after that response, and
`spawnSessionWithCwd` would have to return more than a broker pid — changing
three call sites plus the seam the dialog and quick controllers share, to save
one local request. Rejected: client-side numbering by scraping the directory
(races across tabs/devices, and it puts a path convention in the browser).

## 5. Server changes

1. `server_config.py`: `QUICK_DIR = Path.home() / "codoxear-quick"` — a plain
   constant next to `PROC_ROOT`, deliberately **not** readable from the
   environment — and `QUICK_STATE_PATH = app_dir / "quick_sessions.json"`.
2. New leaf module `codoxear/quick_workspace.py`:
   - `quick_dir_name(slot)`, `quick_display_name(slot)` (`f"{slot:02d}"`),
   - `quick_cwd(root, slot)`,
   - `is_quick_cwd(root, cwd)` — containment on paths resolved on both sides
     (`Path.is_relative_to`), not string prefix and not a lexical root, so
     `/home/u/codoxear-quick-x` is not mistaken for the root and a symlinked
     root still matches,
   - `next_slot(state, existing_slots)` — invariant 5 as a pure function,
   - `allocate(root, persist)` returning `(slot, name, cwd)`.
   Pure functions plus one persistence callback; no imports beyond
   `pathlib`/`dataclasses`, so it is testable without a server.
3. `session_store.py` / `session_manager_store.py` / `AppDirPaths`: add
   `quick: Path` to `SessionStorePaths`, `quick_next_slot` to the store with
   load/save through the existing `load_json_file` / `atomic_write_json` /
   `save_*` coordinator pattern, and `quick_root` so the store can answer
   `is_quick_cwd` for the recent-cwd filter. `copy_session_store_state`
   (`session_manager_store.py`) carries both fields on rebind.
4. **Recent-cwd filter** — `SessionStore.note_recent_cwd` and
   `list_recent_cwds` skip paths under the quick root. Discovery remembers
   every discovered session's cwd (`session_discovery_registry.py`), so without
   this the New-session cwd picker fills with `quick-NN` entries and the
   256-entry cap starves real directories. Filtering at write *and* read keeps
   already-written `recent_cwds.json` files clean.
5. `session_routes.py`: `handle_session_post_route` gains
   `if path == "/api/quick-sessions": _handle_quick_allocate(...)`; the handler
   authorizes, allocates through the manager, returns the JSON above. An
   `OSError` (unwritable root, bad path) returns 500 with a redacted message and
   burns no slot.
6. **Delete leftover report** (recommended, small): `control_routes._handle_delete`
   looks up the session's cwd before deleting. If it is a quick cwd, remove the
   directory when it is empty (`rmdir`, never recursive) and respond
   `{"ok": true, "leftover_dir": "<abs>"}` when it still holds entries.
   Rationale: deleting a session drops the last reference to that workspace, so
   the user should be told where the files are instead of finding out later.

## 6. Client changes

1. `app_quick_session.js::start()`
   - keep the gesture critical section synchronous — `deselectSession()`,
     `changed()`, `closeSidebar()`, `textarea.focus()` — because iOS only opens
     the keyboard when the focus call happens inside the gesture; the new
     `await` must come *after* `focus()`;
   - `const reserved = await reserveQuickSession()` (new injected seam);
   - pass `reserved.cwd` and `reserved.name` into the existing
     `spawnSession(cwd, null, null, sessionName, …)` call.
     `spawnSessionWithCwd` already renames the launched session to
     `sessionName` once the catalog row for its broker pid appears, so the
     alias needs no new client code;
   - pending copy: `Starting Pi · model-b · low in a fresh workspace…` before
     the reserve resolves, `Starting … in Quick Session 07…` after; the button
     title/aria label loses its `~`: `Quick session — own workspace · Pi · …`;
   - `fail()` / launch-failure keeps prefilling the dialog with `cwd: "~"`
     (unchanged); nothing is left on disk because the allocator never created
     the directory;
   - keep `reserved` on the `pending` record so `inspectCatalog`/`ready` can
     name it.
2. `app_application_composition.js` + `app_wiring.js`: expose
   `reserveQuickSession` to the quick controller only
   (`select(deps, [ … 'reserveQuickSession'])`), implemented as a thin
   `api("/api/quick-sessions", {method: "POST", body: {}})`. The dialog
   controller does not receive it.
3. Delete toast (only with 5.6): the three delete call sites
   (`app_sessions.js`, `app_message_history.js`, `app_transcript_render.js`)
   share one helper that toasts `kept <leftover_dir>` when the response carries
   it. If that is one helper too many, drop 5.6 and this half together — the
   plan is complete without it.
4. Display: **no change** to `app_display.js` / `session_display.py` for real
   session rows — the alias stays the label and the cwd basename stays the
   fallback. The one extra rule is confined to *provisional launch-attempt
   rows* (`launch_ledger.py`, rows with `launch_state`): those have no alias by
   construction and their `display_name` is computed from the cwd alone, so
   they can take the derived quick name from `quick_workspace` and the row
   reads `Quick Session 07` from its first paint instead of `quick-07`. That
   confines the path convention to the one row class that can never have an
   alias, and leaves the documented single display authority intact for
   sessions.
   Without it, the provisional row shows `quick-07` for as long as it exists
   (the whole startup window, which for a pending/tmux launch is until the
   backend log binds) and the rename only lands afterwards. Plumbing: the
   derived name goes through `launch_attempt_row` (`launch_ledger.py:343`, an
   optional `quick_root` keyword) and the two module-level wrappers in
   `server.py` / `session_lifecycle.py` that pass `ServerConfig` constants
   today.

## 7. Verification

Docker-isolated and behavioral (extend `scripts/docker_quick_session_verify.sh`;
never the live deployment):

- VM test (`tests/test_quick_session.py`, existing harness): reserve is called
  exactly once per gesture and before `spawnSession`; the spawn call receives
  `reserved.cwd` as `args[0]` and `reserved.name` as `args[3]`; the copy shows
  the name after the reserve resolves and the placeholder before it; a reserve
  failure opens the preset dialog with the draft preserved and never calls
  `spawnSession`; double-fire still reserves once.
- New `tests/test_quick_workspace.py`: padding/name round trip; **slot 100
  renders `Quick Session 100` / `quick-100` and allocates cleanly**; a
  malformed or missing `next_slot` falls back to 1 while the clamp still
  protects existing directories; slot monotonicity across allocations; clamp to
  `max(existing)+1` when the state file is missing or stale; `is_quick_cwd`
  true for the slot dir, false for a same-prefix sibling, true through a
  symlinked root; allocate never creates a directory.
- Store test: `quick-NN` cwds never enter `recent_cwds` (write or read) while
  ordinary cwds still do.
- Browser/Docker: launch #1 → cwd `<HOME>/codoxear-quick/quick-01`, sidebar row
  and tmux window title `Quick Session 01`, launch body carried that absolute
  path; launch #2 → `quick-02` / `Quick Session 02` with both rows
  distinguishable; `ls <HOME>/codoxear-quick` shows only the two slot dirs; the
  dialog's cwd picker shows no `quick-NN` entries; deleting the empty #2 and
  launching again yields `quick-03` (no reuse) with no third directory left for
  a launch that failed; failure path → dialog prefills `~`, draft preserved, no
  new directory under the root; one CC quick launch whose first send reaches
  the backend (proves no fresh-directory trust modal); snapshot restore of a
  quick session keeps its `Quick Session NN` name (the alias remap already
  covers it — this pins it).
- The existing harness assertions `s['cwd'] == '/home/tester'` and
  `b['cwd'] == '~'` must be rewritten to the slot path; leaving them pins the
  old behavior.

## 8. Docs

- AGENTS.md, "Quick session launches on the gesture, never on Send": add the
  allocation / name / directory contract in two or three lines, pointing at
  `quick_workspace.py`.
- `app_quick_session.js` header comment: the cwd comes from the server
  allocator, not from `~`.

## 9. Non-goals

- No migration of quick sessions already living in `$HOME`; their dir and label
  stay as launched.
- No automatic pruning of quick directories, and no `git init` (it would make
  every directory non-empty and add state that cannot be safely pruned).
- No per-day or per-device numbering, no content-derived titles.
- No new UI surface: sidebar rows, tmux titles and notifications carry the
  name; there is no "quick sessions" pane or filter.
