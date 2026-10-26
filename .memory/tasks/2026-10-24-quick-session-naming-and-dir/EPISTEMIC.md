# EPISTEMIC — quick sessions get a name and their own workspace

## Phenomenon

A quick launch (`r` / `N` / "Quick start") used to run in `$HOME` with no alias.
The display authority (`session_display.sidebar_display_name`) names a row by
alias else `basename(cwd)`, so every quick session was titled with the account
name — indistinguishable from each other and from any session opened in `~` —
and all of them shared one cwd.

## Shipped mechanism

One slot per quick launch, and the slot is the identity:

- `codoxear/quick_workspace.py` owns the format (`quick-07` ⇄ `Quick Session 07`,
  minimum-width padding), containment, the counter
  (`<app_dir>/quick_sessions.json` → `{"next_slot": N}`), and the
  dirs-are-the-floor clamp (`max(counter, highest existing slot + 1)`).
- `POST /api/quick-sessions` reserves; it creates no directory and spawns
  nothing. The launch path already mkdirs a missing cwd, so a failed start leaves
  no filesystem state and only burns a number.
- The label is an alias, set by the client's existing
  `spawnSessionWithCwd(..., sessionName)` rename after the broker-pid row
  appears; the directory basename (`quick-07`) is the fallback label.
- Recent-cwd entries under the root are dropped on write and read; provisional
  launch-attempt rows derive their label from the slot; delete trims an empty
  workspace with `rmdir`.

## Live vs rejected alternatives

- Reuse/smallest-free numbering: rejected because abandonment and
  just-allocated are both an empty directory; any rule that guesses wrong puts
  two sessions in one directory under one name. Monotonic has no failure mode.
- Deriving the display name from the cwd for real rows: rejected to keep
  `session_display` the single authority; the one exception is confined to
  launch-attempt rows, which can never have an alias.
- Env knob for the root: rejected by explicit user decision; relocation is a
  symlink, which is why containment resolves both sides.
- Rollover past 99 or a code-wide 3-digit pad: rejected; `{slot:02d}` is a
  minimum width, so 100 is `quick-100`. Only `ls` lexicographic order breaks.

## Evidence

Docker browser run against 33889b8d: `PASS` (`/tmp/codoxear-quick-verify.FoJ4H4`)
— cwd `/home/tester/codoxear-quick/quick-01`, alias/display/sidebar text
`Quick Session 01`, launch body carried the workspace path, `quick_dirs ==
['quick-01']` after two deliberately failed starts, `recent_cwds == ['/tmp']`,
draft handoff + broker ACK + mobile drawer/focus green, 45 in-container tests.

Host suite: 2613 passed, 1 failed — `draftUndoBtn` hint coverage, reproduced at
b7544dcb, pre-existing.

Deployed 60505805 from the snapshot; `/` 200, `GET /api/sessions` 401,
`POST /api/quick-sessions` 401 while an unknown POST path is 404 (route
registered and auth-gated), bundle contains `reserveQuickSession`, and all live
brokers predate the restart.

## Open boundary

Snapshot restore of a quick session keeping its label relies on the existing
alias remap; it is reasoned, not measured. Attachments on a swallowed first send
remain uncovered (unchanged by this feature).
