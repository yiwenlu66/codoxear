# EPISTEMIC — default session name

## Phenomenon explained

An unnamed codoxear session displayed a name in the sidebar (cwd basename
fallback) but had **no** session name in the backend. Consequences: Pi's
own identity (`session_info`, TUI footer/title, `pi --resume` rows) was empty,
and Pi's intercom presence / subagent orchestrator target fell back to the
generated `subagent-chat-<session id>` label; Codex threads and Claude Code
resume rows stayed untitled.

## Mechanism (live, now implemented)

- The name authority is the alias store (`session_aliases.json`, per registry
  session id). `sidebar_display_name` renders alias-else-basename; the backend
  title is a projection of the alias (`SessionNameProjection.reconcile`).
- The bug was a hole in that chain: nothing ever *put* a name into the store
  unless the user renamed. The fallback lived only in the renderer.
- Fix: `reconcile` now materializes a default (cwd basename; quick workspaces
  use their declared label; `-N` suffix against names live sessions hold) via
  `SessionUiStateCoordinator.alias_assign_default`, then projects it. One
  producer, one store, so sidebar / tmux title / backend title cannot disagree.
- Wrong-default risk handled by provenance: the applied record carries
  `origin: "default"` (including on failed/unsupported pushes), so a title the
  user types in the backend's own TUI supersedes an assigned default through the
  existing adoption path; a name chosen in codoxear never yields. An explicit
  clear (`""` entry) is sticky and suppresses defaults.

## Ruled out and why

- **`--name` at launch (broker/server argv).** Covers Pi's birth moment but is a
  second producer: its value can disagree with the store's derivation (which
  sees aliases and other sessions) and cause `session_info` churn or duplicate
  names; terminal-launched brokers and Pi `/new` get nothing from it either.
- **Deriving the default on read** (no store write). Numbering would be
  recomputed from the live set, so names renumber as sessions die, and the
  backend would need its own derivation.
- **Discovery-time assignment.** Rejected as a second write site; the product
  reads (and therefore reconciles) a session when it is opened, which is also
  when its backend title matters. Residual gap: a fresh session never opened in
  the browser keeps an empty backend title until first view — accepted, since Pi
  persists its title in the transcript once set.
- **Counting a still-unnamed session's cwd basename as "taken"** (first rule).
  Measured via the integration test: it inverts numbering (older session gets
  `-2`). Replaced by "assigned names only".

## Anomalies / open questions

- None observed. The pre-existing `test_hint_mode_coverage::draftUndoBtn`
  failure is unrelated (fails on a stashed tree too).
- Untested behaviorally: numbering across two real sessions in one cwd
  (covered by the store-level and integration tests, not the Docker script).

## Current claim

A session with no user-chosen name gets a meaningful default name in both
codoxear and its backend, at the first reconcile after the transcript binds, and
the user's own (UI or terminal) name still wins. Confidence: high — one real Pi
session verified end to end in Docker; rule and store behavior pinned by tests.
