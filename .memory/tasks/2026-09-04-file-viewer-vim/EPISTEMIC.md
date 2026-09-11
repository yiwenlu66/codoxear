# EPISTEMIC — current model

## Phenomenon

File-viewer vim keybindings (plan commits 1–6) are implemented, tested at the VM-behavior level, and committed. The app-wide Esc-never-dismisses policy is in force.

## Live mechanisms (evidence → OPS.md)

- Esc policy: the only modal-dismissal Escape handlers that existed were editor-ops (removed), chat-search input (removed), new-session document handler (removed). All remaining Escape consumers are transient-mode exits or dropdown/focus management, classified in OPS.md.
- Vim layer (`app_file_vim.js`): one capture keydown listener registered before `bindFileEditorInteractions`; guard chain viewer-open → blocking-modal → hint-mode → touch-select → text-entry-target. Consumed keys `preventDefault` + `stopImmediatePropagation`.
- readOnly is single-sourced: `fileEditorCapabilities` includes `!vimNormalActive()`; verbs lift readOnly transiently and re-sync. `activeFileEditorInsertWritable()` is the gate-ignoring variant verbs use; `activeFileEditorInsertIdleTextWritable()` is the same suspension for the Ctrl-S save gate (vim normal must still save).
- Edit-mode flips notify the vim layer (`onFileEditModeChanged` from `setFileEditMode` and `resetActiveFileBufferState`) → sub-mode resets to insert, chip re-renders through one writer.
- `f` calls hint `enter()` explicitly (Monaco's focused textarea otherwise blocks activation); the vim layer defers entirely while hint mode is active.
- Key-dispatch invariants (fix-up round): verbs/prefixes require unmodified keys (Ctrl/Alt/Meta fall through); pending `g`/`d` reject modified completion keys and clear; Space passes through unconsumed in view+normal (Close-button activation); `dd` on the last line deletes the preceding line break; pendingPrefix clears on every Escape and on viewer close (finishHide → `syncEditMode`).

## Ruled out / dead

- "Esc closes viewer" semantics (user-rejected, plan §requested behavior 5).
- Raw `setScrollTop` half-page on cursor surfaces (deviation, see OPS).
- Vim char-find `f`/`t` — permanently displaced by user decision.

## Anomalies / open questions

- None unresolved at the module level. Unverified in a real browser: Monaco cursor visibility on first motion (focus-before-trigger assumed sufficient), hint badges over viewer buttons with Monaco focused (mechanism wired; needs Docker/agent-browser pass).
- `key === "$"` assumes US-layout shift+4; other layouts won't emit `$` for that chord (inherent to key-based matching; same as the plan's key set).

## Most model-changing next step

Deploy (`scripts/deploy.sh cd203ca6`) when the user asks; the fix-up round's Docker smoke gate passed on the rebuilt bundle. A real-browser pass of the fixed keys (Ctrl-S in NORMAL, Space on Close, d/Ctrl-d, dd last line) through a live session remains the only unexercised surface.
