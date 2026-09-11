# File viewer vim keybindings — implementation plan

## Requested behavior

1. Vim-style navigation (j/k/h/l/w/b/e etc.) in view (non-edit) mode.
2. `f` activates hint-mode letter badges over the viewer's clickable buttons.
3. Edit mode starts in "insert" sub-mode (normal Monaco typing).
4. `Esc` in edit mode switches to vim "normal" sub-mode; `i` switches back to insert.
5. **Global policy change (user-stated design rule): `Esc` must not close any modal dialog, anywhere in the app.** Rationale: Esc-closes-modal causes mis-operations. This overrides the earlier plan note where Esc in normal mode closed the viewer.

## Key architectural facts (verified by code exploration)

- View-mode File/Diff rendering is a **read-only Monaco editor** in `#fileDiff`, not a `<pre>`. Monaco shows and moves a cursor even when `readOnly: true`. Markdown preview is a scrollable DOM div (`.fileMarkdownPreview`); PDF renders canvases into `#fileDiff`; images/video use `#fileImage`/`#fileVideo`.
- Keyboard routing has **no central router**: document capture listeners (`app_file_editor_ops.js:44-69`) run first, then element listeners (picker), then document bubble listeners in registration order: hint mode (`app_hint_mode.js:282`) → modal letter activation (`app_transcript_render.js:293`) → direct shortcuts (`app_transcript_render.js:299`, disabled while viewer is open).
- **Collision**: `activateModalButtonForKey` (`app_modal.js:211-228`) currently activates viewer buttons by distinctive letter — typing `e` in the viewer clicks "Edit file" today. Any vim layer must consume keys before this bubble listener.
- Hint mode **already collects viewer controls** when the viewer is open (`app_hint_mode.js:146-157`, viewer is a modal isolation target). But activation is refused when the event target or `document.activeElement` is a text-entry element (`app_hint_mode.js:240-242`) — and Monaco's hidden `.inputarea` textarea counts. So `f` is effectively dead in the common case (Monaco focused).
- Existing capture Escape handler (`app_file_editor_ops.js:49-69`) closes/cancels: `#appConfirm`, file paste dialog, unsaved dialog, the file viewer itself, and send-choice/queue/help/diagnostics/voice-settings/session-edit/new-session dialogs. It does **not** check `defaultPrevented`, and two capture listeners on `document` are same-node/same-phase, so precedence requires explicit delegation, not just ordering.
- Touch-selection mode (coarse pointers) already consumes `h/j/k/l` (`app_file_ops.js:177-208`). The vim layer must defer to it.
- Nested dialogs (unsaved-changes, paste, confirm) have pinned keyboard precedence (`tests/test_modal_keyboard_nested_target_order.py`). The vim layer must not intercept while one is open.
- No `keyup` listeners exist; pending-prefix state (`gg`) needs an explicit reset policy.
- Modal isolation (`inert` on the app root) means every keydown while the viewer is open originates inside the viewer — the vim layer can scope on "viewer open" alone, plus target guards.

## Global Esc policy (new, app-wide)

**Rule: `Esc` never closes or cancels a modal dialog.** This is a semantic invariant, not a per-dialog tweak.

What changes:

- `app_file_editor_ops.js:49-69`: the entire Escape dismissal branch is removed — no more Esc-close for `#appConfirm`, paste dialog, unsaved dialog, file viewer, send-choice, queue, help, diagnostics, voice settings, session edit, new-session. The `#appConfirm` **Tab focus trap stays** (it is focus management, not dismissal).
- Every dialog already has an explicit close/cancel button; keyboard users reach them via distinctive-letter activation (`activateModalButtonForKey`, which stays) or `f`-hints.
- **Audit step (required during implementation)**: grep all `Escape`/`key === "Escape"` handlers app-wide and classify each as *modal dismissal* (remove) vs *transient-mode exit* (keep). Known keeps:
  - Hint mode: Esc exits hint mode (`app_hint_mode.js:252-255`) — transient key-capture mode, not a dialog.
  - File picker menu: Esc closes the dropdown (`app_file_picker.js:786-791`) — dropdown, and focus stays in the input.
  - Composer model picker: Esc closes the picker (`app_composer.js:602-607`) — dropdown.
  - Composer textarea blur on Esc (`app_composer.js:608-611`) — focus management, not a modal.
  - Touch-selection mode: Esc exits selection (`app_file_ops.js:185-189`) — transient mode.
  - Chat search and any other Escape consumer found in the audit must be classified explicitly; chat search close-on-Esc (if present) is a dismissal and goes.
- Help overlay text (`app_shell.js:397-399`) documents Escape behavior — must be rewritten to state the new rule.
- Existing tests that pin Esc-closes-modal behavior must be updated to pin Esc-does-NOT-close (behavior reversal is asserted, not just deleted). Likely affected: `tests/test_file_viewer_source.py` (viewer Esc), modal keyboard suites, composer focus tests. The audit lists them all before changes.

**Consequence for the viewer vim chain**: Esc in insert mode → normal (vim mode transition, not a modal dismissal — stays). Esc in normal mode must **not** close the viewer. New semantics below.

## Viewer vim design

### State model

One new state axis, owned by one new module:

```
viewerMode: "view" | "edit"        (existing fileEditMode — unchanged owner)
vimSubMode: "insert" | "normal"    (new; only meaningful when viewerMode === "edit")
```

Invariants:

- Entering edit mode ⇒ `vimSubMode = "insert"` (request 3). Exiting edit mode resets to view; the sub-mode is forgotten.
- `Esc` in insert ⇒ normal (consumed). `Esc` in normal ⇒ **exit edit mode back to view mode** — but only when the buffer is clean; when dirty, Esc is a no-op (status hint "unsaved changes"), because silently discarding edits on a stray Esc is exactly the mis-operation class the user is eliminating. This replaces the original "Esc closes viewer" idea, which the user explicitly rejected.
- `i` in normal ⇒ insert. Companions `a`/`A`/`o`/`O` included in v1.
- In view mode there is no sub-mode; vim keys act directly.
- Closing the viewer is keyboard-reachable via `f`-hint on the Close button (and mouse/backdrop as today). The unsaved guard on that path is unchanged.

### New module: `app_file_vim.js` — `createFileVimController(options)`

Follows the widget rule (element + render + subscription in one module) and the wiring guard (per-controller `select()` contract in `app_wiring.js`, instantiated in `app_file_ops.js` composition).

Owns:

1. **A document capture-phase keydown listener**, registered via `addAppEvent` **before** `bindFileEditorInteractions()`. Guard chain (first match wins):
   - viewer not open → return
   - unsaved/paste/confirm dialog open → return (nested-dialog precedence)
   - hint mode active → return (hint mode's bubble listener owns the keys)
   - touch-selection mode active → return (existing hjkl owner)
   - event target is a text-entry element **other than** Monaco's `.inputarea` inside `#fileDiff` (i.e. picker input, paste textarea) → return
   - dispatch by (viewerMode, vimSubMode, key)
   - on consume: `preventDefault()` + `stopImmediatePropagation()` (same-phase sibling listeners in `app_file_editor_ops`)

2. **Motion dispatch** against the active Monaco editor (view File/Diff mode, edit normal mode):
   - `h`/`l`/`j`/`k`: `editor.trigger("file-vim", "cursorLeft"/"cursorRight"/"cursorDown"/"cursorUp", null)` — cursor commands are permitted in read-only Monaco. Diff view: drive `getModifiedEditor()`.
   - `w`/`b`/`e`: Monaco word-motion cursor commands (word-start right/left, word-end right — verify exact command IDs in the vendored Monaco at implementation time).
   - `0`/`$`: `cursorHome`/`cursorEnd`.
   - `gg`/`G`: `setPosition` + `revealPosition` top/bottom; `g` sets a pending-prefix flag consumed by the next key and cleared by any other key, mode change, or nested-dialog open. No timeout in v1 (vim has none).
   - `Ctrl-d`/`Ctrl-u`: `setScrollTop(±viewportHeight/2)`.
   - First nav key calls `editor.focus()` so the cursor is actually visible (Monaco hides the cursor in a blurred editor).

3. **Scroll fallback** for non-cursor surfaces (markdown preview, PDF, image, video, plain-text fallback): `j`/`k`/`d`/`u`/`gg`/`G` scroll the active scroll element (`.fileMarkdownPreview` or `#fileDiff`); `w`/`b`/`e`/`0`/`$` are no-ops there.

4. **Insert-mode text entry in normal mode** (edit mode only):
   - `i`: readOnly → false, focus, sub-mode → insert.
   - `a`: cursor right one, then as `i`. `A`: `cursorEnd`, then as `i`.
   - `o`/`O`: temporary readOnly toggle, `executeEdits` newline below/above, then as `i`.
   - `x`, `dd`, `u`, `Ctrl-r`: temporary readOnly toggle + `executeEdits`/`trigger("undo"/"redo")`, restore readOnly.
   - In normal mode Monaco is `readOnly: true` **and** the capture listener stops unconsumed printable keys, so nothing reaches the editor.

5. **Explicit hint entry**: `f` in view mode or edit-normal mode calls the hint controller's `enter()` (expose it on the hint controller's public API if not already exposed) and consumes the key. This bypasses the text-entry activation guard that currently kills `f` when Monaco is focused. Hint labels are unconsumed by the vim layer because guard 3 defers while hint mode is active. **`f` belongs to hints everywhere outside insert mode** — vim char-find (`f`/`t`) is permanently displaced by this decision (see roadmap).

6. **Mode chip** (`#fileVimMode`): small square monospace badge in the viewer header showing `NORMAL`/`INSERT`, visible only in edit mode; single writer function called on every transition (state-authority principle). Element created in `app_shell.js`, styled in `app.css` per the paper design language (square, `--font-mono`, ink-on-paper; no new spacing tokens).

### Escape integration (explicit, not ordering-dependent)

The vim controller exposes `handleEscape()` returning true when it consumed Esc (insert→normal, or normal→view-mode exit when clean). Any residual Escape routing consults it first. With the global Esc-dismissal branch removed from `app_file_editor_ops.js`, there is no competing close-viewer path left to preempt; the capture-order requirement reduces to "vim capture listener runs before Monaco sees keys," which capture phase guarantees against Monaco's bubble/target handlers.

### Interaction changes (behavior deltas to call out)

- **Esc no longer dismisses any modal dialog app-wide.** Close/cancel buttons, distinctive-letter activation, `f`-hints, and backdrop clicks remain. This is the largest user-visible change in the plan and touches dialogs unrelated to the file viewer — it is the user's stated rule, implemented exactly.
- **Direct letter activation of viewer buttons is replaced by `f`-hints** while the viewer is open (except nested dialogs, which keep distinctive-letter activation). Today `e` opens Edit; after this change `e` is word-end motion and `f`+letter activates buttons.
- Monaco's textarea no longer blocks `f`; hint mode becomes reliably reachable in the viewer.
- View-mode `j`/`k` move a **cursor**, not a scroll offset, in File/Diff view (vim semantics). Scroll-surface behavior applies only where no cursor exists.

### Files touched

| File | Change |
|---|---|
| `codoxear/static/app_file_vim.js` | New controller (state, capture listener, motions, chip render, `handleEscape`) |
| `codoxear/static/app_wiring.js` | `createFileVimOptions` select contract |
| `codoxear/static/app_file_ops.js` | Instantiate vim controller; expose active-editor accessor + touch-selection state to it |
| `codoxear/static/app_file_editor_ops.js` | **Remove the entire Escape dismissal branch** (global Esc policy); keep the `#appConfirm` Tab trap |
| `codoxear/static/app_file_viewer_operations.js` / `app_file_mode.js` | Notify vim controller on edit-mode enter/exit (default insert); dirty-check support for Esc-in-normal |
| `codoxear/static/app_hint_mode.js` | Expose `enter()` publicly if not exposed |
| `codoxear/static/app_shell.js` | `#fileVimMode` chip element; help-overlay entries for viewer vim keys **and rewritten Esc documentation** |
| `codoxear/static/app.css` | Chip styles (square, mono, ink) |
| Other `Escape` consumers found in the audit (e.g. chat search) | Remove modal-dismissal behavior per classification |

No new allowlist entries expected in `scripts/wiring_guard_allowlist.json`; pinned counts in `tests/test_check_wiring_guard.py` unchanged (verify).

## Testing plan

New VM behavior suite `tests/test_frontend_file_vim_module_source.py` (harness pattern from `tests/test_frontend_hint_mode_module_source.py`, fake DOM + fake Monaco editor recording `trigger`/`setPosition`/`updateOptions` calls):

- Edit-mode entry defaults to insert; `Esc` → normal (chip text, readOnly true); `i` → insert.
- Esc in normal with clean buffer exits edit mode to view; with dirty buffer is a no-op.
- View-mode `j`/`k`/`h`/`l`/`w`/`b`/`e`/`0`/`$` dispatch the expected Monaco cursor commands; `gg` two-key sequence works, `g` then other key resets.
- Scroll fallback: markdown-preview surface receives scrollTop deltas; word motions are no-ops.
- Guards: keys pass through untouched when picker input focused, nested dialog open, hint mode active, touch-selection active.
- `f` calls hint `enter()` and does not reach Monaco.
- Normal-mode printable keys never reach Monaco (stopImmediatePropagation observed).

Global Esc policy tests:

- **Reverse-pinned**: for each previously Esc-dismissable dialog (viewer, unsaved, paste, confirm, help, queue, diagnostics, voice settings, session edit, new session, send-choice), a behavior test asserts Esc leaves it open; the kept transient exits (hint mode, picker menu, model picker, composer blur, touch selection) assert Esc still works there. Update the existing tests that pinned the old behavior rather than adding parallel files where a suite already owns the dialog.

Existing suites that must still pass (regression): `test_frontend_hint_mode_module_source.py`, `test_hint_mode_coverage.py` (hint map pins — chip is non-interactive, no hint), `test_frontend_modal_keyboard_module_source.py`, `test_modal_keyboard_nested_target_order.py`, `test_frontend_file_viewer_module_source.py`, `test_frontend_extracted_controller_modules.py`, `test_file_viewer_source.py`, `test_check_wiring_guard.py`.

Run: `.venv/bin/python -m pytest -q tests/<the above>` then full suite; `python3 scripts/check_wiring.py`.

**Behavioral verification (mandatory per AGENTS.md)**: `bash scripts/docker_verify.sh HEAD` for the smoke gate, plus a Docker-isolated agent-browser session that: opens the viewer and presses `j`/`k` with visible cursor motion; presses `f` and sees badges on viewer buttons; enters edit mode and types immediately (insert default); presses `Esc`, sees NORMAL chip, moves cursor with `hjkl`; presses `i`, types; confirms **Esc never closes** the viewer, the unsaved dialog, or the help overlay; closes the viewer via `f`-hint on Close with the unsaved guard intact when dirty. Never against the live deployment.

## Sequencing (atomic commits)

1. Global Esc policy: remove the dismissal branch, audit/remove other Escape dismissals, reverse-pin tests, help text rewrite.
2. Vim controller skeleton: state, chip, Esc/i/normal→view transitions, edit-mode default insert + tests.
3. View/normal-mode motions on Monaco surfaces + tests.
4. Scroll-surface fallback + tests.
5. `f` explicit hint entry + hint-mode `enter()` exposure + tests.
6. Help overlay viewer entries, wiring guard run, full test suite.
7. Docker behavioral verification; then `scripts/deploy.sh <commit>` per deployment policy.

Commit 1 lands first deliberately: it is an independent user-visible policy change, and the vim chain builds on the post-policy Esc semantics.

## Roadmap beyond v1

**v2 — count prefixes and operator-pending motions.** A real pending-buffer parser replaces the single `g` flag: digits accumulate a count (`5j`, `3dd`, `2w`), applied uniformly to motions and verbs. New motions: `{`/`}` paragraph, `%` matching bracket (Monaco bracket-matching API), `Ctrl-f`/`Ctrl-b` full page, `zz`/`zt`/`zb` scroll-centering (`revealLineInCenter` etc.). The pending state machine gets a `statusline`-style echo in the chip area (e.g. `12` while typing a count) so partial input is visible.

**v2 — visual mode.** `v`/`V` enter charwise/linewise visual selection using Monaco selections (`setSelection`); motions extend the selection head; `y` copies via clipboard API, `d`/`x` delete the selection through the readOnly-toggle path, `Esc` exits visual to normal. Visual mode adds a third chip state (`VISUAL`).

**v2 — yank/paste.** `y` (with motion/selection), `p`/`P` paste after/before cursor via `executeEdits` + clipboard read (permissions-gated; falls back to the internal register when clipboard read is denied — internal register is also what makes `dd` then `p` work without clipboard permission).

**v2 — in-viewer search.** `/` in view/normal mode opens Monaco's find widget (`editor.getAction("actions.find")`) for Monaco surfaces; `n`/`N` repeat. For scroll surfaces, a minimal find-in-page bar is a larger piece — defer to v3 unless the find widget covers enough.

**v3+ — explicitly speculative, do not plan in detail now**: marks (`m`/`'`), macros (`q`/`@`), custom text objects, a `:`-command line (e.g. `:w` save, `:q` close-with-guard — natural fit given the no-Esc-close policy).

**Permanently rejected**: vim char-find `f`/`t`/`F`/`T` — `f` belongs to hint mode outside insert mode by user decision; `t` alone without `f` has no vim coherence, so the whole family stays out.

## Scope decisions (confirmed with user)

- v1 key set: `hjkl wbe 0 $ gg G Ctrl-d/u`, insert entries `i a A o O`, verbs `x dd u Ctrl-r`, `f`-hints, mode chip.
- Esc never closes any modal dialog, app-wide. Esc chain in the viewer: insert → normal → (clean only) view mode; dirty buffer makes Esc in normal a no-op.
- `f` = hints everywhere outside insert mode; vim char-find permanently displaced.
- View-mode `j`/`k` move the Monaco cursor (focusing the editor on first key); scroll fallback only on cursor-less surfaces.

---

## Outcome (implemented 2026-09-05)

Shipped as 11 commits, 921323d1..cd203ca6. Full suite 1768 passed + wiring guard, independent critic review, and Docker-isolated browser verification all green (cursor motion, f-hints, insert default, NORMAL/INSERT chip, x, Ctrl-S in normal, Esc-closes-nothing). Known residuals: (1) docker/verify.Dockerfile copies /workspace root-owned while app runs as tester — file-write 403s until ownership repaired container-locally; (2) two unexplained HTTP 409 save-conflict console entries during verification, no UI failure, not root-caused; (3) `$` requires shift+4 (layout-inherent). Deploy pending user decision: scripts/deploy.sh cd203ca6.
