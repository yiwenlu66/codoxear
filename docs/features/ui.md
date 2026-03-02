# Browser UI

The UI is a single-page app served from `codoxear/static/` that polls for sessions and renders chat from rollout logs.

## Session list and selection
How users use it:
Open the UI, pick a session, or create a new web-owned session.

Effect:
The UI polls `/api/sessions`, caches the selection, and calls `/api/sessions/<id>/messages` for the active session.

Files:
- `codoxear/static/index.html`
- `codoxear/static/app.js`
- `codoxear/static/app.css`

Key flow:
1. Load the session list and pick the remembered session.
2. Poll `/api/sessions` on an interval to refresh status.
3. Start message polling for the selected session.

Notes:
- The list shows busy/idle badges and queue length from broker state.
- Web-owned sessions display a delete button.
- New session creation prompts only for `cwd`; CLI comes from the sidebar CLI toggle button (`Codex`/`Claude`/`Gemini`), and duplicate-session keeps the source session CLI.
- The selected spawn CLI is cached in `localStorage` (`codexweb.spawnCli`) as the next default.
- Sessions are grouped by workspace (cwd) and sorted by display name within each workspace.
- Each session card shows a compact CLI logo chip (Codex/Claude/Gemini) before the title.
- Files are displayed once per workspace, not within each session row.
- The refresh button clears the selected session's local cache and re-fetches history.
- Refresh shows a brief toast confirmation on success.
- Sessions are grouped into workspaces by `cwd`, with each workspace showing the sessions and the files opened under that working directory.
- Workspace file history rows include per-file remove buttons and a clear action; both clear entries across all sessions in the workspace so the list stays in sync.
- Workspace file history rows use a consistent 30px row/button height so the remove `x` control stays visually uniform.
- Workspace headers include a `Close` action that batch-deletes web-owned sessions in that workspace; non-web sessions are kept and reported as skipped.
- Session badges (busy/queue/unread) are rendered in a dedicated inline-flex container to avoid Safari layout quirks with the unread dot.
- Sessions with saved drafts show a yellow `draft` indicator in place of the last user summary line.
- Each session row shows a one-line summary of the most recent user message, cached in `localStorage`.
- Sessions with unread assistant responses show a red dot indicator until the session is opened.
- The topbar toast does not reserve space when empty to avoid vertical misalignment.
- Duplicated sessions are auto-renamed with a `duplicate` suffix to make them easier to distinguish.
- The sidebar header includes a relay health indicator next to the Codoxear logo (breathing green when API calls succeed, yellow/red on errors or offline).
- The sidebar polls `/api/update` on a slow timer; when GitHub has newer commits, it shows an `Update` button and a one-time toast for that commit.
- The topbar session tools button copies SSH-friendly status/resume commands, shows a tmux attach command when available, and provides a live tail view for the selected session (ANSI/control sequences stripped for readability). Resume command switches by session CLI (`codex resume`, `claude --resume`, or `gemini --resume`). See `docs/features/session-tools.md` for details.

## Sending messages and local echo
How users use it:
Type a message and press Send. If the session is running, choose Send now or Send after current.

Effect:
The UI immediately appends a local echo bubble, then sends the prompt to the server.

Files:
- `codoxear/static/app.js`

Key flow:
1. Append a pending user message with a local id.
2. POST to `/api/sessions/<id>/send`.
3. Reconcile the pending bubble when a matching user event arrives from the log.

Notes:
- Pending matching uses normalized text to handle trailing whitespace differences.
- Pending local echoes are tracked per session, so switching sessions no longer drops unsynced user bubbles.
- Queued messages are shown in the queue viewer (not as chat bubbles) until they are drained and logged.
- Queued messages are stored server-side via `/api/sessions/<id>/queue`, so they continue even if the browser closes. The broker releases one queued message after each turn end (or idle fallback) to avoid interrupting active replies.
- In the send-choice dialog, `Send after current` clears the composer only after queue push succeeds; if queue push fails, the text stays in the composer.
- The queued-message editor skips list re-renders while a queue textarea is focused to prevent IME interruptions; the list refreshes once focus leaves.
- Composer drafts are stored per session in `localStorage` and restored when switching sessions.

## Polling and chat rendering
How users use it:
The UI keeps the chat view updated as the server returns new log events.

Effect:
Events are merged into a windowed DOM cache, with a small local cache in `localStorage` used as a fallback if the initial history fetch fails.

Files:
- `codoxear/static/app.js`

Key flow:
1. Poll `/api/sessions/<id>/messages` with `offset`.
2. Update the chat list and a small local cache.
3. Track idle vs running and render typing dots.

Notes:
- The UI uses an `init=1` fetch to seed recent history on selection, plus an older-messages pager.
- The UI suppresses near-duplicate user/assistant events (same role + text within a short window) to guard against occasional double renders.

## Attachments and interrupts
How users use it:
Attach images or interrupt a running session.

Effect:
Images are converted to JPEG if needed and sent via `/api/sessions/<id>/inject_image`. Interrupt sends an ESC sequence.

Files:
- `codoxear/static/app.js`
- `codoxear/server.py`

Notes:
- Images are resized and compressed client-side to a 10 MB limit.
- Interrupt button only shows while the session is running.
- Pasting an image into the message box attaches it (Safari supported via clipboard files).
- Upload-path placeholder lines emitted during image attach are filtered out of user chat messages.

## File viewer edit and preview
How users use it:
Open a file, toggle Edit to modify, and Preview to render markdown.

Effect:
The UI reads files with `/api/files/read`, writes changes with `/api/files/write`, and renders markdown client-side.

Files:
- `codoxear/static/app.js`
- `codoxear/server.py`

Notes:
- Edit mode uses a full-width editor; on small screens the font size is forced to 16px to avoid mobile input zoom.
- Edit and preview modes share the same modal sizing as view for consistent layout.
- On small screens the file viewer height tracks `--appH` (visual viewport) so the editor stays visible above the iOS keyboard.
- On desktop the file viewer uses a consistent height across view/edit/preview so the editor does not collapse.
- Markdown preview renders Mermaid code fences (language `mermaid`) as diagrams when Mermaid is available.
- The file viewer can pop out into a dedicated full-screen tab via the "Pop out" action, or by using `?file=...&session_id=...&mode=edit&fullscreen=1` in the URL.
- The file viewer header includes "Copy path" and "Copy name" actions for the current file.
- Full-screen pop-out tabs set the browser tab title to the current filename.
- Full-screen mode uses distinct header/status bars, hides the inline open-row, and keeps view/edit/preview content wrapped in a card so it no longer looks like the floating modal. Use the header "Open" action to enter a new path.
- Full-screen edit view keeps the editor width aligned with the wrapped card so it does not bleed past the right edge.
- The full-screen file viewer polls for on-disk changes and refreshes automatically when the file updates. Auto-refresh pauses while there are unsaved edits to avoid clobbering local changes.
- Auto-refresh reads pass `record_history=false` so background refreshes do not re-add removed file history entries.
- Full-screen mode shows a small sync indicator: green breathing dot when live, yellow/red when the network is unavailable, and gray when manual/paused.
- Markdown preview preserves ordered-list starting numbers (e.g. lists that begin with `2.` keep their numbering).
- Markdown preview renders basic pipe tables with header alignment and a scrollable wrapper.
- Reloading a file preserves the current scroll position in view/edit/preview mode.
- On small screens the file viewer header pins the Close button so it stays accessible when actions wrap.
- On small screens the file viewer header stacks and the action row becomes horizontally scrollable to avoid crowding.
- On iOS Safari the file viewer open buttons use touch-first handlers (`touchend` + `click`) so taps reliably open the modal.
