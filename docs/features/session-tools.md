# Session Tools

The Session Tools modal provides quick SSH commands and a live PTY tail for the selected session.

## Session tools modal
How users use it:
Click the session tools button in the topbar to copy commands or watch live output.

Effect:
The UI renders status/resume commands, optionally shows a tmux attach command, and polls `/api/sessions/<id>/tail` while the modal is open.

Files:
- `codoxear/static/app.js`
- `codoxear/static/app.css`
- `codoxear/server.py`
- `codoxear/broker.py`
- `scripts/codoxear-status`

Key flow:
1. Open modal for the selected session; render command strings.
2. Resume command is selected by session CLI metadata (`codex resume <id>` or `claude --resume <id>`).
3. If `tmux_name` is present, show `tmux attach -t <name>`; otherwise show "Not available".
4. Start tail polling every 1.5s; stop polling when the modal closes.

Call stack:
1. `showSessionTools` → `refreshSessionTail` in `codoxear/static/app.js`
2. `GET /api/sessions/<id>/tail` in `ServerHandler.do_GET`
3. `SessionManager.get_tail`
4. `Broker._handle_conn` (`cmd=tail`)

Notes:
- Tail output is sanitized server-side to strip ANSI/control sequences.
- The modal is disabled when no session is selected.
