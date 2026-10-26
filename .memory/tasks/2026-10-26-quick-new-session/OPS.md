# Evidence trail

## 2026-10-26T21:10:43+08:00 — Initial inspection
- Read AGENTS.md, task PROMPT.md, project architecture and validation memory. Existing unrelated task-memory edits and review artifacts are present; no source edits were present. They will not be staged.
- Confirmed dialog `setBackend` writes last-shown backend; `start` persists provider/model before launch, but effort/fast/tmux have no persisted launch record.
- Lifecycle `openSession` notifies selected subscribers before loading the local composer draft. A quick-start handoff must capture before selection, then restore after the local load through the existing draft writer.
- `spawnSessionWithCwd` currently selects unconditionally when broker appears, and returns a launch id for pending launches. Quick launch needs an explicit selection callback and catalog matching for the pending case, preventing a later launch from stealing a user's manual selection.
- Composer disables textarea without selection. Its owning writer must subscribe to quick-start state so typing is enabled during startup while Send stays disabled.
- Server launch plan calls `resolve_dir_target(request.cwd, field_name="cwd")`; validation will prove `~` becomes the container HOME through the actual API.
