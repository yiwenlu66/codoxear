# Multi-CLI Support

Codoxear supports three first-class CLIs in the same runtime:

- Codex
- Claude Code
- Gemini CLI

The UI, server, broker metadata, resume helpers, and chat parsing pipeline all use one shared model keyed by `cli`.

## Why this matters

- One deployment serves mixed teams/projects without separate frontends.
- Session discovery and chat history work for terminal-owned and web-owned sessions across all three CLIs.
- Queueing and busy/idle behavior stay consistent even though upstream log formats differ.

## Gemini as a major feature

Gemini support is implemented as native support, not as a compatibility shim:

- New-session spawn accepts `cli=gemini`.
- Broker/session metadata carries `cli: "gemini"` for session tools and resume commands.
- Log discovery supports `~/.gemini/tmp/**/chats/session-*.json`.
- Gemini chat JSON is mapped into unified `user` / `assistant` events.
- Turn-end markers are synthesized for Gemini completion rows; thinking/tool-only rows intentionally keep the turn open so long reasoning is not marked idle.

## Runtime configuration by CLI

Server-side web session spawn sets CLI-specific env:

- Codex: `CODEX_HOME`, `CODEX_BIN`
- Claude: `CLAUDE_HOME`, `CLAUDE_BIN`
- Gemini: `GEMINI_HOME`, `GEMINI_BIN`

`GEMINI_BIN` allows host-specific wrappers without changing application code.

## Gemini all-approve mode (host pattern)

For web-owned Gemini sessions that should auto-approve actions, set `GEMINI_BIN` to a wrapper:

```bash
#!/usr/bin/env bash
exec gemini --approval-mode yolo "$@"
```

Recommended host setup used in this repo environment:

- Wrapper path: `/usr/local/bin/gemini-web`
- Supervisor env: `GEMINI_BIN=/usr/local/bin/gemini-web`
- Restart daemon: `supervisorctl restart codoxear`

This keeps behavior scoped to Codoxear-launched Gemini processes and avoids invasive code changes.
