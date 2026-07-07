# EPISTEMIC

## Phenomenon
Codoxear exposes Codex, Pi, and Claude backend tabs in New Session. Recent accepted work proved parser/projection paths with synthetic sidecars, but that does not prove the browser New Session path: backend selection, launch request normalization, broker/session creation, log binding, send boundary, and visible transcript outcome.

## Live mechanisms
1. The path works end-to-end: the browser launches Claude Code with requested model/`max` effort, binds a log, sends a prompt, and renders an assistant answer or terminal backend error/no-response/recovery.
2. The backend cannot run in the Docker sandbox because credentials/CLI/config are intentionally absent; the correct product behavior is a truthful failed-launch row with disabled real-session actions and useful local Details/Copy/New-like-this behavior.
3. A defect exists if launch failure, log bind failure, send failure, or backend termination leaves the browser in silent idle, creates a misleading real-session row, or enables send/queue/attach/file actions against a failed launch row.

## Current claim
The highest-value uncertainty is not whether CC `max` is implemented; it is whether the user-visible backend tab path preserves the selectable-backend promise under current deployment constraints.

## Discriminating evidence
A Docker/browser run from the actual New Session UI can distinguish the mechanisms: launch metadata/argv, session list row type, transcript/tail outcome, failed-row action state, and broker call logs show whether the UI is truthful or silent.
