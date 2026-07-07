# Manifest

Artifact root: `.memory/tasks/2026-07-07-selectable-backend-live-parity/browser-artifacts/backend-parity-19396/`

Committed artifacts are reduced to evidence needed for the failed-launch parity claim. Login credential values, cookie jars, auth headers, runtime private file contents, and generated app private-file inventories are excluded.

## Key evidence

- `VERIFICATION-REPORT.md` — reviewed proof summary.
- `api/sessions-after-failure.json.pretty` — `/api/sessions` failed row and launch defaults.
- `api/failed-launch-tail.json.pretty` — failed transcript projection.
- `api/*failed-row.status` and selected `api/*failed-row*.json` — failed-row API action blocking.
- `browser/snapshot-*.txt` — accessibility snapshots from the actual New Session and failed-row UI flow.
- `browser/eval-*.json` — browser state probes for failed-row controls, details, copy-details, and sanitized New-like-this state.
- `browser/probe-*.js` — JS probes used for browser state extraction.
- `container/runtime-evidence.txt` — Claude executable absence, launch ledger, socks/process/tmux state, with generated secret filenames omitted.
- `container/cc-launch-plan-sonnet-max.txt` — backend launch argv for `sonnet` + `max`.
- `container/docker-logs-after-browser.txt` — server log line for launch failure.
- `docker/preflight.txt`, `docker/smoke-start.txt`, `docker/docker-focused-test.txt`, `docker/local-focused-pytest.txt`, `docker/stop.txt`, `docker/ps-after-stop.txt` — validation lifecycle outputs.
- `manifest.files.txt` — generated committed file list with byte sizes.
