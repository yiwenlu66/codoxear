# Validation commands

Boundary rule: broker/server/session/tmux verification is Docker-only. Host throwaway `HOME` is not isolation; it cannot protect live brokers, tmux, `/tmp`, signals, or systemd. Agent-run validation must not use `pkill -f`, `killall`, or broad `pgrep | xargs kill`; container teardown is the cleanup boundary.

- Full local: `python3 -m pytest -q` (expect ~1349 passed, 136 subtests).
- Docker unit: `CODOXEAR_DOCKER_PORT=190NN scripts/codoxear-docker-sandbox test`
- Docker server smoke: `CODOXEAR_DOCKER_PORT=190NN scripts/codoxear-docker-sandbox smoke` (401 pre-login, 200 post-login).
- Browser: `AGENT_BROWSER_SESSION=<name> agent-browser open http://127.0.0.1:190NN/`, login `test-password`, snapshot/eval. Mobile viewport: agent-browser supports viewport sizing; verify both desktop and ~390x844.

- Staged-upload acceptance requires evidence at the commit-boundary, not just route success: multi-file upload creates server staged entries and browser chips; remove-one/clear-all do not call backend `send` or `keys`; confirmed send produces exactly one backend send payload containing generated `Attachment N: <path>` lines and clears staged entries; commit-unknown/send failure preserves staged entries. A useful Docker proof uses a fake broker that supports confirmed send but not key-write errors to distinguish the new stage-only path from old pre-send key injection.
- Staged-upload producer acceptance requires exercising the real browser listeners and the real `/inject_file` network route: file paste/drop stage entries with zero broker `send`/`keys`; text-only paste is not prevented; off-composer file drop prevents navigation without staging; explicit send remains the only broker `send` boundary. Synthetic event dispatch is acceptable when OS clipboard/drag automation is unavailable, if `defaultPrevented`, server attachment API, and broker command logs are captured.
- Monaco vendor caveat: range-level `git diff --check` over the Monaco introduction reports whitespace in vendored `codoxear/static/monaco/**` / third-party notice files. Treat that as third-party asset hygiene unless the touched diff changes non-vendor code or proof artifacts; do not rewrite vendored Monaco bytes casually after browser proof.
- Stop sandboxes after use: `CODOXEAR_DOCKER_PORT=190NN scripts/codoxear-docker-sandbox stop`.
