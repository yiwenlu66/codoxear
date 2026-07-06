# Validation commands

Boundary rule: broker/server/session/tmux verification is Docker-only. Host throwaway `HOME` is not isolation; it cannot protect live brokers, tmux, `/tmp`, signals, or systemd. Agent-run validation must not use `pkill -f`, `killall`, or broad `pgrep | xargs kill`; container teardown is the cleanup boundary.

- Full local: `python3 -m pytest -q` (expect ~1349 passed, 136 subtests).
- Docker unit: `CODOXEAR_DOCKER_PORT=190NN scripts/codoxear-docker-sandbox test`
- Docker server smoke: `CODOXEAR_DOCKER_PORT=190NN scripts/codoxear-docker-sandbox smoke` (401 pre-login, 200 post-login).
- Browser: `AGENT_BROWSER_SESSION=<name> agent-browser open http://127.0.0.1:190NN/`, login `test-password`, snapshot/eval. Mobile viewport: agent-browser supports viewport sizing; verify both desktop and ~390x844.
- Monaco vendor caveat: range-level `git diff --check` over the Monaco introduction reports whitespace in vendored `codoxear/static/monaco/**` / third-party notice files. Treat that as third-party asset hygiene unless the touched diff changes non-vendor code or proof artifacts; do not rewrite vendored Monaco bytes casually after browser proof.
- Stop sandboxes after use: `CODOXEAR_DOCKER_PORT=190NN scripts/codoxear-docker-sandbox stop`.
