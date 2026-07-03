# Validation commands

- Full local: `python3 -m pytest -q` (expect ~1349 passed, 136 subtests).
- Docker unit: `CODOXEAR_DOCKER_PORT=190NN scripts/codoxear-docker-sandbox test`
- Docker server smoke: `CODOXEAR_DOCKER_PORT=190NN scripts/codoxear-docker-sandbox smoke` (401 pre-login, 200 post-login).
- Browser: `AGENT_BROWSER_SESSION=<name> agent-browser open http://127.0.0.1:190NN/`, login `test-password`, snapshot/eval. Mobile viewport: agent-browser supports viewport sizing; verify both desktop and ~390x844.
- Stop sandboxes after use: `CODOXEAR_DOCKER_PORT=190NN scripts/codoxear-docker-sandbox stop`.
