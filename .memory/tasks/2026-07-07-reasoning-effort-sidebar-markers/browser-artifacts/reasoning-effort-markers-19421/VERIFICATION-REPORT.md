# Reasoning-effort sidebar marker proof

## Outcome
Passed on Docker/browser port `19421`. The sidebar now renders compact effort markers for all supported reasoning efforts, including the previously invisible `max`, `minimal`, and `off` values. Existing `low`/`medium`/`high`/`xhigh` markers remained unchanged.

## Fixture
A Docker-only fake broker script (`fake_effort_sessions.py`) created seven terminal-owned sidecar/socket sessions inside the container app dir:

| Session | Backend | API reasoning_effort | Expected marker |
|---|---:|---:|---:|
| `effort-cc-max` | `cc` | `max` | `M+` |
| `effort-pi-minimal` | `pi` | `minimal` | `m` |
| `effort-pi-off` | `pi` | `off` | `–` |
| `effort-codex-xhigh` | `codex` | `xhigh` | `X` |
| `effort-codex-high` | `codex` | `high` | `H` |
| `effort-codex-medium` | `codex` | `medium` | `M` |
| `effort-codex-low` | `codex` | `low` | `L` |

The fake broker state returned only idle state (`busy=false`, `queue_len=0`, `token=null`); the proof targets sidebar metadata rendering, not backend inference.

## Evidence
- `/api/sessions` exposed the seven expected `reasoning_effort` values with their backends/models (`api/effort-sessions-summary.json`).
- Desktop browser sidebar at `1280x720` rendered:
  - `effort-cc-max` → text `M+`, class `effortMark effort-max`, title `reasoning effort max`.
  - `effort-pi-minimal` → text `m`, class `effortMark effort-minimal`, title `reasoning effort minimal`.
  - `effort-pi-off` → text `–`, class `effortMark effort-off`, title `reasoning effort off`.
  - Existing values stayed `X/H/M/L` with their original classes/titles.
  - `bodyOverflow=false` (`browser/desktop-marker-summary.json`).
- Mobile browser sidebar at `390x844` rendered the same marker text/classes/titles after opening the drawer, with every proof row visible and `bodyOverflow=false` (`browser/mobile-marker-summary.json`).
- Screenshots: `browser/desktop-sidebar.png` and `browser/mobile-sidebar.png`.

## Validation commands
- Functional commit validation before proof: focused local pytest `42 passed`; full local pytest `1814 passed, 134 subtests`; `git diff --check` passed.
- Saved proof validation:
  - `python3 -m pytest -q tests/test_reasoning_effort_source.py tests/test_session_sidebar_priority.py tests/test_static_assets.py` → `42 passed` (`local-focused-pytest.txt`).
  - `python3 -m pytest -q` → `1814 passed, 134 subtests` (`local-full-pytest.txt`).
  - Docker focused tests on port `19422` → `42 passed` (`docker-focused-test-19422.txt`).
  - Docker smoke on port `19423` → pre-login `/api/me` 401, post-login `/api/sessions` 200, app dir `/home/tester/.local/share/codoxear` (`docker-smoke-19423.txt`).
  - `git diff --check` passed (`git-diff-check.txt`).

## Hygiene
No cookies, auth headers, credential values, private-key contents, or bearer tokens are stored. API defaults contain the literal auth-method label `apikey`; that is a static configuration value, not an API key. Cookie jar used for API probes stayed under `/tmp` and was not copied into artifacts. Docker cleanup left no `codoxear-sandbox-19421/19422/19423` containers running (`docker-leftovers.txt`).
