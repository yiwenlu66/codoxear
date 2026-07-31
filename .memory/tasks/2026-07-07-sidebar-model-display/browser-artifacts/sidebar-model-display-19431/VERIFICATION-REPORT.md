# Sidebar model display proof

## Outcome
Passed on Docker/browser port `19431`. Sidebar metadata now includes meaningful model names between age and cwd, omits empty/default models, and keeps long model strings constrained by the existing `.metaText` ellipsis without body overflow.

## Fixture
A Docker-only fake broker script (`fake_model_sessions.py`) created five terminal-owned sidecar/socket sessions inside the container app dir:

| Session | Backend | API model | Expected sidebar model behavior |
|---|---:|---|---|
| `model-codex-gpt` | `codex` | `gpt-5.4` | show `gpt-5.4` |
| `model-cc-sonnet` | `cc` | `claude-sonnet-4-5` | show `claude-sonnet-4-5` |
| `model-pi-long` | `pi` | `provider/very-long-model-name-for-mobile-ellipsis-proof-claude-sonnet-4-5-extra-suffix` | show and truncate/ellipsis in narrow layout |
| `model-default-omitted` | `codex` | `default` | omit model segment |
| `model-empty-omitted` | `pi` | normalized API `null` from empty sidecar model | omit model segment |

The fake broker state returned only idle state (`busy=false`, `queue_len=0`, `token=null`); the proof targets sidebar metadata rendering, not backend inference.

## Evidence
- `/api/sessions` exposed the expected model values or normalized null for the empty-model row (`api/model-sessions-summary.json`).
- Desktop browser sidebar at `1280x720` rendered:
  - `model-codex-gpt` meta text: `3h ago | gpt-5.4 | model-codex-gpt`.
  - `model-cc-sonnet` meta text: `3h ago | claude-sonnet-4-5 | model-cc-sonnet`.
  - `model-pi-long` meta text includes the full long provider/model string and remained inside `.metaText` (`scrollWidth > clientWidth`).
  - `model-default-omitted` meta text: `3h ago | model-default-omitted` (no `default` model segment).
  - `model-empty-omitted` meta text: `3h ago | model-empty-omitted` (no empty segment).
  - `bodyOverflow=false` (`browser/desktop-model-summary.json`).
- Mobile browser sidebar at `390x844` rendered the same model/omission behavior after opening the drawer. The long model row had `metaBox.truncated=true` (`scrollWidth=636`, `clientWidth=280`) and `bodyOverflow=false` (`browser/mobile-model-summary.json`).
- Screenshots: `browser/desktop-sidebar.png` and `browser/mobile-sidebar.png`.

## Validation commands
- Functional commit validation before proof: focused local pytest `40 passed`; full local pytest `1817 passed, 134 subtests`; `git diff --check` passed.
- Saved proof validation:
  - `python3 -m pytest -q tests/test_sidebar_model_display_source.py tests/test_session_sidebar_priority.py tests/test_static_assets.py` → `40 passed` (`local-focused-pytest.txt`).
  - `python3 -m pytest -q` → `1817 passed, 134 subtests` (`local-full-pytest.txt`).
  - Docker focused tests on port `19432` → `40 passed` (`docker-focused-test-19432.txt`).
  - Docker smoke on port `19433` → pre-login `/api/me` 401, post-login `/api/sessions` 200, app dir `/home/tester/.local/share/codoxear` (`docker-smoke-19433.txt`).
  - `git diff --check` passed (`git-diff-check.txt`).

## Hygiene
No cookies, auth headers, credential values, private-key contents, or bearer tokens are stored. API defaults contain the literal auth-method label `apikey`; that is a static configuration value, not an API key. Cookie jar used for API probes stayed under `/tmp` and was not copied into artifacts. Docker cleanup left no `codoxear-sandbox-19431/19432/19433` containers running (`docker-leftovers.txt`).
