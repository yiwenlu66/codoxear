# Coarse-pointer code-copy browser proof (port 19532)

## Scope

Docker/browser verification only for functional commit `4fa373d` on branch `recovery/product-gaps` in `/home/yiwen/codex-web-product-recovery`. Product code was not edited and nothing was committed.

## Docker/API verification

- Sandbox port: `19532`; protected port `8743` was not used.
- `scripts/codoxear-docker-sandbox preflight`: passed isolation guard for `/tmp/codoxear-docker-sandbox-19532`.
- `scripts/codoxear-docker-sandbox test`: passed, `1836 passed, 1 skipped, 134 subtests passed in 46.02s`.
- `scripts/codoxear-docker-sandbox smoke`: passed.
  - pre-login `/api/me`: `401`
  - post-login `/api/sessions`: `200`
  - container app dir: `/home/tester/.local/share/codoxear`

## Fake session

Created one deterministic Docker-only fake broker/session inside container `codoxear-sandbox-19532`:

- session id: `coarse-code-copy-session`
- assistant markdown contains two fenced code blocks; the browser assertions clicked the real rendered `.code-copy-btn` for the first block.
- fake broker call summary: `send=0`, `keys=0`, `shutdown=0`; only `state` polls were observed.

## Browser proof

Browser driver: local `/usr/bin/chromium` headless via CDP on exact temporary port `19533`, with `Emulation.setDeviceMetricsOverride`, `Emulation.setTouchEmulationEnabled`, and `Emulation.setEmulatedMedia` for pointer/hover media. Final run had zero assertion failures.

| Scenario | Media proof | Button rect | Computed size | `<pre>` padding-right | Overflow | Clipboard |
|---|---:|---:|---:|---:|---:|---|
| touch tablet 768x1024 | `(pointer: coarse)=true`, `(hover: none)=true` | `44 x 44` | width/height/min-width/min-height all `44px` | `58px` | none | exact `printf 'alpha <tag> & value'` |
| touch phone 390x844 | `(pointer: coarse)=true`, `(hover: none)=true` | `44 x 44` | width/height/min-width/min-height all `44px` | `58px` | none | exact `printf 'alpha <tag> & value'` |
| desktop 1280x800 | `(pointer: coarse)=false` | `30 x 30` | width/height/min-width/min-height all `30px` | `46px` | none | exact `printf 'alpha <tag> & value'` |

Desktop headless Chromium did not report `(pointer: fine)=true`, but it did prove the required non-coarse desktop branch: `(pointer: coarse)=false` and the `.code-copy-btn` remained compact at `30 x 30`.

## Cleanup

- Browser process from the final CDP run exited with return code `0`; no listener remained on `:19533`.
- Killed only the exact fake broker PID recorded in the container (`52`).
- Stopped only exact container `codoxear-sandbox-19532` via `CODOXEAR_DOCKER_PORT=19532 scripts/codoxear-docker-sandbox stop`.
- Post-cleanup checks: no `codoxear-sandbox-19532` container and no listener on `:19532` or `:19533`.

## Raw artifacts

- `COMMANDS-RUN.md`
- `raw/browser-results.json`
- `raw/broker-call-summary.json`
- `raw/broker-calls.jsonl`
- `raw/fake-broker-startup.log`
- `raw/fake-sidecar.json`
- `raw/fake_coarse_code_copy_session.py`
- `raw/run_cdp_coarse_code_copy_browser.py`
- `raw/sessions-after-fake.json`
- `raw/sessions-after-browser.json`
- `raw/smoke-me-before-login.json`
- `raw/smoke-sessions-after-login.json`
- `raw/smoke-isolation.txt`
- `raw/chromium-stderr.log`

Sanitization: no cookies or auth headers were copied into the artifact directory; the temporary Chrome profile was removed after the run.

## Residual concerns

None blocking. Desktop media evidence is non-coarse rather than positive fine-pointer because this headless Chromium reports `(pointer: fine)=false`; the acceptance prompt explicitly allows desktop proof by showing `(pointer: coarse)=false`.
