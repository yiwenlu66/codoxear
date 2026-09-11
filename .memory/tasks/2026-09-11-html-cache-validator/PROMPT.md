# PROMPT

## Objective

Fix the demonstrated HTML cache-validator defect in `/home/yiwen/codoxear`: `send_static_file` in `codoxear/static_routes.py` derives the weak ETag from the raw `path.stat()` before `read_static_bytes` substitutes `__CODOXEAR_ASSET_VERSION__` and the attachment limit into HTML. When a release changes only assets (index.html bytes and stat unchanged), the served HTML representation changes (new `?v=` asset URLs) but the stat ETag does not, so a normal reload's conditional request gets 304 and the browser keeps old HTML pinning old immutable asset URLs — the user-visible symptom was the old displaced subagent activity layout after a normal release.

## Scope

- Smallest mechanism fix: HTML ETag must be a weak content hash of the FINAL substituted response bytes (one read, no double reads). Non-HTML assets keep their fast stat validators and the immutable cache policy. Weak identity stays coherent across gzip/identity (hash over uncompressed final bytes, `W/` prefix).
- Do NOT edit theme/layout (Paper/Clay layout at 5d62d020 confirmed correct), counters, CSS, or unrelated voice/notification work in progress in the shared checkout.
- Reproduce the defect FIRST in Docker via real HTTP conditional requests (unchanged index bytes+stat across an asset update and server restart), before touching source.
- Browser upgrade proof with one persistent browser profile across the simulated release: old-cached HTML must reach the current bundle through a NORMAL reload (`location.reload()`), with network 200/304 and served asset URLs recorded; assert the corrected subagent count/label beside its marker using the existing fixture if feasible. A fresh browser profile may not substitute for the upgrade scenario.
- All executable tests run in Docker. No live server/API/session/login probes, no deploy.
- Scratch logs/screenshots stay outside the repo; commit only explicit scoped task files after staged-diff review.

## Deliverables

1. Defect reproduction artifacts (pre-fix, Docker, real HTTP) retained outside the repo.
2. The fix in `codoxear/static_routes.py` (+ focused executable tests).
3. Post-fix Docker verification: conditional-request matrix + browser upgrade scenario.
4. Task memory (OPS evidence trail, EPISTEMIC model), commit with exact scope, deployment-scope risk notes for the parent's review.
