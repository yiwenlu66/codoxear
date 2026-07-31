# EPISTEMIC

## Phenomenon
Sidebar session rows are the quick scan surface for backend/model/reasoning state. Valid reasoning efforts must be visible there rather than only buried in Details/diagnostics. Before this slice, `max`, `minimal`, and `off` were valid session values but rendered no compact marker.

## Accepted mechanism
The sidebar renderer now uses a frozen `REASONING_EFFORT_MARKERS` lookup rather than an inline four-value ternary. The accepted mapping is:

- `xhigh -> X`
- `high -> H`
- `medium -> M`
- `low -> L`
- `max -> M+`
- `minimal -> m`
- `off -> –`

Unknown/unrecognized effort strings still map to an empty marker and therefore create no `.effortMark` span. CSS classes remain effort-specific (`effortMark effort-${effortTxt}`), and titles remain `reasoning effort <value>`.

## Accepted claim
The sidebar display-truth gap is fixed in commits `f59086a` (implementation), `937e09e` (proof), and `eb7da3b` (review). CC `max`, Pi `minimal`, and Pi `off` now render visible markers in actual sidebar rows, while existing Codex `low`/`medium`/`high`/`xhigh` markers are unchanged. Backend launch/default/session metadata semantics were not changed.

## Evidence basis
- Source tests assert the full seven-value marker mapping, unknown fallback, class/title templates, and new CSS classes.
- Local focused pytest passed (`42 passed`), full local pytest passed (`1814 passed, 134 subtests`), Docker focused pytest passed (`42 passed`), and Docker smoke passed.
- Docker/browser proof in `browser-artifacts/reasoning-effort-markers-19421/` created seven fake sidecar/socket sessions and verified desktop (`1280x720`) plus mobile (`390x844`) sidebar rows render `M+`, `m`, `–`, `X`, `H`, `M`, and `L` with expected classes/titles and no body overflow.
- Clean-room review in `reviews/reasoning-effort-marker-cleanroom-review.md` accepted the result with no blockers.

## Boundary
This proves sidebar metadata rendering for supported effort values. It does not prove real backend inference behavior or change which efforts backends accept.
