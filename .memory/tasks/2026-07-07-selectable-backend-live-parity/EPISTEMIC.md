# EPISTEMIC

## Phenomenon
Codoxear exposes selectable backend tabs as product promises. The tested question was whether the browser-created Claude Code path with reasoning `max` preserves truthful user-visible semantics when the backend cannot actually start in the Docker sandbox.

## Accepted mechanism
Docker port 19396 does not contain a usable `claude` executable. The browser New Session flow still carries the selected backend/model/effort (`cc`, `sonnet`, `max`) into the launch ledger and broker argv. The broker fails before log bind, and Codoxear projects that as a synthetic failed-launch row instead of a silent idle or misleading real session.

The failed-launch row is intentionally not a real session. `/api/sessions` exposes `launch_state=failed`, `launch_stage=broker_early_exit`, `model=sonnet`, `reasoning_effort=max`, `busy=false`, and `log_path=null`; `/messages/tail` exposes `transcript_state=failed` with an assistant error row. Browser controls for send/composer, queue, attach/capture, file, and unattended are disabled with failed-launch labels; API send/enqueue/file-list/attachments reject the launch id as `404 unknown session`. Local recovery affordances remain available: Details, Copy details, New like this, and Dismiss launch.

## Evidence basis
- Task init: `68b5b51`.
- Proof: `1cbd477` plus clarification `9195f5c`, artifacts under `.memory/tasks/2026-07-07-selectable-backend-live-parity/browser-artifacts/backend-parity-19396/`.
- Clean-room review: `3abc5b6`, accepted with no blockers.
- OPS entries contain the command/test/browser/API evidence summary.

## Ruled out in this environment
The feared Docker failure mode is ruled out for the unavailable-Claude branch: the row did not disappear, did not idle silently, did not become a usable session, and did not enable real-session actions.

## Residual uncertainty
The usable Claude success branch remains unproven because launch never reached a bound Claude transcript log. A configured environment with Claude Code installed/authenticated still needs: browser New Session → Claude/max launch → log bind → sentinel send → visible assistant/error/no-response/recovery outcome; if mapped usage appears, `/api/sessions`, `/messages/tail`, and `#ctxChip` must agree.

## Nonblocking observations
- `snapshot-after-start-3s.txt` was ambiguous and is now documented as intermediate-only, not failed-state proof.
- Raw `/api/sessions` currently carries `provider_choice="openai-api"` on a failed CC row even though Details correctly shows `Provider: -`; review treated this as pre-existing cosmetic raw-JSON noise, not a product blocker.

## Current claim
Selectable-backend failed-launch parity for the Claude/max browser path is accepted. Full live Claude parity remains open until a configured Claude environment can exercise the usable-session branch.
