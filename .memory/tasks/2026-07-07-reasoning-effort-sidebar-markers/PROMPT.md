## Objective
Fix and prove sidebar reasoning-effort markers for every reasoning value Codoxear can launch or display. The slice is done when CC `max` and Pi `off`/`minimal` no longer disappear from session rows, existing `low`/`medium`/`high`/`xhigh` markers are unchanged, and the browser proof shows the visible sidebar metadata communicates the true effort for representative sessions.

## Workbench
1. Confirm the current marker omission mechanism in the sidebar renderer.
2. Implement the smallest display mapping and CSS support for `max`, `minimal`, and `off` without changing launch/default semantics.
3. Add source/runtime tests for all supported reasoning efforts and unchanged existing markers.
4. Run focused/local validation plus Docker/browser proof for actual sidebar rows.
5. Commit functional, proof, review, and memory changes separately.

## Context
Repository: `/home/yiwen/codex-web-product-recovery` on branch `recovery/product-gaps`.
Protected checkout: `/home/yiwen/codex-web` on `main`; do not edit or promote it.
Project memory: `.memory/project/ARCHITECTURE.md` and `.memory/project/VALIDATION.md`.
Current code supports CC reasoning effort `max` and Pi efforts `off`/`minimal`, but the sidebar marker chain only maps `xhigh`, `high`, `medium`, and `low`.

## Task specifications
Sidebar metadata must expose a compact marker for every supported effort value that appears on a session row:
- `xhigh` keeps `X`.
- `high` keeps `H`.
- `medium` keeps `M`.
- `low` keeps `L`.
- `max` gets a distinct compact marker, preferably stronger than `xhigh` while still fitting the sidebar.
- `minimal` gets a distinct compact marker.
- `off` gets a distinct compact marker.

The marker's CSS class must stay effort-specific (`effort-${value}`) so colors can distinguish values. Existing backend launch defaults, request parsing, and session metadata must not change.

Public behavior should be proved from the user's perspective: a session row with CC `max` must show a marker instead of omitting effort, and Pi `off`/`minimal` rows must also show markers. Details/diagnostics already expose raw values; this slice is specifically the compact sidebar scan signal.

Acceptance criteria:
- Existing `low`/`medium`/`high`/`xhigh` markers remain unchanged.
- CC `max`, Pi `off`, and Pi `minimal` render non-empty markers with titles `reasoning effort <value>`.
- Unknown/unrecognized effort strings still render no marker rather than inventing semantics.
- Source tests cover the mapping and CSS classes.
- Docker/browser proof exercises actual sidebar rows and records visible text/title for representative efforts.

## Constraints
Do not edit `/home/yiwen/codex-web` or `main`.
Do not touch live runtime dirs: `~/.local/share/codoxear`, `~/.claude`, `~/.codex`, host Pi logs/sockets, systemd/tailscale.
Use Docker-only for broker/server/session/browser verification; avoid port `8743`.
Do not copy secrets into committed artifacts.
Cleanup must be exact-PID/container scoped; no `pkill -f`, `killall`, or broad kills.
Keep functional, proof/evidence, review, and memory commits separate.
Run clean-room adversarial review before yielding.
