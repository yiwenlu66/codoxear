# OPS — connection icon

Append-only evidence trail. Times are local (host clock).

## 2026-10-24 18:20 — read the current implementation

- `app_shell.js:123` placed `div.networkBanner` as a block sibling in `main`'s
  flex column between `topbar` and `chatWrap`; `app.css:927` gave it border +
  wash + `--font-md`, so its height was text-driven and unbounded.
- Three reasons existed in `app_network.js` (`offline`, stale-served,
  `transportFailed` with a 2-failure threshold); they drove wording only.
- Tests pinning the old sentence: `test_offline_resilience.py`,
  `test_offline_banner_transport_failure.py`, `test_stale_cached_data_banner.py`,
  `test_all_critical_mechanisms.py:232`; end-to-end pin in
  `scripts/docker_stale_banner_verify.sh` (probe on `#networkBanner`).

## 2026-10-24 18:22 — glyph legibility, first attempt failed

Rendered the hand-written arc-fragment path at 18px/24px/48px on paper light and
paper dark (`/tmp/conn-icon-verify/icon-preview*.png`). At 18px the truncated
arcs read as noise; only the fan silhouette survives. Replaced with a full wifi
fan (three arcs + dot) crossed by a slash.

## 2026-10-24 18:26 — unit tests updated, targeted run green

`tests/test_offline_resilience.py`, `test_stale_cached_data_banner.py`,
`test_offline_banner_transport_failure.py`, `test_all_critical_mechanisms.py`
now assert `{state, hidden, title, spoken}` instead of prose in `textContent`.
Added a threshold test (1 failure → hidden, 2 → `failing`).
Result: 20 passed (4 files).

## 2026-10-24 18:27 — guard + bundle

`python3 scripts/check_wiring.py` → `SCAN COVERAGE files=83 option_factories=90`,
passed. `esbuild app.js --bundle --format=esm` → builds clean (1.2mb).

## 2026-10-24 18:28 — commit 64ae53c0, full verify launched

Pushed to origin/main. Docker stale-outage verifier on 64ae53c0:

```
01-baseline        statusHidden=true  spoken=""
02-server-paused   statusHidden=false spoken/title="Connection unavailable — showing cached data. Updates retry automatically."
03-server-recovered statusHidden=true spoken=""
PASS
```

## 2026-10-24 18:29 — full suite exposed a latent harness defect

`test_frontend_message_flow.py` failed with `OSError: [Errno 7] Argument list too
long` (E2BIG on the `node -e` argv). Passed at `64ae53c0^`; the module loader
injects `app_network.js` into the message-flow fixture, so ~1.5 KB of new source
crossed MAX_ARG_STRLEN (128 KiB). Fixed by running that file's fixtures from a
temp file like the sibling VM tests (`5c1c5d56`).

## 2026-10-24 18:35 — phone-width geometry, before and after

Docker container from the built image, agent-browser at 390x844, forced
`navigator.onLine = false` via `Object.defineProperty` (the only way to reach the
`offline` reason in a browser that still has a network). Same probe on the parent
commit (`0ec5c0443d04`) and on the shipped tree (`3450275dba25`):

| measure (390x844)        | banner (0ec5c044) | icon (3450275d) |
|--------------------------|-------------------|-----------------|
| chatTop                  | 156               | 100             |
| chatHeight               | 632               | 688             |
| topbarHeight             | 56                | 56              |
| actionsLeft              | 276               | 276             |
| titleWidth (long title)  | —                 | 208 → 182       |

Reading: the banner cost 56 px of transcript; the icon costs 26 px of *title*
(18 px glyph + 8 px gap) and zero elsewhere — `.topActions` never moves, because
the icon lives inside the flexible `.pill` and is absorbed by the title's
ellipsis slack.

## 2026-10-24 18:32 — final-commit verification

- `scripts/docker_stale_banner_verify.sh 3450275d` → PASS (same three probes,
  icon present in the topbar at all three).
- Full suite on the final tree: **2517 passed, 1 failed** —
  `test_hint_mode_coverage::test_rendered_shell_interactives_are_declared_or_explicitly_excluded`,
  reproduced identically in a clean worktree at `64ae53c0^` (`draftUndoBtn` has
  no `data-hint` at that commit either). Pre-existing, unrelated.
- Screenshots: `/tmp/conn-icon-verify/before-after.png`,
  `topbar-offline-390.png`, `topbar-online-390.png`,
  `topbar-offline-longtitle-zoom.png`.
