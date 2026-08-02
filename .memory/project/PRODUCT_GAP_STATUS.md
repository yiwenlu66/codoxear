# Product gap status

## Current state

Codoxear is on the "paper" design language with a written constitution (AGENTS.md § Design language): radius 0, ink #141111 on white, warm wash #efeee9, inversion for primary/active, mono for data, no alpha colors, no decorative shadows. Viewport branching is restricted to tokens, visibility flips, and layout modes; the five sanctioned component branches are enumerated there.

State display follows the authority principle (ARCHITECTURE.md): every displayed state has one declared authoritative source plus an explicit reconciliation rule. The 21-state authority table was produced by audit (2026-08-02); all violations found were fixed.

## Closed themes (2026-07-31 → 2026-08-02)

- Send-path: one unconditional confirmed-send path; steering on all backends; queue is opt-in.
- State honesty: typing counts monotonic per turn (max of live deltas and snapshots; cross-chunk `meta_turn_open` windows so steers preserve and queued turns reset); Pi error+retry stays busy; terminal Pi errors close via PTY retry-status probe (`Retrying (n/m)`); Pi run settings (model/effort/provider) authoritative from full-log replay; unattended edits reconciled server → in-flight → pending; no control shown without backing (`pi_thinking_command` capability gating).
- Control completeness: `/model` and `/thinking` composer pickers (bridge-registered Pi command with clamp readback); interrupt is topbar-only on all viewports (composerStopBtn removed).
- Design: paper sweep (86 radii zeroed, all alpha colors solidified), chrome 32px + 44px touch hit-slop, sidebar middle-ellipsis model + `·eff` suffix, search bar rewrite (`/` hint), diag view cleanup (no new-like, icon copy buttons).
- Markdown on `marked`; SSE live transcript with polling fallback; subagent narration rows.
- Mobile: 15 surfaces structurally verified at 390px; visual audit by image-capable agent with findings fixed (logo, active-card inversion, picker bounds, details mono/scroll, table palette, typing-row presence).

## Residuals (known, written down)

- Pi retry detector follows Pi 0.82.1 status-text contract (`Retrying (n/m) in Ns`); a future Pi changing it needs a detector update.
- Terminal-error busy→idle relies on a 3s quiet window; pathological >3s PTY lag could briefly show idle before retry rows arrive.
- Full-log replay for run-settings authority may cost I/O on very large Pi logs; a checkpoint optimization must not reintroduce a bounded lossy scan.
- Slash commands sent before a session's first model turn execute but leave no log evidence until the log is created.

## Continuation rule

The user's remaining open item is aesthetic judgment of the paper language on real hardware. Do not re-min fixed surfaces without new observation. Next justified work: fresh user-reported defects, or feature work from the user's direction.
