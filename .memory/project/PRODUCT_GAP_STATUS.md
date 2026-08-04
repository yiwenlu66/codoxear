# Product gap status

## Current state

Codoxear uses the paper design language defined in `AGENTS.md` § Design language: square geometry, warm-charcoal ink and borders (`#2f2b26`), paper/wash backgrounds, inverted primary and active states, monospace data, and no decorative shadows. Viewport branching is limited to tokens, visibility flips, and layout modes; the five sanctioned component branches are enumerated there.

State display follows the authority principle in `ARCHITECTURE.md`: every displayed state has one declared authoritative source plus an explicit reconciliation rule. The 21-state authority table was produced by audit (2026-08-02); all violations found were fixed.

## Reconciled disposition ledger (2026-08-04)

### Done

| Item | Disposition | Evidence |
| --- | --- | --- |
| ISSUE-1 — Inter font | Done — deliberate browser-native `sans-serif` decision. | `39c7ac28` |
| ISSUE-2 — direct send while busy | Done — direct confirmed-send no longer silently queues busy input. | `ddae3e62` |
| ISSUE-3 — direct send behind queued prompts | Done — queue-FIFO precondition removed for direct sends; queue remains opt-in. | `ddae3e62` |
| ISSUE-4 — stale living docs | Done — project docs resynced to the paper design and current UX. | `1c047e9b` |
| ISSUE-5 — mobile navigation | Done — compact mobile chat navigation shipped. | `51a4c098` |
| ISSUE-6 — redundant mobile stop control | Done — interrupt is the unified topbar control. | `83fe9dab` |
| ISSUE-7 — mobile toast placement | Done — toast delivery and fixed-chat-nav interaction reconciled. | `9ac87675` |
| ISSUE-8 — red voice button | Done — error state is gated on voice announcements being enabled. | `a7569c3a` |
| PR #21 — Claude interactive-prompt UI | Done — close without merge; the useful shared-broker Claude Code backend was shipped while the coupled prompt UI was intentionally excluded. | `f4a06a2e` |
| PR #22 — macOS web Codex launch | Done — rewritten launch path shipped; close the stale PR without merging its branch. | `d86bdda6` |
| Design language | Done — paper design implementation. | `81434f04` |
| State honesty | Done — direct-send freshness and state reconciliation are merge-safe. | `ddae3e62` |
| Keyboard map | Done — topbar interrupt uses `z`; composer focus remains `i`. | `83fe9dab` |
| `/model` and `/thinking` control | Done — Pi live settings navigation plus bridge-backed effort control are shipped (`/thinking` remains an alias). | `395ee1c9`, `d16b4690` |
| Full-transcript search UX | Done — search spans transcript windows and sidebar/search display was refined. | `4242ef68`, `1502b1c7` |
| Sidebar display | Done — model and chat-search display refined. | `1502b1c7` |
| Diagnostics view | Done — removed new-like and composer-stop controls; diagnostics rows retain responsive layout. | `408de92f` |
| Unattended persistence | Done — pending and in-flight edits reconcile into the authoritative state. | `c5f0d8d4`, `d6d3a1da` |
| Idle on Pi retry error | Done — retry errors retain busy state. | `d12839e5` |
| Terminal Pi retry outcome | Done — PTY retry state distinguishes terminal error from automatic retry. | `db029b04` |
| Typing counts across steering and queued turns | Done — counts remain continuous for steering and reset for queued turns. | `f83645b3`, `2e5602d5` |

### In progress

None. The repository has unrelated uncommitted work; it is not assigned to any ledger item.

### Open

| Existing item | Next action |
| --- | --- |
| Pi retry detector status-text contract | If Pi changes its `Retrying (n/m) in Ns` status text, capture the new PTY output and update the detector with a regression fixture. |
| Terminal-error quiet window | If a real session exhibits a false idle under delayed PTY output, capture timestamps and tune or replace the fixed 3-second quiet-window rule. |
| Pi full-log run-settings replay cost | Profile a large real Pi log before introducing any checkpoint; preserve full-history correctness in any optimization. |
| Pre-first-turn slash-command log evidence | If this becomes user-visible, define an authoritative pre-log command acknowledgement rather than inferring it from a missing backend log. |
| Paper-language aesthetic judgment on real hardware | Collect fresh user hardware observations or screenshots; act only on an observed defect. |

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

The remaining product-open item is aesthetic judgment of the paper language on real hardware. Do not re-min fixed surfaces without new observation. The next justified work is a fresh user-reported defect, a captured residual failure, or feature work from the user's direction.
