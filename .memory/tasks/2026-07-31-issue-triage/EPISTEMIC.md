# EPISTEMIC — project state model (issue triage)

## Phenomenon being explained
Not a single defect yet. This file holds the operating model for owning project
state as issues arrive. It will be rewritten as evidence accumulates.

## Deployed baseline (verified 2026-07-31)
- `/home/yiwen/codoxear` @ `main` `1976c30b`, recovery/product-gaps merged.
- Service `codoxear-server.service` running latest code (pipx reinstall + restart done).
- Healthy: `/` 200, `/api/sessions` 401 (auth gate), live broker session survived.

## Live mechanisms / invariants (carried from project memory; trusted for routing)
- Non-editable pipx install ⇒ redeploy = `pipx install --force` + `systemctl restart`. Pulling code is insufficient.
- Server restart is safe for live sessions; brokers/CLIs are outside the service cgroup's lifecycle control.
- Busy/idle/readiness authority is `RuntimeStatus` + turn-state reducer; backend adapters own launch/log/argv specifics. Multiple invariants around transcript truthfulness, context chip, attachment staging, confirmations (see `.memory/project/ARCHITECTURE.md`).
- Validation floor: pytest green is NOT acceptance. Docker sandbox + browser evidence for UX claims.

## Ruled out / known dead
- No open bounded product-visible defect in source per PRODUCT_GAP_STATUS.md (as of scout at `53768a3`; current HEAD `1976c30b` is later). Treat fresh user issues as the source of truth, not the prior "clean" verdict.

## Anomalies / open
- (none currently open in living docs; ISSUE-1/2/3 are the active defects, see ISSUES.md)

## DO NOT TOUCH: session card good state (recorded 2026-08-02)
The session sidebar card has a TWO-BRANCH DOM (swipe for touch, sessionDesktopLayout for desktop hover) and an outline-based active edge. This is the user-confirmed GOOD STATE.

### Why the card unification refactor (17c339ca) failed — architecture lesson
The unification idea was sound (one DOM + CSS-driven differences is cleaner than two parallel trees). The execution failed on two levels:
1. **Layout collapse**: the refactor merged desktop into the touch branch's vertical-column sessionInner, but never wrote the CSS to switch sessionInner to horizontal (row) on desktop. Result: desktop got a mobile vertical layout, scattering the meta line (pi/tmux/thinking/cwd/branch) across three lines.
2. **Active-edge interaction**: outline/box-shadow behave differently depending on overflow:hidden clipping (swipe branch) vs no clipping (desktop branch). Patching the edge across a DOM that changed underneath produced double/triple rings.

**Lesson**: unifying two DOMs requires unifying their CSS contracts too — not just markup. The subagent unified markup and deleted desktop CSS without writing replacements. Main agent then patched edge symptoms for multiple turns without recognizing the *layout* was broken, not the styling. If re-attempting unification: first write the full CSS for BOTH viewport modes against the unified tree, verify both render correctly, THEN commit. Do not incrementally patch.

Current state: two-branch DOM is correct and intentional. Do not attempt unification again without addressing the above.

## ARCHITECTURE DECISION: send-path unification (2026-07-31, evidence-confirmed)

### Harness reality (verified across all 3 backends via subagents)
All three agent CLIs read stdin CONTINUOUSLY during a turn. A plain text+Enter PTY write while busy = STEERING the live turn (delivered at next safe boundary — assistant-turn boundary for Pi, sampling boundary for Codex, action boundary for Claude Code). NONE require ESC/interrupt to send. "Send while busy (steer)" is DISTINCT from "send after current (new turn)" — the backend decides based on ITS state and the input key (Pi: Enter=steer/Alt+Enter=after-run; Codex: Enter=steer/Tab=queue; CC: Enter=steer, no native after-current).

### Key reframe
The BROKER already has ONE write primitive: the `send` command (text+Enter + optimistic state assertion + sync commit). The busy-gate and queue-gate are NOT in the broker — they're in the MANAGER layer (require_send_preconditions + send_remote_ready + the silent-enqueue fallback in send()). So this was never "two broker paths"; it's one broker path WRONGLY GATED by the manager. The gate is the fiction.

### Decision: (B) keep the abstraction, make it UNCONDITIONAL (rejected (A) raw-PTY, rejected "route send-now to keys")
- ONE send path (/send → broker send command). Both "send now" and "send after current" eventually call it. User's "don't use different paths" satisfied.
- KEEP: optimistic state assertion (harness-faithful UX), sync commit, commit-unknown tracking, attachment bookkeeping. These are real functionality, not fiction.
- REMOVE: busy-gate (silent enqueue fallback), queue-FIFO-gate for user direct sends. These were the only unclean part.
- Queue ("after current") repositioned: NOT system-imposed deferral, but pure opt-in TEMPORAL deferral (hold message → /send on idle). Maps exactly onto steer-now vs new-turn-after.

### Why (B) over (A)
(A) raw-PTY loses optimistic state assertion (must rebuild optimistically), loses commit-unknown (keys path doesn't track_request_sent), can't own attachments. (B) keeps all three for free by removing only the gates. Full functionality + clean abstraction.

### The one engineering wrinkle (real risk)
broker send_handler's state assertion assumes "starting a turn" (sets busy/turn_open, RESETS turn_has_completion_candidate, CLEARS interrupt timestamps). When steering (already busy mid-flight), resetting these could mis-project the reducer. Must make the assertion MERGE-SAFE when already busy/turn_open (preserve existing turn state, don't reset). Discriminating evidence the fix is complete = no mis-projection during a real busy-steer test.

### Elegant consequence
Frontend sendChoiceNowBtn ALREADY calls sendText → /send. Once /send is unconditional, send-now works through the EXISTING call. No frontend routing change. The fix is almost entirely backend (remove gates + merge-safe assertion). Minimal blast radius.

### Resolves
ISSUE-2 + ISSUE-3 dissolve by removing gates. Steering works. Queue preserved as opt-in. Attachments/commit-unknown preserved.

## Currently justified claim
Deploy baseline is sound and ready to receive issues. No fix in flight.

## Question that would most change the model
The first user issue. Until it arrives, every mechanism here is carry-over; the first real report is where the model gets tested.

## Performance architecture (audit 2026-08-02, subagent report)

### The real problem is NOT the bundle — it's the polling
User intuition was correct: inefficient log reading / event transmission dominates mobile traffic, not package loading.

**Steady-state polling volume (visible tab):**
- Idle: ~6,160 requests/hour (~4.4 MB/hour in empty responses alone)
- Running: ~16,560 requests/hour (~15.9 MB/hour)
- Message poll every 900ms idle / 250ms running; sessions every 2.5s; secondary (voice/notifications) every 10s

**Server-side waste:** `static_asset_version()` re-hashes 17.1 MB (including all Monaco) on EVERY `/api/sessions` poll response to populate an `app_version` field the frontend never reads. ~10.7ms CPU per poll, ~24.6 GB/hour of pointless hashing.

### Bundle is secondary but real
- ~1.07 MB uncompressed JS/CSS, 36 eager scripts, no gzip, no minification, no caching (`no-store` default despite cache-busting `?v=` design)
- Monaco (16MB) is correctly lazy-loaded; only 40KB loader is eager
- Quick win: gzip saves ~850KB/load; proper caching nearly eliminates repeat transfers

### Prioritized fixes
- **P0 (hours):** cache asset version at startup; remove unused `app_version` from API; fix cache headers; add gzip
- **P0 (0.5-1 day):** defer voice/notification startup calls until after transcript
- **P1 (1-3 days):** raise polling intervals; return 304/204 on unchanged state
- **Structural (1-2 weeks):** SSE/long-poll for transcript events; minification pipeline; code-splitting

### Log reading is already efficient
The server DOES use byte-offset tailing (`read_jsonl_from_offset` with seek), not full-file reads. The reverse iterator reads in 64KB blocks from the end. The traffic problem is the RESPONSE frequency and size, not the read mechanism.
