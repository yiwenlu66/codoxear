# Result: DEFECT

The scout found an active, tested-in violation of the core invariant "every sent prompt produces a visible answer/error/no-answer/**interruption**." All three ranked targets attack the same invariant through genuinely distinct triggers and distinct code gaps; the top one is a live defect provable from a 2-record synthetic log.

## Top target

**DEFECT — Interrupting a turn leaves no persistent browser-visible outcome; on Pi the interrupted assistant message (including partial output the user already saw streaming) is actively discarded.**

**1. User-visible contradiction (one sentence).** A user sends a prompt, presses Stop, and after the 2.2s "interrupting…" toast fades the transcript shows only their prompt with the session idle — indistinguishable from a prompt that was ignored — even though "interruption" is one of the four mandated visible outcomes.

**2. Suspected mechanism + exact files/functions.**
- Frontend has no interruption row: `interruptSelectedSession()` (`codoxear/static/app.js:6341`) and both stop controls (`interruptBtn` app.js:6352, `composerStopBtn` app.js:6357) only `setToast("interrupting...")` + `POST /interrupt`; `setToast` (app.js:1993) self-clears after 2200ms. No `app_transcript.js`/`app_message_rows.js` path renders an interruption.
- Backend interrupt route just sends ESC: `_handle_interrupt` (`codoxear/control_routes.py:219`) → `manager.inject_keys(session_id, "\x1b", interrupt=True)`.
- Normalizer suppresses the aborted turn instead of projecting it. Pi: `_single_chat_event` returns `None` on `pi_assistant_is_aborted_turn(obj)` **before** the text check (`codoxear/rollout_chat_events.py:50-52`), so even `stopReason:"aborted"` messages carrying partial text are dropped; the notification path mirrors this (`codoxear/rollout_delivery.py:46`). `pi_assistant_is_aborted_turn` = `stopReason == "aborted"` (`codoxear/pi_message.py:94-100`).
- The no-response injector deliberately excludes abort closes: `_inject_no_response_events` (`codoxear/rollout_chat_events.py:238`) treats only `event_msg` `task_complete`/`turn_complete` (Codex) and `system` `turn_duration`/`api_error` (CC) as closes (lines 294/302); Codex `turn_aborted` is neither a chat event nor a close, so it is unprojected too.
- **Tested-in as current behavior:** `tests/test_codex_no_response_projection.py:197` (`test_pi_aborted_turn_does_not_emit_no_response`) asserts an aborted Pi turn yields events `["user"]` only; `tests/test_server_chat_flags.py:306` shows a partial-text aborted turn is not counted as assistant chat.

**3. Discriminating verification plan (PASS = fixed, DEFECT = current).**
- **Layer A — unit (cheapest, deterministic, no creds).** Feed `_extract_positioned_chat_events` a Pi log `[user "hello"; assistant stopReason:"aborted", content:[]]`. DEFECT: roles == `["user"]`. PASS: `["user", <interruption outcome row>]`. Repeat with `content:[{text:"partial"}]` to prove partial output is preserved, and with Codex `[event_msg user_message; event_msg turn_aborted]`.
- **Layer B — API (Docker, fake broker).** Bind a synthetic Pi/Codex log ending in abort; `GET /messages/tail` → DEFECT returns only the user row, PASS returns user + interruption row. `GET /messages/search?q=<interruption phrase>` → DEFECT `match_count=0`, PASS `1` (transcript-search-preserves-synthetic-rows invariant already established in `bb7d38d`).
- **Layer C — browser (Docker, real Pi where env allows).** New Pi session → send a slow prompt → click Stop. DEFECT: only user bubble + a toast that vanishes in 2.2s; reload/second browser shows only the user bubble. PASS: a persistent interruption row survives reload and fresh-server rediscovery.

**4. Why this outranks the alternatives.** It is the only candidate that violates a literally-enumerated required outcome ("interruption") head-on, rather than an edge of the "no-answer" clause. Stop is a first-class, routinely-used control (two buttons), so the silent path is high-frequency. It is the cheapest to discriminate (a 2-record log already sits in the test suite) yet the most severe form of silence — Pi drops even partial streamed output. The asymmetry (completed-no-answer gets a persistent row via `task_complete`; interrupted-no-answer gets nothing) shows this was never built, and the surface map already declares interruption should render as a visible transcript/recovery message, so it is neither polish nor scope expansion.

## Backup targets

**Backup 1 — DEFECT-leaning: Pi is entirely absent from no-response/outcome injection, so a Pi turn that ends without an answer for any non-`error` reason renders nothing while Codex/CC render a no-response row.**
- Contradiction: the "selectable-backend visible-outcome" promise was implemented for Codex and Claude Code but never extended to Pi, the primary certified backend.
- Mechanism/files: `_inject_no_response_events` (`codoxear/rollout_chat_events.py:238-328`) branches on `event_msg`, `user`, `system` only; Pi `type:"message"` rows fall into `else: continue` (they never even set `user_byte`). `_detect_codex_no_response_closes` is Codex-only. Pi's clean close (`pi_assistant_is_final_turn_end`, `codoxear/pi_message.py:103`) requires assistant text, so a Pi turn producing no text has no close to trigger any row; only `stopReason:"error"` (`pi_assistant_error_text`) is visible.
- Verification: unit — Pi log `[user; assistant stopReason:"stop" with empty/no text]` or a truncated close → DEFECT: `["user"]`; PASS: injected no-response row. API/Docker — same via `/messages/tail`.
- Rank: same invariant, distinct trigger and distinct code location (fixing Target 1's abort suppression does not add a Pi close branch), but rarer than interrupt and needs confirming which real Pi close shapes reach it.

**Backup 2 — SCOUT: A backend process that dies mid-turn *after* log bind produces no visible outcome for any backend.**
- Contradiction: a prompt whose backend crashes/OOMs/is killed mid-turn becomes idle with no answer/error/no-response.
- Mechanism/files: `agent_exit_before_log_bind` covers only the pre-log case (`codoxear/broker.py:877,959`; `launch_ledger.py:267`); grep finds no post-log synthetic exit/outcome event in the rollout/transcript layer. No-response injection needs a close row (`task_complete`/`turn_complete`/`turn_duration`/`api_error`) that a dead process never writes, so death-before-close is silent across backends.
- Verification: Docker — fake broker binds a log, writes a user turn, then exits without a close; observe `/messages/tail` and the sidebar. DEFECT: user row only, session drifts to idle, no outcome. PASS: a persistent "backend exited"/no-response outcome row. Requires confirming the broker-exit ↔ `interrupted_idle` interplay, so it is SCOUT-level, not asserted DEFECT.
- Rank: broadest trigger but lowest frequency and highest verification cost; correct as the exploratory backup.

## Parked user decisions
- **Backend parity (real Codex/Claude Code inference).** Credential/gateway-gated per EPISTEMIC; a product decision on release boundary, not a scoutable code defect. Not selected.
- **Interruption message wording/styling.** If Target 1 is taken, the parent should decide the row's exact text/class (e.g., a distinct "interrupted" class vs. reusing `message_class:"error"`); the mechanism is decided, the copy is a product choice. Note: the fix must *not* reintroduce the wrong generic no-response text for aborts (guarded by `test_pi_aborted_turn_does_not_emit_no_response`) — it needs a distinct interruption outcome.
- **Monaco editor provisioning** remains decision-gated (plain-edit is the certified baseline); unrelated to these targets.

## Git status
`git status --short` → clean (empty). No staged files; branch `recovery/product-gaps`; no edits made (read-only scout).

Ruled out during the hunt (verified single-authority or already certified, so not defects): queue badge (`updateQueueBadge` app.js:2603 derives from server `queue_len`), queue-drain send failure (surfaced via commit-unknown/orphan recovery), context/token usage (centralized in `pi_context.py`), notification dedup (shares `_extract_delivery_messages`), interrupt-that-doesn't-land (busy state stays truthful), mobile paperclip (certified 44×44). The June recon's "toast hidden on mobile at 520px" is stale — `app.css:2622` repositions the toast visibly; it remains transient (2.2s), which is the relevant limitation for Target 1.
