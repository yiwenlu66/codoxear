# SSE phone, Tailscale, and long-idle field test

Run this against the committed snapshot serving your real Codoxear instance, with a real broker-backed session selected in the phone browser. This test checks the delivery path that matters in practice: a transcript must resume from the server log after the browser loses its EventSource connection. It does not require a special mobile build.

## Before each run

1. Open Codoxear from the phone through the normal Tailscale URL and log in.
2. Select one real session whose transcript log is bound (the transcript is visible and the composer is enabled).
3. Send a prompt that causes the agent to emit numbered, distinctive markers while it works. For example:

   ```text
   Work on a real task. At each meaningful checkpoint, begin your visible update with `PHONE-SSE-01`, then increment the number for each later update. Keep the task active for at least ten minutes.
   ```

4. Keep a note of the last complete marker shown before the disruption. Do not use a failed launch row or a session that is already idle: those do not exercise a live stream.

A healthy recovery may show an immediate poll update while EventSource reconnects. The user-visible requirement is that the transcript reaches the newest server-log marker exactly once; it must not remain stale or create a duplicated sequence.

## 1. Sleep / wake

1. With the selected session producing work, record the last marker.
2. Lock the phone and leave it asleep for **at least five minutes**. For a stronger test, repeat with a 30-minute sleep.
3. Unlock it, return to the same browser tab, and wait ten seconds without manually refreshing.
4. Verify that the next visible marker is newer than the pre-sleep marker, no markers are duplicated, and the spinner/status matches the live session.
5. Send one short steering message from the phone. It should reach the same terminal session, and its next answer should appear in the phone transcript.

**Pass:** the transcript catches up within ten seconds of visibility, then continues with the same ordered marker sequence. The browser reconnect controller retries after one second; a poll can supply the catch-up state while the stream reconnects.

## 2. Tailscale / network change

1. Keep the same session selected and record its latest marker.
2. While the page remains open, switch Wi-Fi off so the phone moves to cellular; keep Tailscale enabled. Wait 30 seconds.
3. Switch Wi-Fi back on. If your normal use includes moving between two Tailscale networks, repeat while joining the second network instead.
4. Return to the transcript and wait ten seconds. Then send a short steering message.
5. Compare the visible markers with the terminal transcript or the desktop browser for the same session.

**Pass:** all markers produced during the outage appear once and in order after connectivity returns; the steering message is neither lost nor duplicated. A transient connection failure is expected. A stale transcript after ten seconds, a permanent spinner, a repeated marker, or an extra user message is a failure worth reporting.

## 3. Long-lived session

1. Start a real session that will remain available for **24 hours** (longer is better), and keep the phone tab open in the background after recording the latest marker.
2. At approximately 1 hour, 8 hours, and 24 hours, open the tab and check the session. At one checkpoint, change networks before opening it.
3. At every checkpoint, record: time, last marker before backgrounding, first marker after foregrounding, whether the session is busy or idle, and whether a steering message succeeded.
4. If the mobile OS discards the tab, reopen the normal Codoxear URL, select the same session, and perform the same marker/order check. Tab eviction is acceptable only if loading the session rehydrates the transcript correctly from its log.

**Pass:** every foreground/reopen converges to the server-log transcript without gaps or duplicates, and later messages continue to arrive. The session may be idle, but its historical transcript must remain readable.

## What to capture on failure

Capture enough evidence to locate the boundary between the phone, Tailscale, server, and broker:

- Phone model, OS version, browser/version, and whether the page was locked, backgrounded, or evicted.
- The exact Tailscale URL form used and the network transition (Wi-Fi name/cellular/exit node), without credentials.
- Wall-clock time of the disruption and recovery attempt.
- Last marker before disruption, first marker after recovery, and any duplicated/missing marker range.
- A screen recording or screenshots of the transcript and session status.
- The corresponding broker transcript/log excerpt from the terminal, redacted for secrets.

Do not restart or kill the broker to recover a failed phone test. A server restart is safe for session content, but record it as a separate condition because this runbook is specifically measuring client sleep, network change, and durable cursor recovery.

## Automated counterpart

`tests/test_sse_battle.py` runs the repeatable socket-level portion: abrupt server-side stream loss with cursor recovery, a 30-second idle period with the production heartbeat, 1,000 ordered Codex-native log events, fragmented/slow relay failure with replay, and the browser EventSource controller's one-second retry plus visibility-resume path. It cannot reproduce mobile OS radio suspension, Tailscale path changes, or browser tab eviction; those are the purpose of this field run.
