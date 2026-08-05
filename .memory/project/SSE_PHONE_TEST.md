# SSE phone battle-test runbook

Use this runbook against the normal Codoxear instance on port **8443**. It is a real-phone check of live transcript delivery; do **not** use or restart the live deployment on port 8743.

Before starting, choose a session that is actively producing assistant messages, or send it a request that will produce several updates over the test period. Keep the same session selected throughout the test unless a step says otherwise.

| Step | Do this on the phone | What you should see when SSE works |
| --- | --- | --- |
| 1 | Open `http://[tailscale-ip]:8443/` in the phone browser. Replace `[tailscale-ip]` with the Tailscale IP of the Codoxear host. | The Codoxear login page loads. |
| 2 | Log in. | The session list and main Codoxear UI appear. |
| 3 | Select a session. | Its existing transcript loads and the selected session is shown as active. |
| 4 | Watch the transcript while the assistant is working. Do not manually refresh the page. | New assistant messages appear in the transcript as they are produced. The page does not need a reload or a tap to show them. |
| 5 | Put the phone to sleep for **30 seconds**. | The display locks or turns off; no action is required while it is asleep. |
| 6 | Wake and unlock the phone, returning to the same browser tab. Do not refresh manually. | The transcript reconnects automatically. Any assistant messages produced during sleep appear, in order, and later messages continue to arrive live. |
| 7 | With the tab still open, switch from Wi-Fi to cellular, or from cellular to Wi-Fi. | After the network changes, the transcript reconnects automatically and resumes live updates. Messages produced during the transition appear without a manual refresh. |
| 8 | Keep the same tab open for **1 hour**. | The live connection stays usable through the hour because SSE heartbeats keep the idle connection alive. When the assistant produces a message, it appears without refreshing; the transcript does not freeze. |
| 9 | Switch to another phone tab or app, then return to the Codoxear tab. | The transcript resumes automatically and catches up with any messages produced while it was in the background. New messages continue to appear live. |
| 10 | Report any disconnects, frozen state, or missed messages. | A successful run has no manual refresh requirement, no permanently stale transcript, and no missing assistant messages after sleep, network changes, the one-hour idle period, or tab switching. |

## What to include in a failure report

- Which numbered step failed and the time it happened.
- Phone model, OS version, browser and browser version.
- Whether the phone was asleep, the tab was backgrounded, or the network changed.
- The last message visible before the problem and the first message visible after returning, including any missing or duplicated messages.
- A screenshot or screen recording of the frozen or disconnected state, if possible.

Do not restart the server, broker, or browser to hide the failure before recording these details. A manual refresh that restores the transcript is useful evidence, but it is still a failure for this test because SSE was expected to reconnect and catch up automatically.
