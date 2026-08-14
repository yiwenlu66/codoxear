# Current model

## Phenomenon
The frontend message transport and confirmed-send state were distributed across `app.js` and `app_composer.js`, coupling UI composition to EventSource state, polling timers, live cursor application, typing counters, and send recovery.

## Mechanism now implemented
`app_message_flow.js` is the single message data-flow controller. It owns confirmed sends, initial-tail/live-poll request cancellation, EventSource open/error/retry state, poll timers/backoff/fast windows, the shared SSE/poll `applyLiveMessageData` reducer, and typing-counter reconciliation. `app.js` retains session selection and the selected-session generation (`pollGen`); both are injected, so stale async results remain mechanically rejected after selection changes. `app_composer.js` owns input/picker/modal UI and delegates send execution.

SSE and HTTP polling still converge at one reducer. The active transcript slot remains live-cursor authority. On an SSE error, polling reads from the last cursor applied by the SSE reducer; retry later reads the cursor advanced by fallback polling. The standalone browser observation showed `c1` EventSource → SSE applied `c2` → poll requested `c2` and applied `c3` → EventSource reopened at `c3` (OPS.md).

Typing remains one-store/two-feed: live meta deltas increment `typingRowRuntime`, while session snapshots can only raise counts during an open turn and replace the seed while idle.

## Ruled out
Splitting EventSource construction from polling/live application would create competing lifecycle authorities. Keeping confirmed send in composer would leave data mutation and recovery in an input UI controller. Both were avoided.

## Current justified claim
The extraction preserves original transport and send semantics under executable controller, transcript, route/SSE, cursor, static-asset, and real-browser standalone checks (OPS.md).

## Remaining boundary
A post-deploy live-service/browser send and server-restart SSE-resume check remains the supervisor's release verification. Docker was unavailable locally; no host server or broker was started.
