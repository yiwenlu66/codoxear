# Quick-session causal model

## Contract
A quick launch is a gesture-time fresh launch in server HOME, not a queued send. The last successful full-dialog launch is its sole preference authority; last-shown backend and per-backend provider/model memories remain ordinary dialog preferences. Quick starts never write the last-launch record.

## Mechanism
`app_quick_session.js` owns buttons, tooltip, startup copy and transient pending launch identity. Composer owns textarea/send state and subscribes to startup: typing enabled, Send refused. Lifecycle deselection saves the old draft and invalidates transcript requests. Lifecycle launch accepts an optional selection callback so quick state can suppress late auto-selection after manual navigation. Immediate broker discovery uses this callback; pending launch ids use the existing catalog feed, without a new poll loop.

A launched session's selection synchronously loads its local draft. Quick state captures text before calling selection, then uses the existing restore-draft seam immediately after the local load. That seam refuses nonempty drafts and writes via composer `setDraftFromServer` plus draft-sync `noteDraftEdited`. Failure clears startup and opens the full dialog with an explicit preset; ordinary dialog opening remains unchanged.

## Evidence and boundary
See OPS.md. The 44-test Docker VM subset covers successful-only record writes, defaults/providerless fallback, launch arguments, startup typing/send refusal, capture-before-selection, nonempty target protection, manual-selection cancellation, pending/failed catalog rows, shortcut guards, and failure preset preservation. The assembled Docker/browser interface proves launch settings, server HOME expansion, gesture-time focus, startup send gating, draft handoff, broker-acknowledged Send with real Pi PTY receipt, server-error fallback, and narrow drawer/focus/overflow behavior.

The absence of a successful assistant reply is an environment/provider boundary, not evidence against quick launch: the same prompt through the ordinary full dialog produced the same initial missing-file API-key error. Copying the literal credential dependency got past auth; the ordinary dialog then exposed an upstream 400 rejecting the configured default model `auto-jev-main`. The verifier asserts broker ACK plus real Pi PTY receipt and explicitly records whether an assistant reply occurred. It never equates a UI echo with backend receipt. A working model/provider configuration is still required to verify a successful model answer.
