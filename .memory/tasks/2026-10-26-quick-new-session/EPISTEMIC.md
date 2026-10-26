# Quick-session causal model

## Contract
A quick launch is a gesture-time fresh launch in server HOME, not a queued send. The last successful full-dialog launch is its sole preference authority; last-shown backend and per-backend provider/model memories remain ordinary dialog preferences. Quick starts never write the last-launch record.

## Mechanism
`app_quick_session.js` owns buttons, tooltip, startup copy and transient pending launch identity. Composer owns textarea/send state and subscribes to startup: typing enabled, Send refused. Lifecycle deselection saves the old draft and invalidates transcript requests. Lifecycle launch accepts an optional selection callback so quick state can suppress late auto-selection after manual navigation. Immediate broker discovery uses this callback; pending launch ids use the existing catalog feed, without a new poll loop.

A launched session's selection synchronously loads its local draft. Quick state captures text before calling selection, then uses the existing restore-draft seam immediately after the local load. That seam refuses nonempty drafts and writes via composer `setDraftFromServer` plus draft-sync `noteDraftEdited`. Failure clears startup and opens the full dialog with an explicit preset; ordinary dialog opening remains unchanged.

## Evidence and boundary
See OPS.md. Behavioral VM tests cover successful-only record writes, defaults/providerless fallback, launch arguments, startup typing/send refusal, capture-before-selection, nonempty target protection, manual-selection cancellation, pending/failed catalog rows, shortcut guards, and failure preset preservation. Browser verification must still demonstrate the assembled interface with an actual Docker Pi broker and server HOME expansion.
