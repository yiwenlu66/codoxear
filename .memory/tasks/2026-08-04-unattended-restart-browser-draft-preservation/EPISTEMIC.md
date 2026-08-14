# Current model

## Phenomenon
A server-only restart has three distinct prompt-preservation boundaries: composer text is browser-owned localStorage, queue items are server-owned JSON keyed by the surviving broker session id, and Unattended configuration is server-owned JSON keyed by stable backend thread/log identity. User-authored text is lost only where it has not yet crossed its owning persistence boundary.

## Supported mechanism
Composer input is synchronously written on every input to `codexweb.draft.<session_id>` and restored when the selected session opens. Accepted queue items persist before the API returns. Saved Unattended configuration persists by thread scope. These committed states survive a server process replacement while the broker/socket remains alive.

## Live defect
Unattended edits are debounced. `flushUnattendedSave()` removes a per-session patch from the pending map before POST; a restart/network error then toasts but does not restore or retry that patch. The browser therefore discards the only remaining copy of a user-authored Unattended request. Browser disposal also clears pending timers/maps, so reload during the debounce window has the same mechanism.

## Required intervention
Keep the server as sole authority for committed Unattended configuration and injection budget. Browser durability should cover only unacknowledged user-authored `request` text: write a revisioned per-session request draft at edit time, reconcile it over server GET until acknowledgement, replay only `{request}` after recovery, and compare-and-delete only the acknowledged revision. Whole-config local replay is unsafe because duplicate `remaining_injections` writes from multiple tabs could regrant a budget consumed between replays. Volatile same-tab retries may retain other dirty fields, but reload durability must not create a second budget authority.

## Ruled-out mechanisms
- Fresh `SessionStore` construction does not lose an already-saved queue or Unattended config.
- Server-only restart does not change the broker socket/session id, so ordinary queue-key identity remains valid.
- Thread-scoped Unattended config supports a new broker session id for the same backend thread; ordinary queued prompts intentionally do not claim broker-restart durability.

## Most discriminating evidence still needed
An executable controller test must show request edit → failed POST → controller disposal/recreation or online recovery → request-only replay → draft cleared only after matching acknowledgement. A stale success must not clear a newer revision, and durable replay must never contain `enabled`, cooldown, or remaining-budget fields. A browser proof must observe distinctive composer and Unattended request text across a supported server-only restart without sending either prompt.

Evidence provenance: OPS.md.
