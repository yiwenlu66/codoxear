# Current model

## Phenomenon

A visible idle page used independent periodic transports for three state surfaces: sidebar sessions, transcript delivery, and voice/notification state. The prior log-revision change eliminated unchanged-log settings scans from sidebar polls. The remaining unnecessary work was at the boundaries between transports: voice settings and notification subscription snapshots were requested together, and visibility resume could start SSE while also scheduling its HTTP fallback.

## Mechanisms and intervention

- `VoicePush.settings_snapshot()` already owns both the redacted voice state and the subscription map under one lock. `voice_settings_snapshot_payload()` now includes the public subscription projection. `refreshBackgroundState()` consumes that aggregate snapshot, then fetches the feed only for a local notification opt-in. With neither announcements nor notification opt-in enabled, a secondary tick makes no request.
- An EventSource can be in flight before its `open` callback. That transport has the same cursor authority as the fallback. `openMessageEventSource()` now returns whether a stream is available/in flight, and `resumeLiveDelivery()` starts fallback polling only when it cannot use SSE. The visibility handler no longer independently kicks a message poll.
- The tail page cache was already keyed by immutable log identity `(path, size, mtime_ns, limit)`, and live polling already avoids JSONL reads at EOF. Route coverage now confirms two unchanged tail requests execute the bounded reverse scan only once.

## Evidence

`tests/test_continuous_traffic_budget.py` executes the production polling/API helpers over a deterministic 60-second idle timeline. It records 19 API/stream opens and 10,188 budgeted wire bytes (exact response fixture payloads plus a conservative 512-byte HTTP envelope per exchange): below the acceptance limits of 30 requests and 50 KiB. It includes 12 sidebar polls (1 `200`, 11 `304`), one initial tail, one SSE open, two aggregate voice snapshots, and two empty notification feeds; it makes no subscription-snapshot request.

The earlier authentic browser baseline was 24 continuous API calls / 73,318 bytes in 61.38 seconds with three busy sessions. Its session rows changed, so its 17 sidebar polls produced 11 `200` responses; it is not comparable to the new stable one-idle-session floor. No live deployment browser was used for the current validation. See `PERFORMANCE.md` and OPS.md.

## Current claim

For the explicitly defined warm-cache, one-idle-session case, Codoxear is bounded below 30 opens and 50 KiB in 60 seconds. Active sessions can exceed this byte floor when their authoritative sidebar or transcript state changes; that is product traffic rather than periodic duplicate work.

## Question that would most change the model

A Docker browser capture with a genuinely stable bound session would measure real HTTP framing and browser transfer accounting against the 10,188-byte modeled budget. The current contract is behavioral and deterministic but does not replace that environment-specific measurement.

## Docker capture attempt

A 390×844 isolated browser run was held idle for 60 seconds. Two tcpdump attempts from Docker produced no usable pcap: a container-network capture observed zero packets, and a privileged host-network container was denied capture on host `lo` (`Operation not permitted`). The measured traffic floor therefore remains the exercised deterministic contract: 19 API/stream opens and 10,188 modeled wire bytes in 60 seconds. The missing real packet capture is a verification-environment limitation, not evidence that the floor was observed.

