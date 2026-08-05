# Continuous idle-session traffic audit

## Decision

Codoxear's warm-cache idle budget is **fewer than 25 requests and fewer than
50 KiB per 60 seconds for one visible, selected, bound session**. The current
60-second deterministic profile is **19 requests / 10,188 budgeted wire
bytes**. This is a floor for an idle session, not a promise for a busy session
whose authoritative state is changing.

The visible session-list cadence is fixed at **5,000 ms** per visible session
view (`sessionsPollDelayMs("visible")`), yielding 12 sidebar polls in a
60-second idle window. An unchanged list response is one initial `200` followed
by 11 ETag-driven `304` responses.

Hidden traffic uses a **5–15 second backoff**: the current polling module
uses 5 seconds for transcript fallback and 15 seconds for session-list polls
(15 seconds offline). A previously claimed 3-second hidden envelope was never
implemented; this document records the executable cadence. Keep that distinction
explicit when changing polling: the 60-second floor below measures the visible
5,000-ms session-list path, not a hidden-page path.

## Baseline

The initial authenticated browser capture ran for 61.38 seconds and recorded
**72 requests / 410,941 wire bytes** (about **410 KiB**), including cold HTML
and immutable static assets. Its continuous API subset was 24 requests /
73,318 wire bytes. Three pre-existing sessions were busy, so `/api/sessions`
changed on 11 of its 17 requests and accounted for 70,130 bytes.

That capture identifies the source of the former traffic, but it is not a
before/after comparison for the current floor: it includes cold-load assets and
changing session rows. The acceptance profile deliberately uses one stable
idle session after assets are warm.

## Current 60-second floor

| Component | Requests | Why it is bounded |
| --- | ---: | --- |
| Bootstrap | 1 | One authenticated page bootstrap call. |
| Session list | 12 | Five-second visible cadence; one `200`, then 11 `304`s for unchanged data. |
| Transcript tail | 1 | Initial selected-session page only. |
| SSE | 1 | The stream is the live-transcript authority. |
| Voice settings | 2 | Thirty-second secondary cadence. |
| Notification feed | 2 | Thirty-second secondary cadence. |
| **Total** | **19** | **10,188 budgeted wire bytes; below 25 requests / 50 KiB.** |

The budget counts response payloads exactly and reserves 512 bytes for each
HTTP exchange's request/response framing, headers, and cookies. It excludes
HTML and versioned static assets because those are cold-load costs rather than
continuous session traffic.

## Fixes that produced the reduction

| Change | Mechanism | Traffic or repeated-work effect |
| --- | --- | --- |
| Poll retuning (`2bf0f325`, then `47bcb687`) | Raised/tuned visible and hidden polling intervals and removed diagnostics from periodic responses. | The visible session-list policy is now five seconds, rather than an aggressive general-purpose refresh loop. |
| SSE live delivery (`29910fc4`) | A selected bound transcript opens one SSE stream; HTTP live polling is a fallback. | Idle transcript state no longer requires repeated tail payload fetches while the stream is healthy. |
| Bounded tail cache (`685782e8`; regression floor `8b7a6a33`, represented in this history by `28dc6b94`) | Initial tail pages are bounded to 8 MiB and cached by `(path, size, mtime_ns, limit)`; an EOF live cursor skips JSONL reopening/parsing. | Unchanged tail/resume reads reuse the cached page instead of replaying a large log. The original measurement improved a 2.245 GiB Pi-log open from 10.7 s to 33 ms and a cached poll to 0.34 ms. |
| Run-settings revision gate (`1ea5b5d8`) | Session-list enrichment records the log's `(device, inode, size, mtime_ns)` revision and replays model/provider/effort only when it changes. | An ETag-cached sidebar poll no longer also scans an unchanged multi-megabyte log. |
| Aggregate VoicePush snapshot | `/api/settings/voice` carries the subscription projection owned by the same VoicePush snapshot. | Visibility/bootstrap refresh does not also fetch `/api/notifications/subscription`; background refresh does nothing when both local announcements and browser notifications are disabled. |
| SSE ownership on visibility resume | An in-flight EventSource has the same transcript/cursor authority as an open one. | Resume reuses the stream rather than opening a duplicate fallback HTTP poll for the same cursor. |

## How to measure

Run the focused contract from the repository root:

```sh
/home/yiwen/.local/share/pipx/venvs/codoxear/bin/python -m pytest -q tests/test_traffic_floor.py tests/test_continuous_traffic_budget.py tests/test_perf_floor_regression.py
```

The tests are local; they do not contact the live deployment.

- `tests/test_traffic_floor.py` runs production create/list handlers behind a
  real loopback HTTP/1.1 server. Its virtual 30-second quiet window executes
  the actual browser-visible cadence and asserts fewer than 25 requests, fewer
  than 50 KiB of measured response wire bytes, fewer than 10 KiB per
  `/api/sessions` response, both `200` and `304` behavior, and no endpoint
  average faster than one request every five seconds.
- `tests/test_continuous_traffic_budget.py` executes the production polling and
  API helpers over the full 60-second warm-cache profile. It asserts exactly
  19 opens, below 25 requests and 50 KiB; 12 session-list polls with 11 `304`s;
  one tail and one SSE open; two voice and two notification-feed refreshes; and
  no redundant subscription snapshot request.
- `tests/test_perf_floor_regression.py` proves that the bounded-tail cache and
  EOF live-poll path keep a one-GiB log within their read-time floor, so the
  request budget cannot hide a repeated log replay. This is the regression
  floor introduced by `8b7a6a33` (the equivalent committed test in this
  branch is `28dc6b94`).

## Remaining gaps and boundary

- There is no valid post-change real-browser capture of one stable idle session.
  The prior post-deploy attempts were invalid because concurrent frontend boot
  failures prevented the session API profile from running; this report does
  not use the live deployment for validation.
- The 60-second byte result is a deterministic wire budget, not browser DevTools
  transfer accounting. The direct HTTP check measures real HTTP/1.1 response
  framing for the session-list path; a Docker-isolated browser capture remains
  the missing end-to-end measurement.
- Busy sessions, changed sidebar rows, transcript events, SSE reconnects,
  enabled push subscriptions, and cold assets legitimately add traffic. They
  are outside this idle floor and should be measured separately rather than
  silently folded into it.
