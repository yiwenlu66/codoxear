# Continuous traffic floor

## Scope

The traffic floor is the warm-cache API budget for an authenticated page with
one selected, bound, **idle** web-owned session. It excludes HTML and static
assets because immutable versioned assets are cold-load dependent rather than
continuous traffic.

Two CI checks cover complementary parts of the floor:

- `tests/test_traffic_floor.py` starts a real local HTTP/1.1 server running the
  production session create/list handlers. It creates the temporary web-owned
  session through `POST /api/sessions`, opens it through the list endpoint, and
  uses direct Python `http.client` fetches. Its virtual 30-second window follows
  the executed visible browser polling policy from `app_polling.js`, so it
  completes in under 30 seconds without a browser or deployed server.
- `tests/test_continuous_traffic_budget.py` executes the production polling and
  API helpers against a stable ETag-capable response fixture. It covers the
  broader 60-second warm-cache browser API profile: bootstrap, sidebar, initial
  tail, one SSE open, and enabled notification refreshes.

The 30-second direct-HTTP contract is:

| Contract | Floor |
| --- | --- |
| Total requests | fewer than 25 |
| Response wire bytes | fewer than 50 KiB (status line, headers, and body) |
| Per-endpoint cadence | no endpoint averages more than one request per 5 seconds |
| `/api/sessions` response | fewer than 10 KiB each |

The visible session-list cadence is five seconds. The traffic test evaluates
that policy rather than duplicating its value, so reducing it causes the
per-endpoint cadence assertion to fail.

## Mechanisms that keep the floor low

- Session-list requests carry an ETag; an unchanged one-session row produces
  `304` after the initial response. Log run-settings replay is separately gated
  by file revision (`1ea5b5d8`), so unchanged polls also avoid repeated log
  scans.
- The initial transcript tail uses the bounded `(path, size, mtime_ns, limit)`
  cache. Repeating that tail for an unchanged log returns the cached page; live
  reads at EOF do not reopen or parse the JSONL file.
- `/api/settings/voice` includes the VoicePush subscription snapshot. A
  visibility/bootstrap refresh therefore does not also request
  `/api/notifications/subscription`. Secondary refreshes do no work while both
  announcements and browser notifications are locally disabled.
- An in-flight or open SSE connection is authoritative for live transcript
  delivery. Visibility resume reuses it instead of also scheduling the HTTP
  fallback for the same cursor.

## Baseline and current evidence

The pre-change authenticated browser capture (61.38 seconds, three existing
busy sessions) recorded 72 requests / 410,941 wire bytes including cold assets;
its continuous API subset was 24 requests / 73,318 bytes. `/api/sessions`
dominated it: 17 calls, 70,130 bytes, because those active session rows changed.

The current one-idle-session 60-second fixture has 19 requests / 10,188
budgeted wire bytes: 12 sidebar requests (one `200`, 11 `304`), one initial
tail, one SSE open, two aggregate voice snapshots, two notification feeds, and
one bootstrap call. The scenarios differ: the baseline contains cold assets and
continuously changing sessions, while the current number is the deliberately
stable acceptance contract. No live deployment browser is used for this
validation.
