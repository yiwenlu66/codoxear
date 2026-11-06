# Federation: node ↔ hub

Goal: machine B (node) joins machine A (hub). A's web UI shows B's sessions
with a host badge and offers the full session experience (view transcript,
send, interrupt, files, git) exactly as if local. Both directions are
optional: without federation env vars each server is standalone.

Non-goal for v1: cross-node search across rotated history logs, unread badges
for remote sessions, video/office preview transcoding on remote files (these
fall back gracefully or proxy to the node, see below).

## Why these seams (from code survey)

Local architecture today:

- Brokers/sessiond listen on unix sockets `APP_DIR/socks/*.sock` with a
  sidecar `<stem>.json` meta (see "Sidecar schema" below).
- Server discovers sessions by scanning that dir (session_discovery.py),
  liveness = `pid_alive(broker_pid)` (watchdog, 30s sweep, 60s grace, then
  UNLINKS sock+json) and stale rules requiring BOTH broker_pid and codex_pid
  dead. A session registers iff sock+json exist, meta validates, and a sock
  RPC `{"cmd":"state"}` answers within 0.5s.
- Control ops (broker_control.py:200-207): `state`, `tail`, `send`, `keys`,
  `settings` (broker only), `shutdown` — JSON-line request/response over the
  unix socket.
- Content routes (`/api/sessions/<id>/messages/*`, `/file/*`, git views) read
  LOCAL files keyed by session.log_path / session.cwd, scattered across many
  low-level readers — no choke point.

Design consequences:

1. **Control plane: mirror + relay.** Hub mirrors each node session's sidecar
   (with rewrites below) into its own socks dir and creates a per-session
   relay unix socket that forwards the JSON-line RPC to the node. The entire
   discovery/watchdog/control path runs unchanged.
2. **Content plane: HTTP reverse proxy.** A node's own codoxear-server already
   serves all content routes correctly for its local sessions. When hub's HTTP
   layer sees `/api/sessions/<id>/...` for a remote session id, it proxies the
   request (method, path, query, body, streaming SSE) to the node server. Zero
   changes to message/file/git handlers.
3. Session rows carry `node: <name>`; the sidebar shows a small host badge and
   the create dialog offers a host selector (only when nodes are attached).

## Sidecar mirror rewrites (validated against discovery + watchdog)

When hub mirrors node sidecar meta:

- `broker_pid` → a live HUB pid (the federation listener process = the hub
  server itself, `os.getpid()`). This is the only liveness authority the
  watchdog checks.
- `codex_pid` → same hub pid (required int; both-dead rules then never fire).
- `log_path` → key present, value `null` (hub must not read node paths;
  content routes for remote sessions are proxied instead). Discovery accepts
  null log_path when pids are alive.
- keep `session_id`, `cwd`, `start_ts`, `owner`, `agent_backend`,
  `control_protocol_version`, `control_capabilities`, `slash_commands`,
  run-settings fields as-is.
- Hub adds its own bookkeeping OUTSIDE the json (node name), keyed by
  session_id in the federation registry; Session objects get `.node_name`.

Mirrored files: `APP_DIR/socks/<session_id>.sock` (relay listener) +
`<session_id>.json` (rewritten meta). Node session ids are agent thread
UUIDs — collision-safe across hosts.

## Wire protocol (hub listener :8744, length-prefixed JSON frames)

4-byte big-endian length + UTF-8 JSON. First frame from node:
`{"type":"hello","name":...,"token":HMAC...,"version":1}` — hub answers
`{"type":"welcome"}` or closes. Auth: `token = sha256(shared_secret + ":" +
node_name)` hex; shared secret from env on both sides.

Node → hub:

- `{"type":"register","session":{...sidecar meta...}}` (also used for meta
  updates — hub rewrites sidecar; watchdog-refresh)
- `{"type":"unregister","session_id":...}`
- `{"type":"sock_result","rid":...,"ok":true,"response":{...}}` /
  `"ok":false,"error":...`
- `{"type":"launch_result","rid":..., ...}`

Hub → node:

- `{"type":"sock_rpc","rid":...,"session_id":...,"request":{...broker op...}}`
  — node runs `call_control_socket(local sock, request)` and replies.
- `{"type":"launch","rid":...,"params":{cwd,agent_backend,model,...}}` — node
  spawns `python -m codoxear.broker` / sessiond with the same conventions as
  session_launcher (backend from `agent_backend`, args after `--`); the new
  session arrives via the node's own watcher `register`. The hub's create
  route (`POST /api/sessions`) accepts an optional `"node": <name>` field
  that dispatches here; without it, launch stays local.

Reconciliation: on (re)connect, node sends `register` for every live session;
hub drops mirrored sessions of a node when its connection closes (unlink
relay sock + sidecar + registry row; the sessions reappear on rejoin).

## HTTP content proxy (hub)

In the hub request dispatcher, before routing: if path matches
`/api/sessions/<id>/...` and `<id>` belongs to a node → reverse-proxy to
`http://<node_host>:<node_http_port><original path and query>`, streaming the
response (SSE safe: no buffering; the hub copies chunks as they arrive and
closes the client connection at end of body). Node advertises its HTTP port
in `hello`. The browser's normal cookie auth is enforced on the hub before
proxying; hub → node requests authenticate with the header
`X-Federation-Token: sha256(secret + ":" + node_name)`, which the node's auth
check accepts for API routes when it is configured as a federation node.

v1 deviations from the original sketch:

- Only `/api/sessions/<id>/...` paths are proxied; global file routes whose
  request *body* names a remote session are not (the frontend uses
  session-scoped routes for remote sessions).
- On node disconnect the hub unlinks the relay sock + sidecar AND drops the
  mirrored rows from the session registry outright. Mirrored sidecars point
  both pids at the hub's own live process, so the watchdog would immediately
  clear any `lost` tombstone; removal is the only consistent state, and a
  rejoining node simply re-registers.

## Config

- Hub: `CODEX_WEB_FEDERATION_LISTEN=0.0.0.0:8744`,
  `CODEX_WEB_FEDERATION_TOKEN=<shared secret>`
- Node: `CODEX_WEB_FEDERATION_HUB=<hub-host>:8744`,
  `CODEX_WEB_FEDERATION_TOKEN=<same>`, `CODEX_WEB_NODE_NAME=<display name>`
  (default: hostname)

Absence of these vars = standalone behavior, zero overhead.

## Files

- `codoxear/federation_proto.py` — frame codec, auth, message helpers
- `codoxear/federation_node.py` — node client: watcher, RPC executor, launcher
- `codoxear/federation_hub.py` — listener, node registry, relay sockets,
  sidecar mirror, launch dispatch
- `server.py` — start/stop hooks, content-proxy dispatch, `node` in session
  rows, create-route node dispatch
- frontend: `app_sessions*.js` badge, create dialog host selector
- tests: `tests/test_federation_*.py` — two in-process servers on loopback

## Security note

v1 is LAN-scoped: HMAC token auth, no TLS on the federation port. For
untrusted networks, tunnel: `ssh -L 8744:127.0.0.1:8744 hub`. The token never
appears in URLs or the browser.
