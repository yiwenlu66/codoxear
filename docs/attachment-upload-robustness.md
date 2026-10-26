# Attachment uploads: consistency and recovery

Attachments stage files only. Sending a prompt remains an explicit, separate
operation; upload retries, status recovery and cancellation never send prompts.
These contracts address concrete consistency and interruption gaps. They do not
establish the cause of a previously reported cellular/Tailscale upload failure.

## HTTP API

`POST /api/sessions/<id>/inject_file` accepts either:

- `Content-Type: application/octet-stream`, a raw binary body with known
  `Content-Length`, `X-Codoxear-Upload-Id`, and `X-Codoxear-Filename` containing an
  `encodeURIComponent`-encoded filename; or
- the legacy JSON `{filename, data_b64, upload_id?}` representation.

The browser uses the raw Blob path, avoiding base64 expansion. The normal file
limit remains 16 MiB (subject to server configuration); oversized/HEIC images may
be converted client-side before transfer. Transfer framing must be fixed-length
and unambiguous. Invalid framing, oversized bodies, incomplete bodies and early
route/auth rejections close the HTTP/1 connection if bytes remain unread.

Identified operations return `ok`, `upload_id`, `state`, `active` when committed,
optional `attachment`, and the current attachment resource:

```json
{"attachments": [], "pending_attachment": false, "attachments_revision": 12}
```

`GET /api/sessions/<id>/attachments/upload-status?upload_id=<id>` returns state
`absent`, `pending`, `committed`, `failed` or `cancelled`. An unknown ID is
`200/absent`; an unknown session remains 404. This is a verification read: the
browser requests `cache: "no-store"`, the service worker passes it through, and
the server returns `Cache-Control: no-store`, including validation errors.

`POST /api/sessions/<id>/attachments/upload-cancel` takes `{upload_id}`. Cancelling
an already committed upload removes its active attachment without resurrecting
it on a later retry. List/delete/clear routes return the same current resource.
Public responses omit server filesystem paths.

## Identity and commit boundary

One random upload ID names one logical selected file and is reused for every
retry. The server grants one active writer per ID. A committed identity points
to its durable result, including after removal or send consumption, so replay
cannot stage a duplicate. Incompatible reuse is rejected. Identity records are
retained with the session resource; there is currently no per-ID expiry or
bounded tombstone pruning. Explicit session deletion removes the resource.

File staging and durable resource publication precede the success response.
Persistence writes a temporary envelope, fsyncs it, atomically replaces the
resource file and fsyncs the parent directory. A directory-fsync failure after
replacement is an uncertain durability outcome, not permission to delete bytes
referenced by the published envelope. Send consumption targets the captured
attachment IDs: an upload that arrives during a send remains staged.

A lost upload response is therefore not a failed commit. The client first looks
up the same ID through the verification endpoint. Committed results reconcile
immediately; retired committed results stay removed. Absent/retryable failures
permit a same-ID retry, with at most two POST attempts per automatic run.
Pending replies permit three status polls with short backoff, never another
independent writer. Status reads have a 15-second complete-response budget;
upload POSTs have no arbitrary short total timeout.

A failed status lookup leaves a persistent **Upload outcome unknown** chip with
its reason, Retry and Cancel. The File is retained in memory for recovery;
Retry checks status before uploading again. Reload does not persist local File
bytes or unfinished operations. Switching sessions aborts local work and
initiates identified cancellation; failures remain recoverable when returning
to the originating session. These are not resumable/chunked transfers: a
pre-commit interruption retries the whole file.

## Revision authority and reload

`attachments_revision` is a nonnegative server-generated monotonic counter for
the session's list and pending flag. Registration, removal, clear, pending
changes and consumption advance it; an empty list retains its revision. The
catalog reconciles these fields together, rejecting older revisions while
allowing unrelated session status fields to refresh normally. A later deletion
wins; list unions and monotonic counts are not used.

The browser stores a best-effort metadata-only checkpoint in localStorage under
`codoxear.attachment-resources.v1`, keyed to session ID and broker `start_ts`.
This prevents an older HTTP cache from erasing acknowledged attachments after
reload without leaking state across a reused broker ID. Limits are 100 session
resources, 32 KiB per resource and 256 KiB overall. Storage denial/eviction means
reload protection is best effort; normal live revision reconciliation remains.

Server `staged_attachments.json` now writes a version-2 envelope:

```json
{"version": 2, "sessions": {"session-id": {"revision": 12, "pending": false, "uploads": {}, "attachments": []}}}
```

Legacy session-to-list maps remain readable and acquire revision 0; subsequent
writes publish v2. The v2 aggregate becomes the list/pending/revision/identity
authority rather than combining stale detached snapshots. Persisted `pending`
upload records become retryable failures on restart because their writer no
longer exists. Committed identities and revisions survive restart and session
remap. Downgrading to a server that only understands the old list format is not
a supported in-place storage migration. Use the explicit downgrade procedure
below instead of simply deploying an older commit.

### Downgrade procedure (operator action, not automatic deployment)

1. Stop **only** `codoxear-server.service`; leave brokers and agent processes
   running. Close browser tabs or otherwise prevent new staging mutations.
2. Back up the complete application state directory, including
   `staged_attachments.json`, `pending_attachments.json`, and `uploads/`, with
   restrictive permissions. Keep this original v2 backup unchanged.
3. From the new-code checkout, run
   `python3 scripts/export_legacy_attachments.py <backup>/staged_attachments.json <new-export-directory>`.
   The tool refuses an existing output directory and never edits its input.
   Inspect the exported lists and confirm every referenced upload file remains.
4. While the server is still stopped, install the exported
   `staged_attachments.json` and `pending_attachments.json` into the application
   state directory with the same owner and mode 0600. Keep the existing upload
   files; the export does not relocate them.
5. Deploy the chosen reviewed older snapshot through `scripts/deploy.sh` (which
   starts the server). Check the staged files before permitting sends/uploads.

Old code loses upload deduplication, committed-ID retirement and revision
ordering. Discard old browser upload retries and reload clients after downgrade.
Re-upgrading cannot recover identity outcomes lost during old-code writes. Do
not restore an earlier v2 envelope over later old-code mutations: that would
undo real attachment changes. Preserve both snapshots for manual recovery.

## Cache delivery and connectivity

A cached payload establishes data freshness, not server unreachability. Slow
cache fallback and locally chosen deadlines do not by themselves light an
unavailable indicator. Browser link loss and actual transport/body failures are
separate evidence. Caller cancellation does not count as outage evidence.

Equivalent finite network-first reads coalesce through full body consumption.
The worker bounds retained reads (8), background reads (3), cache reads (8) and
cache writes (4), and applies per-key refresh backoff. Late healthy responses
can refresh caches; cache write faults do not invalidate a valid live response.
Streaming/range/live routes retain their passthrough behavior. No-store status
reads never reconcile against an offline cache.

## Verification

Run `scripts/docker_upload_robustness_verify.sh <commit>` with Docker and
agent-browser installed. It uses the standard `codoxear-iso3:latest` sandbox
image; set `CODOXEAR_UPLOAD_VERIFY_IMAGE` to a compatible Python application
image if that base is unavailable. `CODOXEAR_UPLOAD_VERIFY_PORT` chooses the
loopback browser port (default 19881; 8743 is forbidden). The harness serves a committed snapshot in an owned
container behind a test-only HTTP proxy; it requires no provider credentials.
It uses the real file input, raw HTTP body, production catalog and worker,
truncated committed responses, failed status reads, partial request bodies and
slow catalog responses. JSON metadata, screenshots and server logs are retained
under `/tmp/codoxear-upload-results.*`; file contents and auth cookies are not
written to diagnostic logs. A read-only fake broker records every mutating
command and the driver requires zero commands. The fixture validates staging
and recovery mechanics, not real provider execution or an iPhone radio path.
