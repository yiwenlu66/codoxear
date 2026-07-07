---

## Verdict: **ACCEPT**

### No blockers found.

### Product invariant verification

The invariant holds: known transcript export-size 413 responses produce a specific conversation-too-large toast; unrelated copy/export/clipboard failures remain generic.

**Discrimination mechanism** (`transcriptExportTooLargeCopyMessage`): three-gate check requiring all of (1) `status === 413`, (2) `obj.max_bytes` property exists, (3) error text matches transcript-export-specific patterns. Only the transcript export route in `message_routes.py:332-334` and `message_routes.py:310-317` produce this response shape. The only other 413 source (`server_http.py:41` for `RequestPayloadTooLargeError`, `control_routes.py:356` for file upload) returns `{"error": ...}` without `max_bytes` — structurally impossible to match the filter.

**Server cap unchanged**: zero diff in `message_routes.py`, `server_config.py`, `server_route_deps.py`. Default remains `50 * 1024 * 1024`.

**Route shape preserved**: the 413 response body `{error, max_bytes}` is unchanged; new test `test_messages_export_active_session_retains_size_cap` explicitly asserts `status == 413`, `body["max_bytes"] == 8`, and `"too large to export" in error`.

**No auth/clipboard regression**: `app_clipboard.js` has zero diff. `app_api.js` has zero diff. The only catch-path change is replacing inline `"copy failed: ..."` with `copyConversationFailureToast(err)` which returns the same generic text for non-export errors.

**No committed secrets or bulky artifacts**: proof directory totals 124 KB across 24 files (largest 4.4 KB). Searched for password/secret/token/cookie/hmac/credential patterns — all hits are structural nulls (`"token": null`, `"preferred_auth_method": null`). No oversized log file committed.

### Evidence quality assessment

**Lowered-cap browser proof is valid.** The browser proof used `CODEX_WEB_TRANSCRIPT_EXPORT_MAX_BYTES=1024` to trigger a 413 from a 2285-byte runtime-only log, avoiding committing a >50 MiB artifact. This tests the full end-to-end UI mechanism (button click → API fetch → 413 discrimination → toast rendering). The unit test `test_transcript_export_too_large_helper_recognizes_api_error_shape` separately exercises `max_bytes: 52428800` through `formatCopyLimitBytes` and asserts `"50 MiB"` appears in the output. The two together cover both the mechanism and the production formatting.

**Test coverage is thorough.** Five discrimination cases tested (known export error, tagged form, unrelated 413, missing max_bytes, network error). Generic failure preservation tested (`"copy failed: denied"`, `"copy failed: unknown error"`). Guard-check updated to require `transcriptExportTooLargeCopyMessage` on the module. Both server 413 paths (active session ValueError, missing-recovery manual size check) have route-level tests.

### Decision frame

This is a genuine product gap. Before this change, users saw internal byte counts and "transcript log" terminology when hitting the export cap — implementation leakage that mischaracterizes a deliberate size limit as a copy failure. The fix is scoped entirely to frontend error presentation, preserves the server guard, and suggests actionable alternatives. It belongs to the problem domain.

### Non-blocking concerns

1. **`formatCopyLimitBytes` rounding edge case**: Values like `0.95 * 1024 * 1024` (just under 1 MiB) would show as `~950 KiB` rather than `~0.9 MiB`. This is cosmetically acceptable — the 50 MiB default renders cleanly, and fractional-MiB caps are unlikely in practice.

2. **Proof artifacts add ~124 KB to the repo permanently.** Not excessive, but the `api/sessions-after-proof.pretty.json` (4.4 KB, 132 lines of session defaults) could have been trimmed to the relevant session entry. Not worth blocking over.