# Copy conversation oversized export message verification

## Claim
When transcript export is rejected by the known export-size guard, the browser shows a specific conversation-too-large toast rather than generic `copy failed` clipboard/error wording.

## Evidence

- Functional commit: `8d95188 Clarify oversized conversation copy failures`.
- Local validation before commit: focused subset `11 passed`; full local suite `1826 passed, 134 subtests`; `git diff --check` passed.
- Docker focused validation on port 19461: same focused subset `11 passed`.
- Docker smoke on port 19462: pre-login `/api/me=401`, post-login `/api/sessions=200`, container app dir `/home/tester/.local/share/codoxear`.
- Browser proof on port 19464 used an isolated Docker container with `CODEX_WEB_TRANSCRIPT_EXPORT_MAX_BYTES=1024` so the proof could create a tiny runtime-only oversized log and avoid committing bulky data. The committed artifact records the lowered cap; unit tests cover the default 50MiB formatting path.

## Browser observations

`browser/copy-too-large-proof-result.json` proves the real browser path:

- Selected session: `export-too-large-session`.
- Copy Conversation button was enabled before click.
- Clicking the real `#copyConversationBtn` produced toast: `Conversation too large to copy (max 1 KiB). Use search or copy a smaller range.`
- Toast did not match generic `copy failed` wording (`hasGenericCopyFailed:false`).
- Browser-side clipboard write was instrumented and not reached (`clipboardWriteCount:0`), proving failure occurred at export guard projection rather than clipboard write.
- Direct export route returned `413` with `{"error":"transcript log is too large to export (2285 bytes > 1024 bytes)", "max_bytes":1024}`.
- Page had no horizontal overflow during the proof.

## Runtime boundaries

`container/runtime-state.txt` records the isolated app dir, lowered export cap, and runtime-only log size (`2285` bytes). The oversized log itself was not copied into the repo. Container cleanup used exact container name `codoxear-export-too-large-19464` via `docker rm -f`.

## Boundary

The browser proof uses a lowered Docker export cap to avoid committing a >50MiB proof log. The mechanism is the same server 413/max_bytes route shape that default-cap tests cover; no claim is made about changing the export cap.
