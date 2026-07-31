# Codoxear Browser/API Certification — HEAD fb42cfd

**Repo under test:** `/home/yiwen/codex-web-product-recovery` (git HEAD `fb42cfdfd3f7c145093cbcc29b21d01761f33428`, working tree clean — no edits, no commits).
**Harness:** `scripts/codoxear-docker-sandbox` on port **19200** (not 8743), image `codoxear-sandbox:latest`, throwaway home `/tmp/codoxear-cert-19200/home` (isolation preflight passed; host live runtime `~/.local/share/codoxear` never touched).
**Browser:** system Chromium 146 (`/usr/bin/chromium`) driven via `puppeteer-core`.
**Artifacts root:** `/tmp/codoxear-cert-19200/cert-artifacts/`

## Method note (fake broker)

The Docker sandbox has no real backend credentials, so the harness fabricates the session state the task explicitly permits: a Python fake broker (`fake_broker_harness.py`) creates Unix control sockets + sidecar JSON + Codex rollout logs under the container app dir and answers the real control-socket protocol (`{"cmd":"state"}` newline-JSON, identical wire shape to `codoxear.broker_control`). The running server discovers these through its **real** discovery + listing + message + control code paths — no product code was stubbed or bypassed. Each sidecar advertises `control_protocol_version: 2` + `control_capabilities: {sync_send, key_write_errors}` so confirmed sends and file-attachment injection are allowed, exactly as a real broker publishes.

For Claim 5 a **stateful** broker (`stateful_broker.py`) models the interrupt→resume→complete state-machine transitions, flipping its `state` response on `/send` so the real `resolve_runtime_status` resolver observes genuine transitions.

---

## Claim 1 — Attachment indicator truth: **PASS**

The paperclip/count badge (`#attachBadge`) is projected by `projectSelectedAttachmentIndicator` = `max(localAttachedFiles, serverPending?1:0)`, reading the server-authoritative `pending_attachment` from `/api/sessions`. Evidence via the real `/inject_file`, `/send` (allow_pending_attachment), and `/pending_attachment/clear` endpoints against session `cert-attach-a`/`cert-attach-b`.

| State | Badge `display` / text | Evidence |
|---|---|---|
| Before attach | `none` / "" | `browser-results.json` `beforeAttach` |
| After `/inject_file` (server pending=true) | `flex` / "1", visible | `afterAttach`; screenshot `claim1-after-attach.png` |
| After full page reload (reads server pending) | `flex` / "1", visible | `afterReload`; screenshot `claim1-after-reload.png` |
| After **send** with allow_pending_attachment | hidden in **40 ms** (before next poll) | `sendIndicator.hiddenAtMs=40`; `afterBadge.display=none`; screenshot `claim1-after-send.png` |
| After re-attach then **clear** | hidden in 1646 ms | `afterClearB.hiddenAtMs`; screenshot `claim1-after-clear.png` |

Server-side confirmation: `/inject_file` → `pending_attachment=true`; `/send` allow_pending → `pending_attachment=false`; `/pending_attachment/clear` → `{ok, pending_attachment:false}`.

**Conclusion:** badge appears after attach, survives reload from server pending state, and disappears immediately after sending or clearing — without waiting for a session refresh.

---

## Claim 2 — Deleted-session attachment cleanup: **PASS**

Cleanup runs in `session_store` via `file_upload.remove_session_uploads(upload_root, session_id)` on `delete_session`. Two evidence paths:

**A. End-to-end via `POST /api/sessions/cert-cleanup/delete`** (fake broker answers `shutdown`):
- Before (`cleanup-before.txt`): `uploads/cert-cleanup/report.pdf` present alongside sibling `cert-sibling/keep.txt`, `cert-attach-a/b`, and a symlink `cert-outside-link -> /home/tester/secret-outside-uploads.txt`.
- After (`cleanup-after.txt`): `uploads/cert-cleanup/` **removed**; `cert-sibling/`, `cert-attach-a/b`, and the symlink all **preserved**; outside target `secret-outside-uploads.txt` intact (42 bytes, original content).

**B. Symlink branch** (`test_symlink_cleanup.py` invokes the real `remove_session_uploads` against a symlink entry whose target is outside uploads) — `symlink-cleanup-result.json`:
- Before: link `cert-symlinksess -> /home/tester/symlink-target-outside.txt`, target 56 bytes.
- After: link **unlinked**, outside target **survives** (`target_content` unchanged). `removed_returned=true`.

**Conclusion:** the deleted session's upload dir is removed; sibling dirs and symlink targets outside `uploads/` are not removed (the link is unlinked, never followed).

---

## Claim 3 — Mobile composer controls ≥ 44×44 CSS px: **PASS**

Viewport 390×844. The 44px floor lives in `@media (max-width: 520px) { .composer .icon-btn { min-width:44px; min-height:44px; } }` (`app.css` ~2718), held regardless of the later coarse-pointer `--composerCtl:40px` override because min-width/min-height dominate.

`browser-results.json` `claim3` (measured both desktop-pointer and touch-emulated):

| Control | width | height | meets44 |
|---|---|---|---|
| attachBtn (paperclip) | 44 | 44 | ✓ |
| queueBtn | 44 | 44 | ✓ |
| sendBtn | 44 | 44 | ✓ |
| composerStopBtn (stop) | 44 | 44 | ✓ |

Horizontal overflow: `composer.scrollWidth == composer.clientWidth == 390`; `body.scrollWidth == 390`; `horizontalOverflow=false`. Screenshot `claim3-mobile-composer.png`.

**Conclusion:** all four composer controls meet 44×44; no horizontal overflow at 390px.

---

## Claim 4 — Transcript/no-answer projection: **PASS**

A Codex rollout log (`cert-noresp`) with a non-empty `user_message` then `task_complete` (no assistant event) is normalized by `rollout_chat_events._inject_no_response_events`, surfacing `_NO_RESPONSE_TEXT`.

**API** (`GET /api/sessions/cert-noresp/messages/tail` → `noresp-tail.json`): two events — `user` "please summarize the report", then `assistant` `"The backend completed this turn without producing a response."` with `message_class: "error"`, `transcript_state: "bound"`.

**Browser** (`browser-results.json` `claim4Browser`): the transcript DOM contains the exact text (`hasNoResponseText: true`, `hitText` matches). Screenshot `claim4-browser-noresp.png`.

**Control** (`cert-normal`, `normal-tail.json`): a normal answered turn renders `assistant` "world" (`message_class: final_response`) and emits **no** no-response event — confirming no false positives.

**Conclusion:** a completed turn with user input but no assistant text renders an explicit no-response error transcript message in both API and browser, not silent idle.

---

## Claim 5 — Idle projection after interrupt/resume: **PASS**

`interrupted_idle` is an internal field (stripped from the public API) that feeds the final `busy` resolution `busy = not (log_idle or interrupted_idle_override)` (`session_runtime.resolve_runtime_status`). The fake broker cannot perform a real Ctrl-C interrupt inside the agent, so the harness seeds the post-interrupt state directly (broker `state` = `{busy:false, interrupted_idle:true}`) and drives resumed activity via the real `/send` endpoint — the same transition a real broker makes when the agent resumes.

**Stateful broker transition** — API timeline (`claim5-api-timeline.json`) and browser sidebar state-dot (`claim5-browser.json`):

| Phase | API `busy` | Sidebar `.stateDot` | Meaning |
|---|---|---|---|
| 0 — post-interrupt idle | `false` | `idle` (gray) | interrupted idle |
| 1 — resumed activity (after `/send`) | **`true`** | **`busy` (blue)** | running — **not falsely idle** |
| 2 — turn completed | `false` | `idle` (gray) | clean idle — **not falsely busy** |

The `/send` response itself returns `busy:true`. Sendability (`remote_ready`) is correctly gated: idle phases are sendable, the running phase is not.

**Missing discriminator / scope note:** the real bug-class this guards — a broker that keeps reporting `interrupted_idle:true` while post-interrupt log activity arrives — is cleared by the stale-interrupted-idle guard in `session_log_runtime` (log growth past `interrupted_idle_log_off` with user/assistant events sets `clear_interrupted_idle`). That guard is exercised at unit level by the existing `tests/test_codex_no_response_projection.py` live-split suite but was **not** separately re-driven through a log-only-without-broker-flip harness here; the end-to-end evidence above proves the user-visible projection via the broker-state path, which is the primary mechanism. No evidence was invented for the log-only variant.

Screenshots: `claim5-phase0-idle.png`, `claim5-phase1-running.png`, `claim5-phase2-complete.png`.

---

## Artifact index (`/tmp/codoxear-cert-19200/cert-artifacts/`)

- `browser-results.json` — Claims 1, 3, 4-browser DOM/JSON summaries.
- `claim1-after-attach.png`, `claim1-after-reload.png`, `claim1-after-send.png`, `claim1-after-clear.png`.
- `claim3-mobile-composer.png`.
- `claim4-browser-noresp.png`; `noresp-tail.json`, `normal-tail.json` (API).
- `cleanup-before.txt`, `cleanup-after.txt`, `symlink-cleanup-result.json`.
- `claim5-api-timeline.json`, `claim5-browser.json`, `claim5-phase{0,1,2}-*.png`.
- Harness sources: `fake_broker_harness.py`, `stateful_broker.py`, `browser_harness.js`, `claim5_browser.js`, `test_symlink_cleanup.py`, `claim5_drive.sh`.

## Residual gaps

- Claim 5 log-only interrupted-idle clear path (broker keeps `interrupted_idle:true` while log grows) was not re-driven end-to-end; covered by broker-state path + existing unit tests. See Claim 5 note.
- Screenshots are real captures but were not OCR-verified in this session (image rendering unavailable to the agent); the DOM/JSON summaries alongside each screenshot are the primary evidence.

## Cleanup

Only the container/processes this task started were touched: `codoxear-sandbox-19200` was stopped via the sandbox `stop` command. Pre-existing containers (`codoxear-sandbox-19130`, `codoxear-nonutf-open-19136`, `taskwarrior-webui`) and the live port-8743 server were not touched. No code was edited, staged, or committed.
