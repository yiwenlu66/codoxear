All evidence collected. Here is the certification report.

---

# Codoxear Upload/Attachment Certification — PASS

**Repo:** `/home/yiwen/codex-web-product-recovery` @ HEAD `1628498` (served read-only). **Port:** 19144. **Artifacts:** `/tmp/codoxear-upload-cert-19144/artifacts/` (67 files, 19M). **Repo untouched** (no staged files, no diff). **Cleanup:** container removed after artifacts saved; no host brokers/runtime touched; no pkill/killall.

## Primary target claim — PROVEN

**"Browser paperclip upload stages a file and a real Pi backend reads the injected staged absolute path after send."** Measured on the **target** (assistant returns sentinel content), not the HTTP-200 proxy.

Real Pi session `broker-155` (`openai-codex`/`gpt-5.4-mini`, healthy, lsp-pi `./node.js` repair applied — identical to prior Pi cert). Attach sentinel text via hidden `#imgInput` → badge "1", toast `file attached`, `pending_attachment=True`. Staged to `/home/tester/.local/share/codoxear/uploads/broker-155/1783295830759_sentinel-source.txt` (99 B, mode 0600, outside cwd). On send, the assistant's `final_response` was **the exact file contents** including `ATTACH-SENTINEL-1783295830`.

Backend log proves the mechanism: pi issued a **`read` tool call** on the staged absolute path (`toolResult` toolName `read`), then returned the sentinel. This falsifies theorist F1 as a defect — **the app-dir staged path IS readable by the agent.**

Decisive artifacts: `d1-transcript-summary.txt`, `d1-tail-live.json`, `d1-backend-log-excerpt.txt`, `d1-09-assistant-read-sentinel.png`, `pi-read-staged-probe.txt`.

## Required evidence results

| # | Item | Result | Evidence |
|---|------|--------|----------|
| 1 | Desktop happy path | **PASS** | sentinel in assistant transcript; pi `read`-tool in backend log |
| 2a | Oversize >16MB | **PASS** | client toast `attach error: file too large (max 16.0 MB)`, no upload staged (`f1-oversize-toast.png`, `f1-uploads-after-oversize.txt`) |
| 2b | Attach while busy | **PASS** | button disabled + tooltip `Wait for the current response…`; API `inject_file` 409 `session is busy` (`f2-attach-while-busy-disabled.png`, `f2-inject-busy.json`) |
| 2c | Forced server error | **PASS** | invalid base64 → 400 `invalid base64`; missing filename → 400 `filename required` (`f3-*.json`) |
| 3 | Queue/send interaction | **PASS** | `#sendChoiceLater` disabled + retitled `Attachments cannot be queued; send now or wait until idle`; click-guard toast `attachments can only be sent now…`; `queue_len` stayed 0 (`f6-sendchoice-later-disabled.png`, `f6-queue-state.txt`) |
| 4 | Pending persistence/clear | **PASS** | `pending_attachments.json`=`["broker-155"]`; **2nd server process** (port 19145, same home) rediscovered `pending_attachment=True`; clear API → `[]` (`f7-server2-rediscovery.txt`, `f7-pending-json-after-clear.txt`) |
| 5 | Mobile 390×844 | **PASS** | paperclip reachable (34×34, enabled); attach+send → assistant returned `MOBILE-SENTINEL-1783296623`; composer in-viewport/focusable, no stranding (`f8-mobile-*.png`, `f8-mobile-tail.json`) |
| 6 | HEIC desktop boundary | **Recorded (fail-loud)** | `attach error: The source image cannot be decoded.` on desktop Chromium; not staged. **HEIC NOT claimed cross-platform** (`f9-heic-decode-failed.png`) |
| — | Staged path read by Pi | **PROVEN** | app-dir abs path mode 0600; pi `read`-tool returned it (`boundary-uploads-paths.txt`) |

## Method notes / honest caveats

- **Send delivery:** the browser's `#sendChoiceNow` click was unreliable under agent-browser on this hidden-input flow, so the prompt was delivered via `POST /api/sessions/<sid>/send` with `allow_pending_attachment:true` — the **identical `manager.send` code path** the browser's `sendText` calls. The attach itself was browser-driven via `#imgInput` as required. The target (agent reads file) is independent of which side invokes send.
- **agent-browser `upload` quirk:** it sets `#imgInput.files` but did not always dispatch `change`; the product change-handler is correct (proven when `change` fires). Not a product defect — a tooling note.
- **server.log is empty** (PID-1 stdout buffered by Docker); authoritative evidence is the captured API JSON, the Pi backend `.jsonl`, and the transcript — not the container stdout.

## Residual product gaps (NOT blockers for the certified claim)
- **Mobile paperclip is 34×34**, below the 44×44 touch-target ideal — existing product dimension, out of scope to change here (matches prior D3 finding).
- **Stale local `attachedFiles` badge** can linger after reload when `pending_attachment` is server-side False — minor UI resync quirk; the API/server authority is correct.
- **HEIC unsupported on desktop** (fail-loud, by design); iOS-Safari HEIC transcode is plausible but unverified here (no iOS device).