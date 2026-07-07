import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


class TestAttachButtonSource(unittest.TestCase):
    def test_attach_button_reflects_session_selection(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")

        self.assertIn('attachBtn.disabled = true;', source)
        self.assertIn('if (captureBtn) {\n            captureBtn.disabled = true;\n            captureBtn.title = "Select a session to attach a file";\n            captureBtn.setAttribute("aria-label", "Select a session to attach a file");\n          }', source)
        self.assertIn('function attachmentBlockerForSession(sessionId, sessionInfo = null) {', source)
        self.assertIn('if (!sessionId) return "Select a session to attach a file";', source)
        self.assertIn('if (info && sessionLaunchFailed(info)) return "Failed launch cannot receive file attachments";', source)
        self.assertIn('if (info && sessionHasUnknownSend(info)) return "Resolve the unknown send before attaching a file";', source)
        self.assertIn('if (info && sessionIsOrphanRecovery(info)) return "Missing session can only be reviewed";', source)
        self.assertIn('if (info && sessionHasOrphanQueueRecovery(info)) return "Review preserved queued recovery items before attaching a file";', source)
        self.assertIn('if (currentRunning) return "Wait for the current response to finish before attaching a file";', source)
        self.assertIn('if (sending) return "Wait for the current send to finish before attaching a file";', source)
        self.assertIn('function syncAttachButtonState() {', source)
        self.assertIn('const selectedInfo = selected ? sessionIndex.get(selected) || null : null;', source)
        self.assertIn('const attachBlocker = attachmentBlockerForSession(selected, selectedInfo);', source)
        self.assertIn('const attachLabel = attachBlocker || `Attach file (max ${fmtBytes(ATTACH_UPLOAD_MAX_BYTES)})`;', source)
        self.assertIn('const captureLabel = attachBlocker || `Capture photo (max ${fmtBytes(ATTACH_UPLOAD_MAX_BYTES)})`;', source)
        self.assertIn('attachControl.disabled = Boolean(attachBlocker);', source)
        self.assertIn('attachControl.setAttribute("aria-label", attachLabel);', source)
        self.assertIn('captureControl.disabled = Boolean(attachBlocker);', source)
        self.assertIn('captureControl.setAttribute("aria-label", captureLabel);', source)
        self.assertIn('syncAttachButtonState();\n          updateQueueBadge();', source)

    def test_file_view_button_blocks_failed_launches(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")

        self.assertIn('const fileViewerBlocked = Boolean(selected && selectedSessionLaunchFailed());', source)
        self.assertIn('fileViewerBlocked ? "Failed launch has no file browser" : "View file"', source)
        self.assertIn('fileBtn.disabled = !selected || fileViewerBlocked;', source)
        self.assertIn('fileBtn.setAttribute("aria-label", fileViewerLabel);', source)
        self.assertIn('if (selectedSessionLaunchFailed()) {\n            setToast("failed launch has no file browser");\n            return false;\n          }', source)

    def test_attach_upload_uses_shared_stage_files_pipeline(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")

        self.assertIn('async function stageFiles(files, { sid = selected, source = "picker" } = {}) {', source)
        self.assertIn('const sessionId = sid || selected;', source)
        self.assertIn('const uploadFiles = Array.from(files || []).filter(Boolean);', source)
        self.assertIn('const sessionInfo = sessionId ? sessionIndex.get(sessionId) || null : null;', source)
        self.assertIn('const attachBlocker = attachmentBlockerForSession(sessionId, sessionInfo);', source)
        self.assertIn('if (attachBlocker) {\n            if (!sessionId || selected === sessionId) setToast(attachBlocker);\n            return false;\n          }', source)
        self.assertIn('let uploadName = f.name || (producer === "paste" ? pastedFileName(f, fileIndex, producerNameSeed) : producer === "capture" ? capturedFileName(f, fileIndex, producerNameSeed) : "file");', source)
        self.assertIn('const stem = safeAttachmentStem(uploadName);', source)
        self.assertIn('api(`/api/sessions/${sessionId}/inject_file`', source)
        self.assertNotIn('attachment_index:', source)
        self.assertIn('if (selected === sessionId && res && res.ok) {', source)
        self.assertIn('setSelectedSessionStagedAttachments(Array.isArray(res.attachments) ? res.attachments : []);', source)
        self.assertIn('display_name: String(item.display_name || item.filename || "file"),', source)
        self.assertNotIn('display_name: String(item.display_name || item.filename || item.path || "file")', source)
        self.assertNotIn('path: String(item.path || "")', source)
        self.assertIn('return id ? `${name} · ${size} · attachment ${id}` : `${name} · ${size}`;', source)
        self.assertNotIn('if (path) return path;', source)
        self.assertIn('const failures = [];', source)
        self.assertIn('setToast(`attach error: ${failures[0]}`);', source)
        self.assertIn('handleAppAuthLoss();\n                return false;', source)
        self.assertIn('void refreshSessions().catch((refreshErr) => {', source)
        self.assertEqual(source.count('/inject_file'), 1)

        self.assertIn('imgInput.addEventListener("change", async () => {', source)
        self.assertIn('const sid = selected;\n          if (!sid) return;\n          const files = Array.from(imgInput.files || []);\n          imgInput.value = "";\n          await stageFiles(files, { sid, source: "picker" });', source)
        picker_start = source.index('imgInput.addEventListener("change", async () => {')
        picker_end = source.index('captureBtn.onclick = () => {', picker_start)
        picker_block = source[picker_start:picker_end]
        self.assertNotIn('api(`/api/sessions/${sid}/inject_file`', picker_block)
        self.assertNotIn('sessionHasUnknownSend', picker_block)
        self.assertNotIn('sessionIsOrphanRecovery', picker_block)
        self.assertNotIn('sessionHasOrphanQueueRecovery', picker_block)

    def test_capture_producer_routes_through_stage_files(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")

        self.assertIn('id: "captureBtn",', source)
        self.assertIn('title: "Capture photo",', source)
        self.assertIn('"aria-label": "Capture photo",', source)
        self.assertIn('html: iconSvg("camera"),', source)
        self.assertIn('el("input", { id: "captureInput", type: "file", accept: "image/*", capture: "environment", style: "display:none" })', source)
        self.assertIn('function capturedFileName(file, index, seed) {', source)
        self.assertIn('return `captured-${seed}${suffix}.${ext}`;', source)
        self.assertIn('producer === "capture" ? capturedFileName(f, fileIndex, producerNameSeed) : "file"', source)
        self.assertIn('captureBtn.onclick = () => {\n          const sid = selected;\n          const sessionInfo = sid ? sessionIndex.get(sid) || null : null;\n          const attachBlocker = attachmentBlockerForSession(sid, sessionInfo);', source)
        self.assertIn('captureInput.value = "";\n          captureInput.click();', source)
        self.assertIn('captureInput.addEventListener("change", async () => {\n          const sid = selected;\n          if (!sid) return;\n          const files = Array.from(captureInput.files || []);\n          captureInput.value = "";\n          await stageFiles(files, { sid, source: "capture" });\n        });', source)
        self.assertEqual(source.count('/inject_file'), 1)
        capture_start = source.index('captureBtn.onclick = () => {')
        capture_end = source.index('textarea.addEventListener("paste"', capture_start)
        capture_block = source[capture_start:capture_end]
        self.assertNotIn('api(`/api/sessions/${sid}/inject_file`', capture_block)
        self.assertNotIn('sessionHasUnknownSend', capture_block)
        self.assertNotIn('sessionIsOrphanRecovery', capture_block)
        self.assertNotIn('sessionHasOrphanQueueRecovery', capture_block)

    def test_paste_and_drop_producers_route_through_stage_files(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")

        self.assertIn('textarea.addEventListener("paste", (e) => {\n          const files = extractFilesFromClipboardData(e.clipboardData);\n          if (!files.length) return;\n          e.preventDefault();\n          void stageFiles(files, { sid: selected, source: "paste" });\n        });', source)
        self.assertLess(source.index('if (!files.length) return;\n          e.preventDefault();'), source.index('void stageFiles(files, { sid: selected, source: "paste" });'))
        self.assertIn('addAppEvent(composer, "dragover", (e) => {\n          if (!dataTransferHasFiles(e.dataTransfer)) return;\n          e.preventDefault();\n          if (e.dataTransfer) e.dataTransfer.dropEffect = "copy";\n          setComposerDropActive(true);\n        }, { passive: false });', source)
        self.assertIn('addAppEvent(composer, "drop", (e) => {\n          if (!dataTransferHasFiles(e.dataTransfer)) return;\n          e.preventDefault();\n          composerDragDepth = 0;\n          setComposerDropActive(false);\n          const files = extractFilesFromDropData(e.dataTransfer);\n          if (!files.length) return;\n          void stageFiles(files, { sid: selected, source: "drop" });\n        }, { passive: false });', source)
        self.assertIn('composer.classList.toggle("drop-active", Boolean(active));', source)
        self.assertIn('addAppEvent(window, "dragover", (e) => {\n          if (!dataTransferHasFiles(e.dataTransfer)) return;\n          e.preventDefault();\n        }, { passive: false });', source)
        self.assertIn('addAppEvent(window, "drop", (e) => {\n          if (!dataTransferHasFiles(e.dataTransfer)) return;\n          e.preventDefault();\n          composerDragDepth = 0;\n          setComposerDropActive(false);\n        }, { passive: false });', source)
        window_drop_start = source.index('addAppEvent(window, "drop", (e) => {')
        window_drop_end = source.index('}, { passive: false });', window_drop_start)
        self.assertNotIn('stageFiles(', source[window_drop_start:window_drop_end])
        producer_start = source.index('textarea.addEventListener("paste", (e) => {')
        producer_end = source.index('function clearComposer()', producer_start)
        producer_block = source[producer_start:producer_end]
        self.assertNotIn('sessionHasUnknownSend', producer_block)
        self.assertNotIn('sessionIsOrphanRecovery', producer_block)
        self.assertNotIn('sessionHasOrphanQueueRecovery', producer_block)
        self.assertNotIn('sessionLaunchFailed', producer_block)

    def test_running_turn_cannot_split_attachments_into_queue(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")

        self.assertIn('sendChoicePending = { sid: selected, text: raw, attachmentCount: stagedAttachments.length };', source)
        self.assertIn('const hasAttachments = Boolean(sendChoicePending && sendChoicePending.attachmentCount > 0);', source)
        self.assertIn('laterBtn.disabled = hasAttachments;', source)
        self.assertIn('"Attachments cannot be queued; send now or wait until idle"', source)
        self.assertIn('setToast("attachments can only be sent now; wait until idle to queue text with files");', source)

    def test_attach_indicator_reconciles_selected_session_pending_attachment(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")

        # One projection function renders the visible badge from the server-owned
        # staged attachment list, with pending_attachment as compatibility fallback.
        self.assertIn('function projectSelectedAttachmentIndicator() {', source)
        self.assertIn("const sessionInfo = selected ? sessionIndex.get(selected) : null;", source)
        self.assertIn("const serverPending = Boolean(sessionInfo && sessionInfo.pending_attachment);", source)
        self.assertIn("const visible = Math.max(stagedAttachments.length, serverListCount, serverPending ? 1 : 0);", source)
        # setAttachCount must not render the badge inline; it delegates to the
        # projection so the selected session's pending_attachment is honored.
        self.assertIn('attachedFiles = Math.max(0, Number(n) || 0);', source)
        self.assertIn('projectSelectedAttachmentIndicator();\n        };', source)
        self.assertNotIn('attachBadgeEl.textContent = String(next);', source)
        # The projection is called where the selected session's sessionInfo is
        # refreshed, so server-side flips reconcile without a local counter change.
        self.assertIn(
            'applySessionListTranscriptIdentity(selected, sessionIndex.get(selected));\n                syncRecoveryUiForSession(selected);\n              }\n              if (selected) syncStagedAttachmentsFromSelectedSession();',
            source,
        )
        # Send authority uses the selected staged list count and the server
        # pending flag still prompts explicit send for legacy pending state.
        self.assertIn('const localAttachmentCount = renderHere ? stagedAttachments.length : normalizedStagedAttachments(sessionInfo && sessionInfo.staged_attachments).length;', source)
        self.assertIn('let allowPendingAttachment = Boolean(localAttachmentCount > 0);', source)
        self.assertIn('if (!allowPendingAttachment && sessionInfo && sessionInfo.pending_attachment) {', source)

    def test_selected_session_pending_attachment_cache_updates_on_direct_evidence(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")

        # Helper mutates the cached pending_attachment only for the selected
        # session, then re-projects. Without it, setAttachCount(0) after a send
        # or clear re-renders against stale pending_attachment=true.
        self.assertIn('function setSelectedSessionPendingAttachment(value) {', source)
        self.assertIn("info.pending_attachment = Boolean(value);", source)
        self.assertIn("sessionIndex.set(selected, info);", source)
        self.assertIn("projectSelectedAttachmentIndicator();", source)
        # Successful attach returns the authoritative staged list.
        self.assertIn('setSelectedSessionStagedAttachments(Array.isArray(res.attachments) ? res.attachments : []);', source)
        # Successful send with allow_pending_attachment is direct evidence the
        # pending attachment was consumed only if post-send staged cleanup also
        # succeeded; cleanup failure means the delivered turn must not erase the
        # still-staged browser/server projection.
        self.assertIn('const attachmentCleanupErrorRaw = res && (res.attachment_cleanup_error || res.attachments_cleanup_error);', source)
        self.assertIn('if (attachmentCleanupError) cleanupWarnings.push(`attachment cleanup failed: ${attachmentCleanupError}`);', source)
        self.assertIn('if (sendStateCleanupError) cleanupWarnings.push(`send state cleanup failed: ${sendStateCleanupError}`);', source)
        self.assertIn(
            'if (allowPendingAttachment && !attachmentCleanupError) {\n              setSelectedSessionPendingAttachment(false);\n              setAttachCount(0);\n            }',
            source,
        )
        # Successful pending_attachment/clear is direct evidence the flag is now false.
        self.assertIn(
            'if (selected === sessionId) setSelectedSessionPendingAttachment(false);',
            source,
        )
        # The refresh path still re-projects from server truth, so a cache update
        # never shadows an authoritative refresh.
        self.assertIn(
            'applySessionListTranscriptIdentity(selected, sessionIndex.get(selected));\n                syncRecoveryUiForSession(selected);\n              }\n              if (selected) syncStagedAttachmentsFromSelectedSession();',
            source,
        )


if __name__ == "__main__":
    unittest.main()
