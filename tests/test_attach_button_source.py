import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


class TestAttachButtonSource(unittest.TestCase):
    def test_attach_button_reflects_session_selection(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")

        self.assertIn('attachBtn.disabled = true;', source)
        self.assertIn('function syncAttachButtonState() {', source)
        self.assertIn('"Select a session to attach a file"', source)
        self.assertIn('"Failed launch cannot receive file attachments"', source)
        self.assertIn('} else if (selectedSessionLaunchFailed()) {\n            attachLabel = "Failed launch cannot receive file attachments";\n            disabled = true;\n          } else if (selectedSessionHasUnknownSend()) {', source)
        self.assertIn('"Resolve the unknown send before attaching a file"', source)
        self.assertIn('"Missing session can only be reviewed"', source)
        self.assertIn('"Wait for the current response to finish before attaching a file"', source)
        self.assertIn('"Wait for the current send to finish before attaching a file"', source)
        self.assertIn('attachControl.disabled = disabled;', source)
        self.assertIn('attachControl.setAttribute("aria-label", attachLabel);', source)
        self.assertIn('syncAttachButtonState();\n          updateQueueBadge();', source)

    def test_file_view_button_blocks_failed_launches(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")

        self.assertIn('const fileViewerBlocked = Boolean(selected && selectedSessionLaunchFailed());', source)
        self.assertIn('fileViewerBlocked ? "Failed launch has no file browser" : "View file"', source)
        self.assertIn('fileBtn.disabled = !selected || fileViewerBlocked;', source)
        self.assertIn('fileBtn.setAttribute("aria-label", fileViewerLabel);', source)
        self.assertIn('if (selectedSessionLaunchFailed()) {\n            setToast("failed launch has no file browser");\n            return false;\n          }', source)

    def test_attach_upload_uses_selection_captured_at_file_pick(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")

        self.assertIn('const sid = selected;\n\t\t          if (!sid) return;', source)
        self.assertIn('if (selectedSessionLaunchFailed()) {\n\t            setToast("failed launch cannot receive file attachments");\n\t            return;\n\t          }', source)
        self.assertIn('const sessionInfo = sessionIndex.get(sid) || null;\n\t\t          if (sessionInfo && sessionLaunchFailed(sessionInfo)) {', source)
        self.assertIn('imgInput.value = "";\n\t\t            if (selected === sid) setToast("failed launch cannot receive file attachments");\n\t\t            return;', source)
        self.assertIn('const attachmentIndex = attachedFiles + 1;', source)
        self.assertIn('if (currentRunning) {\n\t\t            if (selected === sid) setToast("wait for the current response before attaching a file");\n\t\t            return;\n\t\t          }', source)
        self.assertIn('api(`/api/sessions/${sid}/inject_file`', source)
        self.assertIn('attachment_index: attachmentIndex', source)
        self.assertIn('if (selected === sid) {', source)
        self.assertIn('setAttachCount(attachmentIndex);', source)
        self.assertIn('const commitUnknown = Boolean(e && e.obj && e.obj.commit_unknown);', source)
        self.assertIn('setToast("attachment status unknown; check before retrying");', source)
        self.assertIn('setToast(`attach error: ${e.message}`);', source)

    def test_running_turn_cannot_split_attachments_into_queue(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")

        self.assertIn('sendChoicePending = { sid: selected, text: raw, attachmentCount: attachedFiles };', source)
        self.assertIn('const hasAttachments = Boolean(sendChoicePending && sendChoicePending.attachmentCount > 0);', source)
        self.assertIn('laterBtn.disabled = hasAttachments;', source)
        self.assertIn('"Attachments cannot be queued; send now or wait until idle"', source)
        self.assertIn('setToast("attachments can only be sent now; wait until idle to queue text with files");', source)


if __name__ == "__main__":
    unittest.main()
