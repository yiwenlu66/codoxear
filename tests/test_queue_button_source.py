import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


class TestQueueButtonSource(unittest.TestCase):
    def test_queue_button_reflects_session_selection(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")

        self.assertIn('function selectedSessionHasUnknownSend() {', source)
        self.assertIn('function selectedSessionIsOrphanRecovery() {', source)
        self.assertIn('function selectedSessionHasOrphanQueueRecovery() {', source)
        self.assertIn('function syncQueueSubmitState() {', source)
        self.assertIn('const queueControl = $("#queueBtn");', source)
        self.assertIn('const unknownSend = selectedSessionHasUnknownSend();', source)
        self.assertIn('const orphanQueueRecovery = selectedSessionHasOrphanQueueRecovery();', source)
        self.assertIn('queueControl.disabled = !!queueSubmitBusy || !selected || (unknownSend && !orphanQueueRecovery);', source)
        self.assertIn('Resolve the unknown send before queueing', source)
        self.assertIn('Review preserved queued recovery items', source)
        self.assertIn('missing session can only be reviewed', source)
        self.assertIn('queueControl.setAttribute("aria-label", queueLabel);', source)
        self.assertIn('syncQueueSubmitState();\n          syncSendButtonState();\n          diagBtn.disabled = !selected;', source)

    def test_commit_unknown_queue_items_are_visible_and_not_mutated(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")

        self.assertIn('commitUnknown: !!item.commit_unknown', source)
        self.assertIn('const commitUnknown = !!item.commitUnknown;', source)
        self.assertIn('if (commitUnknown) actions.appendChild(el("div", { class: "queueSendingTag warning", text: "Commit unknown" }));', source)
        self.assertIn('const locked = sending || commitUnknown || queueMutationLocks.has(itemId);', source)
        self.assertIn('const queueMoveCrossesBarrier = (fromIdx, toIdx) => {', source)
        self.assertIn('if (candidate && (candidate.sending || candidate.commitUnknown)) return true;', source)
        self.assertIn('up.disabled = locked || idx <= 0 || queueMoveCrossesBarrier(idx, idx - 1);', source)
        self.assertIn('down.disabled = locked || idx >= q.length - 1 || queueMoveCrossesBarrier(idx, idx + 1);', source)
        self.assertIn('del.disabled = sending || queueMutationLocks.has(itemId);', source)
        self.assertIn('if (res && res.commit_unknown) setToast("send status unknown; queued item needs review");', source)
        self.assertIn('if (selectedInfo && selectedInfo.orphan_recovery && Number(selectedInfo.queue_len || 0) > 0) {\n              showQueueViewer();\n              return;\n            }', source)
        self.assertLess(source.index('if (selectedInfo && selectedInfo.orphan_recovery && Number(selectedInfo.queue_len || 0) > 0)'), source.index('const raw = $("#msg") ? $("#msg").value : "";'))
        self.assertIn('const commitUnknown = Boolean(item && item.commitUnknown);', source)
        self.assertIn('Delete this queued item only after checking the transcript or terminal.', source)
        self.assertIn('body: { id: key, allow_commit_unknown: commitUnknown }', source)


if __name__ == "__main__":
    unittest.main()
