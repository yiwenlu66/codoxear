import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


class TestQueueButtonSource(unittest.TestCase):
    def test_queue_button_reflects_session_selection(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")

        self.assertIn('function syncQueueSubmitState() {', source)
        self.assertIn('const queueControl = $("#queueBtn");', source)
        self.assertIn('queueControl.disabled = !!queueSubmitBusy || !selected;', source)
        self.assertIn('const queueLabel = selected ? "Queued messages" : "Select a session to view queued messages";', source)
        self.assertIn('queueControl.setAttribute("aria-label", queueLabel);', source)
        self.assertIn('syncQueueSubmitState();\n          syncSendButtonState();\n          diagBtn.disabled = !selected;', source)

    def test_commit_unknown_queue_items_are_visible_and_not_mutated(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")

        self.assertIn('commitUnknown: !!item.commit_unknown', source)
        self.assertIn('const commitUnknown = !!item.commitUnknown;', source)
        self.assertIn('if (commitUnknown) actions.appendChild(el("div", { class: "queueSendingTag warning", text: "Commit unknown" }));', source)
        self.assertIn('const locked = sending || commitUnknown || queueMutationLocks.has(itemId);', source)
        self.assertIn('del.disabled = sending || queueMutationLocks.has(itemId);', source)


if __name__ == "__main__":
    unittest.main()
