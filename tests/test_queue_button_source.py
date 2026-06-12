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
        self.assertIn('syncQueueSubmitState();\n          diagBtn.disabled = !selected;', source)


if __name__ == "__main__":
    unittest.main()
