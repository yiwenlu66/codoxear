import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


class TestSendButtonSource(unittest.TestCase):
    def test_send_button_reflects_session_and_sending_state(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")

        self.assertIn('function syncSendButtonState() {', source)
        self.assertIn('const sendControl = $("#sendBtn");', source)
        self.assertIn('sendControl.disabled = !!sending || !selected;', source)
        self.assertIn('const sendLabel = selected ? "Send" : "Select a session to send";', source)
        self.assertIn('sendControl.setAttribute("aria-label", sendLabel);', source)
        self.assertIn('syncSendButtonState();\n          diagBtn.disabled = !selected;', source)
        self.assertIn('sending = true;\n          syncSendButtonState();', source)
        self.assertIn('sending = false;\n            syncSendButtonState();', source)
        self.assertIn('setToast("select a session first");', source)


if __name__ == "__main__":
    unittest.main()
