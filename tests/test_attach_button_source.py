import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


class TestAttachButtonSource(unittest.TestCase):
    def test_attach_button_reflects_session_selection(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")

        self.assertIn('attachBtn.disabled = true;', source)
        self.assertIn('"Select a session to attach a file"', source)
        self.assertIn('const attachControl = $("#attachBtn");', source)
        self.assertIn('attachControl.disabled = !selected;', source)
        self.assertIn('const attachLabel = selected ? `Attach file (max ${fmtBytes(ATTACH_UPLOAD_MAX_BYTES)})` : "Select a session to attach a file";', source)
        self.assertIn('attachControl.setAttribute("aria-label", attachLabel);', source)


if __name__ == "__main__":
    unittest.main()
