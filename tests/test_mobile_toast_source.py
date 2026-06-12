import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"
APP_CSS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.css"


class TestMobileToastSource(unittest.TestCase):
    def test_toast_is_visible_feedback_on_mobile(self) -> None:
        js = APP_JS.read_text(encoding="utf-8")
        css = APP_CSS.read_text(encoding="utf-8")
        self.assertIn('id: "toast", role: "status", "aria-live": "polite"', js)
        mobile_start = css.index("@media (max-width: 520px)")
        mobile_block = css[mobile_start:]
        toast_start = mobile_block.index(".toast {")
        toast_end = mobile_block.index("}", toast_start)
        toast_block = mobile_block[toast_start:toast_end]
        self.assertNotIn("display: none", toast_block)
        self.assertIn("position: fixed;", toast_block)
        self.assertIn("bottom: calc(76px + env(safe-area-inset-bottom));", toast_block)
        self.assertIn("pointer-events: none;", toast_block)
        self.assertIn(".toast:empty", css)


if __name__ == "__main__":
    unittest.main()
