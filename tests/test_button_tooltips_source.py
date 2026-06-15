import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_DISPLAY_JS = ROOT / "codoxear" / "static" / "app_display.js"


class TestButtonTooltipsSource(unittest.TestCase):
    def test_button_helper_has_default_tooltip_fallback(self) -> None:
        app_source = APP_JS.read_text(encoding="utf-8")
        display_source = APP_DISPLAY_JS.read_text(encoding="utf-8")
        self.assertIn("function defaultButtonTooltip(attrs = {}, node = null)", display_source)
        self.assertIn('attrs["aria-label"]', display_source)
        self.assertIn('attrs["data-tooltip"]', display_source)
        self.assertIn("attrs.text", display_source)
        self.assertIn('return codoxearDisplay.defaultButtonTooltip(attrs, node);', app_source)
        self.assertIn('if (tag === "button" && !n.getAttribute("title"))', app_source)
        self.assertIn('n.setAttribute("title", tooltip)', app_source)


if __name__ == "__main__":
    unittest.main()
