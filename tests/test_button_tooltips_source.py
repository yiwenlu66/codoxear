import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


class TestButtonTooltipsSource(unittest.TestCase):
    def test_button_helper_has_default_tooltip_fallback(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("function defaultButtonTooltip(attrs = {}, node = null)", source)
        self.assertIn('attrs["aria-label"]', source)
        self.assertIn('attrs["data-tooltip"]', source)
        self.assertIn("attrs.text", source)
        self.assertIn('if (tag === "button" && !n.getAttribute("title"))', source)
        self.assertIn('n.setAttribute("title", tooltip)', source)


if __name__ == "__main__":
    unittest.main()
