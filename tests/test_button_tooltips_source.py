import re
import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


class TestButtonTooltipsSource(unittest.TestCase):
    def test_button_helper_uses_common_tooltip_fallbacks(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("function defaultButtonTooltip", source)
        self.assertIn('attrs["aria-label"]', source)
        self.assertIn('attrs["data-tooltip"]', source)
        self.assertIn("attrs.text", source)
        self.assertIn('if (tag === "button" && !n.getAttribute("title"))', source)

    def test_button_literals_include_some_label_source(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        missing: list[int] = []
        for match in re.finditer(r'el\("button",\s*\{', source):
            start = match.start()
            line = source.count("\n", 0, start) + 1
            snippet = source[start : start + 320]
            if not any(token in snippet for token in ('title:', '"title"', 'aria-label', 'text:', '"text"')):
                missing.append(line)
        self.assertEqual(missing, [], f"button definitions without tooltip label source: {missing}")


if __name__ == "__main__":
    unittest.main()
