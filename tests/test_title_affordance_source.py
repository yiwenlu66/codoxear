import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


class TestTitleAffordanceSource(unittest.TestCase):
    def test_title_edit_affordance_requires_selected_session(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")

        self.assertIn('const titleLabel = el("div", { id: "threadTitle", text: "No session selected" });', source)
        self.assertIn('titleLabel.style.cursor = "default";', source)
        self.assertIn('titleLabel.title = "No session selected";', source)
        self.assertIn('titleLabel.style.cursor = selected ? "pointer" : "default";', source)
        self.assertIn('titleLabel.title = selected ? "Edit conversation" : "No session selected";', source)
        self.assertIn('if (!selected) return;\n              openEditSession(selected);', source)


if __name__ == "__main__":
    unittest.main()
