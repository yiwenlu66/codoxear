import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


class TestDiagnosticsSource(unittest.TestCase):
    def test_diagnostics_render_is_bound_to_captured_session(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("async function showDiagViewer() {")
        end = source.index("function hideDiagViewer()", start)
        block = source[start:end]

        self.assertIn("const sid = selected;\n          if (!sid) return;", block)
        self.assertIn("api(`/api/sessions/${sid}/diagnostics`)", block)
        self.assertIn("if (selected !== sid) return;\n            diagStatus.textContent = \"\";", block)
        self.assertIn("catch (e) {\n            if (selected !== sid) return;", block)
        self.assertNotIn("/api/sessions/${selected}/diagnostics", block)


if __name__ == "__main__":
    unittest.main()
