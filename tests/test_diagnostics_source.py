import json
import subprocess
import textwrap
import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


def eval_diagnostics_copy_text() -> str:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("function diagnosticsCopyText(sessionId, rows) {")
    end = source.index("diagCopyBtn.onclick", start)
    snippet = source[start:end]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{}};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet + "\nglobalThis.__test = diagnosticsCopyText;\n")}, ctx);
        const out = ctx.__test("sid-1", [["Session", "sid-1"], ["", "ignored"], ["CWD", "/tmp/repo"], ["Empty", ""]]);
        process.stdout.write(out);
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.stdout


class TestDiagnosticsSource(unittest.TestCase):
    def test_diagnostics_render_is_bound_to_captured_session(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("async function showDiagViewer({ opener = null } = {}) {")
        end = source.index("function hideDiagViewer()", start)
        block = source[start:end]

        self.assertIn("const sid = selected;\n          if (!sid) return;", block)
        self.assertIn("api(`/api/sessions/${sid}/diagnostics`)", block)
        self.assertIn("if (selected !== sid) return;\n            diagStatus.textContent = \"\";", block)
        self.assertIn("catch (e) {\n            if (selected !== sid) return;", block)
        self.assertNotIn("/api/sessions/${selected}/diagnostics", block)

    def test_diagnostics_has_copy_details_action_for_rendered_rows(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('id: "diagCopyBtn"', source)
        self.assertIn('title: "Copy details"', source)
        self.assertIn('"aria-label": "Copy details"', source)
        self.assertIn('let diagCopyText = "";', source)
        self.assertIn('diagCopyBtn.disabled = true;', source)
        self.assertIn('el("div", { class: "actions" }, [diagCopyBtn, diagCloseBtn])', source)
        self.assertIn('function diagnosticsCopyText(sessionId, rows) {', source)
        self.assertIn('diagRows.push([cleanLabel, v]);', source)
        self.assertIn('diagCopyText = diagnosticsCopyText(sid, diagRows);', source)
        self.assertIn('diagCopyBtn.disabled = !diagCopyText;', source)
        self.assertIn('diagCopyText = "";\n            diagCopyBtn.disabled = true;\n            diagStatus.textContent = `error:', source)
        self.assertIn('await copyToClipboard(diagCopyText);', source)
        self.assertIn('setToast("Copied details");', source)

    def test_diagnostics_copy_formatter_uses_label_value_rows(self) -> None:
        out = eval_diagnostics_copy_text()
        self.assertEqual(
            out,
            "Codoxear session details\nSession: sid-1\nCWD: /tmp/repo\nEmpty: -",
        )


if __name__ == "__main__":
    unittest.main()
