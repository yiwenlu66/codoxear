import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_SESSION_HELPERS_JS = ROOT / "codoxear" / "static" / "app_session_helpers.js"


def eval_diagnostics_helpers() -> dict:
    source = APP_SESSION_HELPERS_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx);
        const helpers = ctx.window.CodoxearSessionHelpers;
        const copyText = helpers.diagnosticsCopyText("sid-1", [["Session", "sid-1"], ["", "ignored"], ["CWD", "/tmp/repo"], ["Empty", ""]]);
        const copyTextWithInsertedSession = helpers.diagnosticsCopyText("sid-2", [["CWD", "/tmp/repo"]]);
        process.stdout.write(JSON.stringify({{
          piAbsent: helpers.diagnosticsProviderDisplay({{ model_provider: null, provider_choice: "openai-api" }}, "pi"),
          piActual: helpers.diagnosticsProviderDisplay({{ model_provider: "anthropic", provider_choice: "openai-api" }}, "pi"),
          codexChoice: helpers.diagnosticsProviderDisplay({{ model_provider: "openai", provider_choice: "chatgpt" }}, "codex"),
          codexModelProviderFallback: helpers.diagnosticsProviderDisplay({{ model_provider: "openai", provider_choice: "   " }}, "codex"),
          ccIgnored: helpers.diagnosticsProviderDisplay({{ model_provider: "anthropic", provider_choice: "anthropic" }}, "cc"),
          nullRow: helpers.diagnosticsProviderDisplay(null, "codex"),
          copyText,
          copyTextWithInsertedSession,
        }}));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


class TestDiagnosticsSource(unittest.TestCase):
    def test_diagnostics_render_is_bound_to_captured_session(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("async function showDiagViewer({ opener = null } = {}) {")
        end = source.index("function hideDiagViewer", start)
        block = source[start:end]

        self.assertIn("const sid = selected;\n          if (!sid) return;", block)
        self.assertIn("api(`/api/sessions/${sid}/diagnostics`)", block)
        self.assertIn("const selectedInfo = sessionIndex.get(sid) || null;", block)
        self.assertIn("if (sessionLaunchFailed(selectedInfo)) {", block)
        self.assertIn("diagCopyText = recoveryDetailsText(sid, selectedInfo);", block)
        self.assertIn("diagNewLikeSession = launchPresetFromSessionInfo(selectedInfo);", block)
        self.assertLess(block.index("if (sessionLaunchFailed(selectedInfo)) {"), block.index("api(`/api/sessions/${sid}/diagnostics`)"))
        self.assertIn("if (selected !== sid) return;\n            diagStatus.textContent = \"\";", block)
        self.assertIn("catch (e) {\n            if (selected !== sid) return;", block)
        self.assertNotIn("/api/sessions/${selected}/diagnostics", block)

    def test_diagnostics_provider_display_is_backend_aware(self) -> None:
        result = eval_diagnostics_helpers()
        self.assertEqual(result["piAbsent"], "-")
        self.assertEqual(result["piActual"], "anthropic")
        self.assertEqual(result["codexChoice"], "chatgpt")
        self.assertEqual(result["codexModelProviderFallback"], "openai")
        self.assertEqual(result["ccIgnored"], "-")
        self.assertEqual(result["nullRow"], "-")

    def test_diagnostics_has_copy_details_action_for_rendered_rows(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('id: "diagCopyBtn"', source)
        self.assertIn('title: "Copy details"', source)
        self.assertIn('"aria-label": "Copy details"', source)
        self.assertIn('let diagCopyText = "";', source)
        self.assertIn('diagCopyBtn.disabled = true;', source)
        self.assertIn('el("div", { class: "actions" }, [diagNewLikeBtn, diagCopyBtn, diagCloseBtn])', source)
        self.assertIn('typeof codoxearSessionHelpers.diagnosticsProviderDisplay !== "function"', source)
        self.assertIn('typeof codoxearSessionHelpers.diagnosticsCopyText !== "function"', source)
        self.assertIn('function diagnosticsProviderDisplay(d) {\n        return codoxearSessionHelpers.diagnosticsProviderDisplay(d, sessionAgentBackend(d));\n      }', source)
        self.assertIn('addRow("Provider", diagnosticsProviderDisplay(d));', source)
        self.assertIn('function diagnosticsCopyText(sessionId, rows) {\n        return codoxearSessionHelpers.diagnosticsCopyText(sessionId, rows);\n      }', source)
        self.assertIn('function diagnosticsProviderDisplay(d, backend) {', APP_SESSION_HELPERS_JS.read_text(encoding="utf-8"))
        self.assertIn('function diagnosticsCopyText(sessionId, rows) {', APP_SESSION_HELPERS_JS.read_text(encoding="utf-8"))
        self.assertIn('diagRows.push([cleanLabel, v]);', source)
        self.assertIn('diagCopyText = diagnosticsCopyText(sid, diagRows);', source)
        self.assertIn('diagCopyBtn.disabled = !diagCopyText;', source)
        self.assertIn('diagCopyText = "";\n            diagNewLikeSession = null;\n            diagNewLikeBtn.disabled = true;\n            diagCopyBtn.disabled = true;\n            diagStatus.textContent = `error:', source)
        self.assertIn('await copyToClipboard(diagCopyText);', source)
        self.assertIn('setToast("Copied details");', source)

    def test_diagnostics_copy_formatter_uses_label_value_rows(self) -> None:
        result = eval_diagnostics_helpers()
        self.assertEqual(
            result["copyText"],
            "Codoxear session details\nSession: sid-1\nCWD: /tmp/repo\nEmpty: -",
        )
        self.assertEqual(
            result["copyTextWithInsertedSession"],
            "Codoxear session details\nSession: sid-2\nCWD: /tmp/repo",
        )


if __name__ == "__main__":
    unittest.main()
