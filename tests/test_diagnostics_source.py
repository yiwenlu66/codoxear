import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_SESSION_HELPERS_JS = ROOT / "codoxear" / "static" / "app_session_helpers.js"
APP_DIAGNOSTICS_JS = ROOT / "codoxear" / "static" / "app_diagnostics.js"


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
    def test_app_js_delegates_diag_rendering_to_controller_module(self) -> None:
        """The captured-session binding, stale guard, and error path now live in
        the CodoxearDiagnostics controller module (see
        test_frontend_diagnostics_module_source.py for the executable behavior).
        app.js keeps only DOM construction, the Details opener, and thin
        delegating wrappers."""
        source = APP_JS.read_text(encoding="utf-8")
        module = APP_DIAGNOSTICS_JS.read_text(encoding="utf-8")
        # app.js delegates to the controller instead of owning the show body.
        self.assertIn("const diagController = (function instantiateDiagnosticsController() {", source)
        self.assertIn("return diagController.show(opts);", source)
        self.assertIn("return diagController.hide(opts);", source)
        # The captured-session / stale / error guards now live in the module.
        self.assertIn("const sid = getSelected();", module)
        self.assertIn("if (!sid) return;", module)
        self.assertIn("api(`/api/sessions/${sid}/diagnostics`)", module)
        self.assertIn("const selectedInfo = getSessionInfo(sid) || null;", module)
        self.assertIn("if (sessionLaunchFailed(selectedInfo)) {", module)
        self.assertIn("diagCopyText = recoveryDetailsText(sid, selectedInfo);", module)
        self.assertIn("diagNewLikeSession = launchPresetFromSessionInfo(selectedInfo);", module)
        self.assertLess(module.index("if (sessionLaunchFailed(selectedInfo)) {"), module.index("api(`/api/sessions/${sid}/diagnostics`)"))
        self.assertIn("if (getSelected() !== sid) return;\n        renderLiveRows(sid, d);", module)
        self.assertIn("catch (e) {\n        if (getSelected() !== sid) return;", module)
        self.assertNotIn("/api/sessions/${selected}/diagnostics", module)
        # app.js must not carry the old inline rendering authority.
        self.assertNotIn("async function showDiagViewer({ opener = null } = {}) {", source)
        self.assertNotIn("const d = await api(`/api/sessions/${sid}/diagnostics`);", source)

    def test_diagnostics_provider_display_is_backend_aware(self) -> None:
        result = eval_diagnostics_helpers()
        self.assertEqual(result["piAbsent"], "-")
        self.assertEqual(result["piActual"], "anthropic")
        self.assertEqual(result["codexChoice"], "chatgpt")
        self.assertEqual(result["codexModelProviderFallback"], "openai")
        self.assertEqual(result["ccIgnored"], "-")
        self.assertEqual(result["nullRow"], "-")

    def test_diagnostics_has_copy_details_action_wired_to_controller(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        module = APP_DIAGNOSTICS_JS.read_text(encoding="utf-8")
        # DOM construction for the copy/new-like buttons stays in app.js.
        self.assertIn('id: "diagCopyBtn"', source)
        self.assertIn('title: "Copy details"', source)
        self.assertIn('"aria-label": "Copy details"', source)
        self.assertIn('el("div", { class: "actions" }, [diagNewLikeBtn, diagCopyBtn, diagCloseBtn])', source)
        self.assertIn('typeof codoxearSessionHelpers.diagnosticsProviderDisplay !== "function"', source)
        self.assertIn('typeof codoxearSessionHelpers.diagnosticsCopyText !== "function"', source)
        self.assertIn('function diagnosticsProviderDisplay(d) {\n        return codoxearSessionHelpers.diagnosticsProviderDisplay(d, sessionAgentBackend(d));\n      }', source)
        self.assertIn('function diagnosticsCopyText(sessionId, rows) {\n        return codoxearSessionHelpers.diagnosticsCopyText(sessionId, rows);\n      }', source)
        self.assertIn('function diagnosticsProviderDisplay(d, backend) {', APP_SESSION_HELPERS_JS.read_text(encoding="utf-8"))
        self.assertIn('function diagnosticsCopyText(sessionId, rows) {', APP_SESSION_HELPERS_JS.read_text(encoding="utf-8"))
        # The copy/new-like state and click behavior moved into the module.
        self.assertIn("let diagCopyText = \"\";", module)
        self.assertIn('setToast("Copied details");', module)
        self.assertIn('await copyToClipboard(diagCopyText);', module)
        self.assertIn('diagCopyBtn.disabled = !diagCopyText;', module)
        # app.js wires the buttons to the controller; it no longer owns the state.
        self.assertIn("diagCopyBtn.onclick = (e) => diagController.onCopyClick(e);", source)
        self.assertIn("diagNewLikeBtn.onclick = (e) => diagController.onNewLikeClick(e);", source)
        self.assertNotIn("let diagCopyText = \"\";", source)

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
