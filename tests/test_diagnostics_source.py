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

    def test_diagnostics_provider_display_is_backend_aware(self) -> None:
        result = eval_diagnostics_helpers()
        self.assertEqual(result["piAbsent"], "-")
        self.assertEqual(result["piActual"], "anthropic")
        self.assertEqual(result["codexChoice"], "chatgpt")
        self.assertEqual(result["codexModelProviderFallback"], "openai")
        self.assertEqual(result["ccIgnored"], "-")
        self.assertEqual(result["nullRow"], "-")


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
