import json
import re
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_CONVERSATION_COPY_JS = ROOT / "codoxear" / "static" / "app_conversation_copy.js"
INDEX_HTML = ROOT / "codoxear" / "static" / "index.html"


def eval_conversation_copy(events) -> dict:
    source = APP_CONVERSATION_COPY_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx);
        const helpers = ctx.window.CodoxearConversationCopy;
        process.stdout.write(JSON.stringify({{
          text: helpers.formatConversationForCopy({json.dumps(events)}),
          frozen: Object.isFrozen(helpers),
        }}));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


def run_app_conversation_copy_guard(setup_js: str = "") -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("const codoxearConversationCopy = window.CodoxearConversationCopy;")
    end = source.index("function normalizeAgentBackendName", start)
    guard_source = source[start:end]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        try {{
          vm.runInContext({json.dumps(setup_js + "\n" + guard_source)}, ctx);
          process.stdout.write(JSON.stringify({{ ok: true, message: "" }}));
        }} catch (err) {{
          process.stdout.write(JSON.stringify({{ ok: false, message: String(err && err.message || err) }}));
        }}
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


class TestFrontendConversationCopySource(unittest.TestCase):
    def test_index_loads_conversation_copy_before_app(self) -> None:
        source = INDEX_HTML.read_text(encoding="utf-8")
        self.assertIn('app_conversation_copy.js?v=__CODOXEAR_ASSET_VERSION__', source)
        self.assertLess(source.index('app_polling.js?v=__CODOXEAR_ASSET_VERSION__'), source.index('app_conversation_copy.js?v=__CODOXEAR_ASSET_VERSION__'))
        self.assertLess(source.index('app_conversation_copy.js?v=__CODOXEAR_ASSET_VERSION__'), source.index('app.js?v=__CODOXEAR_ASSET_VERSION__'))

    def test_app_js_requires_conversation_copy_helper_without_fallback(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        helper_source = APP_CONVERSATION_COPY_JS.read_text(encoding="utf-8")
        self.assertIn("const codoxearConversationCopy = window.CodoxearConversationCopy;", source)
        self.assertIn('throw new Error("Codoxear conversation-copy helpers failed to load")', source)
        self.assertIn('typeof codoxearConversationCopy.formatConversationForCopy !== "function"', source)
        self.assertIn("function formatConversationForCopy(events)", source)
        self.assertIn("return codoxearConversationCopy.formatConversationForCopy(events);", source)
        self.assertIn("window.CodoxearConversationCopy = Object.freeze({", helper_source)
        self.assertIn('parts.push(`## ${role}${when}\\n\\n${text}`);', helper_source)
        self.assertNotIn('parts.push(`## ${role}${when}\\n\\n${text}`);', source)

    def test_app_conversation_copy_guard_throws_for_missing_or_partial_helper(self) -> None:
        missing = run_app_conversation_copy_guard()
        self.assertEqual(missing, {"ok": False, "message": "Codoxear conversation-copy helpers failed to load"})
        partial = run_app_conversation_copy_guard("window.CodoxearConversationCopy = {};")
        self.assertEqual(partial, {"ok": False, "message": "Codoxear conversation-copy helpers failed to load"})
        complete = run_app_conversation_copy_guard("window.CodoxearConversationCopy = { formatConversationForCopy() {} };")
        self.assertEqual(complete, {"ok": True, "message": ""})

    def test_format_conversation_for_copy_preserves_existing_contract(self) -> None:
        result = eval_conversation_copy(
            [
                {"role": "system", "text": "ignored", "ts": 1},
                {"role": "user", "text": "  hello user  \n\n", "ts": 0},
                {"role": "assistant", "text": "assistant answer\t \n", "ts": "not-a-number"},
                {"role": "assistant", "text": "   ", "ts": 2},
                {"role": "user", "text": None, "ts": 3},
                None,
            ]
        )
        self.assertTrue(result["frozen"])
        self.assertRegex(
            result["text"],
            re.compile(r"^## User \(.+\)\n\n  hello user\n\n---\n\n## Assistant\n\nassistant answer$", re.DOTALL),
        )

    def test_format_conversation_for_copy_returns_empty_for_no_copyable_text(self) -> None:
        self.assertEqual(eval_conversation_copy({"events": []})["text"], "")
        self.assertEqual(
            eval_conversation_copy([
                {"role": "system", "text": "ignored"},
                {"role": "assistant", "text": "\n\t  "},
                {"role": "user", "text": None},
                {"role": "assistant", "text": 0},
                {"role": "user", "text": False},
            ])["text"],
            "",
        )


if __name__ == "__main__":
    unittest.main()
